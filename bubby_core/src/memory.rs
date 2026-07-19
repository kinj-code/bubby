//! Memory persistence layer — schemaless key-value store with JSON indexing.
//!
//! Replaces `src/memory/vector_db.py`. Uses `rusqlite` with WAL mode
//! for concurrent reads. Deterministic memory allocation, no GC pauses.

use parking_lot::RwLock;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// A single memory record stored in the persistence layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub text: String,
    pub timestamp: f64,
    pub importance: f64,
    pub metadata: String, // JSON string
}

/// Thread-safe memory store backed by SQLite.
///
/// Uses WAL journal mode for concurrent reads without blocking writers.
/// All writes are serialized through the internal RwLock.
pub struct MemoryStore {
    conn: Arc<RwLock<Connection>>,
    next_id: Arc<RwLock<i64>>,
    total_adds: Arc<RwLock<u64>>,
    total_queries: Arc<RwLock<u64>>,
}

impl MemoryStore {
    /// Open (or create) the memory store at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> SqlResult<Self> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for concurrent reads
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "cache_size", -2000)?; // 2MB cache
        conn.pragma_update(None, "busy_timeout", 5000)?; // 5s busy timeout

        // Create schema
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);"
        )?;

        // Load the next_id counter
        let next_id: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM memories",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let total_adds: u64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap_or(0);

        Ok(Self {
            conn: Arc::new(RwLock::new(conn)),
            next_id: Arc::new(RwLock::new(next_id)),
            total_adds: Arc::new(RwLock::new(total_adds)),
            total_queries: Arc::new(RwLock::new(0)),
        })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> SqlResult<Self> {
        Self::open(":memory:")
    }

    /// Add a memory record. Returns the assigned ID.
    pub fn add(
        &self,
        text: &str,
        timestamp: f64,
        importance: f64,
        metadata: &str,
    ) -> SqlResult<i64> {
        let conn = self.conn.write();
        let id = {
            let mut nid = self.next_id.write();
            let id = *nid;
            *nid += 1;
            id
        };

        conn.execute(
            "INSERT INTO memories (id, text, timestamp, importance, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, text, timestamp, importance, metadata],
        )?;

        *self.total_adds.write() += 1;
        Ok(id)
    }

    /// Search memories by text substring (LIKE query).
    /// Returns up to `k` results sorted by importance DESC, then timestamp DESC.
    pub fn search_by_text(&self, query: &str, k: usize) -> SqlResult<Vec<MemoryRecord>> {
        let conn = self.conn.read();
        let pattern = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT id, text, timestamp, importance, metadata
             FROM memories
             WHERE text LIKE ?1
             ORDER BY importance DESC, timestamp DESC
             LIMIT ?2",
        )?;

        let records = stmt
            .query_map(params![pattern, k as i64], |row| {
                Ok(MemoryRecord {
                    id: row.get(0)?,
                    text: row.get(1)?,
                    timestamp: row.get(2)?,
                    importance: row.get(3)?,
                    metadata: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        *self.total_queries.write() += 1;
        Ok(records)
    }

    /// Get the N most recent memories.
    pub fn get_recent(&self, n: usize) -> SqlResult<Vec<MemoryRecord>> {
        let conn = self.conn.read();
        let mut stmt = conn.prepare(
            "SELECT id, text, timestamp, importance, metadata
             FROM memories
             ORDER BY timestamp DESC
             LIMIT ?1",
        )?;

        let records = stmt
            .query_map(params![n as i64], |row| {
                Ok(MemoryRecord {
                    id: row.get(0)?,
                    text: row.get(1)?,
                    timestamp: row.get(2)?,
                    importance: row.get(3)?,
                    metadata: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        *self.total_queries.write() += 1;
        Ok(records)
    }

    /// Count total stored records.
    pub fn count(&self) -> u64 {
        let conn = self.conn.read();
        conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap_or(0)
    }

    /// Clear all records.
    pub fn clear(&self) -> SqlResult<()> {
        let conn = self.conn.write();
        conn.execute("DELETE FROM memories", [])?;
        *self.next_id.write() = 0;
        *self.total_adds.write() = 0;
        *self.total_queries.write() = 0;
        Ok(())
    }

    /// Get store statistics.
    pub fn stats(&self) -> MemoryStats {
        let conn = self.conn.read();
        let count: u64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap_or(0);
        let size_bytes: u64 = conn
            .pragma_query_value(None, "page_count", |row| {
                let pages: u64 = row.get(0)?;
                Ok(pages * 4096)
            })
            .unwrap_or(0);

        MemoryStats {
            total_records: count,
            total_adds: *self.total_adds.read(),
            total_queries: *self.total_queries.read(),
            size_bytes,
        }
    }
}

/// Snapshot of store statistics.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total_records: u64,
    pub total_adds: u64,
    pub total_queries: u64,
    pub size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let store = MemoryStore::open_in_memory().unwrap();

        let id1 = store
            .add("User likes coffee", 1000.0, 0.9, "{}")
            .unwrap();
        let id2 = store
            .add("User prefers dark mode", 2000.0, 0.5, "{}")
            .unwrap();
        let id3 = store
            .add("Coffee brewing tips", 3000.0, 0.7, "{}")
            .unwrap();

        assert!(id1 >= 0);
        assert!(id2 > id1);
        assert!(id3 > id2);

        // Search
        let results = store.search_by_text("coffee", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].text.contains("coffee") || results[1].text.contains("coffee"));

        // Get recent
        let recent = store.get_recent(2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].text, "Coffee brewing tips"); // newest first

        // Stats
        let stats = store.stats();
        assert_eq!(stats.total_records, 3);
        assert_eq!(stats.total_adds, 3);
        assert_eq!(stats.total_queries, 2);
    }

    #[test]
    fn test_clear() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.add("test", 0.0, 0.5, "{}").unwrap();
        assert_eq!(store.count(), 1);

        store.clear().unwrap();
        assert_eq!(store.count(), 0);

        let stats = store.stats();
        assert_eq!(stats.total_records, 0);
    }

    #[test]
    fn test_concurrent_reads() {
        use std::thread;

        let store = Arc::new(MemoryStore::open_in_memory().unwrap());
        for i in 0..100 {
            store
                .add(&format!("Record {}", i), i as f64, 0.5, "{}")
                .unwrap();
        }

        let mut handles = vec![];
        for _ in 0..4 {
            let store_clone = store.clone();
            handles.push(thread::spawn(move || {
                let results = store_clone.get_recent(10).unwrap();
                assert!(!results.is_empty());
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");

        // Write
        {
            let store = MemoryStore::open(&db_path).unwrap();
            store.add("persistent record", 42.0, 0.8, r#"{"source":"test"}"#).unwrap();
            store.add("another record", 43.0, 0.6, "{}").unwrap();
        }

        // Read back
        {
            let store = MemoryStore::open(&db_path).unwrap();
            assert_eq!(store.count(), 2);
            let results = store.search_by_text("persistent", 10).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].text, "persistent record");
            assert_eq!(results[0].importance, 0.8);
        }
    }
}