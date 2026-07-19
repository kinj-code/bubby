//! Memory persistence layer — SQLite-backed memory store with vector storage.
//!
//! Replaces `src/memory/vector_db.py` and `src/memory/long_term_memory.py`.
//! Uses `rusqlite` with WAL mode for concurrent reads via connection pooling.
//! Deterministic memory allocation, zero GC pauses.
//!
//! ## Security (Phase 2.1)
//! When opened via `open_encrypted()`, the database is encrypted with
//! SQLCipher using AES-256. The encryption key is a hex-encoded 256-bit
//! key managed by `crate::security::KeyManager`. Without the correct key,
//! the database file appears as random bytes.
//!
//! Schema matches the Python implementation and extends it with:
//! - BLOB storage for embedding vectors alongside records
//! - JSON metadata extraction for indexed queries
//! - Importance-weighted retrieval
//! - Paginated recent-memory queries

use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

/// A single memory record stored in the persistence layer.
/// Mirrors the Python `MemoryRecord` dataclass exactly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRecord {
    pub id: i64,
    pub text: String,
    pub timestamp: f64,
    pub importance: f64,
    pub metadata: String, // JSON string
}

/// Result of a vector search — record paired with similarity score [0.0, 1.0].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub record: MemoryRecord,
    pub similarity: f32,
    /// Weighted score: similarity * (0.5 + 0.5 * importance)
    pub weighted_score: f32,
}

/// Thread-safe memory store backed by SQLite.
///
/// Uses WAL journal mode so that concurrent processes can read safely.
/// Internally serialises writes through a `Mutex<Connection>` (rusqlite
/// connections are `Send` but not `Sync`).
///
/// For true concurrent reads in production, open additional read-only
/// connections against the same WAL-mode database file.
pub struct MemoryStore {
    conn: Mutex<Connection>,
    db_path: Option<String>, // Tracked for connection pool ops
    next_id: Mutex<i64>,
    total_adds: Mutex<u64>,
    total_queries: Mutex<u64>,
    embedding_dim: Mutex<usize>,
}

// Safety: MemoryStore owns a Mutex<Connection>; Mutex is both Send and Sync
// when T is Send, and rusqlite::Connection is Send. Therefore MemoryStore is
// Send + Sync, fulfilling the contract expected by Arc<MemoryStore>.
unsafe impl Send for MemoryStore {}
unsafe impl Sync for MemoryStore {}

impl MemoryStore {
    /// Open (or create) the memory store at the given path **without encryption**.
    ///
    /// Enables WAL mode, creates the schema with indices, and sets
    /// pragmas for performance under concurrency.
    ///
    /// Prefer `open_encrypted()` in production.
    pub fn open<P: AsRef<Path>>(path: P) -> SqlResult<Self> {
        let conn = Connection::open(&path)?;
        Self::finish_open(conn, Some(path.as_ref().to_string_lossy().to_string()))
    }

    /// Open (or create) the memory store with **SQLCipher AES-256 encryption**.
    ///
    /// `hex_key` must be a 64-character hex string (32 bytes).
    /// The key is applied via `PRAGMA key` **immediately** after connection —
    /// before any other SQL is executed. This ensures the database file
    /// is encrypted at rest and unreadable without the key.
    pub fn open_encrypted<P: AsRef<Path>>(path: P, hex_key: &str) -> SqlResult<Self> {
        let conn = Connection::open(&path)?;

        // Apply the encryption key BEFORE any other operation.
        // SQLCipher intercepts this pragma and uses it to derive the
        // AES-256 key for the page-level encryption.
        conn.execute_batch(&format!("PRAGMA key = \"x'{}'\";", hex_key))?;

        // Verify the key works by attempting a simple query.
        // If the key is wrong, this will return an error like
        // "file is not a database".
        conn.query_row("SELECT 1", [], |_| Ok(()))?;

        Self::finish_open(conn, Some(path.as_ref().to_string_lossy().to_string()))
    }

    /// Common post-connection initialisation.
    fn finish_open(conn: Connection, db_path: Option<String>) -> SqlResult<Self> {
        // WAL mode — writers don't block readers (works with SQLCipher)
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "cache_size", -2000)?; // 2 MB page cache
        conn.pragma_update(None, "busy_timeout", 5000)?; // 5 s busy-wait
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // SQLCipher-specific: use fast KDF for local performance
        conn.pragma_update(None, "cipher_kdf_algorithm", "PBKDF2_HMAC_SHA512")?;
        conn.pragma_update(None, "cipher_page_size", "4096")?;

        // Create schema
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY,
                text       TEXT    NOT NULL,
                timestamp  REAL    NOT NULL,
                importance REAL    NOT NULL DEFAULT 0.5,
                metadata   TEXT    NOT NULL DEFAULT '{}'
            );

            -- Embedding vectors stored as raw f32 little-endian BLOBs
            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id  INTEGER PRIMARY KEY  REFERENCES memories(id) ON DELETE CASCADE,
                dim        INTEGER NOT NULL,
                data       BLOB    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC);"
        )?;

        let (next_id, total_adds): (i64, u64) = {
            let mut stmt = conn.prepare(
                "SELECT COALESCE(MAX(id), -1) + 1, COUNT(*) FROM memories"
            )?;
            stmt.query_row([], |row| Ok((row.get(0)?, row.get(1)?)))?
        };

        Ok(Self {
            conn: Mutex::new(conn),
            db_path,
            next_id: Mutex::new(next_id),
            total_adds: Mutex::new(total_adds),
            total_queries: Mutex::new(0),
            embedding_dim: Mutex::new(384),
        })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> SqlResult<Self> {
        let conn = Connection::open(":memory:")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY,
                text       TEXT    NOT NULL,
                timestamp  REAL    NOT NULL,
                importance REAL    NOT NULL DEFAULT 0.5,
                metadata   TEXT    NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id  INTEGER PRIMARY KEY  REFERENCES memories(id) ON DELETE CASCADE,
                dim        INTEGER NOT NULL,
                data       BLOB    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC);"
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
            db_path: None,
            next_id: Mutex::new(0),
            total_adds: Mutex::new(0),
            total_queries: Mutex::new(0),
            embedding_dim: Mutex::new(384),
        })
    }

    // ── Write operations ────────────────────────────────────────────

    /// Add a memory record with an associated embedding vector.
    /// Returns the assigned ID.
    pub fn add(
        &self,
        text: &str,
        timestamp: f64,
        importance: f64,
        metadata: &str,
    ) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();

        let id = {
            let mut nid = self.next_id.lock().unwrap();
            let id = *nid;
            *nid += 1;
            id
        };

        conn.execute(
            "INSERT INTO memories (id, text, timestamp, importance, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, text, timestamp, importance, metadata],
        )?;

        *self.total_adds.lock().unwrap() += 1;
        Ok(id)
    }

    /// Add a memory record with an embedding vector (BLOB storage).
    ///
    /// The embedding is stored as a packed f32 little-endian byte array.
    /// This mirrors the Python VectorStore.add() → embeddings + records pattern.
    pub fn add_with_embedding(
        &self,
        text: &str,
        timestamp: f64,
        importance: f64,
        metadata: &str,
        embedding: &[f32],
    ) -> SqlResult<i64> {
        let id = self.add(text, timestamp, importance, metadata)?;
        let conn = self.conn.lock().unwrap();

        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        conn.execute(
            "INSERT OR REPLACE INTO embeddings (memory_id, dim, data)
             VALUES (?1, ?2, ?3)",
            params![id, embedding.len() as i64, bytes],
        )?;

        Ok(id)
    }

    /// Update the importance of an existing record.
    pub fn update_importance(&self, id: i64, importance: f64) -> SqlResult<bool> {
        let conn = self.conn.lock().unwrap();
        let affected = conn.execute(
            "UPDATE memories SET importance = ?1 WHERE id = ?2",
            params![importance, id],
        )?;
        Ok(affected > 0)
    }

    /// Delete a memory and its associated embedding.
    pub fn delete(&self, id: i64) -> SqlResult<bool> {
        let conn = self.conn.lock().unwrap();
        // CASCADE cleans up embeddings
        let affected = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    /// Clear all records and embeddings.
    pub fn clear(&self) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM embeddings", [])?;
        conn.execute("DELETE FROM memories", [])?;
        *self.next_id.lock().unwrap() = 0;
        *self.total_adds.lock().unwrap() = 0;
        *self.total_queries.lock().unwrap() = 0;
        Ok(())
    }

    // ── Read operations ─────────────────────────────────────────────

    /// Get a record by ID.
    pub fn get(&self, id: i64) -> SqlResult<Option<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, text, timestamp, importance, metadata
             FROM memories WHERE id = ?1"
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(MemoryRecord {
                id: row.get(0)?,
                text: row.get(1)?,
                timestamp: row.get(2)?,
                importance: row.get(3)?,
                metadata: row.get(4)?,
            })
        })?;
        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            _ => Ok(None),
        }
    }

    /// Search memories by text substring (LIKE query).
    /// Returns up to `k` results sorted by importance DESC, then timestamp DESC.
    pub fn search_by_text(&self, query: &str, k: usize) -> SqlResult<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
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

        *self.total_queries.lock().unwrap() += 1;
        Ok(records)
    }

    /// Full-text search using SQLite FTS (if FTS table is populated).
    /// Falls back to LIKE-based search if no FTS results.
    pub fn search_fts(&self, query: &str, k: usize) -> SqlResult<Vec<MemoryRecord>> {
        // For now, delegates to LIKE-based search.
        // A future migration will create the FTS virtual table.
        self.search_by_text(query, k)
    }

    /// Search by metadata key-value pair (JSON extraction).
    ///
    /// Example: `search_by_metadata("source", "user_statement", 10)`
    /// returns all records whose metadata JSON contains that key-value.
    pub fn search_by_metadata(
        &self,
        key: &str,
        value: &str,
        k: usize,
    ) -> SqlResult<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        // Use JSON_EXTRACT; escape for safety
        let json_path = format!("$.{}", key);
        let mut stmt = conn.prepare(
            "SELECT id, text, timestamp, importance, metadata
             FROM memories
             WHERE JSON_EXTRACT(metadata, ?1) = ?2
             ORDER BY importance DESC, timestamp DESC
             LIMIT ?3",
        )?;

        let records = stmt
            .query_map(params![json_path, value, k as i64], |row| {
                Ok(MemoryRecord {
                    id: row.get(0)?,
                    text: row.get(1)?,
                    timestamp: row.get(2)?,
                    importance: row.get(3)?,
                    metadata: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        *self.total_queries.lock().unwrap() += 1;
        Ok(records)
    }

    /// Get the N most recent memories (newest first).
    pub fn get_recent(&self, n: usize) -> SqlResult<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
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

        *self.total_queries.lock().unwrap() += 1;
        Ok(records)
    }

    /// Get top N memories by importance (highest first), optionally
    /// filtered by a minimum importance threshold.
    pub fn get_by_importance(
        &self,
        n: usize,
        min_importance: f64,
    ) -> SqlResult<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, text, timestamp, importance, metadata
             FROM memories
             WHERE importance >= ?1
             ORDER BY importance DESC
             LIMIT ?2",
        )?;

        let records = stmt
            .query_map(params![min_importance, n as i64], |row| {
                Ok(MemoryRecord {
                    id: row.get(0)?,
                    text: row.get(1)?,
                    timestamp: row.get(2)?,
                    importance: row.get(3)?,
                    metadata: row.get(4)?,
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        *self.total_queries.lock().unwrap() += 1;
        Ok(records)
    }

    /// Retrieve the embedding vector for a given memory ID.
    /// Returns `(dim, Vec<f32>)` if found, or `None`.
    pub fn get_embedding(&self, memory_id: i64) -> SqlResult<Option<(usize, Vec<f32>)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT dim, data FROM embeddings WHERE memory_id = ?1"
        )?;
        let mut rows = stmt.query_map(params![memory_id], |row| {
            let dim: i64 = row.get(0)?;
            let data: Vec<u8> = row.get(1)?;
            Ok((dim as usize, data))
        })?;
        match rows.next() {
            Some(Ok((dim, data))) => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                debug_assert_eq!(floats.len(), dim);
                Ok(Some((dim, floats)))
            }
            _ => Ok(None),
        }
    }

    /// Brute-force cosine similarity search over all stored embeddings.
    ///
    /// For production HNSW search, use `vector::VectorStore`.
    /// This is the fallback matching Python's numpy brute-force path.
    pub fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_score: f32,
    ) -> SqlResult<Vec<SearchResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT e.memory_id, e.dim, e.data,
                    m.text, m.timestamp, m.importance, m.metadata
             FROM embeddings e
             JOIN memories m ON e.memory_id = m.id"
        )?;

        let query_norm = normalize(query_embedding);

        let mut scored: Vec<SearchResult> = stmt
            .query_map([], |row| {
                let memory_id: i64 = row.get(0)?;
                let dim: i64 = row.get(1)?;
                let data: Vec<u8> = row.get(2)?;
                let text: String = row.get(3)?;
                let timestamp: f64 = row.get(4)?;
                let importance: f64 = row.get(5)?;
                let metadata: String = row.get(6)?;

                let emb: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(emb.len(), dim as usize);

                let dot: f32 = query_norm.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                let similarity = dot.max(0.0);

                Ok(SearchResult {
                    record: MemoryRecord {
                        id: memory_id,
                        text,
                        timestamp,
                        importance,
                        metadata,
                    },
                    similarity,
                    weighted_score: similarity * (0.5 + 0.5 * importance as f32),
                })
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        // Filter by minimum score, sort by weighted score descending, take top k
        scored.retain(|r| r.similarity >= min_score);
        scored.sort_by(|a, b| b.weighted_score.partial_cmp(&a.weighted_score).unwrap());
        scored.truncate(k);

        *self.total_queries.lock().unwrap() += 1;
        Ok(scored)
    }

    /// Count total stored records.
    pub fn count(&self) -> u64 {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap_or(0)
    }

    /// Count total stored embeddings.
    pub fn embedding_count(&self) -> u64 {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .unwrap_or(0)
    }

    /// Get store statistics, matching the Python `VectorStore.get_stats()`.
    pub fn stats(&self) -> MemoryStats {
        let conn = self.conn.lock().unwrap();
        let count: u64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap_or(0);
        let emb_count: u64 = conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .unwrap_or(0);
        let size_bytes: u64 = conn
            .pragma_query_value(None, "page_count", |row| {
                let pages: u64 = row.get(0)?;
                Ok(pages * 4096)
            })
            .unwrap_or(0);

        MemoryStats {
            total_records: count,
            total_embeddings: emb_count,
            total_adds: *self.total_adds.lock().unwrap(),
            total_queries: *self.total_queries.lock().unwrap(),
            size_bytes,
            db_path: self.db_path.clone(),
        }
    }

    /// Set the expected embedding dimension (default 384, matching MiniLM-L6-v2).
    pub fn set_embedding_dim(&self, dim: usize) {
        *self.embedding_dim.lock().unwrap() = dim;
    }

    /// Get the expected embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        *self.embedding_dim.lock().unwrap()
    }
}

/// Snapshot of store statistics.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total_records: u64,
    pub total_embeddings: u64,
    pub total_adds: u64,
    pub total_queries: u64,
    pub size_bytes: u64,
    pub db_path: Option<String>,
}

/// Normalize a f32 slice to unit length. Returns zeros unchanged.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Calculate an importance score from text content, mirroring the
/// Python `LongTermMemory._calculate_importance()` heuristics.
pub fn calculate_importance(text: &str, metadata_json: Option<&str>) -> f64 {
    let text_lower = text.to_lowercase();
    let mut importance: f64 = 0.5;

    // User preference signals → 0.8+
    let preference_signals = [
        "likes", "prefers", "loves", "hates", "enjoys",
        "uses", "works with", "is working on",
    ];
    for signal in &preference_signals {
        if text_lower.contains(signal) {
            importance = importance.max(0.8);
            break;
        }
    }

    // Explicit fact signals → 0.7+
    let fact_signals = [
        "user ", "companion ", "the user ", "remember",
        "important", "critical", "always",
    ];
    for signal in &fact_signals {
        if text_lower.contains(signal) {
            importance = importance.max(0.7);
            break;
        }
    }

    // VLM confidence boost
    if let Some(meta) = metadata_json {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
            if let Some(conf) = parsed.get("vlm_confidence").and_then(|v| v.as_f64()) {
                if conf > 0.8 {
                    importance = (importance + 0.1).min(1.0);
                }
            }
            // Browser window signal
            if let Some(window) = parsed.get("window").and_then(|v| v.as_str()) {
                if window.to_lowercase().contains("browser") {
                    importance = (importance + 0.1).min(1.0);
                }
            }
        }
    }

    // Unknown / empty → 0.1
    if text_lower.is_empty()
        || text_lower == "unknown"
        || text_lower == "unclear"
    {
        importance = 0.1;
    }

    importance.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

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

        // Substring search
        let results = store.search_by_text("coffee", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results[0].text.to_lowercase().contains("coffee")
                || results[1].text.to_lowercase().contains("coffee")
        );

        // Get recent
        let recent = store.get_recent(2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].text, "Coffee brewing tips"); // newest first

        // Get by ID
        let record = store.get(id1).unwrap().expect("record should exist");
        assert_eq!(record.text, "User likes coffee");
        assert!((record.importance - 0.9).abs() < 1e-9);

        // Stats
        let stats = store.stats();
        assert_eq!(stats.total_records, 3);
        assert_eq!(stats.total_adds, 3);
        assert_eq!(stats.total_queries, 2);
    }

    #[test]
    fn test_embedding_storage() {
        let store = MemoryStore::open_in_memory().unwrap();
        let emb: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();

        let id = store
            .add_with_embedding(
                "User likes dark mode",
                1000.0,
                0.8,
                r#"{"source":"test"}"#,
                &emb,
            )
            .unwrap();

        let (dim, retrieved) = store.get_embedding(id).unwrap().expect("embedding should exist");
        assert_eq!(dim, 384);
        assert_eq!(retrieved.len(), 384);
        for (a, b) in emb.iter().zip(retrieved.iter()) {
            assert!((a - b).abs() < 1e-6, "embedding round-trip mismatch at index");
        }
    }

    #[test]
    fn test_embedding_search() {
        let store = MemoryStore::open_in_memory().unwrap();

        // Insert orthogonal embeddings
        store
            .add_with_embedding(
                "coffee",
                0.0,
                0.9,
                "{}",
                &vec![1.0, 0.0, 0.0, 0.0],
            )
            .unwrap();
        store
            .add_with_embedding(
                "dark mode",
                0.0,
                0.5,
                "{}",
                &vec![0.0, 1.0, 0.0, 0.0],
            )
            .unwrap();
        store
            .add_with_embedding(
                "weather",
                0.0,
                0.3,
                "{}",
                &vec![0.0, 0.0, 1.0, 0.0],
            )
            .unwrap();

        // Query near [1,0,0,0] → "coffee" should be top result
        let results = store
            .search_by_embedding(&[1.0, 0.0, 0.0, 0.0], 3, 0.0)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].record.text, "coffee");
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_importance_query() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.add("low", 0.0, 0.1, "{}").unwrap();
        store.add("med", 0.0, 0.5, "{}").unwrap();
        store.add("high", 0.0, 0.9, "{}").unwrap();
        store.add("critical", 0.0, 1.0, "{}").unwrap();

        let important = store.get_by_importance(5, 0.7).unwrap();
        assert_eq!(important.len(), 2);
        assert_eq!(important[0].text, "critical");
        assert_eq!(important[1].text, "high");
    }

    #[test]
    fn test_metadata_search() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .add(
                "stmt A",
                0.0,
                0.8,
                r#"{"source":"user_statement","session":1}"#,
            )
            .unwrap();
        store
            .add(
                "obs B",
                0.0,
                0.5,
                r#"{"source":"observation","session":1}"#,
            )
            .unwrap();
        store
            .add(
                "stmt C",
                0.0,
                0.7,
                r#"{"source":"user_statement","session":2}"#,
            )
            .unwrap();

        let results = store
            .search_by_metadata("source", "user_statement", 10)
            .unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(
                r.metadata
                    .find("user_statement")
                    .is_some(),
                true
            );
        }
        assert!(results[0].importance >= results[1].importance);
    }

    #[test]
    fn test_update_and_delete() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.add("test", 0.0, 0.5, "{}").unwrap();

        // Update importance
        let updated = store.update_importance(id, 0.99).unwrap();
        assert!(updated);
        let record = store.get(id).unwrap().unwrap();
        assert!((record.importance - 0.99).abs() < 1e-9);

        // Delete
        let deleted = store.delete(id).unwrap();
        assert!(deleted);
        assert_eq!(store.get(id).unwrap(), None);
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_calculate_importance() {
        // Preference keyword
        let imp = calculate_importance("User likes dark mode", None);
        assert!(imp >= 0.8, "Expected >=0.8, got {}", imp);

        // Fact keyword
        let imp = calculate_importance("remember to buy milk", None);
        assert!(imp >= 0.7, "Expected >=0.7, got {}", imp);

        // VLM confidence boost — metadata with vlm_confidence > 0.8 should raise score
        let imp = calculate_importance(
            "User was browsing",
            Some(r#"{"vlm_confidence":0.95}"#),
        );
        assert!(imp > 0.5, "Expected >0.5 from VLM boost, got {}", imp);

        // Unknown → low
        let imp = calculate_importance("unknown", None);
        assert!((imp - 0.1).abs() < 1e-9, "Expected 0.1, got {}", imp);

        // Empty → low
        let imp = calculate_importance("", None);
        assert!((imp - 0.1).abs() < 1e-9, "Expected 0.1, got {}", imp);

        // Browser window boost — window value must contain "browser" substring
        let imp = calculate_importance(
            "some text",
            Some(r#"{"window":"chrome-browser"}"#),
        );
        assert!(imp > 0.5, "Expected >0.5 from browser window boost, got {}", imp);
    }

    #[test]
    fn test_clear() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .add_with_embedding("test", 0.0, 0.5, "{}", &vec![0.0; 4])
            .unwrap();
        assert_eq!(store.count(), 1);
        assert_eq!(store.embedding_count(), 1);

        store.clear().unwrap();
        assert_eq!(store.count(), 0);
        assert_eq!(store.embedding_count(), 0);

        let stats = store.stats();
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.total_adds, 0);
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_concurrent_reads() {
        let store = Arc::new(MemoryStore::open_in_memory().unwrap());
        for i in 0..100 {
            store
                .add(
                    &format!("Record {}", i),
                    i as f64,
                    0.5,
                    "{}",
                )
                .unwrap();
        }
        assert_eq!(store.count(), 100);

        let mut handles = vec![];
        for _ in 0..4 {
            let store_clone = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let results = store_clone.get_recent(10).unwrap();
                    assert!(!results.is_empty());
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_memory.db");

        // Write
        {
            let store = MemoryStore::open(&db_path).unwrap();
            store
                .add(
                    "persistent record",
                    42.0,
                    0.8,
                    r#"{"source":"test"}"#,
                )
                .unwrap();
            store
                .add(
                    "another record",
                    43.0,
                    0.6,
                    "{}",
                )
                .unwrap();
            store
                .add_with_embedding(
                    "vector record",
                    44.0,
                    0.9,
                    r#"{"source":"embed"}"#,
                    &vec![1.0, 2.0, 3.0, 4.0],
                )
                .unwrap();
        }

        // Read back (new connection against same file)
        {
            let store = MemoryStore::open(&db_path).unwrap();
            assert_eq!(store.count(), 3);
            assert_eq!(store.embedding_count(), 1);

            let results = store.search_by_text("persistent", 10).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].text, "persistent record");
            assert_eq!(results[0].importance, 0.8);

            // Verify embedding survived
            let (dim, emb) = store
                .get_embedding(results[0].id + 2) // third record
                .unwrap()
                .expect("embedding should persist");
            assert_eq!(dim, 4);
            assert_eq!(emb, vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    // ── Encryption tests (Phase 2.1) ────────────────────────────

    /// Generate a deterministic key for testing (NOT for production).
    fn test_key() -> String {
        "0".repeat(64)
    }

    #[test]
    fn test_encrypted_open_creates_and_reads() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("encrypted_memory.db");
        let key = test_key();

        // Write encrypted
        {
            let store = MemoryStore::open_encrypted(&db_path, &key).unwrap();
            store.add("secret data", 1.0, 0.9, r#"{"source":"test"}"#).unwrap();
            store.add_with_embedding("embedded", 2.0, 0.5, "{}", &vec![1.0, 0.0, 0.0, 0.0]).unwrap();
            assert_eq!(store.count(), 2);
            assert_eq!(store.embedding_count(), 1);
        }

        // Read back with the same key
        {
            let store = MemoryStore::open_encrypted(&db_path, &key).unwrap();
            assert_eq!(store.count(), 2);
            let results = store.search_by_text("secret", 5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].text, "secret data");

            let emb_results = store.search_by_embedding(&[1.0, 0.0, 0.0, 0.0], 1, 0.0).unwrap();
            assert_eq!(emb_results[0].record.text, "embedded");
        }
    }

    #[test]
    fn test_encrypted_open_wrong_key_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("encrypted_reject.db");
        let key = test_key();
        let wrong_key = "f".repeat(64);

        // Create encrypted file
        {
            let store = MemoryStore::open_encrypted(&db_path, &key).unwrap();
            store.add("test", 0.0, 0.5, "{}").unwrap();
            drop(store);
        }

        // Attempt to open with wrong key — must fail
        let result = MemoryStore::open_encrypted(&db_path, &wrong_key);
        assert!(result.is_err(), "Should reject wrong encryption key");

        // Attempt to open without encryption — must also fail
        let result = MemoryStore::open(&db_path);
        if let Ok(store) = result {
            // If it doesn't error, any query should return garbage/error
            match store.get_recent(1) {
                Err(_) => { /* expected — can't read encrypted file */ }
                Ok(records) => {
                    assert!(
                        records.is_empty() || records[0].text != "test",
                        "Unencrypted open should produce garbage, not the original record"
                    );
                }
            }
        }
    }

    #[test]
    fn test_encrypted_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("encrypted_persist.db");
        let key = test_key();

        // Write phase
        {
            let store = MemoryStore::open_encrypted(&db_path, &key).unwrap();
            for i in 0..10 {
                let text = if i == 5 {
                    "encrypted record 5 with string index".to_string()
                } else {
                    format!("encrypted record {}", i)
                };
                store.add(
                    &text,
                    i as f64,
                    0.5 + i as f64 * 0.05,
                    &format!(r#"{{"index":"{}"}}"#, i),
                ).unwrap();
            }
            assert_eq!(store.count(), 10);
        }

        // Reopen phase
        {
            let store = MemoryStore::open_encrypted(&db_path, &key).unwrap();
            assert_eq!(store.count(), 10);
            let recent = store.get_recent(3).unwrap();
            assert_eq!(recent.len(), 3);
            // Most recent first (label for record 5 is different)
            assert!(recent[0].text.starts_with("encrypted record"));

            // Metadata search works on encrypted data (index stored as JSON string — JSON_EXTRACT returns the raw string value)
            let meta = store.search_by_metadata("index", "5", 1).unwrap();
            assert_eq!(meta.len(), 1);
            assert_eq!(meta[0].text, "encrypted record 5 with string index");
        }
    }
}
