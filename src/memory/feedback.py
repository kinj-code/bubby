"""RAG Feedback Engine: Tracks retrieval quality and self-corrects.

Records per-chunk utility scores, detects weak document indexes,
and triggers automatic re-indexing when retrieval quality degrades.

RAM: ~2MB (sqlite3 + rolling scores).
"""

import logging
import sqlite3
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "knowledge" / "feedback.db"


@dataclass
class RetrievalFeedback:
    """A single feedback record for a RAG retrieval."""
    query_text: str
    source_file: str
    chunk_index: int
    similarity_score: float
    was_helpful: Optional[bool] = None
    user_rating: int = 0
    timestamp: float = field(default_factory=time.time)
    reindex_triggered: bool = False


class FeedbackEngine:
    """Tracks RAG retrieval quality and triggers self-correction."""

    REINDEX_THRESHOLD = 0.35
    MIN_FEEDBACK_FOR_DECISION = 3
    WEAK_SIMILARITY_THRESHOLD = 0.20

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._feedback_count = 0
        logger.info(f"FeedbackEngine initialized (db={self._db_path})")

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                source_file TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                similarity_score REAL NOT NULL DEFAULT 0.0,
                was_helpful INTEGER,
                user_rating INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                reindex_triggered INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS source_file_stats (
                source_file TEXT PRIMARY KEY,
                total_queries INTEGER DEFAULT 0,
                helpful_count INTEGER DEFAULT 0,
                avg_similarity REAL DEFAULT 0.0,
                last_query_time REAL,
                reindex_count INTEGER DEFAULT 0,
                needs_reindex INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_source ON retrieval_feedback(source_file);
            CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON retrieval_feedback(timestamp);
        """)
        self._conn.commit()

    def record_query(
        self, query_text: str, source_file: str,
        chunk_index: int, similarity_score: float,
    ) -> int:
        """Record a retrieval event. Returns feedback record ID."""
        cursor = self._conn.execute(
            """INSERT INTO retrieval_feedback 
               (query_text, source_file, chunk_index, similarity_score, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (query_text[:500], source_file, chunk_index, similarity_score, time.time())
        )
        self._conn.commit()
        self._feedback_count += 1
        # Note: source_file_stats is updated in record_utility, not here
        return cursor.lastrowid

    def record_utility(self, feedback_id: int, was_helpful: bool) -> None:
        """Record whether a retrieval was helpful."""
        self._conn.execute(
            "UPDATE retrieval_feedback SET was_helpful = ? WHERE id = ?",
            (1 if was_helpful else 0, feedback_id)
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT source_file, similarity_score FROM retrieval_feedback WHERE id = ?",
            (feedback_id,)
        ).fetchone()
        if row:
            self._update_source_stats(row["source_file"], row["similarity_score"], was_helpful)

    def _update_source_stats(
        self, source_file: str, similarity_score: float, helpful: Optional[bool],
    ) -> None:
        """Update aggregated stats — called ONCE per query (from record_utility)."""
        row = self._conn.execute(
            "SELECT total_queries, avg_similarity, helpful_count FROM source_file_stats WHERE source_file = ?",
            (source_file,)
        ).fetchone()
        h = 1 if helpful else 0
        if row:
            old_total = row["total_queries"]
            old_avg = row["avg_similarity"]
            old_helpful = row["helpful_count"] or 0
            new_total = old_total + 1
            new_avg = ((old_avg * old_total) + similarity_score) / new_total
            new_helpful = old_helpful + h
            self._conn.execute(
                "UPDATE source_file_stats SET total_queries=?, avg_similarity=?, helpful_count=?, last_query_time=? WHERE source_file=?",
                (new_total, new_avg, new_helpful, time.time(), source_file)
            )
        else:
            self._conn.execute(
                "INSERT INTO source_file_stats (source_file,total_queries,helpful_count,avg_similarity,last_query_time) VALUES (?,1,?,?,?)",
                (source_file, h, similarity_score, time.time())
            )
        self._check_reindex(source_file)
        self._conn.commit()

    def _check_reindex(self, source_file: str) -> bool:
        row = self._conn.execute(
            "SELECT total_queries, helpful_count, avg_similarity FROM source_file_stats WHERE source_file = ?",
            (source_file,)
        ).fetchone()
        if not row or row["total_queries"] < self.MIN_FEEDBACK_FOR_DECISION:
            return False
        total = row["total_queries"]
        helpful = row["helpful_count"] or 0
        utility = helpful / total if total > 0 else 0.0
        avg_sim = row["avg_similarity"]
        needs_reindex = utility < self.REINDEX_THRESHOLD
        if needs_reindex:
            self._conn.execute(
                "UPDATE source_file_stats SET needs_reindex=1, reindex_count=reindex_count+1 WHERE source_file=?",
                (source_file,)
            )
            self._conn.commit()
            logger.warning(
                f"REINDEX TRIGGERED '{source_file}': utility={utility:.2f}, avg_sim={avg_sim:.3f}, queries={total}"
            )
        return needs_reindex

    def should_reindex(self, source_file: str) -> bool:
        row = self._conn.execute(
            "SELECT needs_reindex FROM source_file_stats WHERE source_file = ?", (source_file,)
        ).fetchone()
        return bool(row and row["needs_reindex"])

    def get_files_needing_reindex(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT source_file FROM source_file_stats WHERE needs_reindex = 1"
        ).fetchall()
        return [r["source_file"] for r in rows]

    def get_file_stats(self, source_file: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM source_file_stats WHERE source_file = ?", (source_file,)
        ).fetchone()
        return dict(row) if row else None

    def get_overall_health(self) -> Dict[str, Any]:
        stats = self._conn.execute(
            """SELECT COUNT(*) as total_queries,
               SUM(CASE WHEN was_helpful=1 THEN 1 ELSE 0 END) as helpful,
               SUM(CASE WHEN was_helpful=0 THEN 1 ELSE 0 END) as unhelpful,
               AVG(similarity_score) as avg_similarity
               FROM retrieval_feedback"""
        ).fetchone()
        reindex = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM source_file_stats WHERE needs_reindex = 1"
        ).fetchone()
        return {
            "total_queries": stats["total_queries"] or 0,
            "helpful_count": stats["helpful"] or 0,
            "unhelpful_count": stats["unhelpful"] or 0,
            "avg_similarity": round(stats["avg_similarity"] or 0.0, 3),
            "utility_rate": ((stats["helpful"] or 0) / (stats["total_queries"] or 1)) if stats["total_queries"] else 0.0,
            "files_needing_reindex": reindex["cnt"] or 0,
        }

    def clear_feedback(self, source_file: Optional[str] = None) -> None:
        if source_file:
            self._conn.execute("DELETE FROM retrieval_feedback WHERE source_file = ?", (source_file,))
            self._conn.execute("DELETE FROM source_file_stats WHERE source_file = ?", (source_file,))
        else:
            self._conn.execute("DELETE FROM retrieval_feedback")
            self._conn.execute("DELETE FROM source_file_stats")
        self._conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_feedback": self._feedback_count,
            "db_path": str(self._db_path),
            "health": self.get_overall_health(),
        }

    def close(self) -> None:
        self._conn.close()