"""Lightweight vector store using SQLite + sqlite-vec.

This module provides a persistent vector store that stores embeddings
in an SQLite database using the sqlite-vec extension. It supports
insertion, retrieval by similarity, and deletion.

Usage::

    from local_paper_qa.vector_store import VectorStore

    store = VectorStore("papers/.research_index/vector_store.db")
    store.insert(chunk_id, embedding, metadata)
    results = store.query(query_embedding, limit=10)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import sqlite_vec
except Exception:
    sqlite_vec = None  # type: ignore


class VectorStore:
    """A minimal persistent vector store backed by SQLite + sqlite-vec."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if sqlite_vec:
            sqlite_vec.load(conn)
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            if sqlite_vec:
                conn.execute("SELECT vec_init(db) FROM pragma_database_list")
            conn.commit()

    def insert(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Insert a single vector with its metadata."""
        blob = json.dumps(embedding).encode()
        meta_blob = json.dumps(metadata).encode()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vectors (id, embedding, metadata) VALUES (?, ?, ?)",
                (chunk_id, blob, meta_blob),
            )
            conn.commit()

    def insert_many(self, items: List[tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Insert multiple vectors at once."""
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (id, embedding, metadata) VALUES (?, ?, ?)",
                [
                    (chunk_id, json.dumps(embedding).encode(), json.dumps(metadata).encode())
                    for chunk_id, embedding, metadata in items
                ],
            )
            conn.commit()

    def delete(self, chunk_id: str) -> None:
        """Delete a vector by ID."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM vectors WHERE id = ?", (chunk_id,))
            conn.commit()

    def query(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the top-K nearest neighbors to the query embedding."""
        if sqlite_vec is None:
            # Fallback to no vector store – caller should handle
            return []

        blob = json.dumps(query_embedding).encode()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, metadata, distance
                FROM vectors
                WHERE embedding MATCH ?
                LIMIT ?
                """,
                (blob, limit),
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append({
                "id": row["id"],
                "metadata": json.loads(row["metadata"]),
                "distance": row["distance"],
            })
        return results

    def count(self) -> int:
        """Return the number of vectors stored."""
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) as cnt FROM vectors").fetchone()["cnt"]

    def clear(self) -> None:
        """Delete all vectors."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM vectors")
            conn.commit()

    def close(self) -> None:
        """Close any open connections."""
        pass
