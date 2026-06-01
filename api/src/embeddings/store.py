import hashlib
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config.settings import settings


def _get_conn() -> sqlite3.Connection:
    Path(settings.embeddings_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.embeddings_db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_embeddings (
            user_id    TEXT PRIMARY KEY,
            bio_hash   TEXT NOT NULL,
            embedding  BLOB NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _bio_hash(biography: str) -> str:
    return hashlib.md5(biography.encode()).hexdigest()


def _serialize(vector: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, vector)
    return buf.getvalue()


def _deserialize(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


class EmbeddingStore:
    def get(self, user_id: str, biography: str) -> np.ndarray | None:
        conn = _get_conn()
        try:
            cur = conn.execute(
                "SELECT bio_hash, embedding FROM user_embeddings WHERE user_id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            if row and row[0] == _bio_hash(biography):
                return _deserialize(row[1])
            return None
        finally:
            conn.close()

    def save(self, user_id: str, biography: str, vector: np.ndarray) -> None:
        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT INTO user_embeddings (user_id, bio_hash, embedding, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    bio_hash   = excluded.bio_hash,
                    embedding  = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    _bio_hash(biography),
                    _serialize(vector),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()


embedding_store = EmbeddingStore()
