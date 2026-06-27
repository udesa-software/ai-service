import hashlib
import logging
from datetime import datetime, timezone
import numpy as np
from psycopg_pool import ConnectionPool

from src.config.settings import settings

logger = logging.getLogger(__name__)


def _bio_hash(biography: str) -> str:
    return hashlib.md5(biography.encode()).hexdigest()


def _serialize(vector: np.ndarray) -> str:
    # Format the vector as "[val1,val2,...]" for pgvector
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def _deserialize(val: str) -> np.ndarray:
    # Convert pgvector string "[val1,val2,...]" back to numpy array
    cleaned = val.strip("[]")
    if not cleaned:
        return np.array([], dtype=np.float32)
    return np.array([float(x) for x in cleaned.split(",")], dtype=np.float32)


class EmbeddingStore:
    def __init__(self):
        sslmode = "require" if settings.users_db_ssl else "disable"
        self._pool = ConnectionPool(
            kwargs={
                "host": settings.users_db_host,
                "port": settings.users_db_port,
                "dbname": settings.users_db_name,
                "user": settings.users_db_user,
                "password": settings.users_db_password,
                "connect_timeout": 10,
                "sslmode": sslmode,
            },
            min_size=1,
            max_size=5,
            open=False,
        )

    def open(self) -> None:
        if self._pool.closed:
            self._pool.open()
            logger.info("Postgres embedding pool opened")

    def close(self) -> None:
        self._pool.close()
        logger.info("Postgres embedding pool closed")

    def _connection(self):
        if self._pool.closed:
            self.open()
        return self._pool.connection()

    def get(self, user_id: str, biography: str) -> np.ndarray | None:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT bio_hash, embedding::text FROM user_embeddings WHERE user_id = %s",
                        (user_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0] == _bio_hash(biography):
                        return _deserialize(row[1])
            return None
        except Exception as e:
            logger.error(f"Error reading embedding from Postgres: {e}")
            return None

    def save(self, user_id: str, biography: str, vector: np.ndarray) -> None:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_embeddings (user_id, bio_hash, embedding, updated_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id) DO UPDATE SET
                            bio_hash   = EXCLUDED.bio_hash,
                            embedding  = EXCLUDED.embedding,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            user_id,
                            _bio_hash(biography),
                            _serialize(vector),
                            datetime.now(timezone.utc),
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving embedding to Postgres: {e}")
            raise

    def find_nearest_candidates(
        self,
        vector: np.ndarray,
        exclude_ids: set[str],
        limit: int,
    ) -> list[dict]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT u.id::text, u.username, u.profile_photo_url, p.biography
                        FROM user_embeddings ue
                        JOIN users u ON u.id = ue.user_id
                        JOIN preferences p ON p.user_id = u.id
                        WHERE u.deleted_at IS NULL
                          AND u.is_suspended = FALSE
                          AND p.biography IS NOT NULL
                          AND trim(p.biography) != ''
                          AND ue.bio_hash = md5(p.biography)
                          AND u.id != ALL(%s::uuid[])
                        ORDER BY ue.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (list(exclude_ids), _serialize(vector), limit),
                    )
                    return [
                        {"id": row[0], "username": row[1], "profile_photo_url": row[2], "biography": row[3]}
                        for row in cur.fetchall()
                    ]
        except Exception as e:
            logger.error(f"Error searching nearest embeddings in Postgres: {e}")
            return []


embedding_store = EmbeddingStore()
