"""
Capa de persistencia SQLite para el corpus sintético.

Diseño:
- personas: los atributos usados para generar cada bio (para reproducibilidad)
- generated_bios: el output + metadata de generación

El pipeline es idempotente: al reiniciar detecta qué usernames ya existen
y los saltea, permitiendo reanudar una generación interrumpida.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..personas.sampler import Persona


SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    career        TEXT    NOT NULL,
    year          INTEGER NOT NULL,
    personality   TEXT    NOT NULL,
    interests     TEXT    NOT NULL,   -- JSON array
    origin        TEXT    NOT NULL,
    writing_style TEXT    NOT NULL,
    template_id   INTEGER NOT NULL,
    temperature   REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS generated_bios (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id   INTEGER NOT NULL REFERENCES personas(id),
    username     TEXT    UNIQUE NOT NULL,
    email        TEXT    UNIQUE NOT NULL,
    age          INTEGER NOT NULL,
    bio          TEXT    NOT NULL,
    bio_length   INTEGER NOT NULL,
    model        TEXT    NOT NULL,
    template_id  INTEGER NOT NULL,
    temperature  REAL    NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 1,
    is_valid     INTEGER NOT NULL DEFAULT 1,
    is_duplicate INTEGER NOT NULL DEFAULT 0,
    generated_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bios_career     ON personas(career);
CREATE INDEX IF NOT EXISTS idx_bios_style      ON personas(writing_style);
CREATE INDEX IF NOT EXISTS idx_bios_duplicate  ON generated_bios(is_duplicate);
"""


class BioDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # permite lecturas concurrentes
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def username_exists(self, username: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM generated_bios WHERE username = ?", (username,)
        ).fetchone()
        return row is not None

    def save(
        self,
        persona: Persona,
        bio: str,
        model: str,
        attempts: int,
    ) -> int:
        """
        Persiste la persona y la bio. Devuelve el id de generated_bios.
        Es atómico: si algo falla, ninguna de las dos tablas queda a medias.
        """
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO personas
                    (career, year, personality, interests, origin, writing_style,
                     template_id, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persona.career,
                    persona.year,
                    persona.personality,
                    json.dumps(persona.interests, ensure_ascii=False),
                    persona.origin,
                    persona.writing_style,
                    persona.template_id,
                    persona.temperature,
                ),
            )
            persona_id = cur.lastrowid

            cur2 = self._conn.execute(
                """
                INSERT INTO generated_bios
                    (persona_id, username, email, age, bio, bio_length, model,
                     template_id, temperature, attempts, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persona_id,
                    persona.username,
                    persona.email,
                    persona.age,
                    bio,
                    len(bio),
                    model,
                    persona.template_id,
                    persona.temperature,
                    attempts,
                    datetime.utcnow().isoformat(),
                ),
            )
            return cur2.lastrowid

    def mark_duplicate(self, bio_id: int):
        with self._conn:
            self._conn.execute(
                "UPDATE generated_bios SET is_duplicate = 1 WHERE id = ?",
                (bio_id,),
            )

    def count_valid(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM generated_bios WHERE is_valid = 1 AND is_duplicate = 0"
        ).fetchone()
        return row[0]

    def count_total(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM generated_bios"
        ).fetchone()
        return row[0]

    def fetch_all_valid(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT gb.id, gb.username, gb.email, gb.age, gb.bio, gb.bio_length,
                   gb.model, gb.template_id, gb.temperature, gb.generated_at,
                   p.career, p.year, p.personality, p.interests, p.origin, p.writing_style
            FROM generated_bios gb
            JOIN personas p ON p.id = gb.persona_id
            WHERE gb.is_valid = 1 AND gb.is_duplicate = 0
            ORDER BY gb.id
            """
        ).fetchall()

    def fetch_all_bios(self) -> list[sqlite3.Row]:
        """Para el paso de deduplicación (necesita todos, incluyendo futuros duplicados)."""
        return self._conn.execute(
            "SELECT id, bio FROM generated_bios WHERE is_valid = 1 ORDER BY id"
        ).fetchall()
