"""
Exporta el corpus desde SQLite a formatos portables para entrenamiento.

JSONL: un JSON por línea, formato estándar para datasets de NLP.
CSV:   para análisis en pandas/Excel.
"""

import csv
import json
from pathlib import Path

from .db import BioDatabase


def to_jsonl(db: BioDatabase, output_path: Path) -> int:
    """
    Exporta todas las bios válidas y no duplicadas a JSONL.
    Devuelve la cantidad de registros exportados.
    """
    rows = db.fetch_all_valid()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "id": row["id"],
                "username": row["username"],
                "age": row["age"],
                "career": row["career"],
                "year": row["year"],
                "origin": row["origin"],
                "personality": row["personality"],
                "interests": json.loads(row["interests"]),
                "writing_style": row["writing_style"],
                "bio": row["bio"],
                "bio_length": row["bio_length"],
                "model": row["model"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(rows)


def to_csv(db: BioDatabase, output_path: Path) -> int:
    rows = db.fetch_all_valid()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "id", "username", "age", "career", "year", "origin",
        "personality", "interests", "writing_style", "bio", "bio_length", "model",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "username": row["username"],
                "age": row["age"],
                "career": row["career"],
                "year": row["year"],
                "origin": row["origin"],
                "personality": row["personality"],
                "interests": row["interests"],  # mantener como JSON string en CSV
                "writing_style": row["writing_style"],
                "bio": row["bio"],
                "bio_length": row["bio_length"],
                "model": row["model"],
            })

    return len(rows)
