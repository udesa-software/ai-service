#!/usr/bin/env python3
"""
Pipeline principal de generación de corpus sintético de biografías.

Uso:
    python pipeline.py --count 5000 --model qwen2.5:14b --concurrency 8
    python pipeline.py --count 500  --model llama3.1:8b  --dry-run
    python pipeline.py --dedup-only   # solo corre la deduplicación
    python pipeline.py --export-only  # solo exporta a JSONL/CSV

El pipeline es idempotente: si se interrumpe, al reiniciar detecta
las bios ya generadas en la BD y las saltea automáticamente.
"""

import argparse
import asyncio
import sqlite3
import sys
import time
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from src.generation.ollama_client import OllamaClient
from src.generation.generator import generate_bio
from src.personas.sampler import generate_personas, Persona
from src.personas.templates import N_TEMPLATES
from src.storage.db import BioDatabase
from src.storage.export import to_jsonl, to_csv
from src.dedup.similarity import run_dedup

DATA_DIR = Path(__file__).parent / "data" / "synthetic"
DB_PATH = DATA_DIR / "corpus.db"
JSONL_PATH = DATA_DIR / "bios.jsonl"
CSV_PATH = DATA_DIR / "bios.csv"

TEMPERATURES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_COUNT = 5000
DEFAULT_CONCURRENCY = 8
DEDUP_THRESHOLD = 0.85


async def generate_batch(
    client: OllamaClient,
    personas: list[Persona],
    db: BioDatabase,
    model: str,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> dict:
    stats = {"saved": 0, "skipped": 0, "failed": 0}

    async def process_one(persona: Persona):
        async with semaphore:
            # Idempotencia: saltar si ya existe en la BD
            if db.username_exists(persona.username):
                stats["skipped"] += 1
                pbar.update(1)
                return

            bio, attempts = await generate_bio(client, persona, model)

            if bio is None:
                stats["failed"] += 1
            else:
                try:
                    db.save(persona, bio, model, attempts)
                    stats["saved"] += 1
                except sqlite3.IntegrityError:
                    # Otra tarea concurrente ya guardó este username mientras
                    # esperábamos la respuesta del LLM. No es un error: saltear.
                    stats["skipped"] += 1

            pbar.update(1)
            pbar.set_postfix(
                saved=stats["saved"],
                failed=stats["failed"],
                skip=stats["skipped"],
            )

    tasks = [process_one(p) for p in personas]
    await asyncio.gather(*tasks)
    return stats


async def run_generation(
    count: int,
    model: str,
    concurrency: int,
    db: BioDatabase,
    dry_run: bool = False,
):
    # Verificar Ollama antes de empezar
    print(f"\n[1/4] Verificando Ollama y modelo '{model}'...")
    await OllamaClient.check_model(model)
    print(f"  OK — modelo disponible.\n")

    # Cuántas personas faltan generar (ya hay algunas en la BD si se retomó)
    already_done = db.count_total()
    remaining = max(0, count - already_done)

    if remaining == 0:
        print(f"  La BD ya tiene {already_done} bios. Nada que generar.")
        return

    print(f"[2/4] Generando {remaining} personas ficticias...")
    personas = generate_personas(remaining, N_TEMPLATES, TEMPERATURES)

    if dry_run:
        print(f"  [DRY-RUN] Se generarían {len(personas)} personas.")
        for p in personas[:3]:
            print(f"    {p.name} | {p.career} | {p.writing_style} | T={p.temperature}")
        print("  [DRY-RUN] Sin llamadas reales a Ollama.")
        return

    semaphore = asyncio.Semaphore(concurrency)

    print(f"[3/4] Generando bios ({concurrency} concurrentes)...")
    t0 = time.time()

    with tqdm(total=len(personas), unit="bio", desc="Generando") as pbar:
        async with OllamaClient() as client:
            stats = await generate_batch(client, personas, db, model, semaphore, pbar)

    elapsed = time.time() - t0
    rate = stats["saved"] / elapsed if elapsed > 0 else 0
    print(
        f"\n  Resultados: {stats['saved']} guardadas | "
        f"{stats['failed']} fallidas | "
        f"{stats['skipped']} salteadas\n"
        f"  Tiempo: {elapsed:.0f}s — {rate:.1f} bios/s\n"
    )


def run_pipeline(args):
    db = BioDatabase(DB_PATH)

    try:
        if not args.dedup_only and not args.export_only:
            asyncio.run(
                run_generation(
                    count=args.count,
                    model=args.model,
                    concurrency=args.concurrency,
                    db=db,
                    dry_run=args.dry_run,
                )
            )

        if not args.export_only:
            total = db.count_total()
            if total == 0:
                print("No hay bios en la BD. Generá primero con --count.")
                return

            print(f"[dedup] Ejecutando deduplicación semántica (threshold={DEDUP_THRESHOLD})...")
            n, marked = run_dedup(db, threshold=DEDUP_THRESHOLD)
            print(f"  {n} bios analizadas → {marked} marcadas como duplicadas.\n")

        valid = db.count_valid()
        print(f"[export] Exportando {valid} bios válidas...")
        n_jsonl = to_jsonl(db, JSONL_PATH)
        n_csv = to_csv(db, CSV_PATH)
        print(f"  JSONL: {JSONL_PATH} ({n_jsonl} registros)")
        print(f"  CSV:   {CSV_PATH}   ({n_csv} registros)")

    finally:
        db.close()

    print("\nCorpus listo.")


def main():
    parser = argparse.ArgumentParser(
        description="Genera un corpus sintético de biografías para UdeSA-migos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help="Total de bios a generar (incluyendo las ya existentes en la BD).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Modelo de Ollama a usar. Ej: qwen2.5:14b, llama3.1:8b",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="Requests concurrentes a Ollama.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra las personas a generar sin llamar a Ollama.",
    )
    parser.add_argument(
        "--dedup-only", action="store_true",
        help="Solo ejecuta la deduplicación sobre la BD existente.",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Solo exporta a JSONL/CSV sin generar ni deduplicar.",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=DEDUP_THRESHOLD,
        help="Umbral de similitud coseno para considerar duplicado (0-1).",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
