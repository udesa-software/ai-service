"""
Deduplicación semántica del corpus usando sentence-transformers.

Estrategia:
1. Calcular embeddings de todas las bios con un modelo multilingüe ligero.
2. Calcular la matriz de similitud coseno entre todos los pares.
3. Para cada par con similitud > threshold, marcar el de menor id como duplicado
   (conservamos el generado con mayor temperatura, que suele ser más diverso;
   como no tenemos ese dato directo al deduplicar, conservamos el más reciente
   que ya pasó por más variación de prompts).

El modelo `paraphrase-multilingual-MiniLM-L12-v2` es:
- 50MB, corre en CPU sin GPU.
- Soporta español nativo.
- Suficientemente bueno para detectar near-duplicates semánticos.

Para corpus grandes (>10K), la comparación O(n²) puede tardar varios minutos.
Se usa numpy para vectorizar el cálculo y hacerlo viable.
"""

from typing import List, Tuple
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from ..storage.db import BioDatabase

DEDUP_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.85
# Tamaño de bloque para la comparación por bloques (evita OOM en >10K bios)
BLOCK_SIZE = 1000


def _cosine_matrix(embs: np.ndarray) -> np.ndarray:
    """Calcula la matriz de similitud coseno entre todos los pares."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embs / norms
    return normed @ normed.T


def run_dedup(
    db: BioDatabase,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 512,
) -> Tuple[int, int]:
    """
    Marca como duplicados los pares de bios con similitud coseno > threshold.

    Devuelve (total_procesadas, marcadas_como_duplicadas).
    """
    if not ST_AVAILABLE:
        raise ImportError(
            "sentence-transformers no está instalado. "
            "Ejecutá: pip install sentence-transformers"
        )

    rows = db.fetch_all_bios()
    if not rows:
        return 0, 0

    ids = [r["id"] for r in rows]
    bios = [r["bio"] for r in rows]

    print(f"  Calculando embeddings para {len(bios)} bios...")
    model = SentenceTransformer(DEDUP_MODEL)
    embeddings = model.encode(
        bios,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # evita dividir por norma después
    )
    # Con normalize_embeddings=True, similitud coseno = producto punto
    embeddings = np.array(embeddings, dtype=np.float32)

    print("  Calculando similitudes por bloques...")
    to_mark_duplicate: set[int] = set()
    n = len(ids)

    for i in range(0, n, BLOCK_SIZE):
        block_embs = embeddings[i : i + BLOCK_SIZE]
        # Similitud del bloque contra todo el corpus
        sims = block_embs @ embeddings.T  # (block, n)

        for local_idx, global_idx in enumerate(range(i, min(i + BLOCK_SIZE, n))):
            row_sims = sims[local_idx]
            # Solo pares donde el otro tiene id mayor (triangular superior)
            # para no marcar ambos como duplicados
            for j in range(global_idx + 1, n):
                if ids[j] in to_mark_duplicate:
                    continue
                if row_sims[j] >= threshold:
                    # Marcamos el de id mayor (el más nuevo) como duplicado,
                    # conservando el más antiguo
                    to_mark_duplicate.add(ids[j])

    print(f"  Marcando {len(to_mark_duplicate)} duplicados en la BD...")
    for bio_id in to_mark_duplicate:
        db.mark_duplicate(bio_id)

    return n, len(to_mark_duplicate)


def quick_stats(
    db: BioDatabase,
    sample_size: int = 200,
) -> dict:
    """
    Calcula estadísticas rápidas de diversidad sin deduplicar.
    Útil para el notebook de análisis.
    """
    if not ST_AVAILABLE:
        raise ImportError("sentence-transformers no está instalado.")

    rows = db.fetch_all_bios()
    if len(rows) > sample_size:
        import random
        rows = random.sample(rows, sample_size)

    bios = [r["bio"] for r in rows]
    model = SentenceTransformer(DEDUP_MODEL)
    embs = model.encode(bios, normalize_embeddings=True)
    embs = np.array(embs, dtype=np.float32)
    sim_matrix = embs @ embs.T

    # Excluir diagonal (similitud consigo mismo = 1.0)
    mask = np.ones(sim_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diag = sim_matrix[mask]

    return {
        "n_bios": len(bios),
        "mean_similarity": float(off_diag.mean()),
        "max_similarity": float(off_diag.max()),
        "p95_similarity": float(np.percentile(off_diag, 95)),
        "pct_above_085": float((off_diag >= 0.85).mean() * 100),
    }
