# data-generation — Corpus Sintético de Biografías

Genera un corpus de biografías ficticias para entrenar modelos de embeddings
que potencian el **Bio Matcher** de UdeSA-migos.

## Setup

```bash
# 1. Instalar dependencias Python (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Instalar y correr Ollama
brew install ollama
ollama serve          # en otra terminal, o en background
ollama pull qwen2.5:14b   # modelo principal (~9GB)
# alternativa rápida:
# ollama pull llama3.1:8b
```

## Uso

```bash
# Generación completa: 5000 bios con 8 requests concurrentes
python pipeline.py --count 5000 --model qwen2.5:14b --concurrency 8

# Prueba rápida (500 bios, modelo más rápido)
python pipeline.py --count 500 --model llama3.1:8b --concurrency 4

# Ver qué se generaría sin llamar al modelo
python pipeline.py --count 100 --dry-run

# Si la generación se interrumpió, retomar donde quedó (idempotente)
python pipeline.py --count 5000 --model qwen2.5:14b

# Solo deduplicar lo que ya hay en la BD
python pipeline.py --dedup-only

# Solo exportar a JSONL/CSV
python pipeline.py --export-only
```

## Output

```
data/synthetic/
├── corpus.db       # SQLite con todas las bios + metadata
├── bios.jsonl      # Corpus final (bios válidas, sin duplicados)
└── bios.csv        # Mismo corpus en CSV para análisis en Excel/pandas
```

## Estructura de una bio en JSONL

```json
{
  "id": 42,
  "username": "mariagomez17",
  "age": 20,
  "career": "Psicología",
  "year": 3,
  "origin": "Córdoba capital",
  "personality": "empática y colaborativa",
  "interests": ["teatro", "yoga", "política estudiantil"],
  "writing_style": "informal con lunfardo porteño",
  "bio": "Psico 3er año, cordobesa perdida en BsAs. Me la paso en el teatro o armando algún proyecto. Si tenés ganas de charlar de todo, mandame un mensaje.",
  "bio_length": 148,
  "model": "qwen2.5:14b"
}
```

## Análisis del corpus

```bash
jupyter notebook notebooks/explore_corpus.ipynb
```

## Tiempo estimado (M3 Pro, 36GB RAM)

| Modelo | Bios | Concurrencia | Tiempo aprox. |
|--------|------|--------------|---------------|
| `qwen2.5:14b` | 5.000 | 8 | ~1.5h |
| `qwen2.5:14b` | 10.000 | 8 | ~3h |
| `llama3.1:8b` | 5.000 | 8 | ~40min |

## Estrategias de diversidad

El pipeline combina cuatro capas para evitar que las bios sean demasiado similares:

1. **Taxonomía amplia**: 14 carreras × 8 estilos de escritura × 10 personalidades × 11 orígenes × pool de 35+ intereses.
2. **8 prompt templates** distintos: cada uno encuadra la tarea diferente.
3. **Temperatura variable**: entre 0.70 y 1.00 por bio.
4. **Deduplicación semántica** post-generación con `paraphrase-multilingual-MiniLM-L12-v2` (similitud coseno > 0.85 → marcado como duplicado).

## Reproducibilidad

El corpus generado **no se commitea** (está en `.gitignore`). Para regenerar
exactamente el mismo corpus, usar el mismo `--model` y asegurarse de que
la BD no tenga datos previos (borrar `data/synthetic/corpus.db`).
