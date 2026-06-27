# Benchmark de recomendaciones

Este benchmark evalúa la calidad del ranking del `ai-service` usando el
`EmbeddingModel` real de Hugging Face y la misma lógica de similitud coseno que
usa producción.

## Qué mide

- `precision@3`
- `recall@3`
- `MRR`
- `hit_rate@3`
- latencia promedio y `p95` del benchmark

## Cómo correrlo

```bash
cd ai-service/api
RUN_REAL_EMBEDDINGS_BENCHMARK=1 .venv/bin/pytest tests/benchmark/test_recommendations_benchmark.py -s
```

## Visualización PCA

Podés generar dos visualizaciones PNG de los embeddings reales proyectados a 2D
con PCA:

```bash
cd ai-service/api
RUN_REAL_EMBEDDINGS_BENCHMARK=1 .venv/bin/python -m tests.benchmark.plot_benchmark_pca
```

El script genera:

- `tests/benchmark/output/pca_expected_groups.png`
- `tests/benchmark/output/pca_benchmark_groups.png`

La primera pinta con el mismo color a usuarios que deberían matchear según el
dataset esperado. La segunda pinta con el mismo color a usuarios que quedaron
conectados por las recomendaciones reales del benchmark.

## Qué valida

- Que el embedding real produzca un ranking razonable sobre un dataset dorado.
- Que la lógica de recomendaciones siga devolviendo vecinos semánticamente cercanos.
- Que podamos comparar corridas reales entre cambios de modelo o dataset.

## Alcance

Este benchmark sí usa embeddings reales, pero todavía no mide:

- búsqueda real en Postgres con `pgvector`
- exclusiones reales provenientes de otros microservicios
- calidad sobre datos productivos

La recuperación en este benchmark se hace en memoria con producto punto entre
vectores L2-normalizados. Eso equivale al ordenamiento por similitud coseno que
usa hoy el servicio con `pgvector`, así que es una aproximación fiel para medir
la calidad del embedding sin depender de infraestructura externa.
