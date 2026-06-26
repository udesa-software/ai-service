import pytest

from tests.benchmark.recommendations_benchmark import (
    REAL_BENCHMARK_ENV,
    TOP_K,
    real_benchmark_enabled,
    run_recommendations_benchmark,
    validate_real_benchmark_env,
)


@pytest.mark.asyncio
async def test_recommendations_benchmark_thresholds():
    if not real_benchmark_enabled():
        pytest.skip(f"Set {REAL_BENCHMARK_ENV}=1 to run the real embeddings benchmark")

    validate_real_benchmark_env()
    summary = await run_recommendations_benchmark()

    print("\n" + summary.pretty_report())

    assert summary.avg_precision_at_k >= 0.55
    assert summary.avg_recall_at_k >= 0.55
    assert summary.mean_reciprocal_rank >= 0.70
    assert summary.hit_rate_at_k >= 0.85

    for query in summary.queries:
        assert len(query.recommended_ids) <= TOP_K
        assert query.reciprocal_rank > 0.0
