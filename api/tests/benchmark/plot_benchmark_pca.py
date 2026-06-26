from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from tests.benchmark.benchmark_dataset import BENCHMARK_USERS
from tests.benchmark.recommendations_benchmark import (
    build_real_embedding_service,
    run_recommendations_benchmark,
    validate_real_benchmark_env,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
EXPECTED_PNG = OUTPUT_DIR / "pca_expected_groups.png"
ACTUAL_PNG = OUTPUT_DIR / "pca_benchmark_groups.png"
GRAPH_TOP_K = 2
SELECTED_USER_IDS = ("user-sports-2", "user-ai-1", "user-art-2")
BACKGROUND_COLOR = "#cbd5e1"
CATEGORY_COLORS = {
    "AI": "#dc2626",
    "Art": "#16a34a",
    "Sports": "#2563eb",
    "Music": "#9333ea",
    "Startup": "#ca8a04",
    "Social": "#0891b2",
    "No vistos": "#94a3b8",
}


def _project_to_2d(vectors: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    user_ids = [user.id for user in BENCHMARK_USERS]
    matrix = np.vstack([vectors[user_id] for user_id in user_ids])
    coords = PCA(n_components=2).fit_transform(matrix)
    return {user_id: (float(x), float(y)) for user_id, (x, y) in zip(user_ids, coords)}


def _legend_handles(group_members: dict[str, list[str]], color_map: dict[str, str]) -> list[Line2D]:
    user_by_id = {user.id: user for user in BENCHMARK_USERS}
    handles: list[Line2D] = []
    for root_id in SELECTED_USER_IDS:
        sample_ids = group_members[root_id]
        names = ", ".join(user_by_id[user_id].username for user_id in sample_ids)
        category = next(user.category for user in BENCHMARK_USERS if user.id == root_id)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_map[root_id],
                markeredgecolor="#0f172a",
                markersize=10,
                label=f"{category}: {names}",
            )
        )
    unseen_usernames = sorted(
        user.username
        for user in BENCHMARK_USERS
        if all(user.id not in members for members in group_members.values())
    )
    if unseen_usernames:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=CATEGORY_COLORS["No vistos"],
                markeredgecolor="#0f172a",
                markersize=10,
                label=f"No vistos: {', '.join(unseen_usernames)}",
            )
        )
    return handles


def _legend_handles_expected(group_members: dict[str, list[str]]) -> list[Line2D]:
    handles: list[Line2D] = []
    for category, members in group_members.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=CATEGORY_COLORS[category],
                markeredgecolor="#0f172a",
                markersize=10,
                label=f"{category}: {', '.join(members)}",
            )
        )
    return handles


def _annotation_offsets(points: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    offsets: dict[str, tuple[float, float]] = {}
    ordered = sorted(points.items(), key=lambda item: (item[1][0], item[1][1]))
    for index, (user_id, _) in enumerate(ordered):
        dx = 8 + (index % 3) * 2
        dy = 8 + (index % 4) * 6
        offsets[user_id] = (dx, dy)
    return offsets


def _plot_groups(
    output_path: Path,
    title: str,
    subtitle: str,
    points: dict[str, tuple[float, float]],
    color_map: dict[str, str],
    group_members: dict[str, list[str]],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    offsets = _annotation_offsets(points)

    for user in BENCHMARK_USERS:
        x, y = points[user.id]
        color = color_map.get(user.id, BACKGROUND_COLOR)
        alpha = 0.95 if user.id in color_map else 0.45
        size = 260 if user.id in color_map else 170
        ax.scatter(x, y, s=size, c=color, edgecolors="#0f172a", linewidths=1.2, zorder=3, alpha=alpha)
        dx, dy = offsets[user.id]
        ax.annotate(
            f"{user.username} ({user.id})",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color="#0f172a" if user.id in color_map else "#475569",
            zorder=4,
            alpha=1.0 if user.id in color_map else 0.8,
        )

    ax.set_title(title, fontsize=24, fontweight="bold", color="#0f172a", loc="left", pad=18)
    fig.text(0.125, 0.905, subtitle, fontsize=13, color="#334155")
    ax.set_xlabel("PCA component 1", fontsize=14, color="#334155", labelpad=12)
    ax.set_ylabel("PCA component 2", fontsize=14, color="#334155", labelpad=12)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")

    legend = ax.legend(
        handles=_legend_handles(group_members, color_map),
        title="Groups",
        title_fontsize=15,
        fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderpad=1.0,
        labelspacing=1.1,
    )
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#cbd5e1")

    fig.tight_layout(rect=(0, 0, 0.82, 0.90))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_expected_categories(
    output_path: Path,
    title: str,
    subtitle: str,
    points: dict[str, tuple[float, float]],
    color_map: dict[str, str],
    group_members: dict[str, list[str]],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    offsets = _annotation_offsets(points)

    for user in BENCHMARK_USERS:
        x, y = points[user.id]
        color = color_map[user.id]
        ax.scatter(x, y, s=260, c=color, edgecolors="#0f172a", linewidths=1.2, zorder=3, alpha=0.95)
        dx, dy = offsets[user.id]
        ax.annotate(
            f"{user.username} ({user.id})",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color="#0f172a",
            zorder=4,
        )

    ax.set_title(title, fontsize=24, fontweight="bold", color="#0f172a", loc="left", pad=18)
    fig.text(0.125, 0.905, subtitle, fontsize=13, color="#334155")
    ax.set_xlabel("PCA component 1", fontsize=14, color="#334155", labelpad=12)
    ax.set_ylabel("PCA component 2", fontsize=14, color="#334155", labelpad=12)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")

    legend = ax.legend(
        handles=_legend_handles_expected(group_members),
        title="Categories",
        title_fontsize=15,
        fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderpad=1.0,
        labelspacing=1.1,
    )
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#cbd5e1")

    fig.tight_layout(rect=(0, 0, 0.82, 0.90))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_expected_groups() -> tuple[dict[str, str], dict[str, list[str]]]:
    color_map: dict[str, str] = {}
    group_members: dict[str, list[str]] = {}

    for user in BENCHMARK_USERS:
        color_map[user.id] = CATEGORY_COLORS[user.category]
        group_members.setdefault(user.category, []).append(user.username)

    ordered_group_members = {
        category: sorted(usernames)
        for category, usernames in group_members.items()
    }
    return color_map, ordered_group_members


def _build_actual_groups(recommended_by_user: dict[str, list[str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    color_map: dict[str, str] = {user.id: CATEGORY_COLORS["No vistos"] for user in BENCHMARK_USERS}
    group_members: dict[str, list[str]] = {}

    for root_id in SELECTED_USER_IDS:
        members = [root_id, *recommended_by_user[root_id][:GRAPH_TOP_K]]
        group_members[root_id] = members
        category = next(user.category for user in BENCHMARK_USERS if user.id == root_id)
        color = CATEGORY_COLORS[category]
        for member_id in members:
            color_map[member_id] = color

    return color_map, group_members


async def main() -> None:
    validate_real_benchmark_env()

    _, cache = build_real_embedding_service()
    summary = await run_recommendations_benchmark()

    recommended_by_user = {
        query.user_id: query.recommended_ids
        for query in summary.queries
    }
    vectors = cache.candidate_vectors
    projected = _project_to_2d(vectors)

    expected_colors, expected_groups = _build_expected_groups()
    actual_colors, actual_groups = _build_actual_groups(recommended_by_user)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_expected_categories(
        output_path=EXPECTED_PNG,
        title="Benchmark PCA by expected matches",
        subtitle="Colored groups correspond to the six expected categories in the benchmark dataset.",
        points=projected,
        color_map=expected_colors,
        group_members=expected_groups,
    )
    _plot_groups(
        output_path=ACTUAL_PNG,
        title="Benchmark PCA by retrieved matches",
        subtitle="K=2. Colored groups correspond to retrieved matches for agus, sofia and juan. The rest are marked as No vistos.",
        points=projected,
        color_map=actual_colors,
        group_members=actual_groups,
    )

    print(summary.pretty_report())
    print("")
    print(f"Generated: {EXPECTED_PNG}")
    print(f"Generated: {ACTUAL_PNG}")


if __name__ == "__main__":
    asyncio.run(main())
