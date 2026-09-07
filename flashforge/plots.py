"""Charts for the five questions.

Each function takes the DataFrame its analysis counterpart produced and returns
a matplotlib Figure. Series colours are assigned from the fixed categorical
order in `viz`; every multi-series chart carries both a legend and direct
end-labels, so identity never rests on colour alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from . import viz


def plot_skew(lorenz: pd.DataFrame, summary: pd.DataFrame):
    """Q1. Lorenz curve of expert usage, plus per-layer concentration."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Single series: no legend box, the title names it.
    left.plot(
        lorenz["expert_fraction"], lorenz["access_fraction"], color=viz.SERIES[0]
    )
    left.plot([0, 1], [0, 1], color=viz.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    # Below the diagonal is dead space on a Lorenz plot — nothing can be drawn
    # there, so it is the one place a label never competes with the data.
    left.annotate(
        "uniform routing",
        xy=(0.34, 0.34),
        xytext=(8, -12),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=9,
        color=viz.INK_MUTED,
    )
    half = float(np.interp(0.5, lorenz["expert_fraction"], lorenz["access_fraction"]))
    left.plot([0.5], [half], marker="o", markersize=5, color=viz.SERIES[0], zorder=5)
    # Below-right of the point: the wedge between the curve and the diagonal is
    # the only reliably empty region on a Lorenz plot.
    left.annotate(
        f"top 50% of experts\nserve {half:.0%} of accesses",
        xy=(0.5, half),
        xytext=(9, -20),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=9,
        color=viz.INK_SECONDARY,
    )
    left.set_xlabel("cumulative share of experts (hottest first)")
    left.set_ylabel("cumulative share of accesses")
    left.set_title("Expert usage is concentrated")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1.02)

    right.bar(
        summary["layer"],
        summary["top10pct_mass"],
        color=viz.SERIES[0],
        width=0.7,
    )
    right.axhline(0.10, color=viz.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    # Bars span the full plot width and start at zero, so there is no in-plot
    # spot for this label that a bar does not already occupy. Reserve margin on
    # the right and put it there.
    last_layer = int(summary["layer"].max())
    right.set_xlim(-0.8, last_layer + 4.4)
    right.annotate(
        "uniform\nbaseline (0.10)",
        xy=(last_layer + 0.9, 0.10),
        va="center",
        ha="left",
        fontsize=9,
        color=viz.INK_MUTED,
    )
    right.set_xlabel("layer")
    right.set_ylabel("access share of hottest 10%")
    right.set_title("Concentration by layer")
    right.set_ylim(0, 1)
    right.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def plot_locality(overlap: pd.DataFrame):
    """Q2. Expert-set overlap against token lag, per layer."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    pivot = overlap.groupby(["lag", "layer"])["mean_overlap"].mean().unstack("lag")

    labels = []
    for slot, lag in enumerate(sorted(pivot.columns)):
        series = pivot[lag]
        color = viz.SERIES[slot % len(viz.SERIES)]
        ax.plot(series.index, series.to_numpy(), color=color, label=f"lag {lag}")
        labels.append((series.index[-1], series.to_numpy()[-1], f"lag {lag}", color))

    ax.set_xlabel("layer")
    ax.set_ylabel("mean fraction of experts reused")
    ax.set_ylim(0, 1)
    ax.margins(x=0.14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # layers are discrete
    viz.label_line_ends(ax, labels)
    viz.titled_legend_above(fig, ax, "Temporal locality of routing", ncols=len(pivot.columns))
    return fig


def plot_predictability(summary: pd.DataFrame, top_k: int):
    """Q3. Recall of the true top-k, by predictor and lookahead distance."""
    budgets = sorted(summary["budget_mult"].unique())
    fig, axes = plt.subplots(
        1, len(budgets), figsize=(5.2 * len(budgets), 4.0), sharey=True, squeeze=False
    )

    order = ["prior", "identity", "stale_router", "probe"]
    present = [name for name in order if name in set(summary["predictor"])]

    handles: list = []
    for column, budget in enumerate(budgets):
        ax = axes[0][column]
        subset = summary[summary["budget_mult"] == budget]
        labels = []
        for slot, name in enumerate(present):
            series = subset[subset["predictor"] == name].sort_values("offset")
            if series.empty:
                continue
            color = viz.SERIES[slot % len(viz.SERIES)]
            line, = ax.plot(
                series["offset"], series["recall"], color=color, marker="o", label=name
            )
            if column == 0:
                handles.append(line)
            labels.append(
                (series["offset"].to_numpy()[-1], series["recall"].to_numpy()[-1], name, color)
            )
        ax.set_xlabel("lookahead (layers ahead predicted)")
        ax.set_title(f"Prefetch budget {budget}x top-{top_k}")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(sorted(summary["offset"].unique()))
        ax.margins(x=0.26)
        viz.label_line_ends(ax, labels)
        if column == 0:
            ax.set_ylabel("recall of true top-k")

    # One figure-level legend: an in-axes legend has nowhere to sit that does
    # not collide with either the lines or their end-labels. "outside upper
    # left" is the constrained_layout-aware locator — a plain bbox_to_anchor
    # above the axes gets no space reserved for it and renders off-canvas.
    fig.legend(
        handles=handles,
        loc="outside upper left",
        ncols=len(handles),
        frameon=False,
        columnspacing=1.6,
        handlelength=1.6,
    )
    return fig


def plot_expansion(summary: pd.DataFrame, top_k: int):
    """Q6. Expert-set growth with block size, and the bytes it saves."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    blocks = summary["block_size"].to_numpy()

    # Left: union size, bracketed by the two reference lines that give it meaning.
    series = [
        ("observed", summary["expansion_ratio"].to_numpy(), viz.SERIES[0], {}),
        ("random routing", summary["random_ratio"].to_numpy(), viz.SERIES[1],
         {"linestyle": (0, (5, 2))}),
        ("perfect reuse", np.ones_like(blocks, dtype=float), viz.SERIES[2],
         {"linestyle": (0, (2, 2))}),
    ]
    labels = []
    for name, values, color, style in series:
        left.plot(blocks, values, color=color, marker="o", label=name, **style)
        labels.append((blocks[-1], values[-1], name, color))

    left.set_xlabel("block size (tokens verified together)")
    left.set_ylabel(f"unique experts / top-{top_k}")
    left.set_xscale("log", base=2)
    left.set_xticks(blocks, [str(b) for b in blocks])
    left.margins(x=0.28)
    viz.label_line_ends(left, labels)
    left.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=3,
        borderaxespad=0.0, columnspacing=1.6, handlelength=1.6,
    )

    # Right: the payoff, single series — the title names it, so no legend box.
    amortization = summary["bytes_amortization"].to_numpy()
    right.plot(blocks, amortization, color=viz.SERIES[0], marker="o")
    right.plot(blocks, blocks, color=viz.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    right.annotate(
        "ideal (no expansion)",
        xy=(blocks[len(blocks) // 2], blocks[len(blocks) // 2]),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=viz.INK_MUTED,
    )
    viz.label_line_end(
        right, blocks[-1], amortization[-1], f"{amortization[-1]:.1f}x", viz.SERIES[0]
    )
    right.set_xlabel("block size (tokens verified together)")
    right.set_ylabel("bytes/token reduction")
    right.set_title("Payoff from batching")
    right.set_xscale("log", base=2)
    right.set_xticks(blocks, [str(b) for b in blocks])
    right.margins(x=0.22)

    fig.suptitle(
        "Expert-set expansion  ·  does batching amortise expert loads?",
        x=0.01, ha="left", fontsize=11, fontweight="600", color=viz.INK_PRIMARY,
    )
    return fig


def plot_domain(result):
    """Q4. Jensen-Shannon divergence between domain routing profiles.

    Takes an analysis.DomainResult. The heatmap shows the full-data estimate;
    the marker on the colourbar is the sample-matched noise floor, which is the
    number the verdict actually rests on.
    """
    matrix, within_baseline = result.matrix, result.within_matched
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    values = matrix.to_numpy()
    image = ax.imshow(values, cmap=viz.SEQUENTIAL, vmin=0.0)

    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.grid(False)

    # Direct value labels — the sequential ramp carries magnitude, the numbers
    # carry precision, so the chart stays readable without the colourbar.
    # Text colour is picked from the *rendered* cell luminance rather than a
    # fraction of the max: the mid-band of a light-to-dark ramp is exactly where
    # a fixed threshold puts dark ink on a mid-blue and loses contrast.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            red, green, blue, _ = image.cmap(image.norm(values[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j, i, f"{values[i, j]:.3f}",
                ha="center", va="center", fontsize=8,
                color="#ffffff" if luminance < 0.55 else viz.INK_PRIMARY,
            )

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("JS divergence", color=viz.INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    if np.isfinite(within_baseline):
        bar.ax.axhline(within_baseline, color=viz.SERIES[1], linewidth=2.0)
        # Offset clears the colourbar's own tick labels, which sit immediately
        # to its right.
        bar.ax.annotate(
            "within-domain\nnoise floor",
            xy=(1.0, within_baseline),
            xytext=(42, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=viz.INK_SECONDARY,
            annotation_clip=False,
        )
    ax.set_title("Routing divergence between domains")
    return fig


def plot_cache(sweep: pd.DataFrame, *, total_slots: int | None = None):
    """Q5. Hit rate against cache capacity, with Belady as the ceiling."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order = ["belady", "lru", "lfu", "static"]
    present = [name for name in order if name in set(sweep["policy"])]

    labels = []
    for slot, policy in enumerate(present):
        series = sweep[sweep["policy"] == policy].sort_values("capacity")
        color = viz.SERIES[slot % len(viz.SERIES)]
        style = {"linestyle": (0, (5, 2))} if policy == "belady" else {}
        ax.plot(
            series["capacity"], series["hit_rate"],
            color=color, marker="o", label=policy, **style,
        )
        labels.append(
            (series["capacity"].to_numpy()[-1], series["hit_rate"].to_numpy()[-1], policy, color)
        )

    ax.set_xscale("log")
    ax.set_xlabel("cache capacity (experts resident)")
    ax.set_ylabel("hit rate")
    ax.set_ylim(0, 1.02)
    ax.margins(x=0.30)

    if total_slots:
        ax.axvline(total_slots, color=viz.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(
            "whole model resident",
            xy=(total_slots, 0.42),
            xytext=(-6, 0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            color=viz.INK_MUTED,
        )

    # Policies converge to 1.0 at full capacity, so the end-labels need
    # de-colliding or they overprint each other into mush.
    viz.label_line_ends(ax, labels)
    viz.titled_legend_above(
        fig, ax, "Cache hit rate vs capacity  ·  belady is the ceiling", ncols=len(present)
    )
    return fig
