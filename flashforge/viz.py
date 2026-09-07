"""Chart styling: one validated palette, applied consistently.

The categorical order below is fixed and must not be cycled or reordered — the
ordering itself is the colour-vision-deficiency safety mechanism (worst adjacent
CVD dE 9.1, normal-vision dE 22.9 for the four slots used here). Three of these
slots sit below 3:1 contrast on the light surface, so every chart built from
them ships either direct labels or the accompanying table — the analysis
functions all return their DataFrame for exactly that reason.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in fixed assignment order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Sequential ramp (single hue, light to dark) for magnitude encodings.
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("ff_sequential", SEQUENTIAL_STEPS)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "system-ui", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SECONDARY,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "lines.solid_capstyle": "round",
    "figure.dpi": 120,
    "figure.constrained_layout.use": True,
}


def use_style() -> None:
    """Apply the chart style globally. Call once per notebook/script."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)


def label_line_end(ax, x, y, text: str, color: str, *, dx: float = 0.0) -> None:
    """Direct-label a single line at its right end.

    Text wears its own ink colour rather than the series colour; the marker
    beside it carries identity. Keeps text legible where the series colour is
    below 3:1 on the surface.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8 + dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.plot([x], [y], marker="o", markersize=5, color=color, zorder=5)


def label_line_ends(ax, entries, *, min_gap_frac: float = 0.055) -> None:
    """Direct-label several lines at their right ends, de-colliding vertically.

    entries: iterable of (x, y, text, color).

    Converging series would otherwise stack their labels into unreadable mush —
    which is exactly what happens on a cache sweep, where every policy tends to
    1.0 at full capacity. Markers stay on the true values; only the text is
    nudged apart.
    """
    entries = list(entries)
    if not entries:
        return
    low, high = ax.get_ylim()
    gap = (high - low) * min_gap_frac

    ordered = sorted(entries, key=lambda item: item[1])
    positions: list[float] = []
    previous = None
    for _, y, _, _ in ordered:
        text_y = y if previous is None else max(y, previous + gap)
        positions.append(text_y)
        previous = text_y

    # Series that converge near the top would otherwise be pushed clean off the
    # axes and vanish. Slide the whole group back into range, preserving order.
    overflow = positions[-1] - (high - gap * 0.5)
    if overflow > 0:
        positions = [p - overflow for p in positions]
    underflow = (low + gap * 0.5) - positions[0]
    if underflow > 0:
        positions = [p + underflow for p in positions]

    for (x, y, text, color), text_y in zip(ordered, positions):
        ax.plot([x], [y], marker="o", markersize=5, color=color, zorder=5)
        ax.annotate(
            text,
            xy=(x, text_y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def titled_legend_above(fig, ax, title: str, *, ncols: int = 4) -> None:
    """Chart title above a legend strip, both above the plot area.

    Charts here carry a legend *and* direct end-labels, so an in-axes legend has
    two chances to collide with the data. Stacking title over legend over plot
    keeps all three apart regardless of panel shape — but the title has to move
    to the figure, since an axes title would occupy the same band as the legend.
    """
    ax.set_title("")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncols=ncols,
        borderaxespad=0.0,
        columnspacing=1.6,
        handlelength=1.6,
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="600", color=INK_PRIMARY)
