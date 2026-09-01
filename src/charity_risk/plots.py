"""Figures for the notebooks and the ``outputs/figures`` directory.

One visual system, applied everywhere: a fixed categorical hue order (never
cycled or reassigned between charts, so a model keeps its colour from figure to
figure), a single blue ramp for magnitude, recessive axes and grid, and thin
marks.  Several of the categorical hues fall below 3:1 contrast on the light
surface, so every multi-series chart carries direct labels as well as a legend
— identity is never colour alone.

Every function takes an ``ax`` so figures can be composed in a notebook, and
returns it.  ``save_figure`` writes to :data:`charity_risk.config.FIGURE_DIR`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import FIGURE_DIR

__all__ = [
    "SERIES_COLORS",
    "SEQUENTIAL_BLUE",
    "SURFACE",
    "use_project_style",
    "series_color",
    "save_figure",
    "plot_exit_rates",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_gains_curves",
    "plot_calibration",
    "plot_decile_lift",
    "plot_metric_bars",
    "plot_permutation_importance",
    "plot_score_response",
]

#: Fixed categorical order.  Validated for adjacent-pair colour-vision
#: separation on the light surface; slots are assigned by position, never
#: recycled, so "model 3" is the same aqua in every figure.
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Single-hue ramp for continuous magnitude, light to dark.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
    "#1c5cab", "#184f95", "#104281", "#0d366b",
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"


def use_project_style() -> None:
    """Apply the project's matplotlib defaults.  Call once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        # Grid behind the marks: a gridline showing through a filled bar reads
        # as a stacked segment boundary that is not there.
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
    })


def series_color(index: int) -> str:
    """Colour for the ``index``-th series, in fixed slot order."""
    if index >= len(SERIES_COLORS):
        raise IndexError(
            f"only {len(SERIES_COLORS)} categorical slots exist; a {index + 1}th "
            "series must fold into 'other' or move to small multiples"
        )
    return SERIES_COLORS[index]


def save_figure(fig: plt.Figure, name: str, directory: Path = FIGURE_DIR) -> Path:
    """Write ``fig`` to ``directory/name.png`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    return path


def _direct_label(ax: plt.Axes, x: float, y: float, text: str, color: str,
                  ha: str = "left") -> None:
    ax.annotate(text, xy=(x, y), xytext=(4 if ha == "left" else -4, 5),
                textcoords="offset points", color=color, fontsize=8.5,
                va="bottom", ha=ha)


def _new_ax(ax: plt.Axes | None, figsize: tuple[float, float]) -> plt.Axes:
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_exit_rates(summary: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Exit rate by year, distinguishing confirmed from provisional labels.

    The final year of the window is shown hollow: with only one later year to
    check, an absence there cannot be told apart from a late filing.
    """
    ax = _new_ax(ax, (7.0, 4.0))
    years = summary.index.to_numpy()
    confirmed = summary["exit_rate"].to_numpy()
    provisional = summary["exit_rate_provisional"].to_numpy()

    trustworthy = np.isfinite(confirmed)
    ax.bar(years[trustworthy], confirmed[trustworthy], width=0.62,
           color=series_color(0), label="Confirmed permanent exit")
    unconfirmed = ~trustworthy & np.isfinite(provisional) & (provisional < 1)
    ax.bar(years[unconfirmed], provisional[unconfirmed], width=0.62,
           facecolor="none", edgecolor=series_color(0), linewidth=1.6,
           hatch="///", label="Provisional (one lookahead year only)")

    for year, value in zip(years, np.where(trustworthy, confirmed, provisional)):
        if np.isfinite(value) and value < 1:
            ax.annotate(f"{value:.1%}", xy=(year, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK_SECONDARY)

    ax.set_xticks([y for y in years if summary.loc[y, "exit_rate_provisional"] < 1])
    ax.set_ylabel("Share of filers not seen again")
    ax.set_title("Charity exit rate by filing year")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return ax


def _curve_plot(ax: plt.Axes, curves: dict[str, tuple[np.ndarray, np.ndarray]],
                highlight: set[str] | None = None) -> None:
    highlight = highlight or set()
    for index, (label, (x, y)) in enumerate(curves.items()):
        color = series_color(index)
        emphasised = (not highlight) or (label in highlight)
        ax.plot(x, y, color=color, linewidth=2.0 if emphasised else 1.4,
                linestyle="-" if emphasised else (0, (4, 2)),
                alpha=1.0 if emphasised else 0.85, label=label, zorder=3)


def plot_roc_curves(y_true: np.ndarray, risks: dict[str, np.ndarray],
                    ax: plt.Axes | None = None) -> plt.Axes:
    """ROC curves for a set of models, plus the no-skill diagonal."""
    ax = _new_ax(ax, (6.4, 5.6))
    curves = {}
    for label, risk in risks.items():
        fpr, tpr, _ = roc_curve(y_true, risk)
        curves[label] = (fpr, tpr)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)),
            zorder=1)
    _curve_plot(ax, curves)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discrimination: ROC, out-of-time test year")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return ax


def plot_pr_curves(y_true: np.ndarray, risks: dict[str, np.ndarray],
                   ax: plt.Axes | None = None) -> plt.Axes:
    """Precision-recall curves, the honest view at a 2% base rate."""
    ax = _new_ax(ax, (6.4, 5.6))
    curves = {}
    for label, risk in risks.items():
        precision, recall, _ = precision_recall_curve(y_true, risk)
        curves[label] = (recall, precision)
    base = float(np.mean(y_true))
    ax.axhline(base, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    _direct_label(ax, 0.02, base, f"no skill ({base:.1%})", INK_MUTED)
    _curve_plot(ax, curves)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, out-of-time test year")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    return ax


def plot_gains_curves(y_true: np.ndarray, risks: dict[str, np.ndarray],
                      ax: plt.Axes | None = None) -> plt.Axes:
    """Share of exits captured against share of the register reviewed.

    This is the chart a supervisor with a fixed review budget should read: the
    x-axis is workload, the y-axis is what that workload catches.
    """
    ax = _new_ax(ax, (6.8, 5.4))
    n = len(y_true)
    total = float(np.sum(y_true))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)),
            zorder=1)
    _direct_label(ax, 0.62, 0.62, "review at random", INK_MUTED)

    for index, (label, risk) in enumerate(risks.items()):
        order = np.argsort(-risk, kind="mergesort")
        captured = np.cumsum(y_true[order]) / total
        reviewed = np.arange(1, n + 1) / n
        color = series_color(index)
        ax.plot(reviewed, captured, color=color, linewidth=2.0, label=label, zorder=3)

    ax.set_xlabel("Share of the register reviewed")
    ax.set_ylabel("Share of exits captured")
    ax.set_title("Screening yield")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower right")
    return ax


def plot_calibration(tables: dict[str, pd.DataFrame], ax: plt.Axes | None = None,
                     log_scale: bool = True) -> plt.Axes:
    """Observed against predicted rate, by decile of predicted risk."""
    ax = _new_ax(ax, (6.2, 5.6))
    limits = []
    for index, (label, table) in enumerate(tables.items()):
        color = series_color(index)
        ax.plot(table["mean_predicted"], table["observed"], color=color,
                marker="o", markersize=5, linewidth=1.6, label=label, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        limits.extend(table["mean_predicted"].tolist() + table["observed"].tolist())

    finite = [v for v in limits if np.isfinite(v) and v > 0]
    low, high = (min(finite), max(finite)) if finite else (1e-4, 1.0)
    ax.plot([low, high], [low, high], color=INK_MUTED, linewidth=1.0,
            linestyle=(0, (3, 3)), zorder=1)
    _direct_label(ax, high, high, "perfect", INK_MUTED, ha="right")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed exit rate")
    ax.set_title("Calibration by risk decile")
    ax.legend(loc="upper left")
    return ax


def plot_decile_lift(table: pd.DataFrame, ax: plt.Axes | None = None,
                     title: str = "Lift by risk decile") -> plt.Axes:
    """Lift over the base rate in each decile, worst risk on the left."""
    ax = _new_ax(ax, (7.0, 4.2))
    deciles = np.arange(1, len(table) + 1)
    values = table["lift"].to_numpy()
    # Sequential ramp: darker means higher lift, so the encoding is redundant
    # with position rather than decorative.
    ranks = np.argsort(np.argsort(values))
    colors = [SEQUENTIAL_BLUE[int(r * (len(SEQUENTIAL_BLUE) - 1) / max(len(values) - 1, 1))]
              for r in ranks]
    ax.bar(deciles, values, width=0.7, color=colors)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
    _direct_label(ax, len(table) + 0.4, 1.0, "base rate", INK_MUTED)
    for decile, value in zip(deciles, values):
        ax.annotate(f"{value:.1f}x", xy=(decile, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_SECONDARY)
    ax.set_xticks(deciles)
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Exit rate relative to base rate")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0.4, len(table) + 1.4)
    return ax


def plot_metric_bars(results: pd.DataFrame, metric: str = "average_precision",
                     ax: plt.Axes | None = None, title: str | None = None,
                     label_format: str = "{:.3f}") -> plt.Axes:
    """Horizontal comparison of one metric across models, best at the top."""
    ax = _new_ax(ax, (7.4, 0.55 * len(results) + 1.8))
    ordered = results.sort_values(metric)
    labels = ordered["label"].tolist()
    values = ordered[metric].to_numpy()
    positions = np.arange(len(ordered))

    # Fitted models in the primary hue; the rule-based and null benchmarks in a
    # muted step, so the eye separates "method" from "reference point".
    colors = [series_color(0) if family == "fitted" else SEQUENTIAL_BLUE[2]
              for family in ordered["family"]]
    ax.barh(positions, values, height=0.62, color=colors)
    for position, value in zip(positions, values):
        ax.annotate(label_format.format(value), xy=(value, position), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by model")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, values.max() * 1.18)
    return ax


def plot_permutation_importance(table: pd.DataFrame, top_n: int = 20,
                                ax: plt.Axes | None = None) -> plt.Axes:
    """Loss in average precision when each feature is shuffled."""
    subset = table.head(top_n).iloc[::-1]
    ax = _new_ax(ax, (7.2, 0.34 * len(subset) + 1.8))
    positions = np.arange(len(subset))
    values = subset["ap_drop_mean"].to_numpy()
    errors = subset["ap_drop_std"].to_numpy()

    ranks = np.argsort(np.argsort(values))
    colors = [SEQUENTIAL_BLUE[3 + int(r * 6 / max(len(values) - 1, 1))] for r in ranks]
    ax.barh(positions, values, height=0.66, color=colors,
            xerr=errors, error_kw={"ecolor": INK_MUTED, "elinewidth": 1.0,
                                   "capsize": 2})
    ax.axvline(0, color=INK_MUTED, linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(subset["feature"], fontsize=8.5)
    ax.set_xlabel("Drop in average precision when shuffled")
    ax.set_title(f"What the model relies on (top {len(subset)})")
    ax.grid(axis="y", visible=False)
    return ax


def plot_score_response(frame: pd.DataFrame, score: str, outcome: str,
                        ax: plt.Axes | None = None,
                        title: str | None = None) -> plt.Axes:
    """Outcome rate by discrete score level, with the count on each bar.

    Used for the Tuckman-Chang 0-4 count, where the whole question is whether
    more flags really do mean more risk.
    """
    ax = _new_ax(ax, (6.6, 4.2))
    grouped = frame.groupby(score, observed=True)[outcome]
    rates = grouped.mean()
    counts = grouped.size()
    positions = np.arange(len(rates))

    ax.bar(positions, rates.to_numpy(), width=0.66,
           color=[SEQUENTIAL_BLUE[2 + int(i * 6 / max(len(rates) - 1, 1))]
                  for i in range(len(rates))])
    base = frame[outcome].mean()
    ax.axhline(base, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.annotate(f"all filers ({base:.1%})", xy=(-0.55, base), xytext=(0, -5),
                textcoords="offset points", color=INK_MUTED, fontsize=8.5,
                ha="left", va="top")

    top = max(rates.max() * 1.30, base * 2.2)
    for position, (rate, count) in enumerate(zip(rates, counts)):
        ax.annotate(f"{rate:.1%}", xy=(position, rate), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK_SECONDARY, fontweight="bold")
        # Counts sit on the axis, clear of the base-rate line and of each other.
        ax.annotate(f"n={count:,}", xy=(position, 0), xytext=(0, -22),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK_MUTED, annotation_clip=False)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(level)}" for level in rates.index])
    ax.set_xlabel("Number of Tuckman-Chang measures in the at-risk quintile",
                  labelpad=18)
    ax.set_ylabel("Exit rate")
    ax.set_title(title or "Does the score rank risk?")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(rates) - 0.4)
    ax.set_ylim(0, top)
    return ax
