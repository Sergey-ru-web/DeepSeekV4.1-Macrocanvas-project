"""Small reusable plotting helpers."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from macrocanvas.viz.style import COLORS


def quarter_axis(
    ax: plt.Axes,
    index: pd.PeriodIndex,
    *,
    max_ticks: int = 7,
    rotation: int = 0,
) -> None:
    """Use equally spaced quarter positions with readable period labels."""
    if len(index) == 0:
        return
    count = min(max_ticks, len(index))
    positions = np.unique(np.linspace(0, len(index) - 1, count).round().astype(int))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(index[position]) for position in positions], rotation=rotation)


def recession_spans(
    index: pd.PeriodIndex,
    episodes: pd.DataFrame,
) -> list[tuple[float, float]]:
    """Translate a table of recession episodes into x-axis spans."""
    spans = []
    for _, row in episodes.iterrows():
        start = list(map(str, index)).index(str(row["start"]))
        end = list(map(str, index)).index(str(row["end"]))
        spans.append((start - 0.5, end + 0.5))
    return spans


def shade_recessions(
    ax: plt.Axes,
    index: pd.PeriodIndex,
    episodes: pd.DataFrame,
    *,
    alpha: float = 0.10,
) -> None:
    for left, right in recession_spans(index, episodes):
        ax.axvspan(left, right, color=COLORS["gray"], alpha=alpha, linewidth=0, zorder=0)


def zero_line(ax: plt.Axes, *, axis: str = "y", label: str | None = None) -> None:
    kwargs = dict(color=COLORS["gray"], linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.7)
    if axis == "y":
        ax.axhline(0, **kwargs)
        if label:
            ax.text(
                0.995,
                0.02,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
                color=COLORS["gray"],
            )
    else:
        ax.axvline(0, **kwargs)


def add_last_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    offset: tuple[float, float] = (6, 0),
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


def add_footer(
    fig: plt.Figure,
    text: str,
    *,
    y: float = 0.005,
) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLORS["gray"],
    )


def add_suptitle(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    top: float = 0.96,
) -> None:
    fig.suptitle(title, fontsize=20, fontweight="bold", color=COLORS["ink"], y=top)
    fig.text(
        0.5,
        top - 0.045,
        subtitle,
        ha="center",
        va="top",
        fontsize=10.5,
        color=COLORS["gray"],
    )


def fan_legend(ax: plt.Axes, *, color: str = COLORS["navy"]) -> None:
    handles = [
        Patch(facecolor=color, alpha=0.10, label="90% интервал"),
        Patch(facecolor=color, alpha=0.20, label="68% интервал"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, handlelength=1.2)


def draw_fan(
    ax: plt.Axes,
    x: Iterable[float],
    lower_outer: Iterable[float],
    upper_outer: Iterable[float],
    lower_inner: Iterable[float],
    upper_inner: Iterable[float],
    median: Iterable[float],
    *,
    color: str = COLORS["navy"],
    central_band: tuple[Iterable[float], Iterable[float]] | None = None,
    label: str | None = None,
    linewidth: float = 2.0,
) -> None:
    x = np.asarray(list(x))
    ax.fill_between(
        x,
        np.asarray(list(lower_outer)),
        np.asarray(list(upper_outer)),
        color=color,
        alpha=0.10,
        linewidth=0,
    )
    ax.fill_between(
        x,
        np.asarray(list(lower_inner)),
        np.asarray(list(upper_inner)),
        color=color,
        alpha=0.20,
        linewidth=0,
    )
    if central_band is not None:
        ax.fill_between(
            x,
            np.asarray(list(central_band[0])),
            np.asarray(list(central_band[1])),
            color=color,
            alpha=0.32,
            linewidth=0,
        )
    ax.plot(x, np.asarray(list(median)), color=color, linewidth=linewidth, label=label)
