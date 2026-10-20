"""Shared visual identity for MacroCanvas figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "ink": "#1F2933",
    "navy": "#17324D",
    "teal": "#2A7F8E",
    "teal_light": "#8CC3C8",
    "gold": "#BE8A3B",
    "red": "#B5544A",
    "red_light": "#E7B7B1",
    "green": "#3F7D5C",
    "gray": "#6B7280",
    "gray_light": "#D9DEE3",
    "sand": "#D8C3A5",
    "surface": "#F7F5F0",
    "white": "#FFFFFF",
}
EQUATION_COLORS = {
    "is": COLORS["navy"],
    "phillips": COLORS["red"],
    "okun": COLORS["teal"],
    "taylor": COLORS["gold"],
}
SCENARIO_COLORS = {
    "baseline": COLORS["navy"],
    "tightening": COLORS["red"],
}
COMPONENT_COLORS = {
    "Инерция выпуска": "#C9D6DF",
    "Реальная ставка": COLORS["navy"],
    "Фискальный импульс": COLORS["gold"],
    "Внешний спрос": COLORS["teal"],
    "Финансовые условия": COLORS["red"],
    "Шок спроса": COLORS["gray"],
}


def apply_style() -> None:
    """Apply a consistent paper-like theme to all subsequent figures."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=[COLORS["navy"], COLORS["teal"], COLORS["gold"], COLORS["red"]],
        rc={
            "figure.facecolor": COLORS["surface"],
            "axes.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["surface"],
            "axes.edgecolor": COLORS["gray_light"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": COLORS["gray_light"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.titlecolor": COLORS["ink"],
            "axes.labelsize": 10.5,
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["gray"],
            "ytick.color": COLORS["gray"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "font.family": "DejaVu Sans",
            "text.color": COLORS["ink"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        },
    )
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
