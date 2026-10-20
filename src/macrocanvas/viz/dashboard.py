"""Six-panel macro dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macrocanvas.viz.style import COLORS, apply_style, save_figure
from macrocanvas.viz.utils import (
    add_footer,
    add_last_value,
    add_suptitle,
    quarter_axis,
    recession_spans,
    shade_recessions,
)


def _label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.085,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLORS["gray"],
    )


def plot_macro_dashboard(
    data: pd.DataFrame,
    *,
    seed: int,
    episodes: pd.DataFrame,
    path: str,
) -> None:
    """Create the main, publication-style overview figure."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.08, hspace=0.48, wspace=0.24)
    add_suptitle(
        fig,
        "Синтетическая макроэкономика: квартальная циклическая динамика",
        (
            f"{data.index[0]}–{data.index[-1]}  •  {len(data)} кварталов  •  "
            f"seed {seed}  •  полностью сгенерированные данные"
        ),
        top=0.965,
    )
    x = np.arange(len(data))
    index = pd.PeriodIndex(data.index)

    # A. GDP growth
    ax = axes[0, 0]
    shade_recessions(ax, index, episodes)
    growth = data["gdp_growth"].to_numpy()
    ax.fill_between(
        x,
        0,
        growth,
        where=growth >= 0,
        color=COLORS["teal"],
        alpha=0.25,
        linewidth=0,
    )
    ax.fill_between(
        x,
        0,
        growth,
        where=growth < 0,
        color=COLORS["red"],
        alpha=0.28,
        linewidth=0,
    )
    ax.plot(x, growth, color=COLORS["navy"], linewidth=1.2)
    moving_average = pd.Series(growth).rolling(4, min_periods=1).mean()
    ax.plot(x, moving_average, color=COLORS["gold"], linewidth=1.6, label="4-квартальное среднее")
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_title("Рост реального ВВП", loc="left")
    ax.set_ylabel("q/q, SAAR, %")
    quarter_axis(ax, index, max_ticks=6)
    add_last_value(
        ax,
        x[-1],
        growth[-1],
        f"{growth[-1]:.1f}%",
        color=COLORS["navy"],
        offset=(5, 8),
    )
    ax.legend(loc="lower left", fontsize=8.5)
    _label_panel(ax, "A")

    # B. Inflation
    ax = axes[0, 1]
    inflation = data["inflation_yoy"].to_numpy()
    ax.plot(x, inflation, color=COLORS["red"], linewidth=1.8)
    ax.axhline(2.0, color=COLORS["gold"], linewidth=1.2, linestyle="--", label="Цель 2%")
    ax.fill_between(x, 2.0, inflation, color=COLORS["red"], alpha=0.10, linewidth=0)
    ax.set_title("Инфляция потребительских цен", loc="left")
    ax.set_ylabel("г/г, %")
    quarter_axis(ax, index, max_ticks=6)
    add_last_value(
        ax,
        x[-1],
        inflation[-1],
        f"{inflation[-1]:.1f}%",
        color=COLORS["red"],
        offset=(5, 8),
    )
    ax.legend(loc="upper right", fontsize=8.5)
    _label_panel(ax, "B")

    # C. Unemployment
    ax = axes[0, 2]
    unemployment = data["unemployment"].to_numpy()
    ax.plot(x, unemployment, color=COLORS["teal"], linewidth=1.8)
    natural_rate = float((data["unemployment"] - data["unemployment_gap"]).iloc[-1])
    ax.axhline(
        natural_rate,
        color=COLORS["gold"],
        linewidth=1.2,
        linestyle="--",
        label=f"Естественный уровень ≈ {natural_rate:.1f}%",
    )
    ax.set_title("Уровень безработицы", loc="left")
    ax.set_ylabel("% рабочей силы")
    quarter_axis(ax, index, max_ticks=6)
    add_last_value(
        ax,
        x[-1],
        unemployment[-1],
        f"{unemployment[-1]:.1f}%",
        color=COLORS["teal"],
        offset=(5, 8),
    )
    ax.legend(loc="lower right", fontsize=8.5)
    _label_panel(ax, "C")

    # D. Policy and real rate
    ax = axes[1, 0]
    ax.plot(x, data["policy_rate"], color=COLORS["navy"], linewidth=1.8, label="Ключевая ставка")
    ax.plot(
        x,
        data["real_policy_rate"],
        color=COLORS["gold"],
        linewidth=1.5,
        label="Реальная ставка ex post",
    )
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_title("Монетарные условия", loc="left")
    ax.set_ylabel("% годовых")
    quarter_axis(ax, index, max_ticks=6)
    add_last_value(
        ax,
        x[-1],
        float(data["policy_rate"].iloc[-1]),
        f"{data['policy_rate'].iloc[-1]:.1f}%",
        color=COLORS["navy"],
        offset=(5, 8),
    )
    ax.legend(loc="upper right", fontsize=8.5)
    _label_panel(ax, "D")

    # E. Output gap
    ax = axes[1, 1]
    shade_recessions(ax, index, episodes)
    gap = data["output_gap"].to_numpy()
    ax.fill_between(x, 0, gap, color=COLORS["navy"], alpha=0.14, linewidth=0)
    ax.plot(x, gap, color=COLORS["navy"], linewidth=1.8)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.9)
    ax.set_title("Разрыв выпуска", loc="left")
    ax.set_ylabel("п.п. от потенциального ВВП")
    quarter_axis(ax, index, max_ticks=6)
    add_last_value(
        ax,
        x[-1],
        gap[-1],
        f"{gap[-1]:+.1f}",
        color=COLORS["navy"],
        offset=(5, 8),
    )
    _label_panel(ax, "E")

    # F. GDP and potential GDP
    ax = axes[1, 2]
    ax.plot(
        x,
        data["gdp_index"],
        color=COLORS["teal"],
        linewidth=1.8,
        label="Реальный ВВП",
    )
    ax.plot(
        x,
        data["potential_gdp_index"],
        color=COLORS["gray"],
        linewidth=1.4,
        linestyle="--",
        label="Потенциальный ВВП",
    )
    ax.fill_between(
        x,
        data["potential_gdp_index"],
        data["gdp_index"],
        color=COLORS["teal"],
        alpha=0.08,
        linewidth=0,
    )
    ax.set_title("Уровень выпуска", loc="left")
    ax.set_ylabel("индекс, 1990Q1 = 100")
    quarter_axis(ax, index, max_ticks=6)
    ax.legend(loc="upper left", fontsize=8.5)
    _label_panel(ax, "F")

    for ax in axes.flat:
        ax.tick_params(axis="both", length=0)
        ax.set_xlim(-1, len(data))
        ax.margins(x=0.01)

    add_footer(
        fig,
        "Синхронизированный синтетический DGP: IS-кривая, гибридная кривая Филлипса, закон Оукена "
        "и инерционное правило Тейлора. Затенены рецессии длительностью ≥ 2 кварталов.",
        y=0.007,
    )
    save_figure(fig, path)
