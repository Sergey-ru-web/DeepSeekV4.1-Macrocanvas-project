"""A vector-rendered conceptual map of the MacroCanvas transmission mechanism."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from macrocanvas.viz.style import COLORS, apply_style, save_figure
from macrocanvas.viz.utils import add_footer, add_suptitle


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    shadow = FancyBboxPatch(
        (x + 0.55, y - 0.65),
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.7",
        facecolor=COLORS["gray"],
        edgecolor="none",
        alpha=0.12,
        zorder=1,
    )
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.7",
        facecolor=color,
        edgecolor=color,
        alpha=0.10,
        linewidth=1.2,
        zorder=2,
    )
    border = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.7",
        facecolor="none",
        edgecolor=color,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(shadow)
    ax.add_patch(box)
    ax.add_patch(border)
    ax.text(
        x + width / 2,
        y + height - 1.8,
        title,
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 1.5,
        body,
        ha="center",
        va="center",
        fontsize=8.3,
        color=COLORS["gray"],
        linespacing=1.5,
        zorder=4,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    style: str = "-|>",
    linestyle: str = "-",
    rad: float = 0.0,
    linewidth: float = 1.35,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=4,
        zorder=5,
    )
    ax.add_patch(arrow)


def plot_model_map(path: str) -> None:
    """Draw a stylised macroeconomic transmission map."""
    apply_style()
    fig, ax = plt.subplots(figsize=(15.2, 9.6))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.045)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 95)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["surface"])

    add_suptitle(
        fig,
        "Карта механизма трансмиссии синтетической модели",
        "IS–PC–Okun–Taylor: спрос, цены, рынок труда и реакция денежной политики",
        top=0.965,
    )

    # Demand block
    _box(
        ax,
        2,
        64,
        18,
        25,
        "Экзогенные драйверы",
        "• внешний спрос x*\n• фискальный импульс f\n• финансовый стресс s",
        COLORS["teal"],
    )
    _box(ax, 2, 46, 18, 12, "Нефтяной шок", "$o_t$ — цены на энергию", COLORS["gold"])
    _box(ax, 24, 64, 18, 13, "IS-кривая", "$x_t = \\rho x_{t-1} - \\sigma(r_{t-1}-r^*) + \\dots$", COLORS["navy"])
    _box(ax, 47, 64, 17, 12, "Разрыв выпуска", "$x_t$ — циклическая позиция", COLORS["navy"])
    _box(
        ax,
        47,
        46,
        17,
        11,
        "Кривая Филлипса",
        "$\\pi_t = \\alpha\\pi_{t-1} + \\kappa x_{t-1} + \\gamma_o o_t + \\varepsilon_t$",
        COLORS["red"],
    )
    _box(ax, 70, 46, 24, 11, "Инфляция", "$\\pi_t$ — годовой темп, п.п.", COLORS["red"])
    _box(
        ax,
        70,
        64,
        24,
        12,
        "Правило Тейлора",
        "$i_t = f(\\pi_t, x_t),\\quad i_t \\geq i_{ELB}$",
        COLORS["gold"],
    )
    _box(ax, 70, 80, 24, 10, "Ключевая ставка", "$i_t$ — номинальная ставка", COLORS["gold"])
    _box(
        ax,
        70,
        28,
        24,
        10,
        "Закон Оукена",
        "$u_t - u^* = -\\beta x_t + \\eta_t$",
        COLORS["teal"],
    )
    _box(ax, 70, 11, 24, 10, "Безработица", "$u_t = u^* + (u_t-u^*)$", COLORS["teal"])
    _box(
        ax,
        2,
        11,
        18,
        13,
        "Реальная ставка",
        "$r_t = i_{t-1} - E_t\\pi_{t+1}$\nадаптивные ожидания",
        COLORS["navy"],
    )

    # Main flows
    _arrow(ax, (20, 74), (24, 70.5), color=COLORS["teal"])
    _arrow(ax, (20, 52), (24, 52), color=COLORS["gold"])
    _arrow(ax, (42, 70.5), (47, 70), color=COLORS["navy"])
    _arrow(ax, (55.5, 64), (55.5, 57), color=COLORS["red"])
    _arrow(ax, (64, 52), (70, 51.5), color=COLORS["red"])
    _arrow(ax, (82, 57), (82, 64), color=COLORS["red"])
    _arrow(ax, (64, 68.5), (70, 70), color=COLORS["navy"])
    _arrow(ax, (82, 76), (82, 80), color=COLORS["gold"])
    _arrow(ax, (64, 66), (70, 35), color=COLORS["teal"], rad=-0.18)
    _arrow(ax, (82, 28), (82, 21), color=COLORS["teal"])

    # Policy feedback loop: policy rate -> real rate -> IS.
    ax.plot(
        [94, 98, 98, 11],
        [85, 85, 4.5, 4.5],
        color=COLORS["navy"],
        linewidth=1.35,
        linestyle=(0, (4, 3)),
        zorder=5,
        clip_on=False,
    )
    _arrow(ax, (11, 4.5), (11, 11), color=COLORS["navy"], linestyle=(0, (4, 3)))
    ax.text(
        55,
        2.5,
        "обратная связь политики",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["navy"],
    )
    _arrow(ax, (20, 20), (24, 68), color=COLORS["navy"], rad=-0.08)

    # Expectations channel.
    _arrow(
        ax,
        (70, 48.5),
        (20, 21.5),
        color=COLORS["gray"],
        linestyle=(0, (4, 3)),
        rad=0.18,
    )
    ax.text(
        37.5,
        29,
        "инфляционные ожидания",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["gray"],
        rotation=-8,
    )

    add_footer(
        fig,
        "Сплошные стрелки — основные каналы; штриховые — ожидания и обратная связь ДКП.\n"
        "Концептуальная схема не является оценкой причинных эффектов; калибровка и данные — синтетические.",
        y=0.006,
    )
    save_figure(fig, path)
