"""Analytical figures: estimated curves, IRFs, projections and diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from macrocanvas.analytics.decomposition import latest_waterfall
from macrocanvas.analytics.diagnostics import correlation_matrix
from macrocanvas.models.ols import EquationEstimate
from macrocanvas.models.scenarios import ProjectionBundle
from macrocanvas.models.var import IRFResult
from macrocanvas.viz.style import (
    COLORS,
    COMPONENT_COLORS,
    EQUATION_COLORS,
    SCENARIO_COLORS,
    apply_style,
    save_figure,
)
from macrocanvas.viz.utils import (
    add_footer,
    add_last_value,
    add_suptitle,
    draw_fan,
    fan_legend,
    quarter_axis,
    zero_line,
)

SHORT_COMPONENTS = {
    "Инерция выпуска": "Инерция",
    "Реальная ставка": "Реальная\nставка",
    "Фискальный импульс": "Фискальный\nимпульс",
    "Внешний спрос": "Внешний\nспрос",
    "Финансовые условия": "Фин.\nусловия",
    "Шок спроса": "Шок\nспроса",
}


def _format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.3f}"


def _relation_panel(
    ax: plt.Axes,
    estimate: EquationEstimate,
    *,
    letter: str,
    point_color: str,
) -> None:
    partial = estimate.partial_data()
    x_values = partial["focus"].to_numpy()
    y_values = partial["dependent"].to_numpy()
    slope = estimate.coefficient
    standard_error = estimate.standard_error
    grid = np.linspace(x_values.min(), x_values.max(), 120)
    fitted = slope * grid
    band = 1.96 * standard_error * np.abs(grid)

    ax.scatter(
        x_values,
        y_values,
        s=26,
        color=point_color,
        alpha=0.52,
        edgecolors="none",
        zorder=3,
    )
    ax.plot(grid, fitted, color=point_color, linewidth=2.1, zorder=4)
    ax.fill_between(grid, fitted - band, fitted + band, color=point_color, alpha=0.13, linewidth=0)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=(0, (2, 2)))
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_title(f"{letter}  {estimate.title}", loc="left", fontsize=13)
    ax.set_xlabel(f"{estimate.focus_label} (частичный остаток)")
    ax.set_ylabel(f"{estimate.dependent_label} (частичный остаток)")
    ax.text(
        0.035,
        0.965,
        (
            f"β = {slope:+.3f}\n"
            f"HAC SE = {standard_error:.3f}\n"
            f"t = {estimate.t_stat:+.2f}; p {_format_p_value(estimate.p_value)}\n"
            f"R² = {estimate.r_squared:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        linespacing=1.45,
        bbox=dict(
            facecolor=COLORS["white"],
            edgecolor=COLORS["gray_light"],
            boxstyle="round,pad=0.35",
            alpha=0.92,
        ),
        zorder=6,
    )


def plot_estimated_relations(
    estimates: dict[str, EquationEstimate],
    *,
    hac_lags: int,
    path: str,
) -> None:
    """Four added-variable plots with HAC confidence bands."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.08, right=0.975, top=0.84, bottom=0.09, hspace=0.38, wspace=0.25)
    add_suptitle(
        fig,
        "Оценённые макроэкономические зависимости",
        "Частичные регрессии (Frisch–Waugh–Lovell); полосы — 95% доверительные интервалы по HAC",
        top=0.965,
    )
    specification = [
        ("is", "A"),
        ("phillips", "B"),
        ("okun", "C"),
        ("taylor", "D"),
    ]
    for ax, (key, letter) in zip(axes.flat, specification, strict=True):
        estimate = estimates[key]
        _relation_panel(
            ax,
            estimate,
            letter=letter,
            point_color=EQUATION_COLORS[key],
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    add_footer(
        fig,
        (
            "Стандартные ошибки Newey–West с лагом "
            f"{hac_lags}. В панели Тейлора показана приведённая форма; "
            "коэффициенты не интерпретируются как структурные параметры."
        ),
        y=0.007,
    )
    save_figure(fig, path)


def plot_irf(irf: IRFResult, *, path: str) -> None:
    """Plot identified monetary-policy impulse responses with bootstrap bands."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.10, hspace=0.40, wspace=0.22)
    add_suptitle(
        fig,
        "Импульсные отклики на ужесточение денежной политики",
        (
            "Шок +1 п.п. к ключевой ставке; рекурсивная идентификация Холецкого "
            "[разрыв выпуска, безработица, инфляция, ставка]"
        ),
        top=0.965,
    )
    panels = [
        ("output_gap", "Разрыв выпуска", "п.п.", COLORS["navy"]),
        ("unemployment_gap", "Разрыв безработицы", "п.п.", COLORS["teal"]),
        ("inflation", "Инфляция", "п.п. в год", COLORS["red"]),
        ("policy_rate", "Ключевая ставка", "п.п.", COLORS["gold"]),
    ]
    horizon = irf.point.index.to_numpy()
    for ax, (variable, title, unit, color) in zip(axes.flat, panels, strict=True):
        ax.fill_between(
            horizon,
            irf.lower_90[variable],
            irf.upper_90[variable],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.fill_between(
            horizon,
            irf.lower_68[variable],
            irf.upper_68[variable],
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            horizon,
            irf.median_bootstrap[variable],
            color=color,
            linewidth=2.2,
        )
        ax.axhline(0, color=COLORS["gray"], linewidth=0.9)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Кварталы после шока")
        ax.set_ylabel(unit)
        ax.set_xlim(0, irf.horizon)
        ax.set_xticks(np.arange(0, irf.horizon + 1, 2))
        if variable == "output_gap":
            handles = [
                Patch(facecolor=color, alpha=0.13, label="90% интервал"),
                Patch(facecolor=color, alpha=0.22, label="68% интервал"),
                Line2D([0], [0], color=color, linewidth=2.2, label="Медиана бутстрапа"),
            ]
            ax.legend(handles=handles, loc="lower right", fontsize=8)
    add_footer(
        fig,
        (
            f"Оценённый VAR({irf.lags}) — "
            f"{irf.bootstrap_reps} стабильных бутстрап-репликаций. "
            "Отклики выражены в процентных пунктах; доверительные полосы — эмпирические квантили."
        ),
        y=0.007,
    )
    save_figure(fig, path)


def plot_projection_fan(
    bundle: ProjectionBundle,
    data: pd.DataFrame,
    *,
    history_quarters: int = 8,
    path: str,
) -> None:
    """Fan chart for the structural baseline projection."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.10, hspace=0.42, wspace=0.22)
    n_paths = f"{bundle.paths['path_id'].nunique():,}".replace(",", " ")
    add_suptitle(
        fig,
        "Условная структурная проекция: базовый сценарий",
        (
            "Не прогноз реальной экономики, а симуляция калиброванного DGP; "
            f"полосы — 68% и 90% интервалы {n_paths} траекторий"
        ),
        top=0.965,
    )
    history = data.iloc[-history_quarters:]
    history_x = np.arange(len(history))
    forecast_x = np.arange(len(history), len(history) + len(bundle.horizon_index))
    combined_index = pd.PeriodIndex(list(history.index) + list(bundle.horizon_index))
    panels = [
        ("output_gap", "output_gap", "Разрыв выпуска", "п.п.", COLORS["navy"], 0.0),
        ("inflation", "inflation", "Инфляция", "п.п. в год", COLORS["red"], 2.0),
        ("policy_rate", "policy_rate", "Ключевая ставка", "% годовых", COLORS["gold"], None),
        ("unemployment", "unemployment", "Безработица", "%", COLORS["teal"], 5.4),
    ]
    for panel_index, (ax, (variable, data_column, title, unit, color, target)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        bands = bundle.band(variable).sort_values("horizon")
        x_fan = np.concatenate([[forecast_x[0] - 1], forecast_x])
        median = np.concatenate([[history[data_column].iloc[-1]], bands["q50"].to_numpy()])
        low_90 = np.concatenate([[history[data_column].iloc[-1]], bands["q05"].to_numpy()])
        high_90 = np.concatenate([[history[data_column].iloc[-1]], bands["q95"].to_numpy()])
        low_68 = np.concatenate([[history[data_column].iloc[-1]], bands["q16"].to_numpy()])
        high_68 = np.concatenate([[history[data_column].iloc[-1]], bands["q84"].to_numpy()])

        ax.plot(history_x, history[data_column], color=color, linewidth=1.9)
        draw_fan(
            ax,
            x_fan,
            low_90,
            high_90,
            low_68,
            high_68,
            median,
            color=color,
            linewidth=2.0,
        )
        ax.axvline(len(history) - 0.5, color=COLORS["gray"], linewidth=0.9, linestyle=(0, (3, 3)))
        if target is not None:
            ax.axhline(target, color=COLORS["gray"], linewidth=1.0, linestyle="--", alpha=0.75)
        zero_line(ax) if variable == "output_gap" else None
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.set_xlim(-0.8, len(combined_index) - 0.2)
        quarter_axis(ax, combined_index, max_ticks=7)
        add_last_value(
            ax,
            forecast_x[-1],
            float(median[-1]),
            f"{median[-1]:+.1f}" if variable == "output_gap" else f"{median[-1]:.1f}",
            color=color,
            offset=(4, 8),
        )
        ax.text(
            len(history) - 0.45,
            0.965,
            " проекция →",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color=COLORS["gray"],
        )
        if panel_index == 0:
            fan_legend(ax, color=color)
    add_footer(
        fig,
        (
            f"Горизонт {len(bundle.horizon_index)} кварталов; шоки общие для всех траекторий. "
            "Ширина интервалов отражает только неопределённость будущих шоков при известной калибровке."
        ),
        y=0.007,
    )
    save_figure(fig, path)


def _scenario_effect(
    bundle: ProjectionBundle,
    variable: str,
) -> pd.DataFrame:
    return (
        bundle.effects[bundle.effects["variable"] == variable]
        .sort_values("horizon")
        .reset_index(drop=True)
    )


def _scenario_band(
    bundle: ProjectionBundle,
    variable: str,
    scenario: str,
) -> pd.DataFrame:
    return (
        bundle.bands[
            (bundle.bands["variable"] == variable)
            & (bundle.bands["scenario"] == scenario)
        ]
        .sort_values("horizon")
        .reset_index(drop=True)
    )


def plot_policy_scenario(
    bundle: ProjectionBundle,
    data: pd.DataFrame,
    *,
    history_quarters: int = 8,
    path: str,
) -> None:
    """Compare baseline and temporary tightening with common random numbers."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.11, hspace=0.42, wspace=0.22)
    add_suptitle(
        fig,
        "Сценарный анализ: временное ужесточение денежной политики",
        (
            "Сдвиг желаемой ключевой ставки на +1 п.п. в течение четырёх кварталов; "
            "общие случайные шоки и медианные траектории"
        ),
        top=0.965,
    )
    history = data.iloc[-history_quarters:]
    history_x = np.arange(len(history))
    forecast_x = np.arange(len(history), len(history) + len(bundle.horizon_index))
    combined_index = pd.PeriodIndex(list(history.index) + list(bundle.horizon_index))
    panels = [
        ("policy_rate", "policy_rate", "Ключевая ставка", "% годовых", 0.0),
        ("output_gap", "output_gap", "Разрыв выпуска", "п.п.", 0.0),
        ("inflation", "inflation", "Инфляция", "п.п. в год", 2.0),
        ("unemployment", "unemployment", "Безработица", "%", 5.4),
    ]
    for ax, (variable, data_column, title, unit, target) in zip(axes.flat, panels, strict=True):
        effect = _scenario_effect(bundle, variable)
        base_band = _scenario_band(bundle, variable, "baseline")
        last_value = float(history[data_column].iloc[-1])
        line_x = np.concatenate([[history_x[-1]], forecast_x])
        baseline_line = np.concatenate([[last_value], effect["baseline_median"].to_numpy()])
        tightening_line = np.concatenate(
            [[last_value], effect["tightening_median"].to_numpy()]
        )
        ax.plot(history_x, history[data_column], color=COLORS["gray"], linewidth=1.6)
        ax.plot(
            line_x,
            baseline_line,
            color=SCENARIO_COLORS["baseline"],
            linewidth=2.0,
            label=bundle.baseline_name,
        )
        ax.plot(
            line_x,
            tightening_line,
            color=SCENARIO_COLORS["tightening"],
            linewidth=2.0,
            label=bundle.alternative_name,
        )
        band_low = np.concatenate([[last_value], base_band["q16"].to_numpy()])
        band_high = np.concatenate([[last_value], base_band["q84"].to_numpy()])
        ax.fill_between(
            line_x,
            band_low,
            band_high,
            color=SCENARIO_COLORS["baseline"],
            alpha=0.10,
            linewidth=0,
        )
        ax.axvspan(
            len(history) - 0.5,
            len(history) + min(4, len(forecast_x)) - 0.5,
            color=COLORS["gold"],
            alpha=0.08,
            linewidth=0,
        )
        ax.axvline(len(history) - 0.5, color=COLORS["gray"], linewidth=0.9, linestyle=(0, (3, 3)))
        if target is not None:
            ax.axhline(target, color=COLORS["gray"], linewidth=1.0, linestyle="--", alpha=0.75)
        zero_line(ax) if variable == "output_gap" else None
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.set_xlim(-0.8, len(combined_index) - 0.2)
        quarter_axis(ax, combined_index, max_ticks=7)
        if variable == "policy_rate":
            ax.legend(loc="upper right", fontsize=8)
    effect_horizon = min(4, len(bundle.horizon_index))
    horizon_four = bundle.scenario_effect_summary(horizon=effect_horizon).set_index(
        "variable"
    )
    effect_text = (
        f"Через {effect_horizon} кв.: "
        f"Δx = {horizon_four.loc['output_gap', 'difference']:+.2f} п.п.; "
        f"Δπ = {horizon_four.loc['inflation', 'difference']:+.2f} п.п.; "
        f"Δu = {horizon_four.loc['unemployment', 'difference']:+.2f} п.п."
    )
    add_footer(
        fig,
        (
            effect_text
            + ". Затенение — четыре квартала действия шока; полоса — 68% интервал базового сценария."
        ),
        y=0.007,
    )
    save_figure(fig, path)


def plot_shock_decomposition(
    decomposition: pd.DataFrame,
    *,
    history_quarters: int = 36,
    path: str,
) -> None:
    """History plus waterfall of the exact IS accounting identity."""
    apply_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.4, 7.0),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.065, right=0.975, top=0.78, bottom=0.17, wspace=0.27)
    latest_quarter = decomposition.index[-1]
    add_suptitle(
        fig,
        "Декомпозиция разрыва выпуска по IS-уравнению",
        (
            "Точная бухгалтерская идентичность: инерция, реальная ставка, фискальный импульс, "
            "внешний спрос, финансовые условия и шок"
        ),
        top=0.965,
    )

    ax = axes[0]
    history = decomposition.iloc[-history_quarters:]
    x_history = np.arange(len(history))
    ax.fill_between(
        x_history,
        0,
        history["output_gap"],
        color=COLORS["navy"],
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(x_history, history["output_gap"], color=COLORS["navy"], linewidth=2.0)
    zero_line(ax)
    ax.axvline(len(history) - 0.5, color=COLORS["red"], linewidth=0.9, linestyle=(0, (3, 3)))
    ax.set_title(f"История показателя, последние {len(history)} кварталов", loc="left")
    ax.set_ylabel("п.п. от потенциального выпуска")
    ax.set_xlim(-1, len(history))
    quarter_axis(ax, pd.PeriodIndex(history.index), max_ticks=7)
    add_last_value(
        ax,
        x_history[-1],
        float(history["output_gap"].iloc[-1]),
        f"{history['output_gap'].iloc[-1]:+.2f}",
        color=COLORS["navy"],
        offset=(-8, 11),
    )
    ax.text(
        len(history) - 0.65,
        0.97,
        f"{latest_quarter} →",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["red"],
    )

    ax = axes[1]
    waterfall = latest_waterfall(decomposition)
    positions = np.arange(len(waterfall))
    total = float(decomposition["output_gap"].iloc[-1])
    for position, (label, row) in zip(positions, waterfall.iterrows()):
        value = float(row["contribution"])
        before = float(row["cumulative_before"])
        after = float(row["cumulative_after"])
        color = COLORS["teal"] if value >= 0 else COLORS["red"]
        ax.bar(
            position,
            abs(value),
            bottom=min(before, after),
            width=0.62,
            color=color,
            alpha=0.82,
            edgecolor=COLORS["white"],
            linewidth=0.8,
        )
        ax.annotate(
            f"{value:+.2f}",
            xy=(position, max(before, after)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["ink"],
        )
        if position < len(waterfall) - 1:
            ax.plot(
                [position + 0.31, position + 0.69],
                [after, after],
                color=COLORS["gray"],
                linewidth=0.7,
                linestyle=(0, (2, 2)),
                zorder=2,
            )
    ax.bar(
        len(waterfall),
        total,
        width=0.62,
        color=COLORS["navy"],
        alpha=0.88,
        edgecolor=COLORS["white"],
        linewidth=0.8,
        label="Итоговый разрыв",
    )
    ax.annotate(
        f"{total:+.2f}",
        xy=(len(waterfall) + 0.34, total),
        xytext=(0, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["navy"],
        clip_on=False,
    )
    ax.axhline(0, color=COLORS["gray"], linewidth=0.9)
    ax.set_xticks(np.arange(len(waterfall) + 1))
    ax.set_xticklabels(
        [SHORT_COMPONENTS.get(label, label) for label in waterfall.index] + ["Итог"],
        fontsize=8.4,
    )
    ax.set_ylabel("вклад, п.п.")
    ax.set_title(f"Водопад вкладов, {latest_quarter}", loc="left")
    ax.set_xlim(-0.65, len(waterfall) + 0.80)
    ax.margins(y=0.18)

    max_error = float(decomposition["identity_error"].abs().max())
    add_footer(
        fig,
        (
            "Компоненты вычислены по калиброванным коэффициентам; "
            f"максимальная ошибка идентичности — {max_error:.2e} п.п. "
            "Это не ортогонализованная историческая декомпозиция шоков."
        ),
        y=0.008,
    )
    save_figure(fig, path)


def plot_correlation_map(
    data: pd.DataFrame,
    *,
    path: str,
) -> None:
    """Annotated Pearson correlation heatmap."""
    import seaborn as sns

    apply_style()
    matrix = correlation_matrix(data)
    fig, ax = plt.subplots(figsize=(13.0, 9.5))
    fig.patch.set_facecolor(COLORS["surface"])
    fig.subplots_adjust(left=0.31, right=0.89, top=0.82, bottom=0.16)
    add_suptitle(
        fig,
        "Корреляционная карта макроэкономических переменных",
        "Коэффициенты Пирсона; описательная статистика, не причинная идентификация",
        top=0.965,
    )
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(
        matrix,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        linewidths=0.8,
        linecolor=COLORS["white"],
        cbar_kws={"shrink": 0.66, "pad": 0.025},
    )
    ax.collections[0].colorbar.set_label("r Пирсона", fontsize=9.5, labelpad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=34, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_footer(
        fig,
        (
            f"N = {len(data)} кварталов. Корреляции чувствительны к периоду и калибровке; "
            "они не заменяют структурную оценку."
        ),
        y=0.008,
    )
    save_figure(fig, path)
