"""Markdown report generation from pipeline artefacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from macrocanvas.analytics.decomposition import COMPONENT_LABELS
from macrocanvas.analytics.diagnostics import SampleStatistics, correlation_extremes
from macrocanvas.config import ProjectConfig
from macrocanvas.data.generator import SimulationResult
from macrocanvas.models.ols import EquationEstimate
from macrocanvas.models.scenarios import ProjectionBundle
from macrocanvas.models.var import IRFResult, VARResult


def _markdown_table(frame: pd.DataFrame, *, float_format: str = "{:.3f}") -> str:
    """Render a compact GitHub-flavoured Markdown table without tabulate."""
    if frame.empty:
        return "_Нет данных._"
    formatted = frame.copy().astype(object)
    for column in formatted.columns:
        formatted[column] = [
            float_format.format(value)
            if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else ("—" if isinstance(value, float) and not np.isfinite(value) else str(value))
            for value in formatted[column]
        ]
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _p_value_text(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def write_analysis_report(
    path: str | Path,
    *,
    simulation: SimulationResult,
    estimates: dict[str, EquationEstimate],
    var_result: VARResult,
    irf: IRFResult,
    projections: ProjectionBundle,
    diagnostics: SampleStatistics,
    config: ProjectConfig,
    figure_files: dict[str, str],
) -> Path:
    """Create an automatically generated analytical summary in Russian."""
    data = simulation.data
    generated = datetime.now().astimezone().strftime("%d.%m.%Y %H:%M %Z")
    start, end = data.index[0], data.index[-1]

    selected_stats = diagnostics.table[
        diagnostics.table["variable"].isin(
            ["gdp_growth", "inflation_yoy", "unemployment", "policy_rate", "output_gap"]
        )
    ][["label", "mean", "std", "min", "max"]].rename(
        columns={
            "label": "Показатель",
            "mean": "Среднее",
            "std": "Ст. откл.",
            "min": "Минимум",
            "max": "Максимум",
        }
    )

    estimate_rows = []
    all_signs_ok = True
    for key, estimate in estimates.items():
        sign_ok = (
            estimate.coefficient > 0
            if estimate.expected_sign == "positive"
            else estimate.coefficient < 0
        )
        all_signs_ok = all_signs_ok and sign_ok
        estimate_rows.append(
            {
                "Уравнение": estimate.title,
                "Фокус": estimate.focus_label,
                "β": estimate.coefficient,
                "HAC SE": estimate.standard_error,
                "t": estimate.t_stat,
                "p": _p_value_text(estimate.p_value),
                "R²": estimate.r_squared,
                "Ожидаемый знак": "да" if sign_ok else "нет",
            }
        )
    estimates_frame = pd.DataFrame(estimate_rows)

    max_horizon = int(projections.effects["horizon"].max())
    four_horizon = min(4, max_horizon)
    horizons = [horizon for horizon in (1, 4, 8, 12) if horizon <= max_horizon]
    projection_rows = []
    for variable, label in (
        ("output_gap", "Разрыв выпуска, п.п."),
        ("inflation", "Инфляция, п.п. в год"),
        ("policy_rate", "Ключевая ставка, %"),
        ("unemployment", "Безработица, %"),
    ):
        subset = projections.effects[
            (projections.effects["variable"] == variable)
            & (projections.effects["horizon"].isin(horizons))
        ]
        row: dict[str, object] = {"Показатель": label}
        for _, item in subset.iterrows():
            row[f"Горизонт {int(item['horizon'])} кв."] = item["baseline_median"]
        projection_rows.append(row)
    projection_frame = pd.DataFrame(projection_rows)

    scenario_four = projections.scenario_effect_summary(horizon=four_horizon).set_index(
        "variable"
    )
    scenario_twelve = projections.scenario_effect_summary(horizon=max_horizon).set_index(
        "variable"
    )
    irf_rows = []
    for variable, label in (
        ("output_gap", "Разрыв выпуска"),
        ("unemployment_gap", "Разрыв безработицы"),
        ("inflation", "Инфляция"),
        ("policy_rate", "Ключевая ставка"),
    ):
        series = irf.point[variable]
        irf_rows.append(
            {
                "Переменная": label,
                "Отклик через 1 кв.": float(series.loc[1]),
                "Минимум": float(series.min()),
                "Квартал минимума": int(series.idxmin()),
                "Максимум": float(series.max()),
                "Квартал максимума": int(series.idxmax()),
            }
        )
    irf_frame = pd.DataFrame(irf_rows)

    residual_frame = diagnostics.residual_diagnostics[
        ["equation", "r_squared", "residual_sd", "residual_autocorrelation_lag1"]
    ].rename(
        columns={
            "equation": "Уравнение",
            "r_squared": "R²",
            "residual_sd": "σ остатков",
            "residual_autocorrelation_lag1": "ACF(1) остатков",
        }
    )

    strongest = correlation_extremes(diagnostics.correlations, top_n=3)
    correlation_sentence = "; ".join(
        f"{row.first} ↔ {row.second}: {row.correlation:+.2f}"
        for row in strongest.itertuples(index=False)
    )

    recession_count = len(diagnostics.recession_episodes)
    n_projection_paths = f"{projections.paths['path_id'].nunique():,}".replace(",", " ")
    reconstructed_gap = (
        config.parameters.output_gap_persistence * simulation.data["output_gap"].shift(1)
        - config.parameters.real_rate_sensitivity
        * (
            simulation.data["real_rate_ex_ante"]
            - config.parameters.neutral_real_rate
        )
        + config.parameters.fiscal_impulse_effect * simulation.data["fiscal_impulse"]
        + config.parameters.external_demand_effect
        * simulation.data["external_demand_gap"]
        - config.parameters.financial_stress_effect
        * simulation.data["financial_stress"]
        + simulation.shocks["output_gap_shock"]
    )
    identity_max = float((reconstructed_gap - simulation.data["output_gap"]).abs().max())
    # The expression above is a direct re-evaluation of the IS identity.  Report
    # the maximum numerical error rather than asserting an exact zero.

    lines = [
        "# MacroCanvas: аналитический отчёт",
        "",
        f"_Сформирован автоматически: {generated}._",
        "",
        "> **Дисклеймер.** Все ряды в отчёте полностью синтетические и получены "
        "из калиброванной стохастической модели. Это учебный и исследовательский "
        "артефакт, а не прогноз и не рекомендация экономической политики.",
        "",
        "## 1. Выборка",
        "",
        f"- Период: **{start} — {end}**; наблюдений: **{len(data)}**; "
        f"seed: **{simulation.seed}**.",
        f"- Средний рост ВВП: **{data['gdp_growth'].mean():.2f}%** SAAR; "
        f"средняя инфляция г/г: **{data['inflation_yoy'].mean():.2f}%**; "
        f"средняя безработица: **{data['unemployment'].mean():.2f}%**.",
        f"- Рецессионных эпизодов длительностью ≥ 2 кварталов: **{recession_count}**.",
        f"- Наиболее сильные парные корреляции: {correlation_sentence}.",
        "",
        _markdown_table(selected_stats, float_format="{:.2f}"),
        "",
        "## 2. Оценка поведенческих уравнений",
        "",
        "Стандартные ошибки — Newey–West (HAC), лаг "
        f"{config.estimation.hac_lags}. Фокус-коэффициенты имеют ожидаемые знаки: "
        f"**{'да' if all_signs_ok else 'нет'}**.",
        "",
        _markdown_table(estimates_frame, float_format="{:.3f}"),
        "",
        "Частичные регрессии показывают наклон после очистки от остальных "
        "регрессоров; они не добавляют причинной идентификации. Оценка правила "
        "Тейлора является приведённой формой и содержит механическую обратную связь.",
        "",
        "## 3. VAR и шок денежной политики",
        "",
        f"- VAR({var_result.lags}) по переменным "
        f"`{', '.join(var_result.variables)}`; стабильность: "
        f"**{'да' if var_result.is_stable else 'нет'}**, "
        f"максимальный модуль корня companion-матрицы: **{var_result.max_root_modulus:.3f}**.",
        f"- Идентификация: рекурсивная (Холецкий). Шок нормализован к **+1 п.п.** "
        f"к ключевой ставке в момент шока.",
        f"- Бутстрап: **{irf.bootstrap_reps}** стабильных репликаций.",
        "",
        _markdown_table(irf_frame, float_format="{:+.3f}"),
        "",
        "Отрицательный отклик разрыва выпуска и инфляции согласуется с "
        "краткосрочным эффектом ужесточения. Ширина интервалов и возможные "
        "аномальные отклики отражают конечную выборку и ограничения рекурсивной "
        "идентификации.",
        "",
        "## 4. Базовая проекция",
        "",
        f"Горизонт — {len(projections.horizon_index)} кварталов, "
        f"{projections.horizon_index[0]}–{projections.horizon_index[-1]}. "
        f"Приведены медианы {n_projection_paths} симуляций.",
        "",
        _markdown_table(projection_frame, float_format="{:.2f}"),
        "",
        "## 5. Сценарий ужесточения",
        "",
        f"Сценарий: сдвиг желаемой ставки на **+1 п.п.** в первые четыре квартала; "
        f"общие случайные шоки с базовым сценарием.",
        "",
        f"- Через {four_horizon} квартала: Δx = **{scenario_four.loc['output_gap', 'difference']:+.2f} п.п.**, "
        f"Δπ = **{scenario_four.loc['inflation', 'difference']:+.2f} п.п.**, "
        f"Δu = **{scenario_four.loc['unemployment', 'difference']:+.2f} п.п.**.",
        f"- Через {max_horizon} кварталов: Δx = **{scenario_twelve.loc['output_gap', 'difference']:+.2f} п.п.**, "
        f"Δπ = **{scenario_twelve.loc['inflation', 'difference']:+.2f} п.п.**, "
        f"Δu = **{scenario_twelve.loc['unemployment', 'difference']:+.2f} п.п.**.",
        "",
        "Знаки эффекта согласуются с механизмом: более высокая ставка повышает "
        "реальную ставку, сжимает разрыв выпуска, умеренно снижает инфляцию и "
        "повышает безработицу. Величины условны и зависят от калибровки.",
        "",
        "## 6. Диагностика и ограничения",
        "",
        _markdown_table(residual_frame, float_format="{:.3f}"),
        "",
        f"- Максимальная численная ошибка IS-идентичности: **{identity_max:.2e} п.п.**",
        "- Адаптивные ожидания и линейная кривая Филлипса упрощают реальный механизм.",
        "- Измеряемые переменные трактуются как наблюдаемые без ошибок; в реальных "
        "данных разрыв выпуска и естественная безработица оцениваются.",
        "- Доверительные интервалы не учитывают неопределённость калибровки DGP.",
        "- Сценарные различия не являются причинными эффектами в реальной экономике.",
        "",
        "## 7. Артефакты",
        "",
    ]
    for label, file_path in figure_files.items():
        lines.append(f"- {label}: `{file_path}`")
    lines.extend(
        [
            "",
            "Полные таблицы оценок, IRF, проекций, декомпозиции и описательной "
            "статистики сохранены в `reports/tables/`.",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
