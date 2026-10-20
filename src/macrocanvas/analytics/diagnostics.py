"""Descriptive diagnostics, recession episodes and sample diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macrocanvas.models.ols import EquationEstimate


@dataclass(frozen=True, slots=True)
class SampleStatistics:
    table: pd.DataFrame
    correlations: pd.DataFrame
    recession_episodes: pd.DataFrame
    residual_diagnostics: pd.DataFrame


VARIABLE_LABELS = {
    "gdp_growth": "Рост ВВП (q/q SAAR, %)",
    "inflation": "Инфляция q/q SAAR (п.п.)",
    "inflation_yoy": "Инфляция г/г (%)",
    "unemployment": "Безработица (%)",
    "policy_rate": "Ключевая ставка (%)",
    "real_policy_rate": "Реальная ставка ex post (%)",
    "output_gap": "Разрыв выпуска (п.п.)",
    "unemployment_gap": "Разрыв безработицы (п.п.)",
    "real_rate_ex_ante": "Реальная ставка ex ante (п.п.)",
    "external_demand_gap": "Внешний спрос (п.п.)",
    "fiscal_impulse": "Фискальный импульс (п.п.)",
    "financial_stress": "Финансовый стресс (индекс)",
}

CORRELATION_VARIABLES = (
    "gdp_growth",
    "output_gap",
    "inflation_yoy",
    "unemployment",
    "policy_rate",
    "real_policy_rate",
    "external_demand_gap",
    "fiscal_impulse",
    "financial_stress",
)


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    variables = [variable for variable in VARIABLE_LABELS if variable in data.columns]
    rows = []
    for variable in variables:
        series = data[variable].dropna()
        rows.append(
            {
                "variable": variable,
                "label": VARIABLE_LABELS[variable],
                "nobs": int(series.size),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    variables = [variable for variable in CORRELATION_VARIABLES if variable in data.columns]
    matrix = data[variables].corr(method="pearson")
    matrix.index = [VARIABLE_LABELS.get(variable, variable) for variable in matrix.index]
    matrix.columns = [VARIABLE_LABELS.get(variable, variable) for variable in matrix.columns]
    return matrix


def find_recession_episodes(
    data: pd.DataFrame,
    *,
    min_duration: int = 2,
) -> pd.DataFrame:
    """Find contiguous runs of negative quarterly GDP growth."""
    if "recession" not in data.columns:
        raise ValueError("The data must contain a 'recession' indicator.")
    flags = data["recession"].astype(bool).to_numpy()
    episodes: list[dict[str, object]] = []
    start: int | None = None
    for position, flag in enumerate(flags):
        if flag and start is None:
            start = position
        if not flag and start is not None:
            end = position - 1
            if end - start + 1 >= min_duration:
                episodes.append(_episode_record(data, start, end))
            start = None
    if start is not None and len(flags) - start >= min_duration:
        episodes.append(_episode_record(data, start, len(flags) - 1))
    return pd.DataFrame(episodes)


def _episode_record(data: pd.DataFrame, start: int, end: int) -> dict[str, object]:
    episode = data.iloc[start : end + 1]
    return {
        "start": str(data.index[start]),
        "end": str(data.index[end]),
        "quarters": end - start + 1,
        "cumulative_gap_change": float(
            data["output_gap"].iloc[end] - data["output_gap"].iloc[max(start - 1, 0)]
        ),
        "min_growth": float(episode["gdp_growth"].min()),
    }


def residual_diagnostics(estimates: dict[str, EquationEstimate]) -> pd.DataFrame:
    """First-order residual autocorrelation and residual standard deviation."""
    rows = []
    for key, estimate in estimates.items():
        residuals = estimate.ols.residuals
        rho_1 = float(residuals.autocorr(lag=1))
        rows.append(
            {
                "equation": key,
                "nobs": estimate.ols.nobs,
                "residual_sd": estimate.ols.residual_sd,
                "residual_autocorrelation_lag1": rho_1,
                "p_value_focus": estimate.p_value,
                "r_squared": estimate.r_squared,
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(
    data: pd.DataFrame,
    estimates: dict[str, EquationEstimate],
) -> SampleStatistics:
    return SampleStatistics(
        table=descriptive_statistics(data),
        correlations=correlation_matrix(data),
        recession_episodes=find_recession_episodes(data),
        residual_diagnostics=residual_diagnostics(estimates),
    )


def correlation_extremes(correlations: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Largest off-diagonal absolute correlations, for the written summary."""
    matrix = correlations.to_numpy(copy=True)
    np.fill_diagonal(matrix, np.nan)
    pairs = []
    labels = list(correlations.columns)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = matrix[i, j]
            if np.isfinite(value):
                pairs.append(
                    {
                        "first": labels[i],
                        "second": labels[j],
                        "correlation": float(value),
                        "absolute": float(abs(value)),
                    }
                )
    return (
        pd.DataFrame(pairs)
        .sort_values("absolute", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
