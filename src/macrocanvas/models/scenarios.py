"""Conditional structural projections and a temporary monetary tightening.

These are *model-consistent scenarios*, not forecasts.  The exercise starts
from the terminal state of the synthetic economy, draws future structural
shocks from the same calibrated distributions, and compares a baseline path
with a path in which the desired policy rate is temporarily raised by one
percentage point for four quarters.  Common random numbers ensure that the
difference between scenarios isolates the policy intervention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macrocanvas.config import ModelParameters
from macrocanvas.data.generator import SimulationResult

PROJECTION_VARIABLES = (
    "output_gap",
    "inflation",
    "policy_rate",
    "unemployment_gap",
    "unemployment",
    "gdp_growth",
)


@dataclass(frozen=True, slots=True)
class ProjectionBundle:
    """Long-form simulation paths, quantile bands and scenario effects."""

    paths: pd.DataFrame
    bands: pd.DataFrame
    effects: pd.DataFrame
    horizon_index: pd.PeriodIndex
    baseline_name: str = "Базовый сценарий"
    alternative_name: str = "Ужесточение +1 п.п. (4 квартала)"

    def band(
        self,
        variable: str,
        scenario: str = "baseline",
    ) -> pd.DataFrame:
        subset = self.bands[
            (self.bands["scenario"] == scenario) & (self.bands["variable"] == variable)
        ]
        return subset.reset_index(drop=True)

    def scenario_effect_summary(self, horizon: int = 4) -> pd.DataFrame:
        return self.effects[self.effects["horizon"] == horizon].reset_index(drop=True)


def _draw_innovations(
    parameters: ModelParameters,
    horizon: int,
    n_paths: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = parameters
    return {
        "eps_x": rng.normal(0.0, p.sigma_output_gap, size=(horizon, n_paths)),
        "eps_pi": rng.normal(0.0, p.sigma_inflation, size=(horizon, n_paths)),
        "eps_u": rng.normal(0.0, p.sigma_unemployment, size=(horizon, n_paths)),
        "eps_i": rng.normal(0.0, p.sigma_policy, size=(horizon, n_paths)),
        "external": rng.normal(
            0.0, p.sigma_external_demand, size=(horizon, n_paths)
        ),
        "fiscal": rng.normal(0.0, p.sigma_fiscal_impulse, size=(horizon, n_paths)),
        "oil": rng.normal(0.0, p.sigma_oil, size=(horizon, n_paths)),
        "stress": rng.normal(
            0.0, p.sigma_financial_stress, size=(horizon, n_paths)
        ),
        "potential": rng.normal(
            0.0, p.sigma_potential_growth, size=(horizon, n_paths)
        ),
    }


def _simulate_paths(
    result: SimulationResult,
    parameters: ModelParameters,
    innovations: dict[str, np.ndarray],
    horizon: int,
    n_paths: int,
    policy_offset: np.ndarray,
    scenario: str,
    horizon_index: pd.PeriodIndex,
) -> pd.DataFrame:
    p = parameters
    last = result.data.iloc[-1]

    output_gap = np.full(n_paths, float(last["output_gap"]))
    inflation = np.full(n_paths, float(last["inflation"]))
    policy_rate = np.full(n_paths, float(last["policy_rate"]))
    unemployment_gap = np.full(n_paths, float(last["unemployment_gap"]))
    unemployment_residual = unemployment_gap + p.okun_coefficient * output_gap
    external = np.full(n_paths, float(last["external_demand_gap"]))
    fiscal = np.full(n_paths, float(last["fiscal_impulse"]))
    oil = np.full(n_paths, float(last["oil_price_shock"]))
    stress = np.full(n_paths, float(last["financial_stress"]))
    potential_growth = np.full(n_paths, float(last["potential_growth"]))

    records: list[pd.DataFrame] = []
    for h in range(horizon):
        # Exogenous AR(1) states continue conditionally on their last values.
        external = p.external_demand_persistence * external + innovations["external"][h]
        fiscal = p.fiscal_impulse_persistence * fiscal + innovations["fiscal"][h]
        oil = p.oil_persistence * oil + innovations["oil"][h]
        stress = p.financial_stress_persistence * stress + innovations["stress"][h]
        potential_growth = (
            p.potential_growth_mean
            + p.potential_growth_persistence * (potential_growth - p.potential_growth_mean)
            + innovations["potential"][h]
        )

        previous_output_gap = output_gap.copy()
        inflation_expectations = (
            p.expectation_weight_on_target * p.inflation_target
            + (1.0 - p.expectation_weight_on_target) * inflation
        )
        real_rate = policy_rate - inflation_expectations

        output_gap = (
            p.output_gap_persistence * output_gap
            - p.real_rate_sensitivity * (real_rate - p.neutral_real_rate)
            + p.fiscal_impulse_effect * fiscal
            + p.external_demand_effect * external
            - p.financial_stress_effect * stress
            + innovations["eps_x"][h]
        )
        inflation = (
            p.inflation_target
            + p.inflation_persistence * (inflation - p.inflation_target)
            + p.output_gap_inflation * previous_output_gap
            + p.oil_pass_through * oil
            + innovations["eps_pi"][h]
        )
        unemployment_residual = (
            p.unemployment_residual_persistence * unemployment_residual
            + innovations["eps_u"][h]
        )
        unemployment_gap = -p.okun_coefficient * output_gap + unemployment_residual

        desired_rate = (
            p.neutral_real_rate
            + p.inflation_target
            + p.inflation_response * (inflation - p.inflation_target)
            + p.output_gap_response * output_gap
            + policy_offset[h]
        )
        policy_rate = np.maximum(
            p.effective_lower_bound,
            p.policy_inertia * policy_rate
            + (1.0 - p.policy_inertia) * desired_rate
            + innovations["eps_i"][h],
        )
        gdp_growth = potential_growth + 400.0 * (output_gap - previous_output_gap)

        records.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "path_id": np.arange(n_paths, dtype=int),
                    "horizon": h + 1,
                    "quarter": str(horizon_index[h]),
                    "output_gap": output_gap,
                    "inflation": inflation,
                    "policy_rate": policy_rate,
                    "unemployment_gap": unemployment_gap,
                    "unemployment": p.natural_unemployment + unemployment_gap,
                    "gdp_growth": gdp_growth,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def _bands(
    paths: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.05, 0.16, 0.50, 0.84, 0.95),
) -> pd.DataFrame:
    grouped = paths.groupby(["scenario", "horizon", "quarter", "variable"], observed=True)
    quantiled = (
        grouped["value"]
        .quantile(quantiles)
        .rename("value")
        .reset_index()
    )
    labels = {
        0.05: "q05",
        0.16: "q16",
        0.50: "q50",
        0.84: "q84",
        0.95: "q95",
    }
    quantile_column = next(
        column
        for column in quantiled.columns
        if column not in {"scenario", "horizon", "quarter", "variable", "value"}
    )
    quantiled["quantile"] = [
        labels[round(float(value), 2)] for value in quantiled[quantile_column]
    ]
    frame = quantiled.pivot(
        index=["scenario", "horizon", "quarter", "variable"],
        columns="quantile",
        values="value",
    ).reset_index()
    frame.columns.name = None
    return frame


def run_projection_scenarios(
    result: SimulationResult,
    parameters: ModelParameters,
    *,
    horizon: int = 12,
    n_paths: int = 1500,
    seed: int = 314159,
    tightening_pp: float = 1.0,
    tightening_quarters: int = 4,
) -> ProjectionBundle:
    """Compare baseline and temporary policy tightening with common shocks."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if n_paths < 100:
        raise ValueError("n_paths must be at least 100.")
    if tightening_quarters < 1:
        raise ValueError("tightening_quarters must be positive.")

    innovations = _draw_innovations(parameters, horizon, n_paths, seed)
    baseline_offset = np.zeros(horizon)
    tightening_offset = np.zeros(horizon)
    tightening_offset[: min(tightening_quarters, horizon)] = tightening_pp

    last_quarter = result.data.index[-1]
    horizon_index = pd.period_range(
        start=last_quarter + 1,
        periods=horizon,
        freq="Q",
    )
    baseline_paths = _simulate_paths(
        result,
        parameters,
        innovations,
        horizon,
        n_paths,
        baseline_offset,
        "baseline",
        horizon_index,
    )
    tightening_paths = _simulate_paths(
        result,
        parameters,
        innovations,
        horizon,
        n_paths,
        tightening_offset,
        "tightening",
        horizon_index,
    )
    paths = pd.concat([baseline_paths, tightening_paths], ignore_index=True)
    long = paths.melt(
        id_vars=["scenario", "path_id", "horizon", "quarter"],
        value_vars=list(PROJECTION_VARIABLES),
        var_name="variable",
        value_name="value",
    )
    bands = _bands(long)

    median = (
        long.groupby(
            ["scenario", "horizon", "quarter", "variable"], observed=True
        )["value"]
        .median()
        .unstack("scenario")
        .reset_index()
        .rename(columns={"baseline": "baseline_median", "tightening": "tightening_median"})
    )
    median["difference"] = median["tightening_median"] - median["baseline_median"]
    median = median.rename(columns={"variable": "variable"})

    return ProjectionBundle(
        paths=paths,
        bands=bands,
        effects=median,
        horizon_index=horizon_index,
    )
