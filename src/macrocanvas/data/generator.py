"""Stochastic simulation of a small semi-structural macroeconomic economy.

The generator is intentionally explicit: every equation can be traced to a
line of code and every structural innovation is retained in a separate table.
No real-world observations are used anywhere in the pipeline.

Units
-----
* Output gap: percentage points of potential output.
* Inflation and interest rates: annualised percentage points.
* Unemployment: percentage points; the gap is relative to the natural rate.
* External demand, fiscal impulse and financial stress: model units in pp.

Timing
------
The IS curve uses the lagged ex-ante real policy rate,
``i[t-1] - E[t] pi[t+1]``.  The Phillips curve uses the lagged output gap.
The policy rule reacts to contemporaneous inflation and the output gap, then
is subject to an effective lower bound.  This recursive timing makes the data
generating process transparent and avoids simultaneity in the core real block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from macrocanvas.config import ModelParameters, SimulationConfig


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Container for one synthetic economy and its structural innovations."""

    data: pd.DataFrame
    shocks: pd.DataFrame
    parameters: ModelParameters
    seed: int
    metadata: dict[str, Any]


def ar1_process(
    n: int,
    persistence: float,
    innovation_sd: float,
    rng: np.random.Generator,
    *,
    mean: float = 0.0,
) -> np.ndarray:
    """Simulate a stationary Gaussian AR(1) process.

    The initial value is drawn from the unconditional distribution, so the
    process does not require a separate burn-in.  The function is pure with
    respect to its arguments and is reused by the structural projection module.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must lie in [0, 1).")
    if innovation_sd < 0:
        raise ValueError("innovation_sd must be non-negative.")
    if innovation_sd == 0:
        return np.full(n, mean, dtype=float)
    stationary_sd = innovation_sd / np.sqrt(1.0 - persistence**2)
    values = np.empty(n, dtype=float)
    values[0] = mean + rng.normal(0.0, stationary_sd)
    innovations = rng.normal(0.0, innovation_sd, size=n)
    for t in range(1, n):
        values[t] = mean + persistence * (values[t - 1] - mean) + innovations[t]
    return values


def _allocate(seed: int, n_draws: int) -> list[np.random.Generator]:
    """Create independent, reproducible random streams for each shock."""
    seeds = np.random.SeedSequence(seed).spawn(n_draws)
    return [np.random.default_rng(child) for child in seeds]


def simulate_economy(
    simulation: SimulationConfig,
    parameters: ModelParameters,
) -> SimulationResult:
    """Simulate the synthetic economy and return observations plus shocks."""
    simulation.validate()
    parameters.validate()

    n_total = simulation.burn_in + simulation.periods
    # Independent streams for structural innovations.  This makes it cheap to
    # add a new shock without altering all pre-existing random draws.
    (
        rng_output,
        rng_inflation,
        rng_unemployment,
        rng_policy,
        rng_external,
        rng_fiscal,
        rng_oil,
        rng_stress,
        rng_potential,
    ) = _allocate(simulation.seed, 9)

    p = parameters
    eps_x = rng_output.normal(0.0, p.sigma_output_gap, size=n_total)
    eps_pi = rng_inflation.normal(0.0, p.sigma_inflation, size=n_total)
    eps_u = rng_unemployment.normal(0.0, p.sigma_unemployment, size=n_total)
    eps_i = rng_policy.normal(0.0, p.sigma_policy, size=n_total)

    external = ar1_process(
        n_total,
        p.external_demand_persistence,
        p.sigma_external_demand,
        rng_external,
    )
    fiscal = ar1_process(
        n_total,
        p.fiscal_impulse_persistence,
        p.sigma_fiscal_impulse,
        rng_fiscal,
    )
    oil = ar1_process(n_total, p.oil_persistence, p.sigma_oil, rng_oil)
    stress = ar1_process(
        n_total,
        p.financial_stress_persistence,
        p.sigma_financial_stress,
        rng_stress,
    )
    potential_growth = ar1_process(
        n_total,
        p.potential_growth_persistence,
        p.sigma_potential_growth,
        rng_potential,
        mean=p.potential_growth_mean,
    )

    output_gap = np.zeros(n_total, dtype=float)
    inflation = np.full(n_total, p.inflation_target, dtype=float)
    inflation_expectations = np.full(n_total, p.inflation_target, dtype=float)
    real_rate_ex_ante = np.full(n_total, p.neutral_real_rate, dtype=float)
    unemployment_gap = np.zeros(n_total, dtype=float)
    unemployment_residual = np.zeros(n_total, dtype=float)
    policy_rate = np.full(n_total, p.neutral_real_rate + p.inflation_target, dtype=float)
    log_potential = np.zeros(n_total, dtype=float)
    log_output = np.zeros(n_total, dtype=float)

    for t in range(n_total):
        previous_policy_rate = policy_rate[t - 1] if t > 0 else (
            p.neutral_real_rate + p.inflation_target
        )
        previous_inflation = inflation[t - 1] if t > 0 else p.inflation_target
        previous_output_gap = output_gap[t - 1] if t > 0 else 0.0
        previous_unemployment_residual = (
            unemployment_residual[t - 1] if t > 0 else 0.0
        )

        # Adaptive one-quarter-ahead inflation expectation.
        inflation_expectations[t] = (
            p.expectation_weight_on_target * p.inflation_target
            + (1.0 - p.expectation_weight_on_target) * previous_inflation
        )
        # Ex-ante real rate inherited from the previous policy decision.
        real_rate_ex_ante[t] = previous_policy_rate - inflation_expectations[t]

        # IS curve / aggregate demand block.
        output_gap[t] = (
            p.output_gap_persistence * previous_output_gap
            - p.real_rate_sensitivity
            * (real_rate_ex_ante[t] - p.neutral_real_rate)
            + p.fiscal_impulse_effect * fiscal[t]
            + p.external_demand_effect * external[t]
            - p.financial_stress_effect * stress[t]
            + eps_x[t]
        )

        # Hybrid-style Phillips curve with anchored expectations.
        inflation[t] = (
            p.inflation_target
            + p.inflation_persistence * (previous_inflation - p.inflation_target)
            + p.output_gap_inflation * previous_output_gap
            + p.oil_pass_through * oil[t]
            + eps_pi[t]
        )

        # Okun's law with a serially correlated labour-market residual.
        unemployment_residual[t] = (
            p.unemployment_residual_persistence * previous_unemployment_residual
            + eps_u[t]
        )
        unemployment_gap[t] = (
            -p.okun_coefficient * output_gap[t] + unemployment_residual[t]
        )

        # Inertial Taylor-type policy rule with an effective lower bound.
        desired_policy_rate = (
            p.neutral_real_rate
            + p.inflation_target
            + p.inflation_response * (inflation[t] - p.inflation_target)
            + p.output_gap_response * output_gap[t]
        )
        policy_rate[t] = max(
            p.effective_lower_bound,
            p.policy_inertia * previous_policy_rate
            + (1.0 - p.policy_inertia) * desired_policy_rate
            + eps_i[t],
        )

        # Potential output follows a stochastic trend; actual output embeds
        # the gap.  Levels are rebased to 100 after the burn-in period.
        log_potential[t] = (
            log_potential[t - 1] + potential_growth[t] / 400.0
            if t > 0
            else 0.0
        )
        log_output[t] = log_potential[t] + output_gap[t] / 100.0

    full = pd.DataFrame(
        {
            "output_gap": output_gap,
            "inflation": inflation,
            "inflation_expectations": inflation_expectations,
            "real_rate_ex_ante": real_rate_ex_ante,
            "unemployment_gap": unemployment_gap,
            "policy_rate": policy_rate,
            "potential_growth": potential_growth,
            "log_potential": log_potential,
            "log_output": log_output,
            "external_demand_gap": external,
            "fiscal_impulse": fiscal,
            "oil_price_shock": oil,
            "financial_stress": stress,
        }
    )
    full["gdp_growth"] = 400.0 * full["log_output"].diff()
    full["potential_growth_rate"] = 400.0 * full["log_potential"].diff()
    full["price_index"] = 100.0 * np.exp(
        np.cumsum(np.log1p(full["inflation"] / 400.0))
    )
    full["inflation_yoy"] = (
        full["price_index"].pct_change(4, fill_method=None) * 100.0
    )
    full["real_policy_rate"] = full["policy_rate"] - full["inflation_yoy"]
    full["real_gdp"] = np.exp(full["log_output"])
    full["real_gdp_growth_yoy"] = (
        full["real_gdp"].pct_change(4, fill_method=None) * 100.0
    )
    full["recession"] = (full["gdp_growth"] < 0.0).astype(int)
    # Growth, year-on-year inflation and the ex-post real rate are computed on
    # the full path so that the first retained observation can use burn-in
    # history.  Only the very first row of the full path has missing growth.

    burn = simulation.burn_in
    sample = full.iloc[burn:].copy()
    period_index = pd.period_range(
        start=simulation.start_period,
        periods=simulation.periods,
        freq="Q",
    )
    sample.index = period_index
    sample.index.name = "quarter"

    base_log_output = sample["log_output"].iloc[0]
    sample["gdp_index"] = 100.0 * np.exp(sample["log_output"] - base_log_output)
    sample["potential_gdp_index"] = 100.0 * np.exp(
        sample["log_potential"] - base_log_output
    )
    sample["price_index"] = (
        100.0 * sample["price_index"] / sample["price_index"].iloc[0]
    )
    sample["unemployment"] = (
        p.natural_unemployment + sample["unemployment_gap"]
    )

    shock_columns = {
        "output_gap_shock": eps_x,
        "inflation_shock": eps_pi,
        "unemployment_shock": eps_u,
        "policy_shock": eps_i,
        "external_demand_shock": external,
        "fiscal_shock": fiscal,
        "oil_shock": oil,
        "financial_stress_shock": stress,
        "potential_growth_shock": potential_growth - p.potential_growth_mean,
    }
    shocks = pd.DataFrame(shock_columns, index=full.index).iloc[burn:].copy()
    shocks.index = period_index
    shocks.index.name = "quarter"

    metadata = {
        "model": "Synthetic semi-structural IS-PC-Okun-Taylor economy",
        "data_status": "fully synthetic; no real observations",
        "seed": simulation.seed,
        "periods": simulation.periods,
        "burn_in": simulation.burn_in,
        "start_period": str(period_index[0]),
        "end_period": str(period_index[-1]),
        "frequency": "quarterly",
        "inflation_unit": "annualised percentage points",
        "parameters": asdict(p),
    }

    return SimulationResult(
        data=sample,
        shocks=shocks,
        parameters=parameters,
        seed=simulation.seed,
        metadata=metadata,
    )
