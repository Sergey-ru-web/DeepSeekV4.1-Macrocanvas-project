"""Econometric estimator and identification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocanvas.config import ModelParameters, SimulationConfig
from macrocanvas.data.generator import simulate_economy
from macrocanvas.models.ols import estimate_equations, fit_ols
from macrocanvas.models.var import bootstrap_irf, compute_irf, fit_var


@pytest.fixture(scope="module")
def estimated_system():
    result = simulate_economy(
        SimulationConfig(seed=7, periods=100, burn_in=30),
        ModelParameters(),
    )
    estimates = estimate_equations(result.data, hac_lags=4)
    return result, estimates


def test_ols_recovers_known_linear_model() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=300)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.25, size=300)
    result = fit_ols(
        pd.Series(y, name="y"),
        pd.DataFrame({"x": x}),
        hac_lags=3,
        name="known",
    )
    assert result.coefficient("const") == pytest.approx(2.0, abs=0.10)
    assert result.coefficient("x") == pytest.approx(3.0, abs=0.10)
    assert result.r_squared > 0.90
    assert (result.std_errors > 0).all()
    eigenvalues = np.linalg.eigvalsh(result.covariance.to_numpy())
    assert eigenvalues.min() > -1e-10


def test_equation_signs_and_inference(estimated_system) -> None:
    _, estimates = estimated_system
    assert estimates["is"].coefficient < 0
    assert estimates["phillips"].coefficient > 0
    assert estimates["okun"].coefficient < 0
    assert estimates["taylor"].coefficient > 0
    for estimate in estimates.values():
        assert (estimate.ols.std_errors > 0).all()
        assert 0.0 <= estimate.p_value <= 1.0
        assert 0.0 <= estimate.r_squared <= 1.0


def test_partial_regression_slope_matches_multiple_regression(estimated_system) -> None:
    _, estimates = estimated_system
    for estimate in estimates.values():
        partial = estimate.partial_data()
        x = partial["focus"].to_numpy()
        y = partial["dependent"].to_numpy()
        slope = float(np.linalg.lstsq(x[:, None], y, rcond=None)[0][0])
        assert slope == pytest.approx(estimate.coefficient, abs=1e-10)


def test_var_is_stable_and_impact_normalised(estimated_system) -> None:
    result, _ = estimated_system
    variables = ("output_gap", "unemployment_gap", "inflation", "policy_rate")
    var_result = fit_var(result.data, variables, lags=1)
    assert var_result.is_stable
    impulse = compute_irf(var_result, "policy_rate", horizon=8)
    assert impulse.loc[0, "policy_rate"] == pytest.approx(1.0)
    assert np.allclose(
        impulse.loc[0, ["output_gap", "unemployment_gap", "inflation"]],
        0.0,
    )
    assert impulse["output_gap"].min() < 0


def test_bootstrap_bands_are_ordered(estimated_system) -> None:
    result, _ = estimated_system
    variables = ("output_gap", "unemployment_gap", "inflation", "policy_rate")
    irf = bootstrap_irf(
        result.data,
        variables,
        "policy_rate",
        lags=1,
        horizon=6,
        reps=25,
        seed=99,
    )
    assert irf.lags == 1
    assert irf.bootstrap_reps == 25
    assert (irf.lower_90 <= irf.median_bootstrap + 1e-10).all().all()
    assert (irf.median_bootstrap <= irf.upper_90 + 1e-10).all().all()
    assert (irf.lower_68 <= irf.upper_68 + 1e-10).all().all()


def test_missing_data_are_dropped() -> None:
    y = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], name="y")
    x = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan, 5.0]})
    result = fit_ols(y, x, hac_lags=1)
    assert result.nobs == 3
