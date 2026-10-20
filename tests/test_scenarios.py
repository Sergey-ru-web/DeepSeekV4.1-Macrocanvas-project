"""Structural scenario and accounting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocanvas.analytics.decomposition import decompose_output_gap, latest_waterfall
from macrocanvas.config import ModelParameters, SimulationConfig
from macrocanvas.data.generator import simulate_economy
from macrocanvas.models.scenarios import run_projection_scenarios


def _projections():
    result = simulate_economy(
        SimulationConfig(seed=2026, periods=92, burn_in=30),
        ModelParameters(),
    )
    bundle = run_projection_scenarios(
        result,
        result.parameters,
        horizon=8,
        n_paths=150,
        seed=1234,
    )
    return result, bundle


def test_projection_horizon_and_bands() -> None:
    result, bundle = _projections()
    assert len(bundle.horizon_index) == 8
    assert bundle.horizon_index[0] == result.data.index[-1] + 1
    assert set(bundle.paths["scenario"]) == {"baseline", "tightening"}
    for scenario in ("baseline", "tightening"):
        band = bundle.band("inflation", scenario)
        assert (band["q05"] <= band["q16"]).all()
        assert (band["q16"] <= band["q50"]).all()
        assert (band["q50"] <= band["q84"]).all()
        assert (band["q84"] <= band["q95"]).all()


def test_common_random_numbers_are_reproducible() -> None:
    result = simulate_economy(SimulationConfig(seed=2, periods=80, burn_in=25), ModelParameters())
    first = run_projection_scenarios(result, result.parameters, horizon=6, n_paths=120, seed=777)
    second = run_projection_scenarios(result, result.parameters, horizon=6, n_paths=120, seed=777)
    pd.testing.assert_frame_equal(first.paths, second.paths)
    pd.testing.assert_frame_equal(first.effects, second.effects)


def test_tightening_raises_policy_rate_and_lowers_output() -> None:
    _, bundle = _projections()
    policy = bundle.scenario_effect_summary(horizon=2).set_index("variable")
    output = bundle.scenario_effect_summary(horizon=4).set_index("variable")
    inflation = bundle.scenario_effect_summary(horizon=8).set_index("variable")
    assert policy.loc["policy_rate", "difference"] > 0
    assert output.loc["output_gap", "difference"] < 0
    assert inflation.loc["inflation", "difference"] <= 0
    assert output.loc["unemployment", "difference"] >= 0


def test_decomposition_waterfall_reconstructs_latest_gap() -> None:
    result = simulate_economy(SimulationConfig(seed=3, periods=70, burn_in=24), ModelParameters())
    decomposition = decompose_output_gap(result.data, result.parameters)
    waterfall = latest_waterfall(decomposition)
    assert waterfall["contribution"].sum() == pytest.approx(
        decomposition["output_gap"].iloc[-1],
        abs=1e-12,
    )
    assert np.abs(decomposition["identity_error"]).max() < 1e-12
