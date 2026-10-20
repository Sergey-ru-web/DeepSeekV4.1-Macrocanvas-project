"""Simulation invariants and reproducibility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocanvas.analytics.decomposition import decompose_output_gap
from macrocanvas.config import ModelParameters, SimulationConfig
from macrocanvas.data.generator import simulate_economy


@pytest.fixture(scope="module")
def small_simulation():
    config = SimulationConfig(seed=42, periods=80, burn_in=30, start_period="2000Q1")
    return config, simulate_economy(config, ModelParameters())


def test_same_seed_is_reproducible() -> None:
    config = SimulationConfig(seed=123, periods=60, burn_in=24)
    first = simulate_economy(config, ModelParameters())
    second = simulate_economy(config, ModelParameters())
    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.shocks, second.shocks)


def test_different_seed_changes_data() -> None:
    first = simulate_economy(SimulationConfig(seed=123, periods=60, burn_in=24), ModelParameters())
    second = simulate_economy(SimulationConfig(seed=124, periods=60, burn_in=24), ModelParameters())
    assert not np.allclose(first.data["output_gap"], second.data["output_gap"])


def test_sample_shape_and_columns(small_simulation) -> None:
    config, result = small_simulation
    assert len(result.data) == config.periods
    assert isinstance(result.data.index, pd.PeriodIndex)
    assert result.data.index[0] == pd.Period("2000Q1", freq="Q")
    expected = {
        "output_gap",
        "inflation",
        "inflation_yoy",
        "unemployment",
        "policy_rate",
        "gdp_growth",
        "gdp_index",
        "potential_gdp_index",
        "price_index",
        "real_policy_rate",
        "recession",
    }
    assert expected <= set(result.data.columns)
    assert result.shocks.shape[0] == config.periods
    assert not result.data.isna().any().any()


def test_accounting_identity_is_exact(small_simulation) -> None:
    _, result = small_simulation
    decomposition = decompose_output_gap(result.data, result.parameters)
    assert decomposition["identity_error"].abs().max() < 1e-12
    assert np.allclose(
        decomposition["sum"],
        decomposition["output_gap"],
        atol=1e-12,
    )


def test_units_and_indicators(small_simulation) -> None:
    _, result = small_simulation
    data = result.data
    assert data["price_index"].iloc[0] == pytest.approx(100.0)
    assert data["gdp_index"].iloc[0] == pytest.approx(100.0)
    assert data["inflation"].between(-10.0, 20.0).all()
    assert data["unemployment"].between(0.0, 25.0).all()
    assert data["policy_rate"].ge(0.25 - 1e-12).all()
    assert (data.loc[data["gdp_growth"] < 0, "recession"] == 1).all()


def test_invalid_simulation_config() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(periods=10).validate()
    with pytest.raises(ValueError):
        SimulationConfig(start_period="1990-Q1").validate()
