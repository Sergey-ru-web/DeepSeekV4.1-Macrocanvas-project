"""Configuration and filesystem contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from macrocanvas.config import ProjectConfig, ProjectPaths


def test_baseline_configuration_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = ProjectConfig.from_toml(root / "configs" / "baseline.toml")
    assert config.simulation.periods == 144
    assert config.simulation.seed == 20261020
    assert config.estimation.var_lags == 1
    assert config.parameters.inflation_target == pytest.approx(2.0)


def test_default_configuration_is_valid() -> None:
    ProjectConfig().validate()


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[simulation]\nperiod = 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        ProjectConfig.from_toml(path)


def test_invalid_parameter_is_rejected() -> None:
    config = ProjectConfig()
    invalid = config.parameters.__class__(inflation_persistence=1.2)
    with pytest.raises(ValueError):
        invalid.validate()


def test_project_paths_are_created(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    assert paths.figures.is_dir()
    assert paths.tables.is_dir()
    assert paths.data_raw.is_dir()
