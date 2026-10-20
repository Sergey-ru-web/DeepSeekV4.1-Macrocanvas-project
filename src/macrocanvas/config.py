"""Typed configuration objects and project path management.

The configuration layer is deliberately free of side effects: loading and
validation never touch the filesystem beyond reading the TOML file.  This
keeps the simulation, estimation and plotting layers independently testable.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

_PERIOD_PATTERN = re.compile(r"^\d{4}Q[1-4]$")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ModelParameters:
    """Structural calibration of the synthetic semi-structural economy."""

    # IS curve
    output_gap_persistence: float = 0.68
    real_rate_sensitivity: float = 0.20
    fiscal_impulse_effect: float = 0.18
    external_demand_effect: float = 0.12
    financial_stress_effect: float = 0.12
    neutral_real_rate: float = 0.75
    sigma_output_gap: float = 0.45

    # Phillips curve
    inflation_persistence: float = 0.58
    inflation_target: float = 2.0
    output_gap_inflation: float = 0.25
    oil_pass_through: float = 0.015
    expectation_weight_on_target: float = 0.35
    sigma_inflation: float = 0.60

    # Okun's law
    natural_unemployment: float = 5.4
    okun_coefficient: float = 0.42
    unemployment_residual_persistence: float = 0.50
    sigma_unemployment: float = 0.15

    # Monetary policy
    policy_inertia: float = 0.78
    inflation_response: float = 1.45
    output_gap_response: float = 0.55
    effective_lower_bound: float = 0.25
    sigma_policy: float = 0.18

    # Exogenous processes
    external_demand_persistence: float = 0.82
    sigma_external_demand: float = 0.90
    fiscal_impulse_persistence: float = 0.55
    sigma_fiscal_impulse: float = 0.45
    oil_persistence: float = 0.65
    sigma_oil: float = 4.50
    financial_stress_persistence: float = 0.82
    sigma_financial_stress: float = 0.20
    potential_growth_mean: float = 1.80
    potential_growth_persistence: float = 0.70
    sigma_potential_growth: float = 0.25

    def validate(self) -> None:
        """Check economically meaningful parameter restrictions."""
        for name in (
            "output_gap_persistence",
            "inflation_persistence",
            "unemployment_residual_persistence",
            "policy_inertia",
            "external_demand_persistence",
            "fiscal_impulse_persistence",
            "oil_persistence",
            "financial_stress_persistence",
            "potential_growth_persistence",
            "expectation_weight_on_target",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1); got {value!r}.")
        for name in (
            "real_rate_sensitivity",
            "fiscal_impulse_effect",
            "external_demand_effect",
            "financial_stress_effect",
            "output_gap_inflation",
            "oil_pass_through",
            "okun_coefficient",
            "inflation_response",
            "output_gap_response",
            "sigma_output_gap",
            "sigma_inflation",
            "sigma_unemployment",
            "sigma_policy",
            "sigma_external_demand",
            "sigma_fiscal_impulse",
            "sigma_oil",
            "sigma_financial_stress",
            "sigma_potential_growth",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)!r}.")
        if self.inflation_response <= 0:
            raise ValueError("inflation_response must be strictly positive for a stabilizing rule.")
        if self.effective_lower_bound < 0:
            raise ValueError("effective_lower_bound cannot be negative.")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Controls for one synthetic sample."""

    seed: int = 20261020
    periods: int = 144
    burn_in: int = 60
    start_period: str = "1990Q1"

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.periods < 40:
            raise ValueError("periods must be at least 40 for stable estimation.")
        if self.burn_in < 20:
            raise ValueError("burn_in must be at least 20 to attenuate initial conditions.")
        if not _PERIOD_PATTERN.match(self.start_period):
            raise ValueError("start_period must use the YYYYQn format, for example 1990Q1.")


@dataclass(frozen=True, slots=True)
class EstimationConfig:
    """Numerical settings for reduced-form estimation and projections."""

    hac_lags: int = 4
    var_lags: int = 1
    bootstrap_reps: int = 300
    projection_paths: int = 1500
    projection_horizon: int = 12
    projection_seed: int = 314159

    def validate(self) -> None:
        if self.hac_lags < 0:
            raise ValueError("hac_lags must be non-negative.")
        if not 1 <= self.var_lags <= 8:
            raise ValueError("var_lags must lie in [1, 8].")
        if self.bootstrap_reps < 50:
            raise ValueError("bootstrap_reps must be at least 50 for stable quantiles.")
        if self.projection_paths < 100:
            raise ValueError("projection_paths must be at least 100 for stable quantiles.")
        if not 1 <= self.projection_horizon <= 40:
            raise ValueError("projection_horizon must lie in [1, 40].")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Top-level, serialisable project configuration."""

    simulation: SimulationConfig = SimulationConfig()
    estimation: EstimationConfig = EstimationConfig()
    parameters: ModelParameters = ModelParameters()

    def validate(self) -> None:
        self.simulation.validate()
        self.estimation.validate()
        self.parameters.validate()
        if self.estimation.var_lags >= self.simulation.periods:
            raise ValueError("var_lags must be smaller than the number of observations.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjectConfig":
        """Load a complete or partial TOML file, filling unspecified defaults."""
        path = Path(path)
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        allowed_sections = {"simulation", "estimation", "parameters"}
        unknown_sections = set(raw) - allowed_sections
        if unknown_sections:
            raise ValueError(f"Unknown TOML section(s): {sorted(unknown_sections)}")
        config = cls(
            simulation=_build_section(SimulationConfig, raw.get("simulation", {}), "simulation"),
            estimation=_build_section(EstimationConfig, raw.get("estimation", {}), "estimation"),
            parameters=_build_section(ModelParameters, raw.get("parameters", {}), "parameters"),
        )
        config.validate()
        return config


def _build_section(dataclass_type: type[T], values: dict[str, Any], section: str) -> T:
    known = {field.name for field in fields(dataclass_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [{section}]: {sorted(unknown)}")
    return dataclass_type(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Canonical directory layout for reproducible artefacts."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        return cls(Path(root) if root is not None else Path.cwd())

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "reports" / "tables"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> None:
        for directory in (
            self.data_raw,
            self.data_processed,
            self.figures,
            self.tables,
            self.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "baseline.toml"
