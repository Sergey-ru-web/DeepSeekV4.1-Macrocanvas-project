"""End-to-end orchestration of simulation, estimation, figures and reports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macrocanvas.analytics.decomposition import (
    contribution_summary,
    contribution_table,
    decompose_output_gap,
    latest_waterfall,
)
from macrocanvas.analytics.diagnostics import SampleStatistics, build_diagnostics
from macrocanvas.config import ProjectConfig, ProjectPaths
from macrocanvas.data.generator import SimulationResult, simulate_economy
from macrocanvas.models.ols import (
    EquationEstimate,
    coefficient_table,
    estimate_equations,
    estimates_table,
)
from macrocanvas.models.scenarios import ProjectionBundle, run_projection_scenarios
from macrocanvas.models.var import (
    IRFResult,
    VARResult,
    bootstrap_irf,
    fit_var,
)
from macrocanvas.reporting import write_analysis_report
from macrocanvas.viz.charts import (
    plot_correlation_map,
    plot_estimated_relations,
    plot_irf,
    plot_policy_scenario,
    plot_projection_fan,
    plot_shock_decomposition,
)
from macrocanvas.viz.dashboard import plot_macro_dashboard
from macrocanvas.viz.model_map import plot_model_map

logger = logging.getLogger(__name__)

VAR_VARIABLES = ("output_gap", "unemployment_gap", "inflation", "policy_rate")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    config: ProjectConfig
    paths: ProjectPaths
    simulation: SimulationResult
    estimates: dict[str, EquationEstimate]
    var_result: VARResult
    irf: IRFResult
    projections: ProjectionBundle
    decomposition: pd.DataFrame
    diagnostics: SampleStatistics
    figure_files: dict[str, str]
    table_files: dict[str, str]
    report_file: str
    metadata_file: str
    summary: dict[str, Any]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _save_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, encoding="utf-8-sig")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Period, pd.Timestamp)):
        return str(value)
    return value


def _build_figure_files(figures: Path, root: Path) -> dict[str, str]:
    names = {
        "macro_dashboard": "01_macro_dashboard.png",
        "model_map": "02_model_map.png",
        "estimated_relations": "03_estimated_relations.png",
        "monetary_irf": "04_monetary_irf.png",
        "projection_fan": "05_projection_fan.png",
        "policy_scenario": "06_policy_scenario.png",
        "shock_decomposition": "07_shock_decomposition.png",
        "correlation_map": "08_correlation_map.png",
    }
    return {label: _relative(figures / filename, root) for label, filename in names.items()}


def run_pipeline(
    config: ProjectConfig,
    *,
    root: str | Path | None = None,
) -> PipelineResult:
    """Run the complete reproducible pipeline and persist all artefacts."""
    config.validate()
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    logger.info("Stage 1/6: simulating the synthetic economy")
    simulation = simulate_economy(config.simulation, config.parameters)
    data = simulation.data
    _save_csv(data, paths.data_raw / "synthetic_economy.csv", index=True)
    _save_csv(simulation.shocks, paths.data_raw / "structural_shocks.csv", index=True)
    _save_csv(data, paths.data_processed / "analysis_dataset.csv", index=True)

    logger.info("Stage 2/6: estimating behavioural equations")
    estimates = estimate_equations(data, hac_lags=config.estimation.hac_lags)

    logger.info("Stage 3/6: estimating VAR and bootstrap impulse responses")
    var_result = fit_var(data, VAR_VARIABLES, lags=config.estimation.var_lags)
    irf = bootstrap_irf(
        data,
        VAR_VARIABLES,
        shock="policy_rate",
        lags=config.estimation.var_lags,
        horizon=config.estimation.projection_horizon,
        reps=config.estimation.bootstrap_reps,
        seed=config.estimation.projection_seed + 1,
    )

    logger.info("Stage 4/6: running structural projections and diagnostics")
    projections = run_projection_scenarios(
        simulation,
        config.parameters,
        horizon=config.estimation.projection_horizon,
        n_paths=config.estimation.projection_paths,
        seed=config.estimation.projection_seed,
    )
    decomposition = decompose_output_gap(data, config.parameters)
    diagnostics = build_diagnostics(data, estimates)

    logger.info("Stage 5/6: rendering figures")
    figure_files = _build_figure_files(paths.figures, paths.root)
    plot_macro_dashboard(
        data,
        seed=simulation.seed,
        episodes=diagnostics.recession_episodes,
        path=str(paths.figures / "01_macro_dashboard.png"),
    )
    plot_model_map(str(paths.figures / "02_model_map.png"))
    plot_estimated_relations(
        estimates,
        hac_lags=config.estimation.hac_lags,
        path=str(paths.figures / "03_estimated_relations.png"),
    )
    plot_irf(irf, path=str(paths.figures / "04_monetary_irf.png"))
    plot_projection_fan(
        projections,
        data,
        path=str(paths.figures / "05_projection_fan.png"),
    )
    plot_policy_scenario(
        projections,
        data,
        path=str(paths.figures / "06_policy_scenario.png"),
    )
    plot_shock_decomposition(
        decomposition,
        path=str(paths.figures / "07_shock_decomposition.png"),
    )
    plot_correlation_map(data, path=str(paths.figures / "08_correlation_map.png"))

    logger.info("Stage 6/6: writing tables, metadata and the analytical report")
    table_files = {
        "estimated_equations": _relative(paths.tables / "estimated_equations.csv", paths.root),
        "focus_coefficients": _relative(paths.tables / "focus_coefficients.csv", paths.root),
        "var_coefficients": _relative(paths.tables / "var_coefficients.csv", paths.root),
        "impulse_responses": _relative(paths.tables / "impulse_responses.csv", paths.root),
        "projection_effects": _relative(paths.tables / "projection_effects.csv", paths.root),
        "projection_bands": _relative(paths.tables / "projection_bands.csv", paths.root),
        "shock_decomposition": _relative(paths.tables / "shock_decomposition.csv", paths.root),
        "shock_contributions_long": _relative(
            paths.tables / "shock_contributions_long.csv", paths.root
        ),
        "contribution_summary": _relative(paths.tables / "contribution_summary.csv", paths.root),
        "latest_waterfall": _relative(paths.tables / "latest_waterfall.csv", paths.root),
        "descriptive_statistics": _relative(paths.tables / "descriptive_statistics.csv", paths.root),
        "correlations": _relative(paths.tables / "correlations.csv", paths.root),
        "recession_episodes": _relative(paths.tables / "recession_episodes.csv", paths.root),
        "residual_diagnostics": _relative(paths.tables / "residual_diagnostics.csv", paths.root),
        "parameters": _relative(paths.tables / "parameters.csv", paths.root),
    }
    _save_csv(estimates_table(estimates), paths.tables / "estimated_equations.csv")
    _save_csv(
        coefficient_table(estimates, only_focus=True),
        paths.tables / "focus_coefficients.csv",
    )
    _save_csv(var_result.equation_table(), paths.tables / "var_coefficients.csv")
    _save_csv(irf.tidy(), paths.tables / "impulse_responses.csv")
    _save_csv(projections.effects, paths.tables / "projection_effects.csv")
    _save_csv(projections.bands, paths.tables / "projection_bands.csv")
    _save_csv(decomposition, paths.tables / "shock_decomposition.csv", index=True)
    _save_csv(contribution_table(decomposition), paths.tables / "shock_contributions_long.csv")
    _save_csv(contribution_summary(decomposition), paths.tables / "contribution_summary.csv")
    _save_csv(latest_waterfall(decomposition).reset_index(), paths.tables / "latest_waterfall.csv")
    _save_csv(diagnostics.table, paths.tables / "descriptive_statistics.csv")
    _save_csv(diagnostics.correlations.reset_index(), paths.tables / "correlations.csv")
    _save_csv(diagnostics.recession_episodes, paths.tables / "recession_episodes.csv")
    _save_csv(diagnostics.residual_diagnostics, paths.tables / "residual_diagnostics.csv")
    parameter_row = {
        field.name: getattr(config.parameters, field.name)
        for field in fields(config.parameters)
    }
    _save_csv(pd.DataFrame([parameter_row]), paths.tables / "parameters.csv")

    projection_horizon = int(projections.effects["horizon"].max())
    scenario_four = projections.scenario_effect_summary(
        horizon=min(4, projection_horizon)
    ).set_index("variable")
    scenario_twelve = projections.scenario_effect_summary(
        horizon=projection_horizon
    ).set_index("variable")
    summary = {
        "generated_at": generated_at,
        "sample": {
            "start": str(data.index[0]),
            "end": str(data.index[-1]),
            "n_obs": int(len(data)),
            "seed": simulation.seed,
        },
        "descriptive": {
            "mean_growth": float(data["gdp_growth"].mean()),
            "mean_inflation_yoy": float(data["inflation_yoy"].mean()),
            "mean_unemployment": float(data["unemployment"].mean()),
            "mean_policy_rate": float(data["policy_rate"].mean()),
            "sd_output_gap": float(data["output_gap"].std(ddof=1)),
            "recession_episodes": int(len(diagnostics.recession_episodes)),
        },
        "equations": {
            key: {
                "focus": estimate.focus,
                "coefficient": float(estimate.coefficient),
                "hac_std_error": float(estimate.standard_error),
                "t_stat": float(estimate.t_stat),
                "p_value": float(estimate.p_value),
                "r_squared": float(estimate.r_squared),
                "expected_sign": estimate.expected_sign,
            }
            for key, estimate in estimates.items()
        },
        "var": {
            "lags": int(var_result.lags),
            "n_obs": int(var_result.nobs),
            "stable": bool(var_result.is_stable),
            "max_root_modulus": float(var_result.max_root_modulus),
            "bootstrap_reps": int(irf.bootstrap_reps),
            "irf_min_output_gap": float(irf.point["output_gap"].min()),
            "irf_min_output_gap_horizon": int(irf.point["output_gap"].idxmin()),
            "irf_min_inflation": float(irf.point["inflation"].min()),
            "irf_min_inflation_horizon": int(irf.point["inflation"].idxmin()),
        },
        "scenario": {
            "tightening_pp": 1.0,
            "duration_quarters": 4,
            "horizon_4": {
                variable: float(scenario_four.loc[variable, "difference"])
                for variable in ("output_gap", "inflation", "unemployment")
            },
            "horizon_12": {
                variable: float(scenario_twelve.loc[variable, "difference"])
                for variable in ("output_gap", "inflation", "unemployment")
            },
        },
        "artifacts": {
            "figures": figure_files,
            "tables": table_files,
        },
    }

    report_path = paths.reports / "analysis_summary.md"
    write_analysis_report(
        report_path,
        simulation=simulation,
        estimates=estimates,
        var_result=var_result,
        irf=irf,
        projections=projections,
        diagnostics=diagnostics,
        config=config,
        figure_files=figure_files,
    )
    metadata_path = paths.reports / "run_metadata.json"
    metadata = {
        "generated_at": generated_at,
        "config": config.to_dict(),
        "data": simulation.metadata,
        "summary": summary,
    }
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Pipeline complete: %s", paths.reports)
    return PipelineResult(
        config=config,
        paths=paths,
        simulation=simulation,
        estimates=estimates,
        var_result=var_result,
        irf=irf,
        projections=projections,
        decomposition=decomposition,
        diagnostics=diagnostics,
        figure_files=figure_files,
        table_files=table_files,
        report_file=_relative(report_path, paths.root),
        metadata_file=_relative(metadata_path, paths.root),
        summary=summary,
    )
