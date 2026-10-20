"""End-to-end smoke test for the full reporting pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from macrocanvas.config import ProjectConfig
from macrocanvas.pipeline import run_pipeline


def test_pipeline_writes_complete_report(tmp_path: Path) -> None:
    base = ProjectConfig()
    config = replace(
        base,
        simulation=replace(base.simulation, seed=314, periods=60, burn_in=24),
        estimation=replace(
            base.estimation,
            bootstrap_reps=50,
            projection_paths=100,
            projection_horizon=6,
            var_lags=1,
        ),
    )
    result = run_pipeline(config, root=tmp_path)

    assert len(result.figure_files) == 8
    for relative_path in result.figure_files.values():
        path = tmp_path / relative_path
        assert path.is_file()
        assert path.stat().st_size > 20_000
    assert (tmp_path / result.report_file).is_file()
    assert (tmp_path / result.metadata_file).is_file()
    assert len(result.table_files) >= 14
    report = (tmp_path / result.report_file).read_text(encoding="utf-8")
    assert "MacroCanvas" in report
    assert "Дисклеймер" in report
    assert result.var_result.is_stable
