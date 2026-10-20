"""Command-line interface for MacroCanvas."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from macrocanvas import __version__
from macrocanvas.config import ProjectConfig
from macrocanvas.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="macrocanvas",
        description=(
            "Generate a synthetic semi-structural macroeconomy, estimate it, "
            "and render an academic visual report."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML configuration file (default: configs/baseline.toml if present).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root for generated data and reports (default: current directory).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the simulation seed.")
    parser.add_argument(
        "--periods",
        type=int,
        default=None,
        help="Override the number of quarterly observations.",
    )
    parser.add_argument(
        "--var-lags",
        type=int,
        default=None,
        help="Override the VAR lag order.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MacroCanvas {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    try:
        if args.config is not None:
            config = ProjectConfig.from_toml(args.config)
        elif (Path.cwd() / "configs" / "baseline.toml").exists():
            config = ProjectConfig.from_toml(Path.cwd() / "configs" / "baseline.toml")
        else:
            config = ProjectConfig()
        simulation = config.simulation
        estimation = config.estimation
        if args.seed is not None:
            simulation = replace(simulation, seed=args.seed)
        if args.periods is not None:
            simulation = replace(simulation, periods=args.periods)
        if args.var_lags is not None:
            estimation = replace(estimation, var_lags=args.var_lags)
        config = replace(config, simulation=simulation, estimation=estimation)
        config.validate()
        result = run_pipeline(config, root=args.output_dir)
    except Exception as exc:  # pragma: no cover - CLI presentation boundary
        logging.getLogger(__name__).error("%s", exc)
        if not args.quiet:
            logging.getLogger(__name__).exception("Pipeline failed")
        return 1

    summary = result.summary
    sample = summary["sample"]
    print()
    print("MacroCanvas завершён")
    print(f"  Период:       {sample['start']}–{sample['end']} ({sample['n_obs']} кв.)")
    print(f"  Seed:         {sample['seed']}")
    print(f"  Отчёт:        {result.report_file}")
    print(f"  Метаданные:   {result.metadata_file}")
    print(f"  Графики:      {len(result.figure_files)} PNG в reports/figures/")
    print(f"  Таблицы:      {len(result.table_files)} CSV в reports/tables/")
    equations = summary["equations"]
    print(
        "  Фокус-оценки: "
        + "; ".join(
            f"{key} β={value['coefficient']:+.3f}" for key, value in equations.items()
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
