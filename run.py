#!/usr/bin/env python3
"""Entry point for the hydro-calibration-harness prototype.

Target usage (spec §9):
    python run.py --config config/basin.yaml

Reproducibly rebuilds all results from the externalized experiment configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data import get_meteo_timezone, load_basin_data
from src.metrics import get_log_nse_epsilon


def load_config(config_path: Path) -> dict:
    """Load and return the basin experiment configuration."""
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hydro-calibration harness (GR4J prototype)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/basin.yaml"),
        help="Path to basin experiment YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for figures, runs.csv, and rapport_calage.md",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for cached API responses",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Run data acquisition and QA pipeline only (Phase 1)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore disk cache and re-fetch external APIs",
    )
    parser.add_argument(
        "--meteo-timezone",
        default=None,
        help="Override Open-Meteo daily aggregation timezone (default: config data.meteo_timezone)",
    )
    parser.add_argument(
        "--gr4j-demo",
        action="store_true",
        help="Run one GR4J demonstration simulation and save a hydrograph PNG (Phase 2)",
    )
    parser.add_argument(
        "--metrics-demo",
        action="store_true",
        help="Evaluate uncalibrated GR4J metrics and write metrics_uncalibrated.csv (Phase 3)",
    )
    parser.add_argument(
        "--hydro-summary",
        action="store_true",
        help="Generate hydrological period characterization (Phase 3B)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run LHS parameter exploration and write runs.csv (Phase 4)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Build GLUE-inspired behavioral ensemble from existing runs.csv (Phase 5)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate rapport_calage.md from experiment outputs (Phase 6)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate final interview demo figures from artifacts (Phase 7)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full reproducible pipeline including demo figures",
    )
    parser.add_argument(
        "--export-demo",
        action="store_true",
        help="Export static interview demo assets from existing artifacts (no recalibration)",
    )
    parser.add_argument(
        "--build-ml-dataset",
        action="store_true",
        help="Build residual-correction ML dataset from best-calibration GR4J (Phase 8A)",
    )
    parser.add_argument(
        "--train-ml-baselines",
        action="store_true",
        help="Train residual ML baselines on calibration and evaluate on validation (Phase 8B)",
    )
    parser.add_argument(
        "--ml-ablation",
        action="store_true",
        help="Residual-correction robustness and ablation analysis (Phase 8C)",
    )
    parser.add_argument(
        "--ml-horizon-forecast",
        action="store_true",
        help="True multi-horizon residual forecasting under oracle weather (Phase 8D)",
    )
    parser.add_argument(
        "--meteo-sensitivity",
        action="store_true",
        help="Synthetic meteorological forcing sensitivity for +24/+48/+72 h (Phase 8E)",
    )
    parser.add_argument(
        "--uncertainty-calibration",
        action="store_true",
        help="Calibrate and evaluate forecast intervals for +24/+48/+72 h (Phase 9)",
    )
    return parser.parse_args(argv)


def print_data_report(df, qa: dict) -> None:
    """Print Phase 1 data pipeline summary to stdout."""
    print("=== Data pipeline report ===")
    print(f"Station:              {qa['station_code']}")
    print(f"Basin area:           {qa['basin_area_km2']} km²")
    print(f"Centroid:             ({qa['centroid_lat']}, {qa['centroid_lon']})")
    print(f"Meteo timezone:       {qa['meteo_timezone']}")
    print()
    print(f"Requested date range: {qa['requested_period']['start']} -> {qa['requested_period']['end']}")
    print(f"Calendar days:        {qa['raw_period']['n_days']}")
    print()
    print("Discharge (raw):      L/s — Hub'Eau QmnJ")
    q_ls = df["discharge_ls"]
    print(f"  Valid obs:          {int(q_ls.notna().sum())}")
    print(f"  Missing:            {int(q_ls.isna().sum())} ({qa['variables']['discharge_ls']['missing_proportion']:.1%})")
    if q_ls.notna().any():
        print(f"  Range:              {q_ls.min():.1f} - {q_ls.max():.1f} L/s")
    print()
    print("Discharge (converted): mm/day")
    q_mm = df["discharge_mm"]
    print(f"  Valid obs:          {int(q_mm.notna().sum())}")
    print(f"  Missing:            {int(q_mm.isna().sum())}")
    if q_mm.notna().any():
        print(f"  Range:              {q_mm.min():.3f} - {q_mm.max():.3f} mm/day")
    print(f"  Formula:            {qa['conversion_formula']}")
    print()
    for var, label in [("precipitation_mm", "Precipitation"), ("et0_mm", "ET0 (FAO-56)")]:
        stats = qa["variables"][var]
        print(f"{label}:")
        print(f"  Valid obs:          {stats['n_valid']}")
        print(f"  Missing:            {stats['n_missing']} ({stats['missing_proportion']:.1%})")
        if stats["first_valid"]:
            series = df[var].dropna()
            print(f"  Range:              {series.min():.3f} - {series.max():.3f} mm/day")
    print()
    usable = qa["usable_period"]
    print(f"Usable period (all variables present):")
    print(f"  {usable['start']} -> {usable['end']}")
    print(f"  Days:               {usable['n_days']}")
    print(f"  Excluded:           {usable['missing_proportion']:.1%} of calendar days")
    print()
    if qa.get("processed_data_path"):
        print(f"Processed data:       {qa['processed_data_path']}")
    dup_audit = qa.get("hubeau_duplicate_audit", {})
    if dup_audit:
        print()
        print("Hub'Eau duplicate audit:")
        print(f"  Identical duplicates resolved: {dup_audit.get('identical_duplicate_dates_resolved', 0)} dates")
        conflicting = dup_audit.get("conflicting_dates", [])
        if conflicting:
            print(f"  Conflicting dates:             {conflicting}")
        else:
            print("  Conflicting dates:             none")
    if qa.get("api_issues"):
        print()
        print("API / spec discrepancies:")
        for issue in qa["api_issues"]:
            print(f"  - {issue}")
    if qa.get("limitations"):
        print()
        print("Stated limitations:")
        for item in qa["limitations"]:
            print(f"  - {item}")


def run_data_pipeline(args: argparse.Namespace, config: dict) -> tuple[object, dict]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timezone = args.meteo_timezone if args.meteo_timezone is not None else get_meteo_timezone(config)
    return load_basin_data(
        config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        timezone=timezone,
        force_refresh=args.force_refresh,
    )


def run_metrics_stage(args: argparse.Namespace, config: dict) -> Path:
    from src.evaluation import (
        evaluate_uncalibrated_run,
        metrics_to_dataframe,
        print_metrics_table,
        save_metrics_csv,
    )

    timezone = args.meteo_timezone if args.meteo_timezone is not None else get_meteo_timezone(config)
    evaluation = evaluate_uncalibrated_run(
        config,
        cache_dir=args.cache_dir,
        timezone=timezone,
    )
    csv_path = save_metrics_csv(
        metrics_to_dataframe(evaluation),
        args.output_dir / "metrics_uncalibrated.csv",
    )
    print_metrics_table(evaluation)
    print()
    print(f"CSV written: {csv_path.resolve()}")
    print(f"log-NSE convention: ln(Q + {get_log_nse_epsilon(config)} mm/day)")
    return csv_path


def run_hydro_summary_stage(args: argparse.Namespace, config: dict):
    from src.hydrology import print_hydrological_summary, run_hydrological_characterization

    processed_path = args.output_dir / "data" / "basin_daily.csv"
    summary = run_hydrological_characterization(
        processed_path=processed_path,
        config=config,
        output_dir=args.output_dir,
    )
    print_hydrological_summary(summary)
    print()
    print(f"CSV written:   {(args.output_dir / 'hydrological_summary.csv').resolve()}")
    print(f"Figure written: {(args.output_dir / 'hydrological_years.png').resolve()}")
    return summary


def run_calibration_stage(args: argparse.Namespace, config_path: Path, config: dict):
    from src.calibration import run_calibration_experiment
    from src.calibration_diagnostics import (
        load_n1000_reference,
        print_phase4b_report,
        save_n1000_reference,
        save_parameter_space_diagnostic_figure,
    )
    from src.experiment_data import load_experiment_data
    from src.experiment_metadata import save_metadata
    from src.validation import select_best_calibration_candidate

    processed_path = args.output_dir / "data" / "basin_daily.csv"
    n1000_ref_path = args.output_dir / "reference" / "n1000_best_calibration.csv"
    runs_path = args.output_dir / "runs.csv"
    if runs_path.is_file() and len(pd.read_csv(runs_path)) == 1000:
        n1000_best = select_best_calibration_candidate(pd.read_csv(runs_path))
        save_n1000_reference(n1000_best, n1000_ref_path)
    data = load_experiment_data(
        config,
        cache_dir=args.cache_dir,
        processed_path=processed_path,
        timezone=args.meteo_timezone if args.meteo_timezone is not None else get_meteo_timezone(config),
    )
    result = run_calibration_experiment(config, data, args.output_dir)
    figure_path = save_parameter_space_diagnostic_figure(
        result.runs,
        args.output_dir / "parameter_space_diagnostic.png",
    )
    n1000_best = load_n1000_reference(n1000_ref_path)
    print_phase4b_report(
        result,
        config,
        n1000_best=n1000_best,
        diagnostic_figure_path=figure_path,
    )
    save_metadata(
        args.output_dir,
        {
            "calibration_runtime_s": result.runtime_total_s,
            "calibration_runtime_per_eval_s": result.runtime_per_run_s,
        },
    )
    return result


def run_ensemble_stage(args: argparse.Namespace, config: dict):
    from src.ensemble import print_ensemble_report, run_ensemble_analysis
    from src.experiment_data import load_experiment_data
    from src.experiment_metadata import save_metadata

    runs_path = args.output_dir / "runs.csv"
    processed_path = args.output_dir / "data" / "basin_daily.csv"
    if not runs_path.is_file():
        raise FileNotFoundError(f"runs.csv not found: {runs_path}")
    runs = pd.read_csv(runs_path)
    data = load_experiment_data(
        config,
        cache_dir=args.cache_dir,
        processed_path=processed_path,
        timezone=args.meteo_timezone if args.meteo_timezone is not None else get_meteo_timezone(config),
    )
    result = run_ensemble_analysis(config, data, runs, args.output_dir)
    print_ensemble_report(result)
    save_metadata(args.output_dir, {"ensemble_runtime_s": result.runtime_total_s})
    return result


def run_report_stage(args: argparse.Namespace, config_path: Path, config: dict) -> Path:
    from src.experiment_metadata import build_reproducibility_block
    from src.rapport_calage import generate_rapport_calage

    required = [
        "runs.csv",
        "metrics_uncalibrated.csv",
        "behavioral_runs.csv",
        "ensemble_timeseries.csv",
        "hydrological_summary.csv",
        "data/basin_daily.csv",
    ]
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {', '.join(missing)}")
    report_path = generate_rapport_calage(config_path, args.output_dir)
    repro = build_reproducibility_block(config_path, config, output_dir=args.output_dir)
    print("=== Phase 6: calibration report ===")
    print(f"Report written: {report_path.resolve()}")
    print(f"Config SHA256:    {repro['config_sha256']}")
    print(f"Git commit:       {repro['git_commit'] or 'not available'}")
    return report_path


def run_demo_stage(args: argparse.Namespace, config_path: Path) -> dict[str, Path]:
    from src.demo import generate_demo_figures

    required = [
        "runs.csv",
        "metrics_uncalibrated.csv",
        "behavioral_runs.csv",
        "ensemble_timeseries.csv",
        "ensemble_threshold_sensitivity.csv",
        "data/basin_daily.csv",
    ]
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {', '.join(missing)}")
    paths = generate_demo_figures(config_path, args.output_dir)
    print("=== Phase 7: demo figures ===")
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")
    return paths


def run_ml_dataset_stage(args: argparse.Namespace, config: dict):
    from src.experiment_data import load_experiment_data
    from src.ml_residual_dataset import (
        print_ml_residual_dataset_report,
        run_ml_residual_dataset_export,
    )

    runs_path = args.output_dir / "runs.csv"
    processed_path = args.output_dir / "data" / "basin_daily.csv"
    if not runs_path.is_file():
        raise FileNotFoundError(f"runs.csv not found: {runs_path}")
    runs = pd.read_csv(runs_path)
    data = load_experiment_data(
        config,
        cache_dir=args.cache_dir,
        processed_path=processed_path,
        timezone=args.meteo_timezone if args.meteo_timezone is not None else get_meteo_timezone(config),
    )
    result = run_ml_residual_dataset_export(data, config, runs, args.output_dir)
    print_ml_residual_dataset_report(result)
    return result


def run_ml_baselines_stage(args: argparse.Namespace, config: dict):
    from src.ml_residual_baselines import (
        print_ml_residual_baselines_report,
        run_ml_residual_baselines_export,
    )

    dataset_path = args.output_dir / "ml_residual_dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"ml_residual_dataset.csv not found: {dataset_path} "
            "(run --build-ml-dataset first)"
        )
    result = run_ml_residual_baselines_export(
        dataset_path,
        args.output_dir,
        config=config,
    )
    print_ml_residual_baselines_report(result)
    return result


def run_ml_ablation_stage(args: argparse.Namespace, config: dict):
    from src.ml_residual_ablation import print_ml_ablation_report, run_ml_ablation_export

    dataset_path = args.output_dir / "ml_residual_dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"ml_residual_dataset.csv not found: {dataset_path} "
            "(run --build-ml-dataset first)"
        )
    result = run_ml_ablation_export(dataset_path, args.output_dir, config=config)
    print_ml_ablation_report(result)
    return result


def run_ml_horizon_stage(args: argparse.Namespace, config: dict):
    from src.ml_horizon_forecast import print_ml_horizon_report, run_ml_horizon_export

    dataset_path = args.output_dir / "ml_residual_dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"ml_residual_dataset.csv not found: {dataset_path} "
            "(run --build-ml-dataset first)"
        )
    result = run_ml_horizon_export(dataset_path, args.output_dir, config=config)
    print_ml_horizon_report(result)
    return result


def run_meteo_sensitivity_stage(args: argparse.Namespace, config: dict):
    from src.meteo_sensitivity import print_meteo_sensitivity_report, run_meteo_sensitivity_export

    dataset_path = args.output_dir / "ml_residual_dataset.csv"
    basin_path = args.output_dir / "data" / "basin_daily.csv"
    runs_path = args.output_dir / "runs.csv"
    missing = [
        str(p)
        for p in (dataset_path, basin_path, runs_path)
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {', '.join(missing)}")
    result = run_meteo_sensitivity_export(
        dataset_path=dataset_path,
        basin_data_path=basin_path,
        runs_path=runs_path,
        output_dir=args.output_dir,
        config=config,
    )
    print_meteo_sensitivity_report(result)
    return result


def run_uncertainty_stage(args: argparse.Namespace, config: dict):
    from src.uncertainty_calibration import print_uncertainty_report, run_uncertainty_export

    dataset_path = args.output_dir / "ml_residual_dataset.csv"
    basin_path = args.output_dir / "data" / "basin_daily.csv"
    runs_path = args.output_dir / "runs.csv"
    ensemble_path = args.output_dir / "ensemble_timeseries.csv"
    missing = [
        str(p)
        for p in (dataset_path, basin_path, runs_path, ensemble_path)
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {', '.join(missing)}")
    result = run_uncertainty_export(
        dataset_path=dataset_path,
        basin_data_path=basin_path,
        runs_path=runs_path,
        ensemble_path=ensemble_path,
        output_dir=args.output_dir,
        config=config,
    )
    print_uncertainty_report(result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()

    if not config_path.is_file():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)

    if args.data_only:
        try:
            df, qa = run_data_pipeline(args, config)
        except Exception as exc:
            print(f"Data pipeline failed: {exc}", file=sys.stderr)
            return 1
        print_data_report(df, qa)
        return 0

    if args.gr4j_demo:
        from scripts.gr4j_demo import run_demo

        try:
            simulated, path = run_demo(
                config_path=config_path,
                cache_dir=args.cache_dir,
                output_path=args.output_dir / "hydrograph_gr4j_demo.png",
            )
        except Exception as exc:
            print(f"GR4J demo failed: {exc}", file=sys.stderr)
            return 1
        print("GR4J demo complete")
        print(f"  hydrograph:  {path.resolve()}")
        print(f"  sim range:   {simulated.min():.3f} - {simulated.max():.3f} mm/d")
        print(f"  parameters:  X1=350, X2=0, X3=90, X4=1.4 (demonstration, not calibrated)")
        return 0

    if args.metrics_demo:
        try:
            run_metrics_stage(args, config)
        except Exception as exc:
            print(f"Metrics evaluation failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.hydro_summary:
        try:
            run_hydro_summary_stage(args, config)
        except Exception as exc:
            print(f"Hydrological summary failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.calibrate:
        try:
            run_calibration_stage(args, config_path, config)
        except Exception as exc:
            print(f"Calibration experiment failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.ensemble:
        try:
            run_ensemble_stage(args, config)
        except Exception as exc:
            print(f"Ensemble analysis failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.report:
        try:
            run_report_stage(args, config_path, config)
        except Exception as exc:
            print(f"Report generation failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.demo:
        try:
            run_demo_stage(args, config_path)
        except Exception as exc:
            print(f"Demo figure generation failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.export_demo:
        from src.demo_export import DemoExportError, export_static_demo

        demo_dir = Path("demo")
        try:
            paths = export_static_demo(config_path, args.output_dir, demo_dir)
        except DemoExportError as exc:
            print(f"Demo export failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Demo export failed: {exc}", file=sys.stderr)
            return 1
        print("=== Static interview demo export ===")
        print(f"demo_data.json: {paths['demo_data'].resolve()}")
        print(f"report HTML:    {paths['report_html'].resolve()}")
        print(f"Open file:      {(demo_dir / 'index.html').resolve()}")
        print("Or serve:        python -m http.server 8000")
        print("Then open:       http://localhost:8000/demo/")
        return 0

    if args.build_ml_dataset:
        try:
            run_ml_dataset_stage(args, config)
        except Exception as exc:
            print(f"ML residual dataset build failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.train_ml_baselines:
        try:
            run_ml_baselines_stage(args, config)
        except Exception as exc:
            print(f"ML residual baselines failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.ml_ablation:
        try:
            run_ml_ablation_stage(args, config)
        except Exception as exc:
            print(f"ML ablation analysis failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.ml_horizon_forecast:
        try:
            run_ml_horizon_stage(args, config)
        except Exception as exc:
            print(f"ML horizon forecast failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.meteo_sensitivity:
        try:
            run_meteo_sensitivity_stage(args, config)
        except Exception as exc:
            print(f"Meteo sensitivity analysis failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.uncertainty_calibration:
        try:
            run_uncertainty_stage(args, config)
        except Exception as exc:
            print(f"Uncertainty calibration failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.all:
        stages = [
            ("data", lambda: run_data_pipeline(args, config)),
            ("metrics", lambda: run_metrics_stage(args, config)),
            ("hydrological summary", lambda: run_hydro_summary_stage(args, config)),
            ("calibration", lambda: run_calibration_stage(args, config_path, config)),
            ("ensemble", lambda: run_ensemble_stage(args, config)),
            ("report", lambda: run_report_stage(args, config_path, config)),
            ("demo figures", lambda: run_demo_stage(args, config_path)),
        ]
        for idx, (label, action) in enumerate(stages, start=1):
            print(f"[{idx}/{len(stages)}] {label}")
            try:
                action()
            except Exception as exc:
                print(f"Stage failed ({label}): {exc}", file=sys.stderr)
                return 1
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("hydro-calibration-harness")
    print(f"  config:      {config_path}")
    print(f"  station:     {config['station']['code']}")
    print(f"  model:       {config['model']['name']}")
    print()
    print("Full pipeline not yet implemented. Use --data-only for Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
