"""Calibration experiment orchestration and runs.csv generation.

Responsibilities (per spec §4, §5):
- Evaluate each sampled parameter set with GR4J continuously (warm-up through validation).
- Compute calibration and validation metrics separately.
- Rank and select candidates using KGE_cal only.
- Write experiment log to runs.csv.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import (
    DEMO_PARAMETERS,
    evaluate_temporal_performance,
    run_continuous_gr4j,
)
from src.gr4j import GR4JParameters
from src.metrics import MetricResult
from src.sampling import sample_parameters, save_parameter_samples
from src.validation import (
    CALIBRATION_RANK_COLUMN,
    rank_by_kge_calibration,
    select_best_calibration_candidate,
    top_calibration_candidates,
)

RUNS_CSV_COLUMNS = [
    "run_id",
    "x1",
    "x2",
    "x3",
    "x4",
    "nse_cal",
    "kge_cal",
    "r_cal",
    "alpha_cal",
    "beta_cal",
    "lognse_cal",
    "bias_cal",
    "nse_val",
    "kge_val",
    "r_val",
    "alpha_val",
    "beta_val",
    "lognse_val",
    "bias_val",
    CALIBRATION_RANK_COLUMN,
]


@dataclass(frozen=True)
class CalibrationExperimentResult:
    """Outputs from a full LHS calibration exploration."""

    runs: pd.DataFrame
    top20: pd.DataFrame
    best: pd.Series
    baseline_row: pd.Series
    runtime_total_s: float
    runtime_per_run_s: float
    parameter_samples_path: Path
    runs_csv_path: Path
    top20_csv_path: Path


def _metric_value(metric: MetricResult) -> float:
    return float(metric.value) if metric.is_defined else float("nan")


def _params_from_row(row: pd.Series) -> GR4JParameters:
    return GR4JParameters(
        X1=float(row["X1"]),
        X2=float(row["X2"]),
        X3=float(row["X3"]),
        X4=float(row["X4"]),
    )


def _evaluation_to_record(
    run_id: int,
    params: GR4JParameters,
    evaluation,
) -> dict[str, Any]:
    cal = evaluation.calibration
    val = evaluation.validation
    return {
        "run_id": run_id,
        "x1": params.X1,
        "x2": params.X2,
        "x3": params.X3,
        "x4": params.X4,
        "nse_cal": _metric_value(cal.nse),
        "kge_cal": _metric_value(cal.kge),
        "r_cal": _metric_value(cal.kge_r),
        "alpha_cal": _metric_value(cal.kge_alpha),
        "beta_cal": _metric_value(cal.kge_beta),
        "lognse_cal": _metric_value(cal.lognse),
        "bias_cal": _metric_value(cal.bias),
        "nse_val": _metric_value(val.nse),
        "kge_val": _metric_value(val.kge),
        "r_val": _metric_value(val.kge_r),
        "alpha_val": _metric_value(val.kge_alpha),
        "beta_val": _metric_value(val.kge_beta),
        "lognse_val": _metric_value(val.lognse),
        "bias_val": _metric_value(val.bias),
    }


def evaluate_baseline(data: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Evaluate the fixed uncalibrated demonstration parameter set."""
    simulated = run_continuous_gr4j(data, config, DEMO_PARAMETERS)
    observed = data.loc[simulated.index, "discharge_mm"]
    evaluation = evaluate_temporal_performance(observed, simulated, config)
    record = _evaluation_to_record(-1, DEMO_PARAMETERS, evaluation)
    record["run_id"] = "baseline"
    record[CALIBRATION_RANK_COLUMN] = np.nan
    return pd.Series(record)


def run_calibration_experiment(
    config: dict[str, Any],
    data: pd.DataFrame,
    output_dir: Path,
) -> CalibrationExperimentResult:
    """Run full parameter exploration and write runs.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_parameters(config)
    samples_path = save_parameter_samples(samples, output_dir / "parameter_samples.csv")

    observed_index = data[["precipitation_mm", "et0_mm"]].dropna().index
    observed = data.loc[observed_index, "discharge_mm"]

    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    for run_id, row in enumerate(samples.itertuples(index=False), start=1):
        params = GR4JParameters(X1=row.X1, X2=row.X2, X3=row.X3, X4=row.X4)
        simulated = run_continuous_gr4j(data, config, params)
        evaluation = evaluate_temporal_performance(observed, simulated, config)
        records.append(_evaluation_to_record(run_id, params, evaluation))

    runtime_total = time.perf_counter() - start
    runs = pd.DataFrame.from_records(records)
    runs[CALIBRATION_RANK_COLUMN] = rank_by_kge_calibration(runs)

    runs_csv_path = output_dir / "runs.csv"
    runs[RUNS_CSV_COLUMNS].to_csv(runs_csv_path, index=False)

    top20 = top_calibration_candidates(runs, n=20)
    top20_csv_path = output_dir / "top20_calibration.csv"
    top20.to_csv(top20_csv_path, index=False)

    best = select_best_calibration_candidate(runs)
    baseline_row = evaluate_baseline(data, config)

    runtime_per_run = runtime_total / len(samples) if len(samples) else float("nan")

    return CalibrationExperimentResult(
        runs=runs,
        top20=top20,
        best=best,
        baseline_row=baseline_row,
        runtime_total_s=runtime_total,
        runtime_per_run_s=runtime_per_run,
        parameter_samples_path=samples_path,
        runs_csv_path=runs_csv_path,
        top20_csv_path=top20_csv_path,
    )


def kge_cal_distribution(runs: pd.DataFrame) -> dict[str, float]:
    series = runs["kge_cal"].dropna()
    if series.empty:
        return {"min": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def threshold_counts(runs: pd.DataFrame) -> dict[str, int]:
    series = runs["kge_cal"].dropna()
    return {
        "kge_cal_gt_0.5": int((series > 0.5).sum()),
        "kge_cal_gt_0.6": int((series > 0.6).sum()),
        "kge_cal_gt_0.7": int((series > 0.7).sum()),
        "kge_cal_gt_0.8": int((series > 0.8).sum()),
    }


def print_calibration_report(result: CalibrationExperimentResult, config: dict[str, Any]) -> None:
    """Print Phase 4 diagnostics to stdout."""
    bounds = config["model"]["parameter_bounds"]
    runs = result.runs
    samples = pd.read_csv(result.parameter_samples_path)

    print("=== Phase 4: LHS parameter exploration ===")
    print(f"Samples:           {len(runs)}")
    print(f"Total runtime:     {result.runtime_total_s:.2f} s")
    print(f"Per GR4J run:      {result.runtime_per_run_s:.4f} s")
    print()
    print("Configured bounds:")
    for name in ("X1", "X2", "X3", "X4"):
        lo, hi = bounds[name]
        print(f"  {name}: [{lo}, {hi}]  sampled [{samples[name].min():.4f}, {samples[name].max():.4f}]")
    print()
    print("Best calibration candidate (max KGE_cal):")
    print(
        f"  run_id={int(result.best['run_id'])}  "
        f"X1={result.best['x1']:.3f} X2={result.best['x2']:.3f} "
        f"X3={result.best['x3']:.3f} X4={result.best['x4']:.3f}"
    )
    print(f"  KGE_cal={result.best['kge_cal']:.4f}  KGE_val={result.best['kge_val']:.4f} (diagnostic only)")
    print()
    dist = kge_cal_distribution(runs)
    print("KGE_cal distribution:")
    print(
        f"  min={dist['min']:.4f}  median={dist['median']:.4f}  "
        f"p90={dist['p90']:.4f}  p95={dist['p95']:.4f}  max={dist['max']:.4f}"
    )
    counts = threshold_counts(runs)
    print("KGE_cal threshold counts:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    corr = runs[["kge_cal", "kge_val"]].dropna()
    if len(corr) >= 2:
        correlation = float(corr["kge_cal"].corr(corr["kge_val"]))
    else:
        correlation = float("nan")
    print(f"corr(KGE_cal, KGE_val): {correlation:.4f}")
    print()
    print("Baseline vs best calibration candidate:")
    print(f"{'':24} {'BASELINE':>12} {'BEST CAL':>12}")
    metrics = [
        ("KGE calibration", "kge_cal"),
        ("KGE validation", "kge_val"),
        ("NSE calibration", "nse_cal"),
        ("NSE validation", "nse_val"),
        ("log-NSE calibration", "lognse_cal"),
        ("log-NSE validation", "lognse_val"),
        ("volume bias calibration", "bias_cal"),
        ("volume bias validation", "bias_val"),
    ]
    for label, col in metrics:
        base = result.baseline_row[col]
        best = result.best[col]
        print(f"{label:<24} {base:12.4f} {best:12.4f}")
    print()
    print("Top 20 by KGE_cal (validation shown after ranking fixed):")
    print(f"{'rank':>4} {'run_id':>7} {'KGE_cal':>8} {'KGE_val':>8}")
    for _, row in result.top20.iterrows():
        print(
            f"{int(row[CALIBRATION_RANK_COLUMN]):>4} "
            f"{int(row['run_id']):>7} "
            f"{row['kge_cal']:8.4f} "
            f"{row['kge_val']:8.4f}"
        )
    print()
    print("Generalization diagnostic:")
    top_kge_val = result.top20["kge_val"].max()
    top_rank_val = int(result.top20.loc[result.top20["kge_val"].idxmax(), CALIBRATION_RANK_COLUMN])
    print(
        f"  Best KGE_val within top-20-by-KGE_cal: {top_kge_val:.4f} (rank {top_rank_val} by KGE_cal)"
    )
    strong_both = result.top20[(result.top20["kge_cal"] > 0.6) & (result.top20["kge_val"] > 0.6)]
    print(f"  Top-20 runs with KGE_cal>0.6 AND KGE_val>0.6: {len(strong_both)}")
    rank1_val = float(result.top20.iloc[0]["kge_val"])
    median_top20_val = float(result.top20["kge_val"].median())
    min_top20_val = float(result.top20["kge_val"].min())
    if min_top20_val >= 0.6 and rank1_val >= 0.6:
        answer = (
            "Yes — the top calibration-ranked solutions largely remain strong in validation "
            f"(top-20 KGE_val range {min_top20_val:.4f}–{top_kge_val:.4f}; "
            f"rank-1 KGE_val={rank1_val:.4f})."
        )
    elif rank1_val >= 0.6:
        answer = (
            "Partially — the best calibration candidate generalizes well "
            f"(KGE_val={rank1_val:.4f}), but some high KGE_cal runs weaken in validation "
            f"(minimum top-20 KGE_val={min_top20_val:.4f})."
        )
    else:
        answer = (
            "No — even the best calibration candidate shows weak validation performance "
            f"(rank-1 KGE_val={rank1_val:.4f}; median top-20 KGE_val={median_top20_val:.4f})."
        )
    print(f"  Do the highest-ranked calibration solutions also remain strong in validation? {answer}")
    print()
    print(f"runs.csv:            {result.runs_csv_path.resolve()}")
    print(f"top20_calibration:   {result.top20_csv_path.resolve()}")
    print(f"parameter_samples:   {result.parameter_samples_path.resolve()}")
