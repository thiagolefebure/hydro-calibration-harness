"""Phase 4B calibration-space and generalization diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import CalibrationExperimentResult
from src.validation import CALIBRATION_RANK_COLUMN, top_calibration_candidates

BOUND_PROXIMITY_FRACTION = 0.02
PARAMETER_COLUMNS = ("x1", "x2", "x3", "x4")
PARAMETER_LABELS = ("X1", "X2", "X3", "X4")


def kge_cal_distribution(runs: pd.DataFrame) -> dict[str, float]:
    """Extended KGE_cal distribution including p99."""
    series = runs["kge_cal"].dropna()
    if series.empty:
        nan = float("nan")
        return {
            "min": nan,
            "median": nan,
            "p90": nan,
            "p95": nan,
            "p99": nan,
            "max": nan,
        }
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def threshold_counts(runs: pd.DataFrame) -> dict[str, int]:
    series = runs["kge_cal"].dropna()
    thresholds = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85)
    return {f"kge_cal_gt_{t:g}": int((series > t).sum()) for t in thresholds}


def kge_cal_val_correlation(runs: pd.DataFrame) -> float:
    corr = runs[["kge_cal", "kge_val"]].dropna()
    if len(corr) < 2:
        return float("nan")
    return float(corr["kge_cal"].corr(corr["kge_val"]))


def parameters_close_to_bounds(
    params: dict[str, float],
    bounds: dict[str, list[float]],
    *,
    fraction: float = BOUND_PROXIMITY_FRACTION,
) -> dict[str, str | None]:
    """Return bound side ('lower', 'upper') for parameters within fraction of range."""
    close: dict[str, str | None] = {}
    for col, name in zip(PARAMETER_COLUMNS, PARAMETER_LABELS):
        lo, hi = bounds[name]
        span = hi - lo
        margin = fraction * span
        value = float(params[col])
        if value <= lo + margin:
            close[name] = "lower"
        elif value >= hi - margin:
            close[name] = "upper"
        else:
            close[name] = None
    return close


def format_bound_proximity(close: dict[str, str | None]) -> str:
    flagged = {name: side for name, side in close.items() if side is not None}
    if not flagged:
        return "none (all parameters >2% from configured bounds)"
    parts = [f"{name} near {side} bound" for name, side in flagged.items()]
    return "; ".join(parts)


def parameter_space_summary(
    runs: pd.DataFrame,
    *,
    subset_label: str,
    kge_threshold: float | None = None,
) -> pd.DataFrame:
    """Summarize parameter ranges for a run subset."""
    if kge_threshold is None:
        subset = runs
    else:
        subset = runs[runs["kge_cal"] > kge_threshold]

    rows: list[dict[str, Any]] = []
    for col, name in zip(PARAMETER_COLUMNS, PARAMETER_LABELS):
        series = subset[col].dropna()
        if series.empty:
            rows.append(
                {
                    "subset": subset_label,
                    "parameter": name,
                    "n_runs": 0,
                    "min": float("nan"),
                    "p10": float("nan"),
                    "median": float("nan"),
                    "p90": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "subset": subset_label,
                "parameter": name,
                "n_runs": int(len(series)),
                "min": float(series.min()),
                "p10": float(series.quantile(0.10)),
                "median": float(series.median()),
                "p90": float(series.quantile(0.90)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame.from_records(rows)


def parameter_space_summaries_all(runs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            parameter_space_summary(runs, subset_label="all", kge_threshold=None),
            parameter_space_summary(runs, subset_label="kge_cal_gt_0.7", kge_threshold=0.7),
            parameter_space_summary(runs, subset_label="kge_cal_gt_0.8", kge_threshold=0.8),
        ],
        ignore_index=True,
    )


def delta_kge_series(runs: pd.DataFrame) -> pd.Series:
    return runs["kge_val"] - runs["kge_cal"]


def delta_kge_summary(runs: pd.DataFrame) -> dict[str, float]:
    delta = delta_kge_series(runs).dropna()
    if delta.empty:
        nan = float("nan")
        return {"median": nan, "p10": nan, "p90": nan, "worst": nan, "best": nan}
    return {
        "median": float(delta.median()),
        "p10": float(delta.quantile(0.10)),
        "p90": float(delta.quantile(0.90)),
        "worst": float(delta.min()),
        "best": float(delta.max()),
    }


def top20_generalization_table(runs: pd.DataFrame) -> pd.DataFrame:
    top20 = top_calibration_candidates(runs, n=20)
    table = top20[
        [
            "run_id",
            CALIBRATION_RANK_COLUMN,
            "kge_cal",
            "kge_val",
            "x1",
            "x2",
            "x3",
            "x4",
        ]
    ].copy()
    table["delta_kge"] = table["kge_val"] - table["kge_cal"]
    return table[
        [
            "run_id",
            CALIBRATION_RANK_COLUMN,
            "kge_cal",
            "kge_val",
            "delta_kge",
            "x1",
            "x2",
            "x3",
            "x4",
        ]
    ]


def save_parameter_space_diagnostic_figure(runs: pd.DataFrame, output_path: Path) -> Path:
    """Scatter each parameter against KGE_cal in a 2x2 panel layout."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, col, label in zip(axes_flat, PARAMETER_COLUMNS, PARAMETER_LABELS):
        ax.scatter(
            runs[col],
            runs["kge_cal"],
            s=8,
            alpha=0.35,
            c="#2563eb",
            edgecolors="none",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("KGE_cal")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title(f"{label} vs KGE_cal")

    fig.suptitle("Parameter-space diagnostic (calibration performance only)", fontsize=12)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_n1000_reference(best_row: pd.Series, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([best_row]).to_csv(output_path, index=False)
    return output_path


def load_n1000_reference(path: Path) -> pd.Series | None:
    if not path.is_file():
        return None
    return pd.read_csv(path).iloc[0]


def print_phase4b_report(
    result: CalibrationExperimentResult,
    config: dict[str, Any],
    *,
    n1000_best: pd.Series | None = None,
    diagnostic_figure_path: Path | None = None,
) -> None:
    """Print Phase 4B extended diagnostics."""
    runs = result.runs
    bounds = config["model"]["parameter_bounds"]

    print("=== Phase 4B: higher-density parameter exploration ===")
    print(f"Samples:           {len(runs)}")
    print(f"Total runtime:     {result.runtime_total_s:.2f} s")
    print(f"Mean eval time:    {result.runtime_per_run_s:.4f} s")
    print()

    dist = kge_cal_distribution(runs)
    print("KGE_cal distribution:")
    print(
        f"  min={dist['min']:.4f}  median={dist['median']:.4f}  "
        f"p90={dist['p90']:.4f}  p95={dist['p95']:.4f}  "
        f"p99={dist['p99']:.4f}  max={dist['max']:.4f}"
    )
    print("KGE_cal threshold counts:")
    for key, value in threshold_counts(runs).items():
        print(f"  {key}: {value}")
    print(f"corr(KGE_cal, KGE_val): {kge_cal_val_correlation(runs):.4f}")
    print()

    best = result.best
    print("Best calibration candidate (N=5000, max KGE_cal):")
    print(
        f"  run_id={int(best['run_id'])}  "
        f"X1={best['x1']:.3f} X2={best['x2']:.3f} "
        f"X3={best['x3']:.3f} X4={best['x4']:.3f}"
    )
    print(f"  KGE_cal={best['kge_cal']:.4f}  KGE_val={best['kge_val']:.4f} (diagnostic only)")
    close = parameters_close_to_bounds(best.to_dict(), bounds)
    print(f"  Close to bound (within 2% of range): {format_bound_proximity(close)}")
    print()

    if n1000_best is not None:
        print("N=1000 vs N=5000 best calibration candidate:")
        print(f"{'':18} {'N=1000':>12} {'N=5000':>12} {'delta':>10}")
        for label, col in [
            ("run_id", "run_id"),
            ("KGE_cal", "kge_cal"),
            ("KGE_val", "kge_val"),
            ("X1", "x1"),
            ("X2", "x2"),
            ("X3", "x3"),
            ("X4", "x4"),
        ]:
            v1000 = n1000_best[col]
            v5000 = best[col]
            if col == "run_id":
                delta = int(v5000) - int(v1000)
                print(f"{label:<18} {int(v1000):12d} {int(v5000):12d} {delta:10d}")
            else:
                delta = float(v5000) - float(v1000)
                print(f"{label:<18} {float(v1000):12.4f} {float(v5000):12.4f} {delta:10.4f}")
        kge_delta = float(best["kge_cal"]) - float(n1000_best["kge_cal"])
        print(
            f"  KGE_cal improvement: {kge_delta:+.4f} "
            "(tiny differences are not necessarily meaningful)"
        )
        print()

    summaries = parameter_space_summaries_all(runs)
    print("Parameter-space summaries:")
    for subset in summaries["subset"].unique():
        block = summaries[summaries["subset"] == subset]
        print(f"  [{subset}] n={int(block['n_runs'].iloc[0])}")
        print(f"  {'param':<6} {'min':>10} {'p10':>10} {'median':>10} {'p90':>10} {'max':>10}")
        for _, row in block.iterrows():
            print(
                f"  {row['parameter']:<6} "
                f"{row['min']:10.3f} {row['p10']:10.3f} {row['median']:10.3f} "
                f"{row['p90']:10.3f} {row['max']:10.3f}"
            )
        print()

    delta = delta_kge_summary(runs)
    print("Generalization (delta_KGE = KGE_val - KGE_cal, all runs):")
    print(
        f"  median={delta['median']:.4f}  p10={delta['p10']:.4f}  "
        f"p90={delta['p90']:.4f}  worst={delta['worst']:.4f}  best={delta['best']:.4f}"
    )
    print()
    print("Top 20 calibration-ranked candidates (validation diagnostic only):")
    top20_table = top20_generalization_table(runs)
    print(
        f"{'run_id':>7} {'rank':>5} {'KGE_cal':>8} {'KGE_val':>8} "
        f"{'delta':>8} {'X1':>8} {'X2':>8} {'X3':>8} {'X4':>8}"
    )
    for _, row in top20_table.iterrows():
        print(
            f"{int(row['run_id']):7d} "
            f"{int(row[CALIBRATION_RANK_COLUMN]):5d} "
            f"{row['kge_cal']:8.4f} "
            f"{row['kge_val']:8.4f} "
            f"{row['delta_kge']:8.4f} "
            f"{row['x1']:8.2f} "
            f"{row['x2']:8.3f} "
            f"{row['x3']:8.2f} "
            f"{row['x4']:8.3f}"
        )
    print()
    if diagnostic_figure_path is not None:
        print(f"Diagnostic figure:   {diagnostic_figure_path.resolve()}")
    print(f"runs.csv:            {result.runs_csv_path.resolve()}")
    print(f"top20_calibration:   {result.top20_csv_path.resolve()}")
    print(f"parameter_samples:   {result.parameter_samples_path.resolve()}")
