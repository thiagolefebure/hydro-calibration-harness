"""Presentation-only demo figures and packaging for Phase 7.

This module reads generated artifacts and config to build interview-ready
figures without changing the scientific calibration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.calibration_diagnostics import kge_cal_val_correlation
from src.ensemble import (
    empirical_validation_coverage,
    get_behavioral_threshold,
    validation_plot_window,
)
from src.evaluation import DEMO_PARAMETERS, run_continuous_gr4j
from src.experiment_metadata import load_metadata
from src.gr4j import GR4JParameters
from src.hydrology import load_processed_daily
from src.validation import select_best_calibration_candidate

DEMO_01_FILENAME = "demo_01_calibration_impact.png"
DEMO_02_FILENAME = "demo_02_validation.png"
DEMO_03_FILENAME = "demo_03_uncertainty.png"

VALIDATION_ISOLATION_NOTE = (
    "Validation observations are never used for parameter fitting or ranking."
)
UNCERTAINTY_ONLY_NOTE = (
    "Parametric uncertainty only — excludes precipitation, observation, "
    "initial-state and structural uncertainty."
)
# Deterministic window rule used by validation_plot_window (not shown on figures).
WINDOW_RULE_NOTE = (
    "Validation window centered on maximum observed validation discharge."
)


@dataclass(frozen=True)
class DemoContext:
    config: dict[str, Any]
    output_dir: Path
    processed_data: pd.DataFrame
    runs: pd.DataFrame
    baseline_metrics: pd.DataFrame
    behavioral_runs: pd.DataFrame
    ensemble_timeseries: pd.DataFrame
    sensitivity: pd.DataFrame
    metadata: dict[str, Any]
    best_run: pd.Series
    baseline_simulation: pd.Series
    calibrated_simulation: pd.Series
    window_start: pd.Timestamp
    window_end: pd.Timestamp


def demo_output_paths(output_dir: Path) -> dict[str, Path]:
    """Stable Phase 7 output paths."""
    return {
        "calibration_impact": output_dir / DEMO_01_FILENAME,
        "validation_summary": output_dir / DEMO_02_FILENAME,
        "uncertainty": output_dir / DEMO_03_FILENAME,
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _baseline_metric(baseline_metrics: pd.DataFrame, name: str, column: str) -> float:
    row = baseline_metrics.loc[baseline_metrics["metric"] == name]
    if row.empty:
        raise KeyError(f"Metric {name!r} not found in baseline metrics")
    return float(row.iloc[0][column])


def _best_params(best_run: pd.Series) -> GR4JParameters:
    return GR4JParameters(
        X1=float(best_run["x1"]),
        X2=float(best_run["x2"]),
        X3=float(best_run["x3"]),
        X4=float(best_run["x4"]),
    )


def _format_signed_pct(value: float) -> str:
    pct = 100.0 * value
    return f"{pct:+.1f}%"


def _format_runtime_minutes(runtime_s: float | None) -> str:
    if runtime_s is None:
        return "n/a"
    minutes = max(1, int(round(float(runtime_s) / 60.0)))
    return f"≈{minutes} min"


def load_demo_context(config_path: Path, output_dir: Path) -> DemoContext:
    """Load generated artifacts and derived series for demo figures."""
    config = _load_config(config_path)
    processed = load_processed_daily(output_dir / "data" / "basin_daily.csv")
    runs = pd.read_csv(output_dir / "runs.csv")
    baseline_metrics = pd.read_csv(output_dir / "metrics_uncalibrated.csv")
    behavioral_runs = pd.read_csv(output_dir / "behavioral_runs.csv")
    ensemble_timeseries = pd.read_csv(output_dir / "ensemble_timeseries.csv")
    sensitivity = pd.read_csv(output_dir / "ensemble_threshold_sensitivity.csv")
    metadata = load_metadata(output_dir)

    best_run = select_best_calibration_candidate(runs)
    baseline_sim = run_continuous_gr4j(processed, config, DEMO_PARAMETERS)
    calibrated_sim = run_continuous_gr4j(processed, config, _best_params(best_run))

    observed = processed.loc[:, "discharge_mm"]
    window_start, window_end = validation_plot_window(observed, config)

    return DemoContext(
        config=config,
        output_dir=output_dir,
        processed_data=processed,
        runs=runs,
        baseline_metrics=baseline_metrics,
        behavioral_runs=behavioral_runs,
        ensemble_timeseries=ensemble_timeseries,
        sensitivity=sensitivity,
        metadata=metadata,
        best_run=best_run,
        baseline_simulation=baseline_sim,
        calibrated_simulation=calibrated_sim,
        window_start=window_start,
        window_end=window_end,
    )


def _apply_demo_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )


def _window_mask(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (index >= start) & (index <= end)


def plot_demo_calibration_impact(context: DemoContext, output_path: Path) -> Path:
    """Figure 1: observed vs uncalibrated vs automatically calibrated."""
    _apply_demo_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = context.processed_data.copy()
    mask = _window_mask(data.index, context.window_start, context.window_end)
    window = data.loc[mask].copy()

    baseline = context.baseline_simulation.loc[window.index]
    calibrated = context.calibrated_simulation.loc[window.index]

    baseline_kge_val = _baseline_metric(context.baseline_metrics, "KGE", "validation")
    calibrated_kge_val = float(context.best_run["kge_val"])

    fig, (ax_p, ax_q) = plt.subplots(
        2,
        1,
        figsize=(11.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55, 3.4], "hspace": 0.08},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10)

    ax_p.bar(
        window.index,
        -window["precipitation_mm"],
        width=1.0,
        color="#60a5fa",
        edgecolor="none",
    )
    ax_p.set_ylabel("Precip.\n(mm/d)")
    ax_p.set_ylim(-max(float(window["precipitation_mm"].max()) * 1.15, 1.0), 0.0)
    ax_p.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax_p.tick_params(labelbottom=False)

    ax_q.plot(window.index, window["discharge_mm"], color="#111827", linewidth=1.8, label="Observed")
    ax_q.plot(
        window.index,
        baseline,
        color="#9ca3af",
        linewidth=1.5,
        linestyle="--",
        label="Uncalibrated",
    )
    ax_q.plot(
        window.index,
        calibrated,
        color="#dc2626",
        linewidth=1.8,
        label="Automatically calibrated",
    )
    ax_q.set_ylabel("Discharge (mm/d)")
    ax_q.set_xlabel("Date")
    ax_q.grid(True, alpha=0.25, linewidth=0.5)
    ax_q.legend(loc="upper right", frameon=True)

    fig.suptitle("Automated calibration — validation hydrograph", fontsize=14, y=0.97)
    ax_q.text(
        0.01,
        0.97,
        f"Validation KGE: {baseline_kge_val:.3f} → {calibrated_kge_val:.3f}",
        transform=ax_q.transAxes,
        fontsize=10,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d1d5db", "pad": 4},
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_demo_validation_summary(context: DemoContext, output_path: Path) -> Path:
    """Figure 2: calibration vs validation technical summary."""
    _apply_demo_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = context.baseline_metrics
    best = context.best_run
    corr = kge_cal_val_correlation(context.runs)
    runtime = context.metadata.get("calibration_runtime_s")
    n_runs = len(context.runs)

    table_rows: list[list[str]] = []
    for metric, best_cal_col, best_val_col in (
        ("KGE", "kge_cal", "kge_val"),
        ("NSE", "nse_cal", "nse_val"),
        ("log-NSE", "lognse_cal", "lognse_val"),
        ("Volume bias", "bias_cal", "bias_val"),
    ):
        base_row = baseline.loc[baseline["metric"] == metric].iloc[0]
        if metric == "Volume bias":
            table_rows.append(
                [
                    metric,
                    _format_signed_pct(float(base_row["calibration"])),
                    _format_signed_pct(float(base_row["validation"])),
                    _format_signed_pct(float(best[best_cal_col])),
                    _format_signed_pct(float(best[best_val_col])),
                ]
            )
        else:
            table_rows.append(
                [
                    metric,
                    f"{float(base_row['calibration']):.3f}",
                    f"{float(base_row['validation']):.3f}",
                    f"{float(best[best_cal_col]):.3f}",
                    f"{float(best[best_val_col]):.3f}",
                ]
            )

    fig = plt.figure(figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.84, bottom=0.14)
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.suptitle(
        "Calibration performance vs independent validation",
        fontsize=14,
        y=0.95,
    )
    fig.text(
        0.5,
        0.88,
        VALIDATION_ISOLATION_NOTE,
        ha="center",
        va="top",
        fontsize=10,
        color="#374151",
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Metric",
            "Uncalibrated\ncalibration",
            "Uncalibrated\nvalidation",
            "Calibrated\ncalibration",
            "Calibrated\nvalidation",
        ],
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.05, 0.18, 0.90, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _col), cell in table.get_celld().items():
        cell.set_height(0.14)
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold", fontsize=10)
        else:
            cell.set_facecolor("#ffffff")

    footer = (
        f"{n_runs:,} LHS runs · selection on KGE_cal only · "
        f"corr(cal,val)={corr:.3f} · runtime{_format_runtime_minutes(runtime)}"
    )
    fig.text(0.5, 0.05, footer, ha="center", va="bottom", fontsize=10, color="#374151")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_demo_uncertainty(context: DemoContext, output_path: Path) -> Path:
    """Figure 3: behavioral envelope on deterministic validation window."""
    _apply_demo_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeseries = context.ensemble_timeseries.copy()
    timeseries["date_ts"] = pd.to_datetime(timeseries["date"])
    mask = (
        (timeseries["period"] == "validation")
        & (timeseries["date_ts"] >= context.window_start)
        & (timeseries["date_ts"] <= context.window_end)
    )
    window = timeseries.loc[mask].copy()
    threshold = get_behavioral_threshold(context.config)
    coverage = empirical_validation_coverage(context.ensemble_timeseries)
    n_members = len(context.behavioral_runs)

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16)

    ax.fill_between(
        window["date_ts"],
        window["q05"],
        window["q95"],
        color="#93c5fd",
        alpha=0.28,
        label="q05–q95 behavioral envelope",
    )
    ax.plot(
        window["date_ts"],
        window["q50"],
        color="#1d4ed8",
        linewidth=1.8,
        label="Behavioral ensemble median q50",
    )
    ax.plot(
        window["date_ts"],
        window["q_obs"],
        color="#111827",
        linewidth=1.6,
        label="Observed",
    )
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Parametric uncertainty — validation period", fontsize=14, y=0.96)
    fig.text(
        0.5,
        0.90,
        (
            f"{n_members} behavioral parameter sets · "
            f"KGE_cal > {threshold:.2f} · "
            f"empirical coverage {coverage * 100.0:.1f}%"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#374151",
    )
    fig.text(
        0.5,
        0.035,
        UNCERTAINTY_ONLY_NOTE,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_demo_figures(config_path: Path, output_dir: Path) -> dict[str, Path]:
    """Generate all three final Phase 7 demo figures from artifacts."""
    context = load_demo_context(config_path, output_dir)
    paths = demo_output_paths(output_dir)
    plot_demo_calibration_impact(context, paths["calibration_impact"])
    plot_demo_validation_summary(context, paths["validation_summary"])
    plot_demo_uncertainty(context, paths["uncertainty"])
    return paths
