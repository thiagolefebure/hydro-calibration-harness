"""Auto-generated report and demonstration figures.

Responsibilities (per spec §6, §7):
- hydrograph.png — observed vs simulated, inverted rain bars, period shading
- metrics.png — calibration | validation metric table (+ Top-N candidate comparison)
- ensemble.png — validation-period q05–q95 envelope with median and observed
- rapport_calage.md — full auditable report with reproducibility metadata and stated limitations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_ensemble_validation_window(
    timeseries: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    output_path: Path,
) -> Path:
    """Plot ~60-day validation window with parametric uncertainty envelope."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = timeseries.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    window = frame[
        (frame["period"] == "validation")
        & (frame["date_ts"] >= window_start)
        & (frame["date_ts"] <= window_end)
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    dates = window["date_ts"]

    ax.fill_between(
        dates,
        window["q05"],
        window["q95"],
        alpha=0.25,
        color="#93c5fd",
        label="q05–q95 behavioral envelope",
    )
    ax.plot(dates, window["q50"], color="#1d4ed8", linewidth=1.6, label="q50 (ensemble median)")
    ax.plot(dates, window["q_obs"], color="#111827", linewidth=1.4, label="Observed")
    ax.plot(
        dates,
        window["q_best_cal"],
        color="#dc2626",
        linewidth=0.9,
        alpha=0.85,
        linestyle="--",
        label="Best-calibration simulation",
    )

    ax.set_ylabel("Discharge (mm/d)")
    ax.set_xlabel("Date")
    ax.set_title("Parametric uncertainty only")
    ax.text(
        0.01,
        0.02,
        "Excludes precipitation, observation and structural uncertainty.",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_ensemble_full_validation(
    timeseries: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Internal diagnostic: full validation-period ensemble envelope."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window = timeseries[timeseries["period"] == "validation"].copy()
    window["date_ts"] = pd.to_datetime(window["date"])
    dates = window["date_ts"]

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.fill_between(
        dates,
        window["q05"],
        window["q95"],
        alpha=0.25,
        color="#93c5fd",
        label="q05–q95 behavioral envelope",
    )
    ax.plot(dates, window["q50"], color="#1d4ed8", linewidth=1.2, label="q50")
    ax.plot(dates, window["q_obs"], color="#111827", linewidth=1.0, label="Observed")
    ax.plot(
        dates,
        window["q_best_cal"],
        color="#dc2626",
        linewidth=0.8,
        alpha=0.8,
        linestyle="--",
        label="Best-calibration",
    )
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_xlabel("Date")
    ax.set_title("Full validation period — parametric uncertainty only")
    ax.text(
        0.01,
        0.02,
        "Excludes precipitation, observation and structural uncertainty.",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_hydrograph(
    data: pd.DataFrame,
    simulated: pd.Series,
    config: dict,
    output_path: Path,
) -> None:
    """Generate observed vs simulated hydrograph with inverted precipitation."""
    raise NotImplementedError("Hydrograph plot not yet implemented")


def plot_metrics_table(
    metrics_cal: dict[str, float],
    metrics_val: dict[str, float],
    top_candidates: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate calibration | validation metrics figure."""
    raise NotImplementedError("Metrics plot not yet implemented")


def generate_report(
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Write rapport_calage.md with reproducibility block and stated limitations."""
    from src.rapport_calage import generate_rapport_calage

    return generate_rapport_calage(config_path, output_dir)
