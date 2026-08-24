"""Descriptive hydrological characterization of experiment periods (Phase 3B).

Uses the processed daily dataset only. No model parameters are inferred or
adjusted from these summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

WET_DAY_THRESHOLD_MM = 1.0


def load_processed_daily(path: Path) -> pd.DataFrame:
    """Load aligned daily processed data written by the data pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed daily dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _finite_series(series: pd.Series) -> pd.Series:
    return series.dropna()


def summarize_period(
    df: pd.DataFrame,
    *,
    label: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Compute hydrological summary statistics for a date-bounded subset."""
    subset = df
    if start is not None and end is not None:
        subset = df.loc[start:end]

    precip = _finite_series(subset["precipitation_mm"])
    discharge = _finite_series(subset["discharge_mm"])

    annual_precip = float(precip.sum()) if len(precip) else float("nan")
    annual_q_depth = float(discharge.sum()) if len(discharge) else float("nan")
    runoff_ratio = (
        annual_q_depth / annual_precip if len(precip) and annual_precip > 0 else float("nan")
    )

    if len(discharge):
        max_q = float(discharge.max())
        max_q_date = discharge.idxmax().strftime("%Y-%m-%d")
        mean_q = float(discharge.mean())
    else:
        max_q = float("nan")
        max_q_date = None
        mean_q = float("nan")

    wet_days = int((precip > WET_DAY_THRESHOLD_MM).sum()) if len(precip) else 0
    max_p = float(precip.max()) if len(precip) else float("nan")

    return {
        "period": label,
        "annual_precipitation_mm": annual_precip,
        "annual_observed_discharge_depth_mm": annual_q_depth,
        "runoff_ratio_qp": runoff_ratio,
        "mean_observed_discharge_mm_day": mean_q,
        "max_daily_observed_discharge_mm_day": max_q,
        "date_of_max_discharge": max_q_date,
        "wet_days_p_gt_1mm": wet_days,
        "max_daily_precipitation_mm_day": max_p,
        "n_days_precip": len(precip),
        "n_days_discharge": len(discharge),
    }


def annual_summaries(df: pd.DataFrame, years: range | list[int]) -> pd.DataFrame:
    """Build one summary row per calendar year."""
    rows = [
        summarize_period(
            df,
            label=str(year),
            start=f"{year}-01-01",
            end=f"{year}-12-31",
        )
        for year in years
    ]
    return pd.DataFrame(rows)


def hydrological_summary_table(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Annual rows (2010-2015) plus calibration and validation aggregates."""
    years = range(2010, 2016)
    annual = annual_summaries(df, years)

    periods = config["periods"]
    cal = summarize_period(
        df,
        label="calibration",
        start=periods["calibration"][0],
        end=periods["calibration"][1],
    )
    val = summarize_period(
        df,
        label="validation",
        start=periods["validation"][0],
        end=periods["validation"][1],
    )

    return pd.concat([annual, pd.DataFrame([cal, val])], ignore_index=True)


def save_hydrological_summary(df: pd.DataFrame, output_path: Path) -> Path:
    export = df.drop(columns=["n_days_precip", "n_days_discharge"], errors="ignore")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)
    return output_path


def plot_hydrological_years(df: pd.DataFrame, output_path: Path) -> Path:
    """Bar chart of annual precipitation and observed discharge depth."""
    annual = df[df["period"].str.fullmatch(r"\d{4}")].copy()
    annual["period"] = annual["period"].astype(int)
    annual = annual.sort_values("period")

    x = annual["period"].to_numpy()
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, annual["annual_precipitation_mm"], width, label="Precipitation (mm)")
    ax.bar(
        x + width / 2,
        annual["annual_observed_discharge_depth_mm"],
        width,
        label="Observed discharge depth (mm)",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Depth (mm)")
    ax.set_title("Annual precipitation and observed discharge depth")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def print_hydrological_summary(summary: pd.DataFrame) -> None:
    """Print concise console tables for annual and aggregate rows."""
    annual = summary[summary["period"].str.fullmatch(r"\d{4}")].copy()
    aggregates = summary[~summary["period"].str.fullmatch(r"\d{4}")]

    print("=== Annual hydrological summary (2010-2015) ===")
    print(
        f"{'year':<6} {'P_mm':>8} {'Q_mm':>8} {'Q/P':>7} "
        f"{'meanQ':>7} {'maxQ':>7} {'wet':>5} {'maxP':>7}"
    )
    for _, row in annual.iterrows():
        qp = row["runoff_ratio_qp"]
        qp_str = f"{qp:7.3f}" if pd.notna(qp) else "    nan"
        print(
            f"{row['period']:<6} "
            f"{row['annual_precipitation_mm']:8.1f} "
            f"{row['annual_observed_discharge_depth_mm']:8.1f} "
            f"{qp_str} "
            f"{row['mean_observed_discharge_mm_day']:7.3f} "
            f"{row['max_daily_observed_discharge_mm_day']:7.3f} "
            f"{int(row['wet_days_p_gt_1mm']):5d} "
            f"{row['max_daily_precipitation_mm_day']:7.1f}"
        )

    print()
    print("=== Period aggregates ===")
    print(
        f"{'period':<14} {'P_mm':>8} {'Q_mm':>8} {'Q/P':>7} "
        f"{'meanQ':>7} {'maxQ':>7} {'maxQ_date':>12} {'wet':>5} {'maxP':>7}"
    )
    for _, row in aggregates.iterrows():
        qp = row["runoff_ratio_qp"]
        qp_str = f"{qp:7.3f}" if pd.notna(qp) else "    nan"
        max_date = row["date_of_max_discharge"] or "-"
        print(
            f"{row['period']:<14} "
            f"{row['annual_precipitation_mm']:8.1f} "
            f"{row['annual_observed_discharge_depth_mm']:8.1f} "
            f"{qp_str} "
            f"{row['mean_observed_discharge_mm_day']:7.3f} "
            f"{row['max_daily_observed_discharge_mm_day']:7.3f} "
            f"{max_date:>12} "
            f"{int(row['wet_days_p_gt_1mm']):5d} "
            f"{row['max_daily_precipitation_mm_day']:7.1f}"
        )


def run_hydrological_characterization(
    processed_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Load processed data, write summary CSV and annual figure."""
    df = load_processed_daily(processed_path)
    summary = hydrological_summary_table(df, config)
    save_hydrological_summary(summary, output_dir / "hydrological_summary.csv")
    plot_hydrological_years(summary, output_dir / "hydrological_years.png")
    return summary
