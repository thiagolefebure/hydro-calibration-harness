"""Phase 8A: supervised residual-correction dataset construction.

Predicts residual error of the best-calibration GR4J simulation:

    residual(t) = Q_obs(t) - Q_phys(t)

Features at time t use only information available at or before t (no future
observed discharge, no future precipitation, no centered rolling windows).
Any fitted preprocessing statistics are estimated on the calibration period
only (validation never enters the fit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ensemble import period_labels
from src.evaluation import simulation_inputs
from src.gr4j import GR4JParameters, run_gr4j_continuous_periods
from src.validation import period_masks, select_best_calibration_candidate

DATASET_COLUMNS = [
    "date",
    "period",
    "q_obs",
    "q_phys",
    "residual",
    "q_obs_lag_1",
    "residual_lag_1",
    "precipitation_1d",
    "precipitation_3d",
    "precipitation_7d",
    "precipitation_30d",
    "et0_current",
    "et0_7d_mean",
    "production_store",
    "routing_store",
    "day_of_year_sin",
    "day_of_year_cos",
    "q_phys_change_1d",
    "q_obs_change_1d",
]

FEATURE_COLUMNS = [
    "q_obs_lag_1",
    "residual_lag_1",
    "precipitation_1d",
    "precipitation_3d",
    "precipitation_7d",
    "precipitation_30d",
    "et0_current",
    "et0_7d_mean",
    "production_store",
    "routing_store",
    "day_of_year_sin",
    "day_of_year_cos",
    "q_phys_change_1d",
    "q_obs_change_1d",
]

TARGET_COLUMN = "residual"


@dataclass(frozen=True)
class MLResidualDatasetResult:
    """Artifacts from residual ML dataset construction."""

    dataset: pd.DataFrame
    summary: dict[str, Any]
    dataset_path: Path
    summary_path: Path
    best_run_id: int


def _params_from_row(row: pd.Series) -> GR4JParameters:
    return GR4JParameters(
        X1=float(row["x1"]),
        X2=float(row["x2"]),
        X3=float(row["x3"]),
        X4=float(row["x4"]),
    )


def trailing_sum(series: pd.Series, window: int) -> pd.Series:
    """Causal trailing sum ending at t (no centered / forward window)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window=window, min_periods=window).sum()


def trailing_mean(series: pd.Series, window: int) -> pd.Series:
    """Causal trailing mean ending at t (no centered / forward window)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window=window, min_periods=window).mean()


def lag_past(series: pd.Series, periods: int = 1) -> pd.Series:
    """Shift series forward in time index so value at t is from t-periods."""
    if periods < 1:
        raise ValueError("lag periods must be >= 1")
    return series.shift(periods)


def day_of_year_cyclical(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Seasonality encoding using day-of-year sine/cosine (leap-year safe)."""
    doy = index.dayofyear.astype(float)
    # Use 365.25 so features stay continuous across year boundaries.
    angle = 2.0 * np.pi * (doy - 1.0) / 365.25
    return (
        pd.Series(np.sin(angle), index=index, name="day_of_year_sin"),
        pd.Series(np.cos(angle), index=index, name="day_of_year_cos"),
    )


def simulate_best_physical(
    data: pd.DataFrame,
    config: dict[str, Any],
    best_run: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Re-simulate best-calibration GR4J with optional daily store diagnostics."""
    params = _params_from_row(best_run)
    inputs = simulation_inputs(data)
    q_phys, _final, states = run_gr4j_continuous_periods(
        inputs,
        params,
        period_bounds=config["periods"],
        return_states=True,
    )
    return q_phys, states


def build_residual_feature_frame(
    data: pd.DataFrame,
    config: dict[str, Any],
    q_phys: pd.Series,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble aligned residual features with strict causal transforms."""
    index = q_phys.index
    observed = data.reindex(index)["discharge_mm"]
    precip = data.reindex(index)["precipitation_mm"]
    et0 = data.reindex(index)["et0_mm"]

    residual = observed - q_phys
    doy_sin, doy_cos = day_of_year_cyclical(pd.DatetimeIndex(index))

    # Concurrent q_obs change uses only past+present obs (not future).
    # For residual learning this is descriptive; lag-1 residual/obs remain primary.
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(index).strftime("%Y-%m-%d"),
            "period": period_labels(pd.DatetimeIndex(index), config).to_numpy(dtype="object"),
            "q_obs": observed.to_numpy(dtype=float),
            "q_phys": q_phys.to_numpy(dtype=float),
            "residual": residual.to_numpy(dtype=float),
            "q_obs_lag_1": lag_past(observed, 1).to_numpy(dtype=float),
            "residual_lag_1": lag_past(residual, 1).to_numpy(dtype=float),
            "precipitation_1d": precip.to_numpy(dtype=float),
            "precipitation_3d": trailing_sum(precip, 3).to_numpy(dtype=float),
            "precipitation_7d": trailing_sum(precip, 7).to_numpy(dtype=float),
            "precipitation_30d": trailing_sum(precip, 30).to_numpy(dtype=float),
            "et0_current": et0.to_numpy(dtype=float),
            "et0_7d_mean": trailing_mean(et0, 7).to_numpy(dtype=float),
            "production_store": states.reindex(index)["production_store"].to_numpy(dtype=float),
            "routing_store": states.reindex(index)["routing_store"].to_numpy(dtype=float),
            "day_of_year_sin": doy_sin.to_numpy(dtype=float),
            "day_of_year_cos": doy_cos.to_numpy(dtype=float),
            "q_phys_change_1d": q_phys.diff(1).to_numpy(dtype=float),
            "q_obs_change_1d": observed.diff(1).to_numpy(dtype=float),
        },
        index=index,
    )
    return frame


def fit_calibration_imputation(
    frame: pd.DataFrame,
    config: dict[str, Any],
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Fit simple fill values on calibration rows only (no validation leakage)."""
    cols = columns if columns is not None else FEATURE_COLUMNS
    masks = period_masks(frame.index, config)
    cal = frame.loc[masks["calibration"], cols]
    return {col: float(cal[col].median(skipna=True)) for col in cols}


def apply_imputation(frame: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    """Apply pre-fitted fill values; does not re-estimate from the applied subset."""
    out = frame.copy()
    for col, value in fill_values.items():
        if col in out.columns and np.isfinite(value):
            out[col] = out[col].fillna(value)
    return out


def drop_unusable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep days with finite target and all required feature columns."""
    required = [TARGET_COLUMN, "q_obs", "q_phys", *FEATURE_COLUMNS]
    mask = frame[required].notna().all(axis=1)
    return frame.loc[mask, DATASET_COLUMNS].reset_index(drop=True)


def summarize_residual_dataset(
    dataset: pd.DataFrame,
    *,
    best_run_id: int,
    imputation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a concise descriptive summary (not used for model fitting)."""
    residual = dataset[TARGET_COLUMN]
    by_period = dataset["period"].value_counts().to_dict()

    missingness = {
        col: float(dataset[col].isna().mean()) for col in DATASET_COLUMNS if col != "date"
    }

    correlations: dict[str, float | None] = {}
    for col in FEATURE_COLUMNS:
        if residual.nunique(dropna=True) < 2 or dataset[col].nunique(dropna=True) < 2:
            correlations[col] = None
        else:
            correlations[col] = float(dataset[col].corr(residual))

    return {
        "best_run_id": int(best_run_id),
        "target": "residual = q_obs - q_phys",
        "total_rows": int(len(dataset)),
        "rows_warmup": int(by_period.get("warmup", 0)),
        "rows_calibration": int(by_period.get("calibration", 0)),
        "rows_validation": int(by_period.get("validation", 0)),
        "feature_missingness": missingness,
        "residual_mean": float(residual.mean()) if len(residual) else None,
        "residual_std": float(residual.std(ddof=1)) if len(residual) > 1 else None,
        "feature_residual_correlation": correlations,
        "imputation_fit_period": "calibration",
        "imputation_values": imputation or {},
        "notes": [
            "Correlations are descriptive only and must not be used as leakage checks.",
            "Rolling precipitation/ET0 windows are trailing (causal) ending at t.",
            "Lag features use past values only (shift).",
            "Validation rows are never used to fit imputation statistics.",
        ],
    }


def build_ml_residual_dataset(
    data: pd.DataFrame,
    config: dict[str, Any],
    runs: pd.DataFrame,
    *,
    apply_calibration_imputation: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    """Construct the residual ML table and summary from best-calibration GR4J.

    By default, rows with incomplete causal features (lags / trailing windows)
    are dropped. Optional median imputation, when enabled, is fitted on the
    calibration period only and then applied globally.
    """
    best = select_best_calibration_candidate(runs)
    q_phys, states = simulate_best_physical(data, config, best)
    raw = build_residual_feature_frame(data, config, q_phys, states)

    fill_values = fit_calibration_imputation(raw, config)
    working = apply_imputation(raw, fill_values) if apply_calibration_imputation else raw
    dataset = drop_unusable_rows(working)

    summary = summarize_residual_dataset(
        dataset,
        best_run_id=int(best["run_id"]),
        imputation=fill_values if apply_calibration_imputation else {},
    )
    return dataset, summary, int(best["run_id"])

def save_ml_residual_dataset(
    dataset: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write CSV dataset and JSON summary under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "ml_residual_dataset.csv"
    summary_path = output_dir / "ml_residual_dataset_summary.json"
    dataset.to_csv(dataset_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return dataset_path, summary_path


def run_ml_residual_dataset_export(
    data: pd.DataFrame,
    config: dict[str, Any],
    runs: pd.DataFrame,
    output_dir: Path,
) -> MLResidualDatasetResult:
    """Build and persist the Phase 8A residual dataset."""
    dataset, summary, best_run_id = build_ml_residual_dataset(data, config, runs)
    dataset_path, summary_path = save_ml_residual_dataset(dataset, summary, output_dir)
    return MLResidualDatasetResult(
        dataset=dataset,
        summary=summary,
        dataset_path=dataset_path,
        summary_path=summary_path,
        best_run_id=best_run_id,
    )


def print_ml_residual_dataset_report(result: MLResidualDatasetResult) -> None:
    """Print a short console report for Phase 8A."""
    s = result.summary
    print("=== Phase 8A: ML residual dataset ===")
    print(f"Best calibration run_id: {result.best_run_id}")
    print(f"Dataset:  {result.dataset_path.resolve()}")
    print(f"Summary:  {result.summary_path.resolve()}")
    print(f"Total rows:        {s['total_rows']}")
    print(f"Calibration rows:  {s['rows_calibration']}")
    print(f"Validation rows:   {s['rows_validation']}")
    print(f"Warm-up rows:      {s['rows_warmup']}")
    if s["residual_mean"] is not None and s["residual_std"] is not None:
        print(f"Residual mean/std: {s['residual_mean']:.6f} / {s['residual_std']:.6f}")
