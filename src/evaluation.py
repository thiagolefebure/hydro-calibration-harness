"""Temporal hydrological performance evaluation (spec §2, §4).

Runs GR4J continuously across warm-up, calibration and validation periods
without resetting model states, then computes metrics on calibration and
validation timestamps only (warm-up excluded from all reported metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import get_meteo_timezone, load_basin_data
from src.gr4j import GR4JParameters, run_gr4j_continuous_periods
from src.metrics import MetricResult, compute_metrics, get_log_nse_epsilon
from src.validation import period_masks

# Uncalibrated demonstration parameters (fixed; not tuned to observations).
DEMO_PARAMETERS = GR4JParameters(X1=350.0, X2=0.0, X3=90.0, X4=1.4)


@dataclass(frozen=True)
class PeriodEvaluation:
    """Metrics for calibration and validation periods."""

    calibration: MetricResult
    validation: MetricResult
    simulated: pd.Series
    observed: pd.Series


def simulation_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Return GR4J forcing columns for continuous simulation."""
    return data[["precipitation_mm", "et0_mm"]].dropna()


def run_continuous_gr4j(
    data: pd.DataFrame,
    config: dict[str, Any],
    params: GR4JParameters,
) -> pd.Series:
    """Run GR4J continuously from warm-up start through validation end."""
    inputs = simulation_inputs(data)
    simulated, _final_state = run_gr4j_continuous_periods(
        inputs,
        params,
        period_bounds=config["periods"],
    )
    return simulated


def evaluate_temporal_performance(
    observed: pd.Series,
    simulated: pd.Series,
    config: dict[str, Any],
) -> PeriodEvaluation:
    """Compute calibration and validation metrics; warm-up excluded."""
    masks = period_masks(observed.index, config)
    epsilon = get_log_nse_epsilon(config)

    cal_metrics = compute_metrics(
        observed,
        simulated,
        period_mask=masks["calibration"],
        epsilon_mm=epsilon,
    )
    val_metrics = compute_metrics(
        observed,
        simulated,
        period_mask=masks["validation"],
        epsilon_mm=epsilon,
    )
    return PeriodEvaluation(
        calibration=cal_metrics,
        validation=val_metrics,
        simulated=simulated,
        observed=observed,
    )


def metrics_to_dataframe(evaluation: PeriodEvaluation) -> pd.DataFrame:
    """Build metrics summary table for CSV export."""
    rows = [
        ("NSE", evaluation.calibration.nse, evaluation.validation.nse),
        ("KGE", evaluation.calibration.kge, evaluation.validation.kge),
        ("r", evaluation.calibration.kge_r, evaluation.validation.kge_r),
        ("alpha", evaluation.calibration.kge_alpha, evaluation.validation.kge_alpha),
        ("beta", evaluation.calibration.kge_beta, evaluation.validation.kge_beta),
        ("log-NSE", evaluation.calibration.lognse, evaluation.validation.lognse),
        ("Volume bias", evaluation.calibration.bias, evaluation.validation.bias),
    ]
    records = []
    for name, cal_mv, val_mv in rows:
        records.append(
            {
                "metric": name,
                "calibration": cal_mv.value if cal_mv.is_defined else float("nan"),
                "validation": val_mv.value if val_mv.is_defined else float("nan"),
                "n_calibration": cal_mv.n_valid,
                "n_validation": val_mv.n_valid,
            }
        )
    return pd.DataFrame.from_records(records)


def save_metrics_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def print_metrics_table(evaluation: PeriodEvaluation) -> None:
    """Print concise console summary."""
    df = metrics_to_dataframe(evaluation)
    print("=== Uncalibrated metrics (warm-up excluded) ===")
    print(f"{'metric':<14} {'calibration':>12} {'validation':>12} {'n_cal':>8} {'n_val':>8}")
    for _, row in df.iterrows():
        cal = evaluation.calibration
        val = evaluation.validation
        metric_key = {
            "NSE": (cal.nse, val.nse),
            "KGE": (cal.kge, val.kge),
            "r": (cal.kge_r, val.kge_r),
            "alpha": (cal.kge_alpha, val.kge_alpha),
            "beta": (cal.kge_beta, val.kge_beta),
            "log-NSE": (cal.lognse, val.lognse),
            "Volume bias": (cal.bias, val.bias),
        }[row["metric"]]
        cal_str = metric_key[0].formatted()
        val_str = metric_key[1].formatted()
        print(
            f"{row['metric']:<14} {cal_str:>12} {val_str:>12} "
            f"{int(row['n_calibration']):>8} {int(row['n_validation']):>8}"
        )


def evaluate_uncalibrated_run(
    config: dict[str, Any],
    cache_dir: Path,
    *,
    timezone: str | None = None,
) -> PeriodEvaluation:
    """Load data, run fixed demo GR4J parameters, evaluate temporal metrics."""
    tz = timezone if timezone is not None else get_meteo_timezone(config)
    data, _qa = load_basin_data(config, cache_dir=cache_dir, timezone=tz)
    simulated = run_continuous_gr4j(data, config, DEMO_PARAMETERS)
    observed = data.loc[simulated.index, "discharge_mm"]
    return evaluate_temporal_performance(observed, simulated, config)
