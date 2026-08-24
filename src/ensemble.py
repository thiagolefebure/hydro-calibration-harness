"""GLUE-inspired behavioral ensemble and parametric uncertainty envelope.

This module implements a prototype **GLUE-inspired behavioral ensemble** — not a
complete GLUE implementation. Ensemble membership is determined exclusively by
calibration-period KGE; validation metrics are never used for selection.

The behavioral threshold (default KGE_cal > 0.80) is a configurable prototype
criterion, not a universal hydrological acceptability threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import run_continuous_gr4j
from src.gr4j import GR4JParameters
from src.metrics import MetricResult, compute_metrics, get_log_nse_epsilon
from src.validation import period_masks, select_best_calibration_candidate

SENSITIVITY_THRESHOLDS = (0.70, 0.75, 0.80, 0.85)
VALIDATION_PLOT_WINDOW_DAYS = 60
WEAKLY_CONSTRAINED_FRACTION = 0.50

BEHAVIORAL_RUNS_COLUMNS = [
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
    "rank_kge_cal",
]

ENSEMBLE_TIMESERIES_COLUMNS = [
    "date",
    "q_obs",
    "q_best_cal",
    "q05",
    "q50",
    "q95",
    "period",
]


@dataclass(frozen=True)
class EnsembleAnalysisResult:
    """Outputs from Phase 5 behavioral ensemble analysis."""

    behavioral_runs: pd.DataFrame
    timeseries: pd.DataFrame
    threshold: float
    n_members: int
    validation_coverage: float
    width_mean: float
    width_median: float
    width_p90: float
    width_relative_to_obs_mean: float
    q50_validation_metrics: MetricResult
    best_cal_validation_metrics: MetricResult
    sensitivity: pd.DataFrame
    parameter_ranges: pd.DataFrame
    weakly_constrained: list[str]
    behavioral_runs_path: Path
    timeseries_path: Path
    validation_figure_path: Path
    full_validation_figure_path: Path
    runtime_total_s: float


def get_behavioral_threshold(config: dict[str, Any]) -> float:
    """Return the official behavioral KGE_cal threshold from config."""
    return float(config["calibration"]["behavioral_kge_threshold"])


def select_behavioral_members(runs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Select GLUE-inspired behavioral ensemble members (KGE_cal > threshold only)."""
    if "kge_cal" not in runs.columns:
        raise ValueError("runs DataFrame missing kge_cal column")
    return runs.loc[runs["kge_cal"] > threshold].copy()


def _params_from_row(row: pd.Series) -> GR4JParameters:
    return GR4JParameters(
        X1=float(row["x1"]),
        X2=float(row["x2"]),
        X3=float(row["x3"]),
        X4=float(row["x4"]),
    )


def run_member_simulations(
    data: pd.DataFrame,
    config: dict[str, Any],
    members: pd.DataFrame,
) -> pd.DataFrame:
    """Run continuous GR4J for each member; columns are run_id strings."""
    if members.empty:
        raise ValueError("Cannot simulate an empty behavioral ensemble")

    simulations: dict[str, pd.Series] = {}
    for row in members.itertuples(index=False):
        params = GR4JParameters(X1=row.x1, X2=row.x2, X3=row.x3, X4=row.x4)
        simulated = run_continuous_gr4j(data, config, params)
        simulations[str(int(row.run_id))] = simulated

    return pd.DataFrame(simulations)


def ensemble_quantiles(
    simulations: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Compute per-timestep quantiles across behavioral ensemble members."""
    values = simulations.to_numpy(dtype=float)
    result = pd.DataFrame(index=simulations.index)
    mapping = {0.05: "q05", 0.5: "q50", 0.95: "q95"}
    for q in quantiles:
        col = mapping.get(q, f"q{int(q * 100):02d}")
        result[col] = np.nanquantile(values, q, axis=1)
    return result


def period_labels(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.Series:
    """Assign warmup / calibration / validation label to each timestamp."""
    masks = period_masks(index, config)
    labels = pd.Series("unknown", index=index, dtype="object")
    labels.loc[masks["warmup"]] = "warmup"
    labels.loc[masks["calibration"]] = "calibration"
    labels.loc[masks["validation"]] = "validation"
    return labels


def build_ensemble_timeseries(
    observed: pd.Series,
    best_cal_simulated: pd.Series,
    quantiles: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Assemble the full-period ensemble timeseries table."""
    index = quantiles.index
    frame = pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "q_obs": observed.reindex(index).to_numpy(dtype=float),
            "q_best_cal": best_cal_simulated.reindex(index).to_numpy(dtype=float),
            "q05": quantiles["q05"].to_numpy(dtype=float),
            "q50": quantiles["q50"].to_numpy(dtype=float),
            "q95": quantiles["q95"].to_numpy(dtype=float),
            "period": period_labels(index, config).to_numpy(dtype="object"),
        }
    )
    return frame[ENSEMBLE_TIMESERIES_COLUMNS]


def empirical_validation_coverage(
    timeseries: pd.DataFrame,
) -> float:
    """Fraction of validation days with q05 <= Qobs <= q95.

    This is the empirical validation coverage of the behavioral envelope.
    It is NOT a calibrated 90% probability interval.
    """
    val = timeseries[timeseries["period"] == "validation"].copy()
    valid = val["q_obs"].notna() & val["q05"].notna() & val["q95"].notna()
    if valid.sum() == 0:
        return float("nan")
    inside = (val.loc[valid, "q_obs"] >= val.loc[valid, "q05"]) & (
        val.loc[valid, "q_obs"] <= val.loc[valid, "q95"]
    )
    return float(inside.sum() / valid.sum())


def envelope_width_series(timeseries: pd.DataFrame) -> pd.Series:
    """Daily envelope width q95 - q05."""
    return timeseries["q95"] - timeseries["q05"]


def envelope_width_diagnostics(
    timeseries: pd.DataFrame,
) -> dict[str, float]:
    """Validation-period envelope width statistics."""
    val = timeseries[timeseries["period"] == "validation"]
    width = envelope_width_series(val).dropna()
    obs_mean = val["q_obs"].dropna().mean()
    if width.empty:
        nan = float("nan")
        return {
            "mean": nan,
            "median": nan,
            "p90": nan,
            "relative_to_obs_mean": nan,
        }
    return {
        "mean": float(width.mean()),
        "median": float(width.median()),
        "p90": float(width.quantile(0.90)),
        "relative_to_obs_mean": float(width.mean() / obs_mean) if obs_mean else float("nan"),
    }


def validation_metrics_for_series(
    observed: pd.Series,
    simulated: pd.Series,
    config: dict[str, Any],
) -> MetricResult:
    """Compute validation-period metrics for a simulated series."""
    masks = period_masks(observed.index, config)
    return compute_metrics(
        observed,
        simulated,
        period_mask=masks["validation"],
        epsilon_mm=get_log_nse_epsilon(config),
    )


def metrics_to_dict(metrics: MetricResult) -> dict[str, float]:
    def _val(mv) -> float:
        return float(mv.value) if mv.is_defined else float("nan")

    return {
        "nse": _val(metrics.nse),
        "kge": _val(metrics.kge),
        "r": _val(metrics.kge_r),
        "alpha": _val(metrics.kge_alpha),
        "beta": _val(metrics.kge_beta),
        "lognse": _val(metrics.lognse),
        "bias": _val(metrics.bias),
    }


def threshold_sensitivity_table(
    runs: pd.DataFrame,
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    simulation_cache: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    """Member counts and diagnostic validation coverage by threshold."""
    sims_low = simulation_cache[min(SENSITIVITY_THRESHOLDS)]
    observed = data.loc[sims_low.index, "discharge_mm"]

    rows: list[dict[str, Any]] = []
    for threshold in SENSITIVITY_THRESHOLDS:
        members = select_behavioral_members(runs, threshold)
        member_ids = {str(int(rid)) for rid in members["run_id"]}
        subset_cols = [c for c in sims_low.columns if c in member_ids]
        quantiles = ensemble_quantiles(sims_low[subset_cols])
        ts = build_ensemble_timeseries(
            observed,
            pd.Series(np.nan, index=quantiles.index),
            quantiles,
            config,
        )
        rows.append(
            {
                "threshold": threshold,
                "n_members": int(len(members)),
                "validation_coverage": empirical_validation_coverage(ts),
            }
        )
    return pd.DataFrame.from_records(rows)


def parameter_range_diagnostics(
    behavioral_runs: pd.DataFrame,
    bounds: dict[str, list[float]],
) -> tuple[pd.DataFrame, list[str]]:
    """Summarize parameter ranges and flag weakly constrained parameters."""
    mapping = [("x1", "X1"), ("x2", "X2"), ("x3", "X3"), ("x4", "X4")]
    rows: list[dict[str, Any]] = []
    weakly: list[str] = []
    for col, name in mapping:
        series = behavioral_runs[col]
        lo_bound, hi_bound = bounds[name]
        span = hi_bound - lo_bound
        pmin = float(series.min())
        pmax = float(series.max())
        pmed = float(series.median())
        sampled_span = pmax - pmin
        rows.append(
            {
                "parameter": name,
                "min": pmin,
                "median": pmed,
                "max": pmax,
                "configured_lower": lo_bound,
                "configured_upper": hi_bound,
                "sampled_span": sampled_span,
                "bound_span": span,
            }
        )
        if span > 0 and sampled_span / span >= WEAKLY_CONSTRAINED_FRACTION:
            weakly.append(name)
    return pd.DataFrame.from_records(rows), weakly


def validation_plot_window(
    observed: pd.Series,
    config: dict[str, Any],
    window_days: int = VALIDATION_PLOT_WINDOW_DAYS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Deterministic ~60-day window centered on maximum validation discharge."""
    masks = period_masks(observed.index, config)
    val_obs = observed.loc[masks["validation"]].dropna()
    if val_obs.empty:
        val_start = pd.Timestamp(config["periods"]["validation"][0])
        return val_start, val_start + pd.Timedelta(days=window_days - 1)

    peak_date = val_obs.idxmax()
    half = window_days // 2
    start = peak_date - pd.Timedelta(days=half)
    end = start + pd.Timedelta(days=window_days - 1)

    val_start = pd.Timestamp(config["periods"]["validation"][0])
    val_end = pd.Timestamp(config["periods"]["validation"][1])
    if start < val_start:
        start = val_start
        end = start + pd.Timedelta(days=window_days - 1)
    if end > val_end:
        end = val_end
        start = end - pd.Timedelta(days=window_days - 1)
    if start < val_start:
        start = val_start
    return start, end


def save_behavioral_runs(behavioral: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    behavioral[BEHAVIORAL_RUNS_COLUMNS].to_csv(path, index=False)
    return path


def save_ensemble_timeseries(timeseries: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(path, index=False)
    return path


def run_ensemble_analysis(
    config: dict[str, Any],
    data: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
) -> EnsembleAnalysisResult:
    """Execute Phase 5 GLUE-inspired behavioral ensemble analysis."""
    import time

    from src.report import plot_ensemble_full_validation, plot_ensemble_validation_window

    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = get_behavioral_threshold(config)
    behavioral = select_behavioral_members(runs, threshold)
    if behavioral.empty:
        raise ValueError(
            f"No behavioral members for KGE_cal > {threshold}; cannot build ensemble"
        )

    start = time.perf_counter()
    simulations = run_member_simulations(data, config, behavioral)
    quantiles = ensemble_quantiles(simulations)

    best = select_best_calibration_candidate(runs)
    best_cal_sim = run_continuous_gr4j(data, config, _params_from_row(best))
    observed = data.loc[quantiles.index, "discharge_mm"]
    timeseries = build_ensemble_timeseries(observed, best_cal_sim, quantiles, config)

    behavioral_path = save_behavioral_runs(
        behavioral, output_dir / "behavioral_runs.csv"
    )
    timeseries_path = save_ensemble_timeseries(
        timeseries, output_dir / "ensemble_timeseries.csv"
    )

    coverage = empirical_validation_coverage(timeseries)
    width = envelope_width_diagnostics(timeseries)
    q50_metrics = validation_metrics_for_series(
        observed, pd.Series(timeseries["q50"].values, index=observed.index), config
    )
    best_cal_metrics = validation_metrics_for_series(observed, best_cal_sim, config)

    members_070 = select_behavioral_members(runs, 0.70)
    missing = members_070[~members_070["run_id"].isin(behavioral["run_id"])]
    if not missing.empty:
        extra_sims = run_member_simulations(data, config, missing)
        all_sims = pd.concat([simulations, extra_sims], axis=1)
    else:
        all_sims = simulations
    sim_cache = {min(SENSITIVITY_THRESHOLDS): all_sims}

    sensitivity = threshold_sensitivity_table(
        runs, data, config, simulation_cache=sim_cache
    )

    param_ranges, weakly = parameter_range_diagnostics(
        behavioral, config["model"]["parameter_bounds"]
    )

    window_start, window_end = validation_plot_window(observed, config)
    val_fig = plot_ensemble_validation_window(
        timeseries,
        window_start=window_start,
        window_end=window_end,
        output_path=output_dir / "ensemble_validation.png",
    )
    full_fig = plot_ensemble_full_validation(
        timeseries,
        output_path=output_dir / "ensemble_full_validation.png",
    )

    sensitivity.to_csv(output_dir / "ensemble_threshold_sensitivity.csv", index=False)

    runtime = time.perf_counter() - start

    return EnsembleAnalysisResult(
        behavioral_runs=behavioral,
        timeseries=timeseries,
        threshold=threshold,
        n_members=len(behavioral),
        validation_coverage=coverage,
        width_mean=width["mean"],
        width_median=width["median"],
        width_p90=width["p90"],
        width_relative_to_obs_mean=width["relative_to_obs_mean"],
        q50_validation_metrics=q50_metrics,
        best_cal_validation_metrics=best_cal_metrics,
        sensitivity=sensitivity,
        parameter_ranges=param_ranges,
        weakly_constrained=weakly,
        behavioral_runs_path=behavioral_path,
        timeseries_path=timeseries_path,
        validation_figure_path=val_fig,
        full_validation_figure_path=full_fig,
        runtime_total_s=runtime,
    )


def print_ensemble_report(result: EnsembleAnalysisResult) -> None:
    """Print Phase 5 summary to stdout."""
    print("=== Phase 5: GLUE-inspired behavioral ensemble ===")
    print(
        "Method: GLUE-inspired behavioral ensemble (not a complete GLUE implementation)."
    )
    print(
        f"Official threshold: KGE_cal > {result.threshold:g} "
        "(configurable prototype criterion; not a universal acceptability threshold)"
    )
    print(f"Ensemble members:      {result.n_members}")
    print(f"Runtime:               {result.runtime_total_s:.2f} s")
    print()
    print(
        "Empirical validation coverage of the behavioral envelope (q05–q95): "
        f"{result.validation_coverage:.1%}"
    )
    print("(Not a calibrated 90% probability interval.)")
    print()
    print("Validation envelope width (q95 - q05):")
    print(f"  mean:                {result.width_mean:.4f} mm/d")
    print(f"  median:              {result.width_median:.4f} mm/d")
    print(f"  p90:                 {result.width_p90:.4f} mm/d")
    print(f"  mean / obs mean Q:   {result.width_relative_to_obs_mean:.4f}")
    print()
    q50 = metrics_to_dict(result.q50_validation_metrics)
    best = metrics_to_dict(result.best_cal_validation_metrics)
    print("Validation metrics — q50 vs best-calibration (diagnostic):")
    print(f"{'metric':<12} {'q50':>10} {'best_cal':>10}")
    for key in ("nse", "kge", "r", "alpha", "beta", "lognse", "bias"):
        print(f"{key:<12} {q50[key]:10.4f} {best[key]:10.4f}")
    print()
    print("Threshold sensitivity (validation coverage is diagnostic only):")
    print(f"{'threshold':>10} {'n_members':>10} {'val_coverage':>14}")
    for _, row in result.sensitivity.iterrows():
        print(
            f"{row['threshold']:10.2f} {int(row['n_members']):10d} "
            f"{row['validation_coverage']:13.1%}"
        )
    print()
    print("Behavioral parameter ranges (KGE_cal > official threshold):")
    print(f"{'param':<6} {'min':>10} {'median':>10} {'max':>10}")
    for _, row in result.parameter_ranges.iterrows():
        print(
            f"{row['parameter']:<6} {row['min']:10.3f} "
            f"{row['median']:10.3f} {row['max']:10.3f}"
        )
    if result.weakly_constrained:
        print(
            "Weakly constrained parameters "
            f"(sampled span >= {WEAKLY_CONSTRAINED_FRACTION:.0%} of configured range): "
            + ", ".join(result.weakly_constrained)
        )
    else:
        print("Weakly constrained parameters: none")
    print()
    print(f"behavioral_runs.csv:   {result.behavioral_runs_path.resolve()}")
    print(f"ensemble_timeseries:   {result.timeseries_path.resolve()}")
    print(f"ensemble_validation:   {result.validation_figure_path.resolve()}")
    print(f"ensemble_full_val:     {result.full_validation_figure_path.resolve()}")
