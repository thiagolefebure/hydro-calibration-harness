"""Phase 9: forecast uncertainty calibration (leakage-safe, time-aware).

Central point forecast (fixed architecture):
    Q_point(t+h|t) = Q_phys(t+h|t) + residual_t
    (physical GR4J + residual persistence)

Interval methods (modular; not used to retune point skill):
    A. empirical_residual — calibration residual quantiles
    B. conditional_quantile — HGB quantile regression on forecast-origin features
    C. split_conformal — absolute-error conformal around Q_point
    R. behavioral_parametric — unchanged Phase 5 q05–q95 reference

Calibration / conformal fitting uses CALIBRATION-period forecast errors only.
VALIDATION is evaluation-only.

CRPS deferred — interval score used as primary probabilistic score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.ensemble import (
    VALIDATION_PLOT_WINDOW_DAYS,
    empirical_validation_coverage,
    validation_plot_window,
)
from src.evaluation import simulation_inputs
from src.gr4j import GR4JParameters
from src.meteo_sensitivity import (
    SCENARIO_MODERATE,
    SCENARIO_ORACLE,
    SCENARIO_LABELS,
    build_origin_state_index,
    forecast_discharge_from_origin,
    generate_scenario_precip,
    get_meteo_config,
)
from src.ml_horizon_forecast import (
    FORECAST_FEATURE_COLUMNS,
    HORIZONS,
    build_forecast_origin_dataset,
    filter_horizon_rows,
)
from src.ml_residual_ablation import (
    HIGH_FLOW_QUANTILE,
    calibration_high_flow_threshold,
    identify_peak_events,
)
from src.ml_residual_baselines import hybrid_discharge, load_ml_residual_dataset
from src.ml_residual_dataset import TARGET_COLUMN
from src.validation import select_best_calibration_candidate

METHOD_EMPIRICAL = "empirical_residual"
METHOD_QUANTILE = "conditional_quantile"
METHOD_CONFORMAL = "split_conformal"
METHOD_BEHAVIORAL = "behavioral_parametric"

METHODS_PRIMARY = (METHOD_EMPIRICAL, METHOD_QUANTILE, METHOD_CONFORMAL)

NOMINAL_LEVELS_PRIMARY = (0.80, 0.90, 0.95)
NOMINAL_LEVELS_RELIABILITY = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

LOW_FLOW_QUANTILE = 0.10
DEFAULT_TRAIN_FRACTION = 0.75
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_TOP_N_EXTREMES = 10
HGB_QUANTILE_CONFIG = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 200,
    "min_samples_leaf": 20,
    "random_state": 42,
}

# Finite-sample split conformal:
#   scores s_i = |y_i - yhat_i| on calibration subset of size n
#   level = ceil((n + 1) * (1 - alpha)) / n
#   qhat = empirical quantile of scores at min(level, 1)
#   interval = [yhat - qhat, yhat + qhat]
CONFORMAL_FORMULA = (
    "split conformal (absolute residual): "
    "s_i = |Q_obs - Q_point| on chronological CALIBRATION_SUBSET (size n); "
    "level = ceil((n+1)*(1-alpha))/n; "
    "q_hat = quantile(s, min(level, 1.0)); "
    "interval = [Q_point - q_hat, Q_point + q_hat]. "
    "Point forecast uses residual persistence (no trainable point model)."
)

CRPS_NOTE = "CRPS deferred — interval score used as primary probabilistic score."

COVERAGE_SUMMARY_COLUMNS = [
    "method",
    "horizon_days",
    "meteo_scenario",
    "nominal_coverage",
    "empirical_coverage",
    "coverage_error",
    "mean_width",
    "median_width",
    "p90_width",
    "mean_width_norm",
    "mean_interval_score",
]

FORECAST_COLUMNS = [
    "date",
    "horizon_days",
    "meteo_scenario",
    "method",
    "q_obs",
    "q_point",
    "lower_80",
    "upper_80",
    "lower_90",
    "upper_90",
    "lower_95",
    "upper_95",
    "period",
]


@dataclass
class UncertaintyResult:
    coverage_summary: pd.DataFrame
    regime_coverage: pd.DataFrame
    extreme_events: pd.DataFrame
    forecasts: pd.DataFrame
    answers: dict[str, str]
    coverage_path: Path
    regime_path: Path
    extreme_path: Path
    forecasts_path: Path
    reliability_path: Path
    rolling_path: Path
    demo_path: Path
    before_after_path: Path
    answers_path: Path = field(default_factory=lambda: Path())
    preferred_method: str = METHOD_CONFORMAL


def get_uncertainty_config(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "train_fraction": DEFAULT_TRAIN_FRACTION,
        "rolling_coverage_window_days": DEFAULT_ROLLING_WINDOW,
        "top_n_extreme_events": DEFAULT_TOP_N_EXTREMES,
        "moderate_realization": 0,
        "meteo_scenarios": [SCENARIO_ORACLE, SCENARIO_MODERATE],
    }
    if not config or "uncertainty" not in config or not config["uncertainty"]:
        return defaults
    user = config["uncertainty"]
    out = dict(defaults)
    out.update({k: user[k] for k in defaults if k in user})
    return out


def _params_from_row(row: pd.Series) -> GR4JParameters:
    return GR4JParameters(
        X1=float(row["x1"]),
        X2=float(row["x2"]),
        X3=float(row["x3"]),
        X4=float(row["x4"]),
    )


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    date_col: str = "origin_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological TRAIN_SUBSET / CALIBRATION_SUBSET split (no random split)."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    ordered = frame.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError(f"need at least 10 rows for chronological split, got {n}")
    n_train = max(1, int(np.floor(n * train_fraction)))
    n_train = min(n_train, n - 1)
    train = ordered.iloc[:n_train].copy()
    calib = ordered.iloc[n_train:].copy()
    if train[date_col].max() >= calib[date_col].min():
        # Equal dates at boundary are allowed only if order preserved; enforce strict time order.
        assert train[date_col].iloc[-1] <= calib[date_col].iloc[0]
    return train, calib


def empirical_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, q, method="linear"))


def conformal_quantile_level(n: int, alpha: float) -> float:
    """Finite-sample split-conformal quantile level: ceil((n+1)(1-alpha))/n."""
    if n < 1:
        raise ValueError("conformal calibration subset must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n))


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Central prediction interval score for nominal coverage 1-alpha."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    return width + (2.0 / alpha) * below + (2.0 / alpha) * above


def clip_nonnegative_interval(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discharge intervals: clip lower at 0 and enforce lower <= upper."""
    lo = np.maximum(np.asarray(lower, dtype=float), 0.0)
    up = np.maximum(np.asarray(upper, dtype=float), lo)
    return lo, up


def build_point_forecast_table(
    forecast_df: pd.DataFrame,
    *,
    horizon: int,
    q_phys_h: np.ndarray,
    origin_mask_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble Q_point = Q_phys(t+h|t) + residual_t for one horizon."""
    frame = origin_mask_frame.copy()
    residual_t = frame["residual_t"].to_numpy(dtype=float)
    q_point = hybrid_discharge(q_phys_h, residual_t)
    out = frame.copy()
    out["q_phys_forecast"] = np.asarray(q_phys_h, dtype=float)
    out["q_point"] = q_point
    out["q_obs_target"] = frame[f"q_obs_h{horizon}"].to_numpy(dtype=float)
    out["error"] = out["q_obs_target"] - out["q_point"]
    out["abs_error"] = np.abs(out["error"])
    out["horizon_days"] = int(horizon)
    out["target_date"] = pd.to_datetime(frame[f"target_date_h{horizon}"])
    return out


def oracle_q_phys_for_horizon(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    return frame[f"q_phys_h{horizon}"].to_numpy(dtype=float)


def moderate_q_phys_for_horizon(
    frame: pd.DataFrame,
    *,
    horizon: int,
    precip_forcing: np.ndarray,
    et0: np.ndarray,
    date_to_pos: dict[pd.Timestamp, int],
    state_by_date: dict,
    params: GR4JParameters,
) -> np.ndarray:
    origins = pd.to_datetime(frame["origin_date"])
    q = np.empty(len(frame), dtype=float)
    for i, origin in enumerate(origins):
        origin_ts = pd.Timestamp(origin)
        pos = date_to_pos[origin_ts]
        future_idx = np.arange(pos + 1, pos + 1 + horizon)
        path = forecast_discharge_from_origin(
            precip_forcing[future_idx],
            et0[future_idx],
            params,
            state_by_date[origin_ts],
        )
        q[i] = path[horizon - 1]
    return q


@dataclass
class EmpiricalResidualInterval:
    horizon: int
    quantiles: dict[float, tuple[float, float]]  # nominal -> (lo_q, hi_q)
    fitted_on: str = "calibration"

    def predict(self, q_point: np.ndarray, nominal: float) -> tuple[np.ndarray, np.ndarray]:
        lo_q, hi_q = self.quantiles[round(float(nominal), 10)]
        lower = q_point + lo_q
        upper = q_point + hi_q
        return clip_nonnegative_interval(lower, upper)


@dataclass
class ConditionalQuantileInterval:
    horizon: int
    models: dict[float, Any]  # quantile level -> fitted HGB
    feature_names: list[str]
    fitted_on: str = "calibration_train_or_full"

    @staticmethod
    def _qkey(q: float) -> float:
        return round(float(q), 10)

    def _predict_q(self, frame: pd.DataFrame, q: float) -> np.ndarray:
        x = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        return np.asarray(self.models[self._qkey(q)].predict(x), dtype=float)

    def predict(self, frame: pd.DataFrame, q_point: np.ndarray, nominal: float) -> tuple[np.ndarray, np.ndarray]:
        alpha = 1.0 - float(nominal)
        lo_err = self._predict_q(frame, alpha / 2.0)
        hi_err = self._predict_q(frame, 1.0 - alpha / 2.0)
        # Ensure ordering of error quantiles
        lo_err, hi_err = np.minimum(lo_err, hi_err), np.maximum(lo_err, hi_err)
        return clip_nonnegative_interval(q_point + lo_err, q_point + hi_err)


@dataclass
class SplitConformalInterval:
    horizon: int
    q_hat: dict[float, float]  # nominal -> conformal radius
    n_calib: int
    fitted_on: str = "calibration_subset"

    def predict(self, q_point: np.ndarray, nominal: float) -> tuple[np.ndarray, np.ndarray]:
        radius = self.q_hat[round(float(nominal), 10)]
        return clip_nonnegative_interval(q_point - radius, q_point + radius)


def fit_empirical_residual(
    cal_errors: np.ndarray,
    *,
    horizon: int,
    nominal_levels: tuple[float, ...] = NOMINAL_LEVELS_RELIABILITY,
) -> EmpiricalResidualInterval:
    quantiles: dict[float, tuple[float, float]] = {}
    for nom in nominal_levels:
        alpha = 1.0 - float(nom)
        quantiles[round(float(nom), 10)] = (
            empirical_quantile(cal_errors, alpha / 2.0),
            empirical_quantile(cal_errors, 1.0 - alpha / 2.0),
        )
    return EmpiricalResidualInterval(horizon=horizon, quantiles=quantiles)


def fit_conditional_quantile(
    train_frame: pd.DataFrame,
    *,
    horizon: int,
    feature_names: list[str] | None = None,
    nominal_levels: tuple[float, ...] = NOMINAL_LEVELS_RELIABILITY,
) -> ConditionalQuantileInterval:
    feats = list(feature_names or FORECAST_FEATURE_COLUMNS)
    forbidden = {"q_obs_target", "error", "abs_error", "q_point", "q_obs"}
    if set(feats) & forbidden:
        raise ValueError("forbidden future/obs columns in quantile features")
    x = train_frame.loc[:, feats].to_numpy(dtype=float)
    y = train_frame["error"].to_numpy(dtype=float)
    # Unique quantile levels needed across nominal coverages
    q_levels = sorted(
        {
            round(a, 10)
            for nom in nominal_levels
            for a in ((1.0 - nom) / 2.0, 1.0 - (1.0 - nom) / 2.0)
        }
    )
    # Always include median for diagnostics
    if 0.5 not in q_levels:
        q_levels.append(0.5)
        q_levels = sorted(q_levels)
    models: dict[float, Any] = {}
    for q in q_levels:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(q),
            **HGB_QUANTILE_CONFIG,
        )
        model.fit(x, y)
        models[round(float(q), 10)] = model
    return ConditionalQuantileInterval(
        horizon=horizon,
        models=models,
        feature_names=feats,
        fitted_on="calibration",
    )


def fit_split_conformal(
    calib_abs_errors: np.ndarray,
    *,
    horizon: int,
    nominal_levels: tuple[float, ...] = NOMINAL_LEVELS_RELIABILITY,
) -> SplitConformalInterval:
    scores = np.asarray(calib_abs_errors, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    q_hat: dict[float, float] = {}
    for nom in nominal_levels:
        alpha = 1.0 - float(nom)
        level = conformal_quantile_level(n, alpha)
        q_hat[round(float(nom), 10)] = empirical_quantile(scores, level)
    return SplitConformalInterval(horizon=horizon, q_hat=q_hat, n_calib=n)


def coverage_and_width_stats(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float,
    mean_obs: float,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return {
            "empirical_coverage": float("nan"),
            "coverage_error": float("nan"),
            "mean_width": float("nan"),
            "median_width": float("nan"),
            "p90_width": float("nan"),
            "mean_width_norm": float("nan"),
            "mean_interval_score": float("nan"),
        }
    yv, lo, up = y[valid], lower[valid], upper[valid]
    inside = (yv >= lo) & (yv <= up)
    width = up - lo
    alpha = 1.0 - float(nominal)
    scores = interval_score(yv, lo, up, alpha=alpha)
    emp = float(np.mean(inside))
    return {
        "empirical_coverage": emp,
        "coverage_error": emp - float(nominal),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "p90_width": float(np.quantile(width, 0.90)),
        "mean_width_norm": float(np.mean(width) / mean_obs) if mean_obs > 0 else float("nan"),
        "mean_interval_score": float(np.mean(scores)),
    }


def build_forecast_bundle_for_scenario(
    *,
    scenario: str,
    forecast_df: pd.DataFrame,
    basin_data: pd.DataFrame,
    runs: pd.DataFrame,
    config: dict[str, Any],
    unc_cfg: dict[str, Any],
) -> dict[int, dict[str, pd.DataFrame]]:
    """Return {horizon: {'calibration': df, 'validation': df}} with q_point/error."""
    best = select_best_calibration_candidate(runs)
    params = _params_from_row(best)
    meteo_cfg = get_meteo_config(config)

    if scenario == SCENARIO_ORACLE:
        precip_forcing = None
        state_by_date = None
        date_to_pos = None
        et0 = None
    else:
        _q_phys, state_by_date, date_index = build_origin_state_index(basin_data, params, config)
        inputs = simulation_inputs(basin_data)
        precip_obs = inputs["precipitation_mm"].to_numpy(dtype=float)
        et0 = inputs["et0_mm"].to_numpy(dtype=float)
        date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(date_index)}
        precip_forcing, _ = generate_scenario_precip(
            precip_obs,
            scenario=scenario,
            realization=int(unc_cfg["moderate_realization"]),
            meteo_cfg=meteo_cfg,
        )

    out: dict[int, dict[str, pd.DataFrame]] = {}
    for h in HORIZONS:
        cal_h = filter_horizon_rows(forecast_df, origin_period="calibration", horizon=h)
        val_h = filter_horizon_rows(forecast_df, origin_period="validation", horizon=h)
        if scenario == SCENARIO_ORACLE:
            q_cal = oracle_q_phys_for_horizon(cal_h, h)
            q_val = oracle_q_phys_for_horizon(val_h, h)
        else:
            assert precip_forcing is not None and et0 is not None
            assert state_by_date is not None and date_to_pos is not None
            q_cal = moderate_q_phys_for_horizon(
                cal_h,
                horizon=h,
                precip_forcing=precip_forcing,
                et0=et0,
                date_to_pos=date_to_pos,
                state_by_date=state_by_date,
                params=params,
            )
            q_val = moderate_q_phys_for_horizon(
                val_h,
                horizon=h,
                precip_forcing=precip_forcing,
                et0=et0,
                date_to_pos=date_to_pos,
                state_by_date=state_by_date,
                params=params,
            )
        out[h] = {
            "calibration": build_point_forecast_table(forecast_df, horizon=h, q_phys_h=q_cal, origin_mask_frame=cal_h),
            "validation": build_point_forecast_table(forecast_df, horizon=h, q_phys_h=q_val, origin_mask_frame=val_h),
        }
    return out


def fit_methods_for_horizon(
    cal_table: pd.DataFrame,
    *,
    horizon: int,
    train_fraction: float,
) -> dict[str, Any]:
    """Fit A/B/C on calibration only. Validation never enters."""
    train, calib = chronological_split(cal_table, train_fraction=train_fraction)
    # Method A: full calibration errors
    empirical = fit_empirical_residual(cal_table["error"].to_numpy(), horizon=horizon)
    # Method B: quantile regression on full calibration (primary). Conformal uses train only.
    quantile = fit_conditional_quantile(cal_table, horizon=horizon)
    # Method C: conformal radii from chronological CALIBRATION_SUBSET abs errors of Q_point
    conformal = fit_split_conformal(calib["abs_error"].to_numpy(), horizon=horizon)
    return {
        METHOD_EMPIRICAL: empirical,
        METHOD_QUANTILE: quantile,
        METHOD_CONFORMAL: conformal,
        "_train": train,
        "_calib": calib,
        "_quantile_train_only": fit_conditional_quantile(train, horizon=horizon),
    }


def predict_intervals(
    method_name: str,
    fitted: dict[str, Any],
    frame: pd.DataFrame,
    *,
    nominal: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_point = frame["q_point"].to_numpy(dtype=float)
    if method_name == METHOD_EMPIRICAL:
        return fitted[METHOD_EMPIRICAL].predict(q_point, nominal)
    if method_name == METHOD_QUANTILE:
        return fitted[METHOD_QUANTILE].predict(frame, q_point, nominal)
    if method_name == METHOD_CONFORMAL:
        return fitted[METHOD_CONFORMAL].predict(q_point, nominal)
    raise ValueError(f"unknown method: {method_name}")


def behavioral_intervals_for_targets(
    val_table: pd.DataFrame,
    ensemble_ts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map Phase 5 same-day q05/q95 onto forecast target dates (reference only)."""
    ts = ensemble_ts.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    by_date = ts.set_index("date")
    targets = pd.to_datetime(val_table["target_date"])
    q_obs = val_table["q_obs_target"].to_numpy(dtype=float)
    lower = []
    upper = []
    for d in targets:
        if d in by_date.index:
            lower.append(float(by_date.loc[d, "q05"]))
            upper.append(float(by_date.loc[d, "q95"]))
        else:
            lower.append(float("nan"))
            upper.append(float("nan"))
    return q_obs, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def evaluate_all_methods(
    *,
    bundles: dict[str, dict[int, dict[str, pd.DataFrame]]],
    fitted_by_scenario: dict[str, dict[int, dict[str, Any]]],
    ensemble_ts: pd.DataFrame,
    cal_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high_thr = calibration_high_flow_threshold(cal_daily, quantile=HIGH_FLOW_QUANTILE)
    low_thr = float(np.quantile(cal_daily["q_obs"].dropna(), LOW_FLOW_QUANTILE))

    coverage_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    forecast_parts: list[pd.DataFrame] = []

    for scenario, by_h in bundles.items():
        for h, tables in by_h.items():
            val = tables["validation"]
            fitted = fitted_by_scenario[scenario][h]
            mean_obs = float(np.nanmean(val["q_obs_target"]))

            for method in METHODS_PRIMARY:
                # Reliability levels + primary
                for nom in NOMINAL_LEVELS_RELIABILITY:
                    lo, up = predict_intervals(method, fitted, val, nominal=nom)
                    stats = coverage_and_width_stats(
                        val["q_obs_target"].to_numpy(),
                        lo,
                        up,
                        nominal=nom,
                        mean_obs=mean_obs,
                    )
                    coverage_rows.append(
                        {
                            "method": method,
                            "horizon_days": h,
                            "meteo_scenario": scenario,
                            "nominal_coverage": float(nom),
                            **stats,
                        }
                    )

                # Regime coverage at 90%
                lo90, up90 = predict_intervals(method, fitted, val, nominal=0.90)
                y = val["q_obs_target"].to_numpy(dtype=float)
                for regime, mask in (
                    ("high_flow", y > high_thr),
                    ("normal_flow", (y >= low_thr) & (y <= high_thr)),
                    ("low_flow", y < low_thr),
                ):
                    if mask.any():
                        inside = (y[mask] >= lo90[mask]) & (y[mask] <= up90[mask])
                        width = up90[mask] - lo90[mask]
                        emp = float(np.mean(inside))
                        mean_w = float(np.mean(width))
                    else:
                        emp = mean_w = float("nan")
                    regime_rows.append(
                        {
                            "method": method,
                            "horizon_days": h,
                            "meteo_scenario": scenario,
                            "nominal_coverage": 0.90,
                            "regime": regime,
                            "empirical_coverage": emp,
                            "mean_width": mean_w,
                            "n_days": int(mask.sum()),
                            "threshold_high": high_thr,
                            "threshold_low": low_thr,
                        }
                    )

                # Forecast artifact columns for 80/90/95
                lo80, up80 = predict_intervals(method, fitted, val, nominal=0.80)
                lo95, up95 = predict_intervals(method, fitted, val, nominal=0.95)
                forecast_parts.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(val["target_date"]).dt.strftime("%Y-%m-%d"),
                            "horizon_days": h,
                            "meteo_scenario": scenario,
                            "method": method,
                            "q_obs": y,
                            "q_point": val["q_point"].to_numpy(dtype=float),
                            "lower_80": lo80,
                            "upper_80": up80,
                            "lower_90": lo90,
                            "upper_90": up90,
                            "lower_95": lo95,
                            "upper_95": up95,
                            "period": "validation",
                        }
                    )
                )

            # Behavioral reference (same-day envelope on target date); nominal ~0.90 interpretive
            y_b, lo_b, up_b = behavioral_intervals_for_targets(val, ensemble_ts)
            stats_b = coverage_and_width_stats(y_b, lo_b, up_b, nominal=0.90, mean_obs=mean_obs)
            coverage_rows.append(
                {
                    "method": METHOD_BEHAVIORAL,
                    "horizon_days": h,
                    "meteo_scenario": scenario,
                    "nominal_coverage": 0.90,
                    **stats_b,
                }
            )
            forecast_parts.append(
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(val["target_date"]).dt.strftime("%Y-%m-%d"),
                        "horizon_days": h,
                        "meteo_scenario": scenario,
                        "method": METHOD_BEHAVIORAL,
                        "q_obs": y_b,
                        "q_point": val["q_point"].to_numpy(dtype=float),
                        "lower_80": np.nan,
                        "upper_80": np.nan,
                        "lower_90": lo_b,
                        "upper_90": up_b,
                        "lower_95": np.nan,
                        "upper_95": np.nan,
                        "period": "validation",
                    }
                )
            )

            # Extreme events at +24h / oracle preferred later; collect for all
            if h == 1:
                frame = pd.DataFrame(
                    {
                        "date": pd.to_datetime(val["target_date"]),
                        "q_obs": y,
                    }
                )
                events = identify_peak_events(frame, high_thr)
                events_sorted = sorted(events, key=lambda e: e["observed_peak"], reverse=True)
                for method in (*METHODS_PRIMARY, METHOD_BEHAVIORAL):
                    if method == METHOD_BEHAVIORAL:
                        lo_e, up_e = lo_b, up_b
                        q_point_e = val["q_point"].to_numpy(dtype=float)
                    else:
                        lo_e, up_e = predict_intervals(method, fitted, val, nominal=0.90)
                        q_point_e = val["q_point"].to_numpy(dtype=float)
                    for event in events_sorted[: DEFAULT_TOP_N_EXTREMES]:
                        idx = int(event["peak_idx"])
                        extreme_rows.append(
                            {
                                "method": method,
                                "horizon_days": h,
                                "meteo_scenario": scenario,
                                "event_id": event["event_id"],
                                "date": pd.Timestamp(event["observed_peak_date"]).strftime("%Y-%m-%d"),
                                "q_obs": float(y[idx]),
                                "q_point": float(q_point_e[idx]),
                                "point_error": float(y[idx] - q_point_e[idx]),
                                "lower_90": float(lo_e[idx]),
                                "upper_90": float(up_e[idx]),
                                "inside_90": bool(lo_e[idx] <= y[idx] <= up_e[idx]),
                                "interval_width": float(up_e[idx] - lo_e[idx]),
                            }
                        )

    coverage = pd.DataFrame(coverage_rows)[COVERAGE_SUMMARY_COLUMNS]
    regime = pd.DataFrame(regime_rows)
    extremes = pd.DataFrame(extreme_rows)
    forecasts = pd.concat(forecast_parts, ignore_index=True)[FORECAST_COLUMNS]
    return coverage, regime, extremes, forecasts


def plot_reliability(
    coverage: pd.DataFrame,
    *,
    scenario: str,
    method: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        sub = coverage.loc[
            (coverage["meteo_scenario"] == scenario)
            & (coverage["method"] == method)
            & (coverage["horizon_days"] == h)
        ].sort_values("nominal_coverage")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
        ax.plot(sub["nominal_coverage"], sub["empirical_coverage"], "o-", label=method)
        ax.set_title(f"+{h * 24} h")
        ax.set_xlabel("Nominal coverage")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Empirical validation coverage")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Reliability diagram — {method} — {scenario}\n(evaluation on validation only)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_rolling_coverage(
    forecasts: pd.DataFrame,
    *,
    scenario: str,
    method: str,
    horizon: int,
    window: int,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = forecasts.loc[
        (forecasts["meteo_scenario"] == scenario)
        & (forecasts["method"] == method)
        & (forecasts["horizon_days"] == horizon)
    ].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date")
    inside = ((sub["q_obs"] >= sub["lower_90"]) & (sub["q_obs"] <= sub["upper_90"])).astype(float)
    roll = inside.rolling(window=window, min_periods=max(10, window // 3)).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sub["date"], roll, color="#1d4ed8", lw=1.8)
    ax.axhline(0.90, color="black", ls="--", lw=1, label="nominal 90%")
    ax.set_ylabel("Trailing empirical coverage")
    ax.set_xlabel("Target date")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        f"Rolling {window}-day coverage — {method} +{horizon * 24}h — {scenario}\n"
        "(diagnostic only; not used for retuning)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_calibrated_demo(
    forecasts: pd.DataFrame,
    ensemble_ts: pd.DataFrame,
    config: dict[str, Any],
    *,
    scenario: str,
    method: str,
    coverage_row: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts = ensemble_ts.copy()
    ts["date_ts"] = pd.to_datetime(ts["date"])
    obs = ts.set_index("date_ts")["q_obs"]
    start, end = validation_plot_window(obs, config, window_days=VALIDATION_PLOT_WINDOW_DAYS)

    sub = forecasts.loc[
        (forecasts["meteo_scenario"] == scenario)
        & (forecasts["method"] == method)
        & (forecasts["horizon_days"] == 1)
    ].copy()
    sub["date_ts"] = pd.to_datetime(sub["date"])
    window = sub.loc[(sub["date_ts"] >= start) & (sub["date_ts"] <= end)].sort_values("date_ts")

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.fill_between(
        window["date_ts"],
        window["lower_90"],
        window["upper_90"],
        color="#93c5fd",
        alpha=0.35,
        label="Nominal 90% interval",
    )
    ax.plot(window["date_ts"], window["q_point"], color="#1d4ed8", lw=1.8, label="Point forecast")
    ax.plot(window["date_ts"], window["q_obs"], color="#111827", lw=1.6, label="Observed")
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    emp = float(coverage_row["empirical_coverage"]) * 100.0
    width = float(coverage_row["mean_width"])
    fig.suptitle("Calibrated forecast interval — +24 h", fontsize=14)
    fig.text(
        0.5,
        0.02,
        (
            f"Nominal coverage: 90% · Observed validation coverage: {emp:.1f}% · "
            f"Mean interval width: {width:.3f} mm/d · Method: {method}\n"
            "Coverage calibrated on pre-validation data only."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_before_after(
    forecasts: pd.DataFrame,
    ensemble_ts: pd.DataFrame,
    config: dict[str, Any],
    *,
    scenario: str,
    method: str,
    behavioral_coverage: float,
    calibrated_coverage: float,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts = ensemble_ts.copy()
    ts["date_ts"] = pd.to_datetime(ts["date"])
    obs = ts.set_index("date_ts")["q_obs"]
    start, end = validation_plot_window(obs, config, window_days=VALIDATION_PLOT_WINDOW_DAYS)

    left = ts.loc[(ts["period"] == "validation") & (ts["date_ts"] >= start) & (ts["date_ts"] <= end)]
    right = forecasts.loc[
        (forecasts["meteo_scenario"] == scenario)
        & (forecasts["method"] == method)
        & (forecasts["horizon_days"] == 1)
    ].copy()
    right["date_ts"] = pd.to_datetime(right["date"])
    right = right.loc[(right["date_ts"] >= start) & (right["date_ts"] <= end)].sort_values("date_ts")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    ax0.fill_between(left["date_ts"], left["q05"], left["q95"], color="#93c5fd", alpha=0.3, label="q05–q95")
    ax0.plot(left["date_ts"], left["q_obs"], color="#111827", lw=1.5, label="Observed")
    ax0.set_title(f"Behavioral parametric\nempirical coverage {behavioral_coverage * 100:.1f}%")
    ax0.set_ylabel("Discharge (mm/d)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    ax1.fill_between(right["date_ts"], right["lower_90"], right["upper_90"], color="#86efac", alpha=0.35, label="90% calibrated")
    ax1.plot(right["date_ts"], right["q_point"], color="#1d4ed8", lw=1.5, label="Point forecast")
    ax1.plot(right["date_ts"], right["q_obs"], color="#111827", lw=1.5, label="Observed")
    ax1.set_title(f"Calibrated interval ({method})\nempirical coverage {calibrated_coverage * 100:.1f}%")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    fig.suptitle(
        "Parametric dispersion → calibrated predictive uncertainty\n"
        "(identical validation window; calibrated on pre-validation data only)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def select_preferred_method(coverage: pd.DataFrame, scenario: str = SCENARIO_ORACLE) -> str:
    """Prefer method closest to nominal 90% with secondary sharpness preference.

    Labelled post-validation comparison only (not an unbiased final claim).
    """
    sub = coverage.loc[
        (coverage["meteo_scenario"] == scenario)
        & (coverage["nominal_coverage"] == 0.90)
        & (coverage["method"].isin(METHODS_PRIMARY))
    ].copy()
    if sub.empty:
        return METHOD_CONFORMAL
    # Score: coverage absolute error (primary) then mean width
    sub["score"] = sub["coverage_error"].abs() + 0.01 * sub["mean_width"]
    # Aggregate over horizons
    ranked = sub.groupby("method")["score"].mean().sort_values()
    return str(ranked.index[0])


def formulate_answers(
    coverage: pd.DataFrame,
    regime: pd.DataFrame,
    *,
    preferred: str,
    behavioral_global_coverage: float,
) -> dict[str, str]:
    def emp(method: str, h: int, scenario: str = SCENARIO_ORACLE, nom: float = 0.90) -> float:
        row = coverage.loc[
            (coverage["method"] == method)
            & (coverage["horizon_days"] == h)
            & (coverage["meteo_scenario"] == scenario)
            & (coverage["nominal_coverage"] == nom)
        ]
        return float(row.iloc[0]["empirical_coverage"]) if not row.empty else float("nan")

    def width(method: str, h: int, scenario: str = SCENARIO_ORACLE) -> float:
        row = coverage.loc[
            (coverage["method"] == method)
            & (coverage["horizon_days"] == h)
            & (coverage["meteo_scenario"] == scenario)
            & (coverage["nominal_coverage"] == 0.90)
        ]
        return float(row.iloc[0]["mean_width"]) if not row.empty else float("nan")

    closest = {}
    for h in HORIZONS:
        errs = {
            m: abs(emp(m, h) - 0.90)
            for m in METHODS_PRIMARY
        }
        closest[h] = min(errs, key=errs.get)

    ans1 = (
        f"+24h closest={closest[1]} (cov={emp(closest[1],1):.3f}); "
        f"+48h closest={closest[2]} (cov={emp(closest[2],2):.3f}); "
        f"+72h closest={closest[3]} (cov={emp(closest[3],3):.3f})."
    )

    # Sharpness among methods with |coverage_error|<0.05 at each horizon
    sharp_bits = []
    for h in HORIZONS:
        candidates = []
        for m in METHODS_PRIMARY:
            if abs(emp(m, h) - 0.90) <= 0.08:
                candidates.append((width(m, h), m))
        if candidates:
            candidates.sort()
            sharp_bits.append(f"+{h*24}h narrowest~comparable={candidates[0][1]} (width={candidates[0][0]:.3f})")
        else:
            sharp_bits.append(f"+{h*24}h: no method within 8pp of 90%")
    ans2 = "; ".join(sharp_bits)

    deg = [emp(preferred, h) for h in HORIZONS]
    ans3 = (
        f"Preferred ({preferred}) 90% coverage by horizon: "
        f"{deg[0]:.3f}/{deg[1]:.3f}/{deg[2]:.3f}. "
        + (
            "Coverage degrades with horizon."
            if deg[0] - deg[2] > 0.02
            else "Coverage remains relatively stable across horizons."
        )
    )

    def regime_cov(method: str, regime_name: str, h: int = 1) -> float:
        row = regime.loc[
            (regime["method"] == method)
            & (regime["regime"] == regime_name)
            & (regime["horizon_days"] == h)
            & (regime["meteo_scenario"] == SCENARIO_ORACLE)
        ]
        return float(row.iloc[0]["empirical_coverage"]) if not row.empty else float("nan")

    ans4 = (
        f"At +24h / oracle / {preferred}: high-flow cov={regime_cov(preferred,'high_flow'):.3f}, "
        f"normal={regime_cov(preferred,'normal_flow'):.3f}, "
        f"low={regime_cov(preferred,'low_flow'):.3f}. "
        + (
            "High-flow coverage deteriorates relative to normal/low flow."
            if regime_cov(preferred, "high_flow") < regime_cov(preferred, "normal_flow") - 0.05
            else "No severe high-flow under-coverage relative to normal flow in this sample."
        )
    )

    o24, m24 = emp(preferred, 1), emp(preferred, 1, SCENARIO_MODERATE)
    o48, m48 = emp(preferred, 2), emp(preferred, 2, SCENARIO_MODERATE)
    o72, m72 = emp(preferred, 3), emp(preferred, 3, SCENARIO_MODERATE)
    meteo_shift = max(abs(o24 - m24), abs(o48 - m48), abs(o72 - m72))
    if meteo_shift < 0.02:
        meteo_note = (
            "On this basin and moderate realization, overall 90% coverage for the preferred "
            "method stays nearly unchanged; widths can still widen at longer horizons. "
            "Honesty under imperfect weather forcing is therefore not guaranteed a priori — "
            "re-evaluate coverage under the operational meteo scenario."
        )
    else:
        meteo_note = (
            "Coverage shifts under moderate forcing: intervals calibrated under oracle "
            "forcing are not automatically honest once weather forcing is imperfect."
        )
    ans5 = (
        f"Oracle vs moderate 90% coverage for {preferred}: "
        f"+24h {o24:.3f}→{m24:.3f}, "
        f"+48h {o48:.3f}→{m48:.3f}, "
        f"+72h {o72:.3f}→{m72:.3f}. "
        f"{meteo_note}"
    )

    cal_cov = emp(preferred, 1)
    ans6 = (
        f"Behavioral envelope global validation coverage={behavioral_global_coverage:.3f} (~59.9% class). "
        f"Preferred calibrated +24h 90% coverage={cal_cov:.3f}. "
        f"Absolute coverage improvement ≈ {(cal_cov - behavioral_global_coverage)*100:.1f} percentage points "
        "(different semantics: same-day parametric dispersion vs lead-time predictive interval)."
    )

    conf_err = np.mean([abs(emp(METHOD_CONFORMAL, h) - 0.90) for h in HORIZONS])
    emp_err = np.mean([abs(emp(METHOD_EMPIRICAL, h) - 0.90) for h in HORIZONS])
    ans7 = (
        f"Mean |coverage_error| at 90%: conformal={conf_err:.3f}, empirical={emp_err:.3f}. "
        + (
            "Split conformal is useful for finite-sample coverage targeting."
            if conf_err <= emp_err + 0.01
            else "Conformal is not clearly better than empirical residual quantiles here."
        )
    )

    q_w = np.mean([width(METHOD_QUANTILE, h) for h in HORIZONS])
    e_w = np.mean([width(METHOD_EMPIRICAL, h) for h in HORIZONS])
    ans8 = (
        f"Mean 90% width: conditional_quantile={q_w:.3f}, empirical_residual={e_w:.3f}. "
        + (
            "Conditional quantile regression materially improves sharpness."
            if q_w < e_w * 0.9
            else "Conditional quantile regression does not materially improve sharpness versus empirical residual intervals."
        )
    )

    ans9 = (
        f"Recommended STRYMO prototype baseline (preferred after validation comparison): {preferred} "
        f"around physical+persistence point forecast, with regime-specific coverage monitoring. "
        f"{CRPS_NOTE}"
    )
    ans10 = (
        "A new untouched test period (or locked holdout after method selection) is required before "
        "operational claims: the preferred method was identified by comparing validation results, "
        "so the same validation cannot serve as an unbiased final performance estimate."
    )

    return {
        "1_closest_to_nominal_90": ans1,
        "2_narrowest_comparable_coverage": ans2,
        "3_coverage_vs_horizon": ans3,
        "4_high_flow_coverage": ans4,
        "5_moderate_meteo_effect": ans5,
        "6_vs_behavioral_envelope": ans6,
        "7_conformal_useful": ans7,
        "8_quantile_sharpness": ans8,
        "9_recommended_baseline": ans9,
        "10_needs_new_test_period": ans10,
        "preferred_method_label": "preferred after validation comparison",
        "preferred_method": preferred,
        "conformal_formula": CONFORMAL_FORMULA,
        "crps_note": CRPS_NOTE,
    }


def run_uncertainty_calibration(
    *,
    daily: pd.DataFrame,
    basin_data: pd.DataFrame,
    runs: pd.DataFrame,
    ensemble_ts: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[int, dict[str, Any]]],
    dict[str, Any],
]:
    unc_cfg = get_uncertainty_config(config)
    forecast_df = build_forecast_origin_dataset(daily)
    cal_daily = daily.loc[daily["period"] == "calibration"].copy()

    scenarios = list(unc_cfg["meteo_scenarios"])
    bundles: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    fitted_by_scenario: dict[str, dict[int, dict[str, Any]]] = {}

    for scenario in scenarios:
        bundles[scenario] = build_forecast_bundle_for_scenario(
            scenario=scenario,
            forecast_df=forecast_df,
            basin_data=basin_data,
            runs=runs,
            config=config,
            unc_cfg=unc_cfg,
        )
        fitted_by_scenario[scenario] = {}
        for h in HORIZONS:
            fitted_by_scenario[scenario][h] = fit_methods_for_horizon(
                bundles[scenario][h]["calibration"],
                horizon=h,
                train_fraction=float(unc_cfg["train_fraction"]),
            )

    coverage, regime, extremes, forecasts = evaluate_all_methods(
        bundles=bundles,
        fitted_by_scenario=fitted_by_scenario,
        ensemble_ts=ensemble_ts,
        cal_daily=cal_daily,
    )
    meta = {
        "unc_cfg": unc_cfg,
        "behavioral_global_coverage": float(empirical_validation_coverage(ensemble_ts)),
        "bundles": bundles,
        "fitted_by_scenario": fitted_by_scenario,
    }
    return coverage, regime, extremes, forecasts, fitted_by_scenario, meta


def run_uncertainty_export(
    *,
    dataset_path: Path,
    basin_data_path: Path,
    runs_path: Path,
    ensemble_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> UncertaintyResult:
    daily = load_ml_residual_dataset(dataset_path)
    basin = pd.read_csv(basin_data_path, parse_dates=["date"]).set_index("date").sort_index()
    runs = pd.read_csv(runs_path)
    ensemble_ts = pd.read_csv(ensemble_path)

    coverage, regime, extremes, forecasts, _fitted, meta = run_uncertainty_calibration(
        daily=daily,
        basin_data=basin,
        runs=runs,
        ensemble_ts=ensemble_ts,
        config=config,
    )
    preferred = select_preferred_method(coverage, scenario=SCENARIO_ORACLE)
    answers = formulate_answers(
        coverage,
        regime,
        preferred=preferred,
        behavioral_global_coverage=meta["behavioral_global_coverage"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "uncertainty_coverage_summary.csv"
    regime_path = output_dir / "uncertainty_regime_coverage.csv"
    extreme_path = output_dir / "uncertainty_extreme_events.csv"
    forecasts_path = output_dir / "uncertainty_forecasts.csv"
    reliability_path = output_dir / "uncertainty_reliability.png"
    rolling_path = output_dir / "uncertainty_rolling_coverage.png"
    demo_path = output_dir / "uncertainty_calibrated_demo.png"
    before_after_path = output_dir / "uncertainty_before_after.png"
    answers_path = output_dir / "uncertainty_answers.json"

    coverage.to_csv(coverage_path, index=False)
    regime.to_csv(regime_path, index=False)
    extremes.to_csv(extreme_path, index=False)
    forecasts.to_csv(forecasts_path, index=False)

    plot_reliability(
        coverage,
        scenario=SCENARIO_ORACLE,
        method=preferred,
        output_path=reliability_path,
    )
    plot_rolling_coverage(
        forecasts,
        scenario=SCENARIO_ORACLE,
        method=preferred,
        horizon=1,
        window=int(meta["unc_cfg"]["rolling_coverage_window_days"]),
        output_path=rolling_path,
    )

    cov_row = coverage.loc[
        (coverage["method"] == preferred)
        & (coverage["horizon_days"] == 1)
        & (coverage["meteo_scenario"] == SCENARIO_ORACLE)
        & (coverage["nominal_coverage"] == 0.90)
    ].iloc[0].to_dict()
    plot_calibrated_demo(
        forecasts,
        ensemble_ts,
        config,
        scenario=SCENARIO_ORACLE,
        method=preferred,
        coverage_row=cov_row,
        output_path=demo_path,
    )
    plot_before_after(
        forecasts,
        ensemble_ts,
        config,
        scenario=SCENARIO_ORACLE,
        method=preferred,
        behavioral_coverage=meta["behavioral_global_coverage"],
        calibrated_coverage=float(cov_row["empirical_coverage"]),
        output_path=before_after_path,
    )

    answers_path.write_text(
        json.dumps(
            {
                "answers": answers,
                "scenario_labels": {k: SCENARIO_LABELS.get(k, k) for k in meta["unc_cfg"]["meteo_scenarios"]},
                "conformal_formula": CONFORMAL_FORMULA,
                "crps_note": CRPS_NOTE,
                "uncertainty_config": meta["unc_cfg"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return UncertaintyResult(
        coverage_summary=coverage,
        regime_coverage=regime,
        extreme_events=extremes,
        forecasts=forecasts,
        answers=answers,
        coverage_path=coverage_path,
        regime_path=regime_path,
        extreme_path=extreme_path,
        forecasts_path=forecasts_path,
        reliability_path=reliability_path,
        rolling_path=rolling_path,
        demo_path=demo_path,
        before_after_path=before_after_path,
        answers_path=answers_path,
        preferred_method=preferred,
    )


def print_uncertainty_report(result: UncertaintyResult) -> None:
    print("=== Phase 9: forecast uncertainty calibration ===")
    print(CRPS_NOTE)
    print(f"Preferred method (after validation comparison): {result.preferred_method}")
    print()
    print(f"Coverage summary: {result.coverage_path.resolve()}")
    print(f"Regime coverage:  {result.regime_path.resolve()}")
    print(f"Extreme events:   {result.extreme_path.resolve()}")
    print(f"Forecasts:        {result.forecasts_path.resolve()}")
    print(f"Reliability:      {result.reliability_path.resolve()}")
    print(f"Rolling coverage: {result.rolling_path.resolve()}")
    print(f"Demo figure:      {result.demo_path.resolve()}")
    print(f"Before/after:     {result.before_after_path.resolve()}")
    print()
    sub = result.coverage_summary.loc[
        (result.coverage_summary["nominal_coverage"] == 0.90)
        & (result.coverage_summary["meteo_scenario"] == SCENARIO_ORACLE)
    ]
    cols = ["method", "horizon_days", "empirical_coverage", "coverage_error", "mean_width", "mean_interval_score"]
    print("--- Oracle 90% coverage ---")
    print(sub[cols].to_string(index=False))
    print()
    print("--- Answers ---")
    for key, text in result.answers.items():
        if key in {"conformal_formula", "crps_note", "preferred_method", "preferred_method_label"}:
            continue
        print(f"{key}: {text}")
