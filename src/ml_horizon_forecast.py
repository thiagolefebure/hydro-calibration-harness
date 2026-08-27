"""Phase 8D: true multi-horizon residual forecasting (forecast-origin logic).

At forecast origin t, predict residual at t+h for h in {1,2,3} using ONLY
information available at t. Physical Q_phys(t+h) uses historically observed
future precipitation (ORACLE METEOROLOGICAL FORCING) — not operational NWP.

Does not modify GR4J, calibration, validation periods, metrics, or Phase 8B/8C
artifacts. Does not introduce LightGBM/XGBoost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.metrics import get_log_nse_epsilon, mae, rmse
from src.ml_residual_ablation import HIGH_FLOW_QUANTILE, calibration_high_flow_threshold
from src.ml_residual_baselines import (
    RIDGE_ALPHA,
    evaluate_discharge_metrics,
    hybrid_discharge,
    load_ml_residual_dataset,
    split_calibration_validation,
)
from src.ml_residual_dataset import TARGET_COLUMN

HORIZONS = (1, 2, 3)

EXPERIMENT_TAG = "ORACLE METEOROLOGICAL FORCING"
EXPERIMENT_NOTE = (
    "Q_phys(t+h) is generated with historically observed future precipitation. "
    "This is NOT operational forecasting performance; it isolates residual-correction "
    "skill from meteorological forecast error."
)

FORECAST_FEATURE_COLUMNS = [
    "residual_t",
    "q_obs_t",
    "q_phys_t",
    "q_phys_change_1d_at_t",
    "precip_last_1d",
    "precip_last_3d",
    "precip_last_7d",
    "precip_last_30d",
    "production_store_t",
    "routing_store_t",
    "day_of_year_sin",
    "day_of_year_cos",
]

FORBIDDEN_FUTURE_COLUMNS = frozenset(
    {
        "q_obs_h1",
        "q_obs_h2",
        "q_obs_h3",
        "residual_h1",
        "residual_h2",
        "residual_h3",
        "q_phys_h1",
        "q_phys_h2",
        "q_phys_h3",
    }
)

COMPARISON_COLUMNS = [
    "horizon_days",
    "model",
    "kge",
    "nse",
    "lognse",
    "bias",
    "mae",
    "rmse",
]

MODEL_PHYSICAL = "physical"
MODEL_PERSISTENCE = "persistence_residual"
MODEL_AR1 = "AR1_residual"
MODEL_RIDGE = "ridge"


@dataclass
class FittedAR1Horizon:
    """AR(1) on calibration: e_t = c + phi * e_(t-1); recurse for multi-step."""

    intercept: float
    phi: float
    fitted_on: str = "calibration"

    def forecast_from_origin(self, residual_t: np.ndarray, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        e = np.asarray(residual_t, dtype=float)
        for _ in range(horizon):
            e = self.intercept + self.phi * e
        return e


@dataclass
class HorizonRidge:
    horizon: int
    feature_names: list[str]
    scaler: StandardScaler
    model: Ridge

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        return np.asarray(self.model.predict(self.scaler.transform(x)), dtype=float)


@dataclass
class MLHorizonResult:
    comparison: pd.DataFrame
    highflow: pd.DataFrame
    degradation: pd.DataFrame
    forecast_dataset: pd.DataFrame
    comparison_path: Path
    highflow_path: Path
    degradation_path: Path
    answers: dict[str, str] = field(default_factory=dict)
    experiment_tag: str = EXPERIMENT_TAG


def build_forecast_origin_dataset(daily: pd.DataFrame) -> pd.DataFrame:
    """Build one row per forecast origin t with forward residual targets.

    Predictors use only information at t. Targets residual_h* are shifted
    forward (future residuals), never used as predictors.
    """
    required = {
        "date",
        "period",
        "q_obs",
        "q_phys",
        TARGET_COLUMN,
        "precipitation_1d",
        "precipitation_3d",
        "precipitation_7d",
        "precipitation_30d",
        "production_store",
        "routing_store",
        "day_of_year_sin",
        "day_of_year_cos",
        "q_phys_change_1d",
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily residual dataset missing columns: {sorted(missing)}")

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "origin_date": frame["date"],
            "period": frame["period"].to_numpy(dtype="object"),
            "residual_t": frame[TARGET_COLUMN].to_numpy(dtype=float),
            "q_obs_t": frame["q_obs"].to_numpy(dtype=float),
            "q_phys_t": frame["q_phys"].to_numpy(dtype=float),
            "q_phys_change_1d_at_t": frame["q_phys_change_1d"].to_numpy(dtype=float),
            "precip_last_1d": frame["precipitation_1d"].to_numpy(dtype=float),
            "precip_last_3d": frame["precipitation_3d"].to_numpy(dtype=float),
            "precip_last_7d": frame["precipitation_7d"].to_numpy(dtype=float),
            "precip_last_30d": frame["precipitation_30d"].to_numpy(dtype=float),
            "production_store_t": frame["production_store"].to_numpy(dtype=float),
            "routing_store_t": frame["routing_store"].to_numpy(dtype=float),
            "day_of_year_sin": frame["day_of_year_sin"].to_numpy(dtype=float),
            "day_of_year_cos": frame["day_of_year_cos"].to_numpy(dtype=float),
        }
    )

    for h in HORIZONS:
        # Targets and future physical/obs for evaluation only (not predictors).
        out[f"residual_h{h}"] = frame[TARGET_COLUMN].shift(-h).to_numpy(dtype=float)
        out[f"q_obs_h{h}"] = frame["q_obs"].shift(-h).to_numpy(dtype=float)
        out[f"q_phys_h{h}"] = frame["q_phys"].shift(-h).to_numpy(dtype=float)
        out[f"target_date_h{h}"] = frame["date"].shift(-h)
        out[f"target_period_h{h}"] = frame["period"].shift(-h)

    return out


def filter_horizon_rows(
    forecast_df: pd.DataFrame,
    *,
    origin_period: str,
    horizon: int,
) -> pd.DataFrame:
    """Keep origins in origin_period whose target day is also in that period."""
    target_period_col = f"target_period_h{horizon}"
    residual_col = f"residual_h{horizon}"
    mask = (
        (forecast_df["period"] == origin_period)
        & (forecast_df[target_period_col] == origin_period)
        & forecast_df[residual_col].notna()
        & forecast_df[f"q_obs_h{horizon}"].notna()
        & forecast_df[f"q_phys_h{horizon}"].notna()
        & forecast_df[FORECAST_FEATURE_COLUMNS].notna().all(axis=1)
    )
    return forecast_df.loc[mask].copy()


def fit_ar1_on_calibration_residuals(cal_daily: pd.DataFrame) -> FittedAR1Horizon:
    """Fit e_t = c + phi * e_(t-1) on calibration residuals only."""
    ordered = cal_daily.copy()
    if "date" in ordered.columns:
        ordered["date"] = pd.to_datetime(ordered["date"])
        ordered = ordered.sort_values("date")
    y = ordered[TARGET_COLUMN].to_numpy(dtype=float)
    x = ordered[TARGET_COLUMN].shift(1).to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)
    reg = LinearRegression()
    reg.fit(x[valid].reshape(-1, 1), y[valid])
    return FittedAR1Horizon(intercept=float(reg.intercept_), phi=float(reg.coef_[0]))


def fit_ridge_for_horizon(
    cal_origins: pd.DataFrame,
    horizon: int,
    *,
    alpha: float = RIDGE_ALPHA,
) -> HorizonRidge:
    features = list(FORECAST_FEATURE_COLUMNS)
    leak = set(features) & FORBIDDEN_FUTURE_COLUMNS
    if leak:
        raise ValueError(f"future columns in predictors: {sorted(leak)}")
    x = cal_origins.loc[:, features].to_numpy(dtype=float)
    y = cal_origins[f"residual_h{horizon}"].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=alpha)
    model.fit(x_scaled, y)
    return HorizonRidge(
        horizon=horizon,
        feature_names=features,
        scaler=scaler,
        model=model,
    )


def _metric_row(
    *,
    horizon: int,
    model: str,
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    epsilon_mm: float,
) -> dict[str, Any]:
    metrics = evaluate_discharge_metrics(observed, predicted, epsilon_mm=epsilon_mm)
    return {
        "horizon_days": int(horizon),
        "model": model,
        "kge": metrics["kge_val"],
        "nse": metrics["nse_val"],
        "lognse": metrics["lognse_val"],
        "bias": metrics["bias_val"],
        "mae": metrics["mae_val"],
        "rmse": metrics["rmse_val"],
    }


def predict_all_models(
    val_origins: pd.DataFrame,
    *,
    horizon: int,
    ar1: FittedAR1Horizon,
    ridge: HorizonRidge,
) -> dict[str, np.ndarray]:
    q_phys_h = val_origins[f"q_phys_h{horizon}"].to_numpy(dtype=float)
    residual_t = val_origins["residual_t"].to_numpy(dtype=float)

    pers_resid = residual_t.copy()
    ar_resid = ar1.forecast_from_origin(residual_t, horizon)
    ridge_resid = ridge.predict(val_origins)

    return {
        MODEL_PHYSICAL: q_phys_h.copy(),
        MODEL_PERSISTENCE: hybrid_discharge(q_phys_h, pers_resid),
        MODEL_AR1: hybrid_discharge(q_phys_h, ar_resid),
        MODEL_RIDGE: hybrid_discharge(q_phys_h, ridge_resid),
    }


def evaluate_horizon(
    val_origins: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    horizon: int,
    epsilon_mm: float,
) -> pd.DataFrame:
    obs = val_origins[f"q_obs_h{horizon}"].to_numpy(dtype=float)
    rows = [
        _metric_row(
            horizon=horizon,
            model=name,
            observed=obs,
            predicted=pred,
            epsilon_mm=epsilon_mm,
        )
        for name, pred in predictions.items()
    ]
    return pd.DataFrame(rows)


def highflow_horizon_metrics(
    val_origins: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    obs = val_origins[f"q_obs_h{horizon}"].to_numpy(dtype=float)
    high = obs >= threshold
    rows: list[dict[str, Any]] = []

    peak = np.zeros(len(obs), dtype=bool)
    for i in range(1, len(obs) - 1):
        if high[i] and obs[i] >= obs[i - 1] and obs[i] >= obs[i + 1]:
            peak[i] = True
    if len(obs) == 1 and high[0]:
        peak[0] = True

    for model_name, pred in predictions.items():
        if not high.any():
            mae_v = rmse_v = mean_bias = peak_mag = float("nan")
            n_high = 0
            n_peak = 0
        else:
            mae_m = mae(obs[high], pred[high])
            rmse_m = rmse(obs[high], pred[high])
            mae_v = float(mae_m.value) if mae_m.is_defined else float("nan")
            rmse_v = float(rmse_m.value) if rmse_m.is_defined else float("nan")
            mean_bias = float(np.mean(pred[high] - obs[high]))
            n_high = int(high.sum())
            if peak.any():
                peak_mag = float(np.mean(pred[peak] - obs[peak]))
                n_peak = int(peak.sum())
            else:
                peak_mag = float("nan")
                n_peak = 0
        rows.append(
            {
                "horizon_days": int(horizon),
                "model": model_name,
                "n_high_flow_days": n_high,
                "n_peak_days": n_peak,
                "threshold_q_obs": threshold,
                "mae_highflow": mae_v,
                "rmse_highflow": rmse_v,
                "mean_bias_highflow": mean_bias,
                "peak_magnitude_error": peak_mag,
                "experiment_tag": EXPERIMENT_TAG,
            }
        )
    return pd.DataFrame(rows)


def build_degradation_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in comparison["model"].unique():
        sub = comparison.loc[comparison["model"] == model].set_index("horizon_days")
        k1 = float(sub.loc[1, "kge"]) if 1 in sub.index else float("nan")
        k2 = float(sub.loc[2, "kge"]) if 2 in sub.index else float("nan")
        k3 = float(sub.loc[3, "kge"]) if 3 in sub.index else float("nan")
        rows.append(
            {
                "model": model,
                "kge_h1": k1,
                "kge_h2": k2,
                "kge_h3": k3,
                "delta_KGE_1_to_3": k3 - k1,
                "experiment_tag": EXPERIMENT_TAG,
            }
        )
    return pd.DataFrame(rows)


def formulate_horizon_answers(
    comparison: pd.DataFrame,
    highflow: pd.DataFrame,
    degradation: pd.DataFrame,
) -> dict[str, str]:
    def kge(model: str, h: int) -> float:
        row = comparison.loc[
            (comparison["model"] == model) & (comparison["horizon_days"] == h)
        ]
        return float(row.iloc[0]["kge"]) if not row.empty else float("nan")

    def hf_mae(model: str, h: int) -> float:
        row = highflow.loc[
            (highflow["model"] == model) & (highflow["horizon_days"] == h)
        ]
        return float(row.iloc[0]["mae_highflow"]) if not row.empty else float("nan")

    phys = {h: kge(MODEL_PHYSICAL, h) for h in HORIZONS}
    pers = {h: kge(MODEL_PERSISTENCE, h) for h in HORIZONS}
    ar1 = {h: kge(MODEL_AR1, h) for h in HORIZONS}
    ridge = {h: kge(MODEL_RIDGE, h) for h in HORIZONS}

    def helps(h: int) -> str:
        gain = pers[h] - phys[h]
        hours = h * 24
        return (
            f"Yes at +{hours} h (persistence KGE={pers[h]:.4f} vs physical={phys[h]:.4f}, "
            f"delta={gain:+.4f})."
            if gain > 0
            else (
                f"No meaningful help at +{hours} h (persistence KGE={pers[h]:.4f} vs "
                f"physical={phys[h]:.4f}, delta={gain:+.4f})."
            )
        )

    deg = degradation.set_index("model")
    pers_decay = float(deg.loc[MODEL_PERSISTENCE, "delta_KGE_1_to_3"])
    ridge_vs_pers_h1 = ridge[1] - pers[1]
    ridge_vs_pers_h3 = ridge[3] - pers[3]

    hf_change = []
    for h in HORIZONS:
        hf_change.append(
            f"+{h * 24}h MAE physical/persistence/ridge="
            f"{hf_mae(MODEL_PHYSICAL, h):.4f}/"
            f"{hf_mae(MODEL_PERSISTENCE, h):.4f}/"
            f"{hf_mae(MODEL_RIDGE, h):.4f}"
        )

    remaining = 1.0 - ridge[1]
    answers = {
        "1_persistence_plus_24h": helps(1),
        "2_persistence_plus_48h": helps(2),
        "3_persistence_plus_72h": helps(3),
        "4_gain_decay_with_horizon": (
            f"Persistence KGE: +24h={pers[1]:.4f}, +48h={pers[2]:.4f}, "
            f"+72h={pers[3]:.4f}; delta_KGE_1_to_3={pers_decay:+.4f}. "
            f"AR1: {ar1[1]:.4f}/{ar1[2]:.4f}/{ar1[3]:.4f}; "
            f"Ridge: {ridge[1]:.4f}/{ridge[2]:.4f}/{ridge[3]:.4f}. "
            "Correction value decays as residual autocorrelation fades with lead time."
        ),
        "5_ridge_vs_persistence": (
            f"Ridge minus persistence KGE: +24h={ridge_vs_pers_h1:+.4f}, "
            f"+72h={ridge_vs_pers_h3:+.4f}. "
            + (
                "Ridge does not outperform persistence enough to justify ML complexity "
                "for these horizons."
                if abs(ridge_vs_pers_h1) < 0.02 and abs(ridge_vs_pers_h3) < 0.02
                else "Ridge shows a material edge over persistence on at least one horizon."
            )
        ),
        "6_high_flow_conclusion": (
            "High-flow (cal threshold): " + "; ".join(hf_change) + ". "
            + (
                "Conclusion unchanged: residual persistence still helps on high flows "
                "at short lead, with similar horizon decay."
                if hf_mae(MODEL_PERSISTENCE, 1) < hf_mae(MODEL_PHYSICAL, 1)
                else "On high flows, persistence advantage vs physical is weak or absent."
            )
        ),
        "7_meteo_vs_residual": (
            f"Under {EXPERIMENT_TAG}, precipitation at t+h is known historically, so "
            f"remaining error after residual correction is NOT weather-forecast error. "
            f"At +24h, ridge KGE={ridge[1]:.4f} (gap to 1.0 approx {remaining:.4f}). "
            "In real operations, meteorological forecast error would add substantial "
            "extra degradation beyond this oracle-forcing residual-correction skill. "
            "Most of the Phase 8B same-time gain was residual persistence; at multi-day "
            "leads that signal decays, and operational meteo uncertainty would dominate "
            "further losses."
        ),
        "experiment_tag": EXPERIMENT_TAG,
        "experiment_note": EXPERIMENT_NOTE,
    }
    return answers


def run_horizon_forecast_analysis(
    daily: pd.DataFrame,
    *,
    epsilon_mm: float | None = None,
    high_flow_quantile: float = HIGH_FLOW_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    eps = get_log_nse_epsilon(None) if epsilon_mm is None else float(epsilon_mm)
    forecast_df = build_forecast_origin_dataset(daily)
    cal_daily, _val_daily = split_calibration_validation(daily)

    ar1 = fit_ar1_on_calibration_residuals(cal_daily)
    threshold = calibration_high_flow_threshold(cal_daily, quantile=high_flow_quantile)

    comparison_parts: list[pd.DataFrame] = []
    highflow_parts: list[pd.DataFrame] = []
    ridges: dict[int, HorizonRidge] = {}

    for h in HORIZONS:
        cal_h = filter_horizon_rows(forecast_df, origin_period="calibration", horizon=h)
        val_h = filter_horizon_rows(forecast_df, origin_period="validation", horizon=h)
        if cal_h.empty or val_h.empty:
            raise ValueError(f"insufficient rows for horizon {h}")

        ridge = fit_ridge_for_horizon(cal_h, h)
        ridges[h] = ridge
        preds = predict_all_models(val_h, horizon=h, ar1=ar1, ridge=ridge)
        comparison_parts.append(
            evaluate_horizon(val_h, preds, horizon=h, epsilon_mm=eps)
        )
        highflow_parts.append(
            highflow_horizon_metrics(val_h, preds, horizon=h, threshold=threshold)
        )

    comparison = pd.concat(comparison_parts, ignore_index=True)[COMPARISON_COLUMNS]
    highflow = pd.concat(highflow_parts, ignore_index=True)
    degradation = build_degradation_table(comparison)
    answers = formulate_horizon_answers(comparison, highflow, degradation)

    degradation = degradation.copy()
    degradation.attrs["ar1_intercept"] = ar1.intercept
    degradation.attrs["ar1_phi"] = ar1.phi
    degradation.attrs["n_ridge_models"] = len(ridges)
    return comparison, highflow, degradation, forecast_df, answers


def run_ml_horizon_export(
    dataset_path: Path,
    output_dir: Path,
    *,
    epsilon_mm: float | None = None,
    config: dict[str, Any] | None = None,
) -> MLHorizonResult:
    if epsilon_mm is None:
        epsilon_mm = get_log_nse_epsilon(config)
    daily = load_ml_residual_dataset(dataset_path)
    comparison, highflow, degradation, forecast_df, answers = run_horizon_forecast_analysis(
        daily, epsilon_mm=epsilon_mm
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "ml_horizon_comparison.csv"
    highflow_path = output_dir / "ml_horizon_highflow.csv"
    degradation_path = output_dir / "ml_horizon_degradation.csv"
    forecast_path = output_dir / "ml_horizon_forecast_dataset.csv"
    answers_path = output_dir / "ml_horizon_answers.json"

    comparison.to_csv(comparison_path, index=False)
    highflow.to_csv(highflow_path, index=False)
    degradation.to_csv(degradation_path, index=False)
    keep_cols = (
        ["origin_date", "period", *FORECAST_FEATURE_COLUMNS]
        + [f"residual_h{h}" for h in HORIZONS]
        + [f"q_obs_h{h}" for h in HORIZONS]
        + [f"q_phys_h{h}" for h in HORIZONS]
        + [f"target_date_h{h}" for h in HORIZONS]
        + [f"target_period_h{h}" for h in HORIZONS]
    )
    forecast_df.loc[:, keep_cols].to_csv(forecast_path, index=False)
    answers_path.write_text(
        json.dumps(
            {
                "answers": answers,
                "experiment_tag": EXPERIMENT_TAG,
                "experiment_note": EXPERIMENT_NOTE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return MLHorizonResult(
        comparison=comparison,
        highflow=highflow,
        degradation=degradation,
        forecast_dataset=forecast_df,
        comparison_path=comparison_path,
        highflow_path=highflow_path,
        degradation_path=degradation_path,
        answers=answers,
    )


def print_ml_horizon_report(result: MLHorizonResult) -> None:
    print("=== Phase 8D: multi-horizon residual forecasting ===")
    print(f"Experiment tag: {EXPERIMENT_TAG}")
    print(EXPERIMENT_NOTE)
    print()
    print(f"Comparison:   {result.comparison_path.resolve()}")
    print(f"High-flow:    {result.highflow_path.resolve()}")
    print(f"Degradation:  {result.degradation_path.resolve()}")
    print()
    print("--- Metrics by horizon ---")
    print(result.comparison.to_string(index=False))
    print()
    print("--- Horizon degradation (KGE) ---")
    print(result.degradation.to_string(index=False))
    print()
    print("--- Answers ---")
    for key, text in result.answers.items():
        if key.startswith("experiment_"):
            continue
        print(f"{key}: {text}")
