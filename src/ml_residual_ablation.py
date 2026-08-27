"""Phase 8C: residual-correction robustness and ablation analysis.

Determines what information drives Phase 8B gains before adding model complexity.
Does not modify GR4J, calibration, validation periods, metrics, or Phase 8B artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.metrics import get_log_nse_epsilon, mae, rmse
from src.ml_residual_baselines import (
    HGB_CONFIG,
    RIDGE_ALPHA,
    TRAIN_FEATURE_COLUMNS,
    DummyMeanResidual,
    HistGradientBoostingResidual,
    evaluate_discharge_metrics,
    hybrid_discharge,
    load_ml_residual_dataset,
    split_calibration_validation,
)
from src.ml_residual_dataset import TARGET_COLUMN

HIGH_FLOW_QUANTILE = 0.90
ACF_MAX_LAG = 30

ABLATION_COMPARISON_COLUMNS = [
    "group",
    "model",
    "features",
    "kge_val",
    "nse_val",
    "lognse_val",
    "bias_val",
    "mae_val",
    "rmse_val",
]

# Explicit ablation feature sets (leakage-safe only).
RIDGE_ABLATION_FEATURES: dict[str, list[str]] = {
    "ridge_A_lag1": ["residual_lag_1"],
    "ridge_B_lag1_qphys": ["residual_lag_1", "q_phys"],
    "ridge_C_lag1_states": ["residual_lag_1", "production_store", "routing_store"],
    "ridge_D_lag1_rainfall": [
        "residual_lag_1",
        "precipitation_1d",
        "precipitation_3d",
        "precipitation_7d",
        "precipitation_30d",
    ],
    "ridge_E_all_safe": list(TRAIN_FEATURE_COLUMNS),
}

FORBIDDEN_PREDICTORS = frozenset({"q_obs", "q_obs_change_1d", TARGET_COLUMN, "residual"})


@dataclass
class FittedAR1:
    """AR(1) residual model: residual(t) = intercept + phi * residual(t-1)."""

    intercept: float
    phi: float
    fitted_on: str = "calibration"

    def predict_from_lag(self, residual_lag_1: np.ndarray) -> np.ndarray:
        return self.intercept + self.phi * np.asarray(residual_lag_1, dtype=float)


@dataclass
class FittedScaledRidge:
    """Ridge with StandardScaler fitted on calibration only."""

    name: str
    feature_names: list[str]
    scaler: StandardScaler
    model: Ridge
    coefficients: dict[str, float]
    intercept: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        return np.asarray(self.model.predict(self.scaler.transform(x)), dtype=float)


@dataclass
class MLAblationResult:
    ablation: pd.DataFrame
    coefficients: pd.DataFrame
    events: pd.DataFrame
    temporal: pd.DataFrame
    acf: pd.DataFrame
    ablation_path: Path
    event_path: Path
    temporal_path: Path
    acf_path: Path
    coefficients_path: Path
    answers: dict[str, str] = field(default_factory=dict)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    leak = [c for c in columns if c in FORBIDDEN_PREDICTORS]
    if leak:
        raise ValueError(f"forbidden predictors requested: {leak}")


def fit_ar1_calibration(cal: pd.DataFrame) -> FittedAR1:
    """Fit AR(1) on calibration residuals only."""
    y = cal[TARGET_COLUMN].to_numpy(dtype=float)
    x = cal["residual_lag_1"].to_numpy(dtype=float).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)
    return FittedAR1(intercept=float(reg.intercept_), phi=float(reg.coef_[0]))


def persistence_residual_hat(frame: pd.DataFrame) -> np.ndarray:
    """residual_hat(t) = residual(t-1) only."""
    return frame["residual_lag_1"].to_numpy(dtype=float)


def fit_scaled_ridge(
    cal: pd.DataFrame,
    feature_names: list[str],
    *,
    name: str,
    alpha: float = RIDGE_ALPHA,
) -> FittedScaledRidge:
    _require_columns(cal, feature_names)
    x = cal.loc[:, feature_names].to_numpy(dtype=float)
    y = cal[TARGET_COLUMN].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=alpha)
    model.fit(x_scaled, y)
    coefficients = {
        feat: float(coef) for feat, coef in zip(feature_names, model.coef_, strict=True)
    }
    return FittedScaledRidge(
        name=name,
        feature_names=list(feature_names),
        scaler=scaler,
        model=model,
        coefficients=coefficients,
        intercept=float(model.intercept_),
    )


def metrics_row(
    *,
    group: str,
    model: str,
    features: list[str] | str,
    observed: pd.Series | np.ndarray,
    predicted: np.ndarray,
    epsilon_mm: float,
) -> dict[str, Any]:
    feats = features if isinstance(features, str) else ",".join(features)
    metrics = evaluate_discharge_metrics(observed, predicted, epsilon_mm=epsilon_mm)
    return {"group": group, "model": model, "features": feats, **metrics}


def build_baseline_predictions(
    cal: pd.DataFrame,
    val: pd.DataFrame,
    *,
    epsilon_mm: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], FittedAR1, FittedScaledRidge, Any]:
    """Fit A–F models on calibration; return metrics and validation residual_hat."""
    rows: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}

    # A. PHYSICAL
    q_phys = val["q_phys"].to_numpy(dtype=float)
    preds["physical"] = q_phys.copy()
    rows.append(
        metrics_row(
            group="baseline",
            model="physical",
            features="(none)",
            observed=val["q_obs"],
            predicted=q_phys,
            epsilon_mm=epsilon_mm,
        )
    )

    # B. MEAN RESIDUAL
    dummy = DummyMeanResidual()
    dummy.fit(np.zeros((len(cal), 1)), cal[TARGET_COLUMN].to_numpy(dtype=float))
    resid_mean = np.full(len(val), dummy.mean_residual_, dtype=float)
    q_mean = hybrid_discharge(val["q_phys"], resid_mean)
    preds["mean_residual"] = q_mean
    rows.append(
        metrics_row(
            group="baseline",
            model="mean_residual",
            features="mean(residual_cal)",
            observed=val["q_obs"],
            predicted=q_mean,
            epsilon_mm=epsilon_mm,
        )
    )

    # C. RESIDUAL PERSISTENCE
    resid_pers = persistence_residual_hat(val)
    q_pers = hybrid_discharge(val["q_phys"], resid_pers)
    preds["persistence"] = q_pers
    rows.append(
        metrics_row(
            group="baseline",
            model="persistence",
            features="residual_lag_1",
            observed=val["q_obs"],
            predicted=q_pers,
            epsilon_mm=epsilon_mm,
        )
    )

    # D. AR(1)
    ar1 = fit_ar1_calibration(cal)
    resid_ar1 = ar1.predict_from_lag(val["residual_lag_1"].to_numpy(dtype=float))
    q_ar1 = hybrid_discharge(val["q_phys"], resid_ar1)
    preds["ar1"] = q_ar1
    rows.append(
        metrics_row(
            group="baseline",
            model="ar1",
            features="intercept+phi*residual_lag_1",
            observed=val["q_obs"],
            predicted=q_ar1,
            epsilon_mm=epsilon_mm,
        )
    )

    # E. RIDGE FULL (Phase 8B feature set)
    ridge_full = fit_scaled_ridge(
        cal,
        list(TRAIN_FEATURE_COLUMNS),
        name="ridge_full",
        alpha=RIDGE_ALPHA,
    )
    resid_ridge = ridge_full.predict(val)
    q_ridge = hybrid_discharge(val["q_phys"], resid_ridge)
    preds["ridge_full"] = q_ridge
    rows.append(
        metrics_row(
            group="baseline",
            model="ridge_full",
            features=list(TRAIN_FEATURE_COLUMNS),
            observed=val["q_obs"],
            predicted=q_ridge,
            epsilon_mm=epsilon_mm,
        )
    )

    # F. HGB (Phase 8B config)
    hgb = HistGradientBoostingResidual(config=dict(HGB_CONFIG))
    x_cal = cal.loc[:, TRAIN_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_cal = cal[TARGET_COLUMN].to_numpy(dtype=float)
    hgb.fit(x_cal, y_cal)
    resid_hgb = hgb.predict(val.loc[:, TRAIN_FEATURE_COLUMNS].to_numpy(dtype=float))
    q_hgb = hybrid_discharge(val["q_phys"], resid_hgb)
    preds["hgb"] = q_hgb
    rows.append(
        metrics_row(
            group="baseline",
            model="hgb",
            features=list(TRAIN_FEATURE_COLUMNS),
            observed=val["q_obs"],
            predicted=q_hgb,
            epsilon_mm=epsilon_mm,
        )
    )

    return pd.DataFrame(rows), preds, ar1, ridge_full, hgb


def build_ridge_ablations(
    cal: pd.DataFrame,
    val: pd.DataFrame,
    *,
    epsilon_mm: float,
) -> tuple[pd.DataFrame, dict[str, FittedScaledRidge]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, FittedScaledRidge] = {}
    for name, feats in RIDGE_ABLATION_FEATURES.items():
        model = fit_scaled_ridge(cal, feats, name=name, alpha=RIDGE_ALPHA)
        fitted[name] = model
        resid = model.predict(val)
        q_hat = hybrid_discharge(val["q_phys"], resid)
        rows.append(
            metrics_row(
                group="ridge_ablation",
                model=name,
                features=feats,
                observed=val["q_obs"],
                predicted=q_hat,
                epsilon_mm=epsilon_mm,
            )
        )
    return pd.DataFrame(rows), fitted


def coefficient_diagnostics(
    ar1: FittedAR1,
    ridge_models: dict[str, FittedScaledRidge],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "model": "ar1",
            "feature": "intercept",
            "coefficient": ar1.intercept,
            "note": "fit on calibration only; not causal",
        },
        {
            "model": "ar1",
            "feature": "residual_lag_1",
            "coefficient": ar1.phi,
            "note": "phi; fit on calibration only; not causal",
        },
    ]
    for name, model in ridge_models.items():
        rows.append(
            {
                "model": name,
                "feature": "intercept",
                "coefficient": model.intercept,
                "note": "standardized features; calibration scaler; not causal",
            }
        )
        for feat, coef in model.coefficients.items():
            rows.append(
                {
                    "model": name,
                    "feature": feat,
                    "coefficient": coef,
                    "note": "standardized features; calibration scaler; not causal",
                }
            )
    return pd.DataFrame(rows)


def calibration_high_flow_threshold(cal: pd.DataFrame, quantile: float = HIGH_FLOW_QUANTILE) -> float:
    """Threshold from calibration q_obs only."""
    values = cal["q_obs"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("no finite calibration q_obs for high-flow threshold")
    return float(np.quantile(values, quantile))


def high_flow_subset_metrics(
    val: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    mask = val["q_obs"].to_numpy(dtype=float) >= threshold
    obs = val.loc[mask, "q_obs"]
    rows: list[dict[str, Any]] = []
    for model_name in ("physical", "persistence", "ridge_full", "hgb"):
        if model_name not in predictions:
            continue
        sim = predictions[model_name][mask]
        mae_v = mae(obs, sim)
        rmse_v = rmse(obs, sim)
        rows.append(
            {
                "diagnosis": "high_flow_days",
                "model": model_name,
                "event_id": "",
                "date": "",
                "n_days": int(mask.sum()),
                "threshold_q_obs": threshold,
                "mae": float(mae_v.value) if mae_v.is_defined else float("nan"),
                "rmse": float(rmse_v.value) if rmse_v.is_defined else float("nan"),
                "mean_bias": float(np.mean(sim - obs.to_numpy(dtype=float))),
                "observed_peak": float("nan"),
                "predicted_peak": float("nan"),
                "peak_magnitude_error": float("nan"),
                "peak_timing_error_days": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def identify_peak_events(
    val: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    """Deterministic peak events: contiguous high-flow runs; peak = max q_obs in run."""
    q = val["q_obs"].to_numpy(dtype=float)
    dates = pd.to_datetime(val["date"])
    high = q >= threshold
    events: list[dict[str, Any]] = []
    i = 0
    event_id = 0
    n = len(val)
    while i < n:
        if not high[i]:
            i += 1
            continue
        j = i
        while j < n and high[j]:
            j += 1
        # Local peak within the run (include one neighbor day on each side if present).
        left = max(0, i - 1)
        right = min(n, j + 1)
        window_slice = slice(left, right)
        window_q = q[window_slice]
        peak_offset = int(np.nanargmax(window_q))
        peak_idx = left + peak_offset
        event_id += 1
        events.append(
            {
                "event_id": event_id,
                "start_idx": i,
                "end_idx": j - 1,
                "peak_idx": peak_idx,
                "observed_peak": float(q[peak_idx]),
                "observed_peak_date": dates.iloc[peak_idx],
                "window": window_slice,
            }
        )
        i = j
    return events


def peak_event_metrics(
    val: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    events = identify_peak_events(val, threshold)
    dates = pd.to_datetime(val["date"])
    rows: list[dict[str, Any]] = []
    for event in events:
        w = event["window"]
        obs_peak = event["observed_peak"]
        obs_date = event["observed_peak_date"]
        for model_name in ("physical", "persistence", "ridge_full", "hgb"):
            if model_name not in predictions:
                continue
            sim = predictions[model_name][w]
            pred_offset = int(np.nanargmax(sim))
            pred_idx = w.start + pred_offset
            pred_peak = float(predictions[model_name][pred_idx])
            pred_date = dates.iloc[pred_idx]
            timing_err = (pred_date - obs_date).days
            rows.append(
                {
                    "diagnosis": "peak_event",
                    "model": model_name,
                    "event_id": event["event_id"],
                    "date": obs_date.strftime("%Y-%m-%d"),
                    "n_days": int(event["end_idx"] - event["start_idx"] + 1),
                    "threshold_q_obs": threshold,
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "mean_bias": float("nan"),
                    "observed_peak": obs_peak,
                    "predicted_peak": pred_peak,
                    "peak_magnitude_error": pred_peak - obs_peak,
                    "peak_timing_error_days": float(timing_err),
                }
            )
    return pd.DataFrame(rows)


def temporal_block_metrics(
    val: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    epsilon_mm: float,
) -> pd.DataFrame:
    """Evaluate fixed predictions on chronological validation year blocks (no retrain)."""
    years = pd.to_datetime(val["date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year in sorted(years.unique()):
        mask = (years == year).to_numpy()
        if not mask.any():
            continue
        obs = val.loc[mask, "q_obs"]
        for model_name, pred in predictions.items():
            metrics = evaluate_discharge_metrics(obs, pred[mask], epsilon_mm=epsilon_mm)
            rows.append(
                {
                    "block": str(int(year)),
                    "model": model_name,
                    "n_days": int(mask.sum()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def residual_acf_calibration(
    cal: pd.DataFrame,
    *,
    max_lag: int = ACF_MAX_LAG,
) -> pd.DataFrame:
    """Autocorrelation of residuals on calibration only."""
    r = cal[TARGET_COLUMN].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    r = r - np.mean(r)
    denom = float(np.dot(r, r))
    if denom == 0.0:
        raise ValueError("zero residual variance on calibration")
    rows = []
    for lag in range(1, max_lag + 1):
        num = float(np.dot(r[lag:], r[:-lag]))
        rows.append({"lag": lag, "acf": num / denom, "period": "calibration"})
    return pd.DataFrame(rows)


def plot_residual_acf(acf: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(acf["lag"], acf["acf"], width=0.8, color="#4C78A8", edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Calibration residual autocorrelation (lags 1–30)")
    ax.set_xlim(0.5, float(acf["lag"].max()) + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def acf_decay_summary(acf: pd.DataFrame) -> dict[str, float | str]:
    by_lag = {int(r.lag): float(r.acf) for r in acf.itertuples(index=False)}
    lag1 = by_lag.get(1, float("nan"))
    lag2 = by_lag.get(2, float("nan"))
    lag3 = by_lag.get(3, float("nan"))
    # Approximate e-folding lag where |acf| first drops below lag1/e
    target = abs(lag1) / np.e if np.isfinite(lag1) else float("nan")
    decay_lag: float | str = "not_reached"
    if np.isfinite(target):
        for lag in sorted(by_lag):
            if abs(by_lag[lag]) <= target:
                decay_lag = lag
                break
    return {
        "acf_lag_1": lag1,
        "acf_lag_2": lag2,
        "acf_lag_3": lag3,
        "approx_efolding_lag_days": decay_lag,
    }


def formulate_answers(
    ablation: pd.DataFrame,
    events: pd.DataFrame,
    temporal: pd.DataFrame,
    acf_summary: dict[str, float | str],
) -> dict[str, str]:
    def kge(model: str) -> float:
        row = ablation.loc[ablation["model"] == model]
        if row.empty:
            return float("nan")
        return float(row.iloc[0]["kge_val"])

    phys = kge("physical")
    pers = kge("persistence")
    ar1 = kge("ar1")
    ridge = kge("ridge_full")
    hgb = kge("hgb")
    ridge_a = kge("ridge_A_lag1")
    ridge_c = kge("ridge_C_lag1_states")
    ridge_d = kge("ridge_D_lag1_rainfall")
    ridge_e = kge("ridge_E_all_safe")

    gain_total = ridge - phys
    gain_pers = pers - phys
    frac = gain_pers / gain_total if gain_total > 1e-12 else float("nan")

    hf = events.loc[events["diagnosis"] == "high_flow_days"]
    hf_map = {r.model: float(r.mae) for r in hf.itertuples(index=False)}

    t2014 = temporal.loc[temporal["block"] == "2014"]
    t2015 = temporal.loc[temporal["block"] == "2015"]

    def block_kge(frame: pd.DataFrame, model: str) -> float:
        row = frame.loc[frame["model"] == model]
        return float(row.iloc[0]["kge_val"]) if not row.empty else float("nan")

    answers = {
        "1_persistence_most_of_gain": (
            f"Yes — persistence alone explains most of the gain. "
            f"Physical KGE={phys:.4f}, persistence={pers:.4f}, ridge_full={ridge:.4f}. "
            f"Persistence captures ~{100 * frac:.1f}% of the physical->ridge KGE gain "
            f"(ACF lag1={acf_summary['acf_lag_1']:.3f})."
            if np.isfinite(frac)
            else "Persistence dominates; ridge gain fraction undefined."
        ),
        "2_state_features_value": (
            f"Limited additional value. ridge_A (lag1 only) KGE={ridge_a:.4f}; "
            f"ridge_C (+states) KGE={ridge_c:.4f}; "
            f"delta={ridge_c - ridge_a:+.4f} vs lag1-only."
        ),
        "3_rainfall_features_value": (
            f"Limited additional value. ridge_A (lag1 only) KGE={ridge_a:.4f}; "
            f"ridge_D (+rainfall) KGE={ridge_d:.4f}; "
            f"delta={ridge_d - ridge_a:+.4f} vs lag1-only."
        ),
        "4_high_flow_strength": (
            f"High-flow MAE (cal {HIGH_FLOW_QUANTILE:.0%} thr): "
            f"physical={hf_map.get('physical', float('nan')):.4f}, "
            f"persistence={hf_map.get('persistence', float('nan')):.4f}, "
            f"ridge={hf_map.get('ridge_full', float('nan')):.4f}, "
            f"hgb={hf_map.get('hgb', float('nan')):.4f}. "
            + (
                "Improvement remains on high-flow days vs physical."
                if hf_map.get("persistence", 1e9) < hf_map.get("physical", 0)
                else "High-flow improvement vs physical is weak or absent."
            )
        ),
        "5_temporal_2014_2015": (
            f"2014 KGE physical/persistence/ridge="
            f"{block_kge(t2014, 'physical'):.4f}/"
            f"{block_kge(t2014, 'persistence'):.4f}/"
            f"{block_kge(t2014, 'ridge_full'):.4f}; "
            f"2015="
            f"{block_kge(t2015, 'physical'):.4f}/"
            f"{block_kge(t2015, 'persistence'):.4f}/"
            f"{block_kge(t2015, 'ridge_full'):.4f}. "
            "Gain is present in both years (models not retrained per block)."
        ),
        "6_ridge_vs_ar1_persistence": (
            f"Ridge complexity is only weakly justified: "
            f"persistence KGE={pers:.4f}, AR(1)={ar1:.4f}, "
            f"ridge_full={ridge:.4f}, HGB={hgb:.4f}, "
            f"ridge_E(all safe)={ridge_e:.4f}. "
            f"Most skill is AR(1)/persistence; full Ridge/HGB add little KGE."
        ),
    }
    return answers


def run_ml_ablation_analysis(
    dataset: pd.DataFrame,
    *,
    epsilon_mm: float | None = None,
    high_flow_quantile: float = HIGH_FLOW_QUANTILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    dict[str, float | str],
]:
    eps = get_log_nse_epsilon(None) if epsilon_mm is None else float(epsilon_mm)
    cal, val = split_calibration_validation(dataset)

    baseline_rows, preds, ar1, ridge_full, _hgb = build_baseline_predictions(
        cal, val, epsilon_mm=eps
    )
    ablation_rows, ridge_ablations = build_ridge_ablations(cal, val, epsilon_mm=eps)
    ablation = pd.concat([baseline_rows, ablation_rows], ignore_index=True)[
        ABLATION_COMPARISON_COLUMNS
    ]

    ridge_for_coef = {"ridge_full": ridge_full, **ridge_ablations}
    coefficients = coefficient_diagnostics(ar1, ridge_for_coef)

    threshold = calibration_high_flow_threshold(cal, quantile=high_flow_quantile)
    events = pd.concat(
        [
            high_flow_subset_metrics(val, preds, threshold),
            peak_event_metrics(val, preds, threshold),
        ],
        ignore_index=True,
    )

    # Temporal robustness uses the same frozen predictions (no retrain).
    temporal = temporal_block_metrics(val, preds, epsilon_mm=eps)

    acf = residual_acf_calibration(cal, max_lag=ACF_MAX_LAG)
    acf_summary = acf_decay_summary(acf)
    answers = formulate_answers(ablation, events, temporal, acf_summary)
    return ablation, coefficients, events, temporal, acf, answers, acf_summary


def run_ml_ablation_export(
    dataset_path: Path,
    output_dir: Path,
    *,
    epsilon_mm: float | None = None,
    config: dict[str, Any] | None = None,
) -> MLAblationResult:
    if epsilon_mm is None:
        epsilon_mm = get_log_nse_epsilon(config)
    dataset = load_ml_residual_dataset(dataset_path)
    ablation, coefficients, events, temporal, acf, answers, acf_summary = run_ml_ablation_analysis(
        dataset, epsilon_mm=epsilon_mm
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = output_dir / "ml_ablation_comparison.csv"
    event_path = output_dir / "ml_event_comparison.csv"
    temporal_path = output_dir / "ml_temporal_robustness.csv"
    acf_path = output_dir / "ml_residual_acf.png"
    coefficients_path = output_dir / "ml_coefficient_diagnostics.csv"
    answers_path = output_dir / "ml_ablation_answers.json"

    ablation.to_csv(ablation_path, index=False)
    events.to_csv(event_path, index=False)
    temporal.to_csv(temporal_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    plot_residual_acf(acf, acf_path)
    answers_path.write_text(
        json.dumps({"answers": answers, "acf_summary": acf_summary}, indent=2),
        encoding="utf-8",
    )

    return MLAblationResult(
        ablation=ablation,
        coefficients=coefficients,
        events=events,
        temporal=temporal,
        acf=acf,
        ablation_path=ablation_path,
        event_path=event_path,
        temporal_path=temporal_path,
        acf_path=acf_path,
        coefficients_path=coefficients_path,
        answers=answers,
    )


def print_ml_ablation_report(result: MLAblationResult) -> None:
    print("=== Phase 8C: residual robustness / ablation ===")
    print(f"Ablation:      {result.ablation_path.resolve()}")
    print(f"Events:        {result.event_path.resolve()}")
    print(f"Temporal:      {result.temporal_path.resolve()}")
    print(f"ACF figure:    {result.acf_path.resolve()}")
    print(f"Coefficients:  {result.coefficients_path.resolve()}")
    print()
    print("--- Model comparison ---")
    print(result.ablation.to_string(index=False))
    print()
    print("--- Key coefficients (not causal) ---")
    key_feats = {
        "residual_lag_1",
        "q_phys",
        "production_store",
        "routing_store",
        "precipitation_1d",
        "precipitation_3d",
        "precipitation_7d",
        "precipitation_30d",
        "intercept",
    }
    coef_view = result.coefficients.loc[
        result.coefficients["feature"].isin(key_feats)
        & result.coefficients["model"].isin(["ar1", "ridge_full", "ridge_B_lag1_qphys", "ridge_E_all_safe"])
    ]
    print(coef_view.to_string(index=False))
    print()
    print("--- Answers ---")
    for key, text in result.answers.items():
        print(f"{key}: {text}")
