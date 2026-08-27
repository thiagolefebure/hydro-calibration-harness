"""Phase 8B: baseline residual machine learning (no GR4J changes).

Trains residual models on the calibration period only and evaluates hybrid
discharge on the validation period only. Hyperparameters are fixed; validation
is never used for model selection or tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.metrics import (
    get_log_nse_epsilon,
    kge,
    log_nse,
    mae,
    nse,
    rmse,
    volume_bias,
)
from src.ml_residual_dataset import FEATURE_COLUMNS, TARGET_COLUMN

COMPARISON_COLUMNS = [
    "model",
    "kge_val",
    "nse_val",
    "lognse_val",
    "bias_val",
    "mae_val",
    "rmse_val",
]

PREDICTION_COLUMNS = [
    "model",
    "date",
    "q_obs",
    "q_phys",
    "residual_obs",
    "residual_pred",
    "q_hybrid",
    "period",
]

# Contemporaneous q_obs_change_1d = q_obs(t)-q_obs(t-1) recovers q_obs(t) with
# q_obs_lag_1, which trivially determines residual(t)=q_obs(t)-q_phys(t).
EXCLUDED_LEAKY_FEATURES = frozenset({"q_obs_change_1d"})
TRAIN_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in EXCLUDED_LEAKY_FEATURES]

MODEL_PHYSICAL = "physical_gr4j"
MODEL_DUMMY = "dummy_mean_residual"
MODEL_RIDGE = "ridge"
MODEL_HGB = "hist_gradient_boosting"

# Fixed first-experiment configuration (no validation-based tuning).
RIDGE_ALPHA = 1.0
HGB_CONFIG = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 200,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": 42,
}


class ResidualModel(Protocol):
    name: str

    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class DummyMeanResidual:
    """Predict the calibration-period mean residual everywhere."""

    name: str = MODEL_DUMMY
    mean_residual_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        _ = x
        if len(y) == 0:
            raise ValueError("cannot fit dummy residual model on empty target")
        self.mean_residual_ = float(np.mean(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean_residual_, dtype=float)


@dataclass
class RidgeResidual:
    """Ridge residual regressor with calibration-only feature scaling."""

    name: str = MODEL_RIDGE
    alpha: float = RIDGE_ALPHA
    _scaler: StandardScaler | None = None
    _model: Ridge | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(x_scaled, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._scaler is None or self._model is None:
            raise RuntimeError("RidgeResidual must be fit before predict")
        return np.asarray(self._model.predict(self._scaler.transform(x)), dtype=float)


@dataclass
class HistGradientBoostingResidual:
    """HistGradientBoosting residual regressor with fixed hyperparameters."""

    name: str = MODEL_HGB
    config: dict[str, Any] | None = None
    _model: HistGradientBoostingRegressor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        params = dict(HGB_CONFIG if self.config is None else self.config)
        self._model = HistGradientBoostingRegressor(**params)
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HistGradientBoostingResidual must be fit before predict")
        return np.asarray(self._model.predict(x), dtype=float)


@dataclass(frozen=True)
class MLResidualBaselineResult:
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    comparison_path: Path
    predictions_path: Path
    improves_physical: bool
    best_ml_model: str | None
    verdict: str


def load_ml_residual_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"ML residual dataset not found: {path}")
    df = pd.read_csv(path)
    required = set(TRAIN_FEATURE_COLUMNS + [TARGET_COLUMN, "q_obs", "q_phys", "period", "date"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    return df


def split_calibration_validation(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = dataset.loc[dataset["period"] == "calibration"].copy()
    val = dataset.loc[dataset["period"] == "validation"].copy()
    if cal.empty:
        raise ValueError("no calibration rows available for residual ML training")
    if val.empty:
        raise ValueError("no validation rows available for residual ML evaluation")
    return cal, val


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, TRAIN_FEATURE_COLUMNS].to_numpy(dtype=float)


def target_vector(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, TARGET_COLUMN].to_numpy(dtype=float)


def hybrid_discharge(q_phys: np.ndarray | pd.Series, residual_hat: np.ndarray) -> np.ndarray:
    """q_hybrid = q_phys + residual_hat, clipped at zero for discharge metrics."""
    q = np.asarray(q_phys, dtype=float) + np.asarray(residual_hat, dtype=float)
    return np.maximum(q, 0.0)


def evaluate_discharge_metrics(
    observed: pd.Series | np.ndarray,
    simulated: pd.Series | np.ndarray,
    *,
    epsilon_mm: float,
) -> dict[str, float]:
    obs = pd.Series(np.asarray(observed, dtype=float))
    sim = pd.Series(np.asarray(simulated, dtype=float))
    kge_val, _, _, _ = kge(obs, sim)
    nse_val = nse(obs, sim)
    lognse_val = log_nse(obs, sim, epsilon_mm=epsilon_mm)
    bias_val = volume_bias(obs, sim)
    mae_val = mae(obs, sim)
    rmse_val = rmse(obs, sim)
    return {
        "kge_val": float(kge_val.value) if kge_val.is_defined else float("nan"),
        "nse_val": float(nse_val.value) if nse_val.is_defined else float("nan"),
        "lognse_val": float(lognse_val.value) if lognse_val.is_defined else float("nan"),
        "bias_val": float(bias_val.value) if bias_val.is_defined else float("nan"),
        "mae_val": float(mae_val.value) if mae_val.is_defined else float("nan"),
        "rmse_val": float(rmse_val.value) if rmse_val.is_defined else float("nan"),
    }


def _prediction_frame(
    *,
    model_name: str,
    frame: pd.DataFrame,
    residual_pred: np.ndarray,
) -> pd.DataFrame:
    q_phys = frame["q_phys"].to_numpy(dtype=float)
    q_hybrid = hybrid_discharge(q_phys, residual_pred)
    return pd.DataFrame(
        {
            "model": model_name,
            "date": frame["date"].to_numpy(),
            "q_obs": frame["q_obs"].to_numpy(dtype=float),
            "q_phys": q_phys,
            "residual_obs": frame[TARGET_COLUMN].to_numpy(dtype=float),
            "residual_pred": np.asarray(residual_pred, dtype=float),
            "q_hybrid": q_hybrid,
            "period": frame["period"].to_numpy(),
        }
    )[PREDICTION_COLUMNS]


def build_default_models() -> list[ResidualModel]:
    return [
        DummyMeanResidual(),
        RidgeResidual(alpha=RIDGE_ALPHA),
        HistGradientBoostingResidual(config=dict(HGB_CONFIG)),
    ]


def run_residual_baselines(
    dataset: pd.DataFrame,
    *,
    epsilon_mm: float | None = None,
    models: list[ResidualModel] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train on calibration only; score hybrid discharge on validation only."""
    eps = get_log_nse_epsilon(None) if epsilon_mm is None else float(epsilon_mm)
    cal, val = split_calibration_validation(dataset)
    x_cal = feature_matrix(cal)
    y_cal = target_vector(cal)
    x_val = feature_matrix(val)

    fitted = models if models is not None else build_default_models()
    comparison_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    # Physical GR4J baseline (no residual correction).
    phys_metrics = evaluate_discharge_metrics(
        val["q_obs"],
        val["q_phys"],
        epsilon_mm=eps,
    )
    comparison_rows.append({"model": MODEL_PHYSICAL, **phys_metrics})
    prediction_parts.append(
        _prediction_frame(
            model_name=MODEL_PHYSICAL,
            frame=val,
            residual_pred=np.zeros(len(val), dtype=float),
        )
    )

    for model in fitted:
        model.fit(x_cal, y_cal)
        residual_hat = model.predict(x_val)
        q_hat = hybrid_discharge(val["q_phys"], residual_hat)
        metrics = evaluate_discharge_metrics(val["q_obs"], q_hat, epsilon_mm=eps)
        comparison_rows.append({"model": model.name, **metrics})
        prediction_parts.append(
            _prediction_frame(model_name=model.name, frame=val, residual_pred=residual_hat)
        )

    comparison = pd.DataFrame.from_records(comparison_rows)[COMPARISON_COLUMNS]
    predictions = pd.concat(prediction_parts, ignore_index=True)[PREDICTION_COLUMNS]
    meta = {
        "n_calibration": int(len(cal)),
        "n_validation": int(len(val)),
        "feature_columns": list(TRAIN_FEATURE_COLUMNS),
        "excluded_leaky_features": sorted(EXCLUDED_LEAKY_FEATURES),
        "ridge_alpha": RIDGE_ALPHA,
        "hgb_config": dict(HGB_CONFIG),
        "tuning": "none — fixed configuration; validation unused for selection",
    }
    return comparison, predictions, meta


def verdict_from_comparison(comparison: pd.DataFrame) -> tuple[bool, str | None, str]:
    """State explicitly whether any ML model beats physical KGE on validation."""
    if comparison.empty or MODEL_PHYSICAL not in set(comparison["model"]):
        return False, None, "Physical baseline metrics unavailable."

    phys = comparison.loc[comparison["model"] == MODEL_PHYSICAL].iloc[0]
    phys_kge = float(phys["kge_val"])
    ml = comparison.loc[comparison["model"] != MODEL_PHYSICAL].copy()
    if ml.empty or not np.isfinite(phys_kge):
        return False, None, "No ML models available for comparison against physical GR4J."

    ml = ml.loc[np.isfinite(ml["kge_val"])]
    if ml.empty:
        return False, None, "ML validation KGE undefined; no improvement claim."

    best_row = ml.loc[ml["kge_val"].idxmax()]
    best_name = str(best_row["model"])
    best_kge = float(best_row["kge_val"])
    if best_kge > phys_kge:
        return (
            True,
            best_name,
            (
                f"Validation KGE improved vs physical GR4J: "
                f"{best_name} KGE_val={best_kge:.4f} > physical KGE_val={phys_kge:.4f}."
            ),
        )
    return (
        False,
        best_name,
        (
            f"ML does not improve the physical GR4J baseline on validation KGE: "
            f"best ML ({best_name}) KGE_val={best_kge:.4f} vs physical KGE_val={phys_kge:.4f}."
        ),
    )


def save_residual_baseline_outputs(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "ml_model_comparison.csv"
    predictions_path = output_dir / "ml_residual_predictions.csv"
    comparison.to_csv(comparison_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    return comparison_path, predictions_path


def run_ml_residual_baselines_export(
    dataset_path: Path,
    output_dir: Path,
    *,
    epsilon_mm: float | None = None,
    config: dict[str, Any] | None = None,
) -> MLResidualBaselineResult:
    """Load Phase 8A dataset, train baselines, write comparison artifacts."""
    if epsilon_mm is None:
        epsilon_mm = get_log_nse_epsilon(config)
    dataset = load_ml_residual_dataset(dataset_path)
    comparison, predictions, _meta = run_residual_baselines(dataset, epsilon_mm=epsilon_mm)
    comparison_path, predictions_path = save_residual_baseline_outputs(
        comparison, predictions, output_dir
    )
    improves, best_ml, verdict = verdict_from_comparison(comparison)
    return MLResidualBaselineResult(
        comparison=comparison,
        predictions=predictions,
        comparison_path=comparison_path,
        predictions_path=predictions_path,
        improves_physical=improves,
        best_ml_model=best_ml,
        verdict=verdict,
    )


def print_ml_residual_baselines_report(result: MLResidualBaselineResult) -> None:
    print("=== Phase 8B: residual ML baselines ===")
    print(f"Comparison:  {result.comparison_path.resolve()}")
    print(f"Predictions: {result.predictions_path.resolve()}")
    print()
    print(result.comparison.to_string(index=False))
    print()
    print(result.verdict)
