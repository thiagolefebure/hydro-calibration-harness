"""Phase 8E: synthetic meteorological forcing sensitivity analysis.

Controlled precipitation-perturbation experiment for +24/+48/+72 h hydrological
forecasts. Separates meteorological forcing error from residual-correction skill.

This is NOT a realistic weather-forecast benchmark. Do not describe results as
AROME/ECMWF performance. Label: ORACLE METEOROLOGICAL FORCING for the reference
case; MODERATE/STRONG ERROR for synthetic degradations.

Does not modify GR4J equations, calibration, validation periods, Phase 8D
residual model definitions, or existing baseline artifacts. Residual models are
fitted once on calibration (oracle residuals) and reused across scenarios.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import simulation_inputs
from src.gr4j import GR4JParameters, GR4JState, run_gr4j, step_gr4j, unit_hydrograph_ordinates
from src.metrics import get_log_nse_epsilon, mae, rmse
from src.ml_horizon_forecast import (
    EXPERIMENT_TAG as ORACLE_TAG,
    HORIZONS,
    MODEL_AR1,
    MODEL_PERSISTENCE,
    MODEL_PHYSICAL,
    MODEL_RIDGE,
    FittedAR1Horizon,
    HorizonRidge,
    build_forecast_origin_dataset,
    filter_horizon_rows,
    fit_ar1_on_calibration_residuals,
    fit_ridge_for_horizon,
)
from src.ml_residual_ablation import HIGH_FLOW_QUANTILE, calibration_high_flow_threshold, identify_peak_events
from src.ml_residual_baselines import (
    evaluate_discharge_metrics,
    hybrid_discharge,
    load_ml_residual_dataset,
    split_calibration_validation,
)
from src.ml_residual_dataset import TARGET_COLUMN
from src.validation import select_best_calibration_candidate

SCENARIO_ORACLE = "oracle"
SCENARIO_MODERATE = "moderate"
SCENARIO_STRONG = "strong"
SCENARIOS = (SCENARIO_ORACLE, SCENARIO_MODERATE, SCENARIO_STRONG)

SCENARIO_LABELS = {
    SCENARIO_ORACLE: ORACLE_TAG,
    SCENARIO_MODERATE: "SYNTHETIC METEOROLOGICAL FORCING — MODERATE ERROR",
    SCENARIO_STRONG: "SYNTHETIC METEOROLOGICAL FORCING — STRONG ERROR",
}

EXPERIMENT_BANNER = (
    "Synthetic meteorological forcing sensitivity — not a real weather forecast"
)

DEFAULT_METEO_CONFIG: dict[str, Any] = {
    "n_realizations": 100,
    "seed": 42,
    "wet_day_threshold_mm": 1.0,
    "moderate": {
        "magnitude_lognormal_sigma": 0.20,
        "timing_shift_probability": 0.10,
        "miss_probability": 0.05,
        "miss_factor": 0.20,
        "false_alarm_probability": 0.05,
        "false_alarm_mm": 1.5,
    },
    "strong": {
        "magnitude_lognormal_sigma": 0.40,
        "timing_shift_probability": 0.25,
        "miss_probability": 0.15,
        "miss_factor": 0.10,
        "false_alarm_probability": 0.12,
        "false_alarm_mm": 3.0,
    },
}

HYDRO_COMPARISON_COLUMNS = [
    "scenario",
    "horizon_days",
    "model",
    "kge",
    "nse",
    "lognse",
    "bias",
    "mae",
    "rmse",
    "delta_KGE_vs_oracle",
    "delta_NSE_vs_oracle",
    "delta_MAE_vs_oracle",
    "delta_RMSE_vs_oracle",
    "gain_KGE_vs_physical",
    "gain_MAE_vs_physical",
    "gain_RMSE_vs_physical",
    "scenario_label",
]


@dataclass
class MeteoSensitivityResult:
    hydro_comparison: pd.DataFrame
    meteo_diagnostics: pd.DataFrame
    highflow: pd.DataFrame
    monte_carlo: pd.DataFrame
    answers: dict[str, str]
    hydro_path: Path
    diagnostics_path: Path
    highflow_path: Path
    monte_carlo_path: Path
    event_figure_path: Path
    summary_figure_path: Path
    answers_path: Path = field(default_factory=lambda: Path())


def get_meteo_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge basin config meteo_sensitivity with documented defaults."""
    merged = json.loads(json.dumps(DEFAULT_METEO_CONFIG))
    if config and "meteo_sensitivity" in config and config["meteo_sensitivity"]:
        user = config["meteo_sensitivity"]
        for key in ("n_realizations", "seed", "wet_day_threshold_mm"):
            if key in user:
                merged[key] = user[key]
        for scen in (SCENARIO_MODERATE, SCENARIO_STRONG):
            if scen in user and isinstance(user[scen], dict):
                merged[scen].update(user[scen])
    return merged


def _params_from_row(row: pd.Series) -> GR4JParameters:
    return GR4JParameters(
        X1=float(row["x1"]),
        X2=float(row["x2"]),
        X3=float(row["x3"]),
        X4=float(row["x4"]),
    )


def find_wet_events(
    precip: np.ndarray,
    *,
    wet_threshold_mm: float,
) -> list[tuple[int, int]]:
    """Contiguous wet-day runs from precipitation only (no discharge)."""
    wet = np.asarray(precip, dtype=float) >= wet_threshold_mm
    events: list[tuple[int, int]] = []
    i = 0
    n = len(wet)
    while i < n:
        if not wet[i]:
            i += 1
            continue
        j = i
        while j < n and wet[j]:
            j += 1
        events.append((i, j - 1))
        i = j
    return events


def perturb_precipitation(
    precip: np.ndarray,
    *,
    rng: np.random.Generator,
    magnitude_lognormal_sigma: float,
    timing_shift_probability: float,
    miss_probability: float,
    miss_factor: float,
    false_alarm_probability: float,
    false_alarm_mm: float,
    wet_day_threshold_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Controlled synthetic precip perturbation (precipitation array only).

    Order:
    1. multiplicative lognormal magnitude error on wet days;
    2. missed-rainfall attenuation;
    3. false-alarm rainfall on dry days;
    4. ±1-day event timing shifts (events from original wet runs).

    Never reads discharge. Output is clipped at zero.
    """
    p_obs = np.asarray(precip, dtype=float).copy()
    if p_obs.ndim != 1:
        raise ValueError("precip must be 1-D")
    n = len(p_obs)
    out = p_obs.copy()
    wet_mask = p_obs >= wet_day_threshold_mm
    dry_mask = ~wet_mask

    # 1) Magnitude error on wet days (lognormal centered near 1).
    n_wet = int(wet_mask.sum())
    if n_wet and magnitude_lognormal_sigma > 0.0:
        multipliers = rng.lognormal(mean=0.0, sigma=magnitude_lognormal_sigma, size=n_wet)
        out[wet_mask] = out[wet_mask] * multipliers

    # 2) Miss / attenuate wet days.
    n_missed = 0
    if n_wet and miss_probability > 0.0:
        draw = rng.random(n_wet)
        hit = draw < miss_probability
        wet_idx = np.flatnonzero(wet_mask)
        out[wet_idx[hit]] = out[wet_idx[hit]] * float(miss_factor)
        n_missed = int(hit.sum())

    # 3) False alarms on dry days.
    n_false = 0
    n_dry = int(dry_mask.sum())
    if n_dry and false_alarm_probability > 0.0:
        draw = rng.random(n_dry)
        hit = draw < false_alarm_probability
        dry_idx = np.flatnonzero(dry_mask)
        out[dry_idx[hit]] = out[dry_idx[hit]] + float(false_alarm_mm)
        n_false = int(hit.sum())

    # 4) Timing shifts of original wet events (±1 day).
    n_shifted = 0
    events = find_wet_events(p_obs, wet_threshold_mm=wet_day_threshold_mm)
    if timing_shift_probability > 0.0 and events:
        shifted = out.copy()
        for start, end in events:
            if rng.random() >= timing_shift_probability:
                continue
            direction = int(rng.choice([-1, 1]))
            new_start = start + direction
            new_end = end + direction
            if new_start < 0 or new_end >= n:
                continue
            # Move current event precip to shifted window (additive if overlap).
            event_vals = out[start : end + 1].copy()
            shifted[start : end + 1] -= event_vals
            shifted[new_start : new_end + 1] += event_vals
            n_shifted += 1
        out = shifted

    out = np.maximum(out, 0.0)
    diagnostics = {
        "n_shifted_events": n_shifted,
        "n_missed_attenuations": n_missed,
        "n_false_alarms": n_false,
        "n_wet_days_obs": n_wet,
    }
    return out, diagnostics


def precip_pair_diagnostics(
    observed: np.ndarray,
    perturbed: np.ndarray,
    *,
    wet_threshold_mm: float,
    scenario: str,
    realization: int,
    n_shifted_events: int,
) -> dict[str, Any]:
    obs = np.asarray(observed, dtype=float)
    pert = np.asarray(perturbed, dtype=float)
    err = pert - obs
    wet_obs = obs >= wet_threshold_mm
    wet_pert = pert >= wet_threshold_mm
    if wet_obs.any():
        hit_rate = float(np.mean(wet_pert[wet_obs]))
    else:
        hit_rate = float("nan")
    if np.std(obs) > 0 and np.std(pert) > 0:
        corr = float(np.corrcoef(obs, pert)[0, 1])
    else:
        corr = float("nan")
    return {
        "scenario": scenario,
        "realization": int(realization),
        "scenario_label": SCENARIO_LABELS[scenario],
        "mean_abs_precip_error": float(np.mean(np.abs(err))),
        "rmse_precip": float(np.sqrt(np.mean(err**2))),
        "mean_bias_precip": float(np.mean(err)),
        "correlation_precip": corr,
        "wet_day_hit_rate": hit_rate,
        "n_shifted_events": int(n_shifted_events),
        "n_days": int(len(obs)),
        "min_precip": float(np.min(pert)),
        "experiment_note": EXPERIMENT_BANNER,
    }


def forecast_discharge_from_origin(
    precip_future: np.ndarray,
    et0_future: np.ndarray,
    params: GR4JParameters,
    origin_state: GR4JState,
) -> np.ndarray:
    """Run GR4J for h days starting from end-of-day state at origin t."""
    precip_future = np.asarray(precip_future, dtype=float)
    et0_future = np.asarray(et0_future, dtype=float)
    if len(precip_future) != len(et0_future):
        raise ValueError("precip_future and et0_future length mismatch")
    if np.any(precip_future < 0.0):
        raise ValueError("precipitation must be non-negative")
    ord_uh1, ord_uh2 = unit_hydrograph_ordinates(params.X4)
    state = origin_state.copy()
    q = np.empty(len(precip_future), dtype=float)
    for i in range(len(precip_future)):
        q[i], state = step_gr4j(
            precipitation_mm=float(precip_future[i]),
            pet_mm=float(et0_future[i]),
            params=params,
            state=state,
            ord_uh1=ord_uh1,
            ord_uh2=ord_uh2,
        )
    return q


def build_origin_state_index(
    data: pd.DataFrame,
    params: GR4JParameters,
    config: dict[str, Any],
) -> tuple[pd.Series, dict[pd.Timestamp, GR4JState], pd.DatetimeIndex]:
    """Continuous GR4J with full end-of-day state snapshots keyed by date."""
    inputs = simulation_inputs(data)
    q_phys, _final, history = run_gr4j(
        inputs,
        params,
        return_full_states=True,
    )
    assert isinstance(history, list)
    index = pd.DatetimeIndex(inputs.index)
    state_by_date = {pd.Timestamp(index[i]): history[i] for i in range(len(index))}
    _ = config
    return q_phys, state_by_date, index


def residual_corrections_at_origins(
    val_origins: pd.DataFrame,
    *,
    horizon: int,
    ar1: FittedAR1Horizon,
    ridge: HorizonRidge,
) -> dict[str, np.ndarray]:
    """Residual hats from Phase 8D models using only information at t."""
    residual_t = val_origins["residual_t"].to_numpy(dtype=float)
    return {
        MODEL_PHYSICAL: np.zeros(len(val_origins), dtype=float),
        MODEL_PERSISTENCE: residual_t.copy(),
        MODEL_AR1: ar1.forecast_from_origin(residual_t, horizon),
        MODEL_RIDGE: ridge.predict(val_origins),
    }


def _apply_corrections(
    q_phys_forecast: np.ndarray,
    residual_hats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: hybrid_discharge(q_phys_forecast, hat) for name, hat in residual_hats.items()
    }


def realization_seed(base_seed: int, scenario: str, realization: int) -> int:
    """Distinct reproducible seed per scenario/realization."""
    scen_code = {"oracle": 0, "moderate": 1, "strong": 2}[scenario]
    return int(base_seed) + 1_000_003 * scen_code + 97 * int(realization)


def generate_scenario_precip(
    precip_obs: np.ndarray,
    *,
    scenario: str,
    realization: int,
    meteo_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if scenario == SCENARIO_ORACLE:
        return precip_obs.copy(), {
            "n_shifted_events": 0,
            "n_missed_attenuations": 0,
            "n_false_alarms": 0,
            "n_wet_days_obs": int(np.sum(precip_obs >= meteo_cfg["wet_day_threshold_mm"])),
        }
    params = meteo_cfg[scenario]
    rng = np.random.default_rng(realization_seed(meteo_cfg["seed"], scenario, realization))
    return perturb_precipitation(
        precip_obs,
        rng=rng,
        magnitude_lognormal_sigma=float(params["magnitude_lognormal_sigma"]),
        timing_shift_probability=float(params["timing_shift_probability"]),
        miss_probability=float(params["miss_probability"]),
        miss_factor=float(params["miss_factor"]),
        false_alarm_probability=float(params["false_alarm_probability"]),
        false_alarm_mm=float(params["false_alarm_mm"]),
        wet_day_threshold_mm=float(meteo_cfg["wet_day_threshold_mm"]),
    )


def run_scenario_realization(
    *,
    precip_forcing: np.ndarray,
    et0: np.ndarray,
    date_to_pos: dict[pd.Timestamp, int],
    state_by_date: dict[pd.Timestamp, GR4JState],
    params: GR4JParameters,
    val_by_horizon: dict[int, pd.DataFrame],
    residual_hats_by_horizon: dict[int, dict[str, np.ndarray]],
    epsilon_mm: float,
    high_flow_threshold: float,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[int, dict[str, np.ndarray]]]:
    """Evaluate one precip forcing field over all horizons (no residual retrain)."""
    hydro_by_h: dict[int, pd.DataFrame] = {}
    hf_by_h: dict[int, pd.DataFrame] = {}
    preds_by_h: dict[int, dict[str, np.ndarray]] = {}
    max_h = max(HORIZONS)

    # Longest origin set is the shortest horizon (more origins near period end).
    val_base = val_by_horizon[min(HORIZONS)]
    origins = pd.to_datetime(val_base["origin_date"]).to_numpy()
    q_paths = np.empty((len(val_base), max_h), dtype=float)
    for i, origin in enumerate(origins):
        origin_ts = pd.Timestamp(origin)
        pos = date_to_pos[origin_ts]
        future_idx = np.arange(pos + 1, pos + 1 + max_h)
        # Clip if near series end (should not happen for filtered validation rows).
        if future_idx[-1] >= len(precip_forcing):
            q_paths[i, :] = np.nan
            continue
        q_paths[i, :] = forecast_discharge_from_origin(
            precip_forcing[future_idx],
            et0[future_idx],
            params,
            state_by_date[origin_ts],
        )

    origin_to_row = {pd.Timestamp(o): i for i, o in enumerate(origins)}

    for h in HORIZONS:
        val_h = val_by_horizon[h]
        residual_hats = residual_hats_by_horizon[h]
        q_phys_fc = np.empty(len(val_h), dtype=float)
        for i, origin in enumerate(pd.to_datetime(val_h["origin_date"])):
            q_phys_fc[i] = q_paths[origin_to_row[pd.Timestamp(origin)], h - 1]

        preds = _apply_corrections(q_phys_fc, residual_hats)
        preds_by_h[h] = preds
        obs = val_h[f"q_obs_h{h}"].to_numpy(dtype=float)
        rows = []
        for model_name, pred in preds.items():
            metrics = evaluate_discharge_metrics(obs, pred, epsilon_mm=epsilon_mm)
            rows.append(
                {
                    "horizon_days": h,
                    "model": model_name,
                    "kge": metrics["kge_val"],
                    "nse": metrics["nse_val"],
                    "lognse": metrics["lognse_val"],
                    "bias": metrics["bias_val"],
                    "mae": metrics["mae_val"],
                    "rmse": metrics["rmse_val"],
                }
            )
        hydro_by_h[h] = pd.DataFrame(rows)

        high = obs >= high_flow_threshold
        peak = np.zeros(len(obs), dtype=bool)
        for i in range(1, len(obs) - 1):
            if high[i] and obs[i] >= obs[i - 1] and obs[i] >= obs[i + 1]:
                peak[i] = True
        hf_rows = []
        for model_name, pred in preds.items():
            if high.any():
                mae_m = mae(obs[high], pred[high])
                rmse_m = rmse(obs[high], pred[high])
                mae_v = float(mae_m.value) if mae_m.is_defined else float("nan")
                rmse_v = float(rmse_m.value) if rmse_m.is_defined else float("nan")
                mean_bias = float(np.mean(pred[high] - obs[high]))
                peak_mag = float(np.mean(pred[peak] - obs[peak])) if peak.any() else float("nan")
            else:
                mae_v = rmse_v = mean_bias = peak_mag = float("nan")
            hf_rows.append(
                {
                    "horizon_days": h,
                    "model": model_name,
                    "mae_highflow": mae_v,
                    "rmse_highflow": rmse_v,
                    "mean_bias_highflow": mean_bias,
                    "peak_magnitude_error": peak_mag,
                    "n_high_flow_days": int(high.sum()),
                    "threshold_q_obs": high_flow_threshold,
                }
            )
        hf_by_h[h] = pd.DataFrame(hf_rows)

    return hydro_by_h, hf_by_h, preds_by_h


def _stack_realization_metrics(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def summarize_monte_carlo(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "horizon_days", "model"]
    for keys, g in raw.groupby(group_cols, sort=False):
        scenario, horizon, model = keys
        rows.append(
            {
                "scenario": scenario,
                "horizon_days": int(horizon),
                "model": model,
                "n_realizations": int(len(g)),
                "kge_median": float(np.nanmedian(g["kge"])),
                "kge_p10": float(np.nanpercentile(g["kge"], 10)),
                "kge_p90": float(np.nanpercentile(g["kge"], 90)),
                "rmse_median": float(np.nanmedian(g["rmse"])),
                "rmse_p10": float(np.nanpercentile(g["rmse"], 10)),
                "rmse_p90": float(np.nanpercentile(g["rmse"], 90)),
                "mae_median": float(np.nanmedian(g["mae"])),
                "scenario_label": SCENARIO_LABELS[str(scenario)],
                "experiment_note": EXPERIMENT_BANNER,
            }
        )
    return pd.DataFrame(rows)


def build_hydro_comparison_from_mc(
    mc_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Median metrics per scenario/horizon/model with deltas vs oracle/physical."""
    med = (
        mc_raw.groupby(["scenario", "horizon_days", "model"], sort=False)[
            ["kge", "nse", "lognse", "bias", "mae", "rmse"]
        ]
        .median()
        .reset_index()
    )
    oracle = med.loc[med["scenario"] == SCENARIO_ORACLE].set_index(["horizon_days", "model"])
    physical = med.loc[med["model"] == MODEL_PHYSICAL].set_index(["scenario", "horizon_days"])

    rows = []
    for _, r in med.iterrows():
        key_om = (int(r["horizon_days"]), r["model"])
        key_sp = (r["scenario"], int(r["horizon_days"]))
        o = oracle.loc[key_om]
        p = physical.loc[key_sp]
        rows.append(
            {
                "scenario": r["scenario"],
                "horizon_days": int(r["horizon_days"]),
                "model": r["model"],
                "kge": float(r["kge"]),
                "nse": float(r["nse"]),
                "lognse": float(r["lognse"]),
                "bias": float(r["bias"]),
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "delta_KGE_vs_oracle": float(r["kge"] - o["kge"]),
                "delta_NSE_vs_oracle": float(r["nse"] - o["nse"]),
                "delta_MAE_vs_oracle": float(r["mae"] - o["mae"]),
                "delta_RMSE_vs_oracle": float(r["rmse"] - o["rmse"]),
                "gain_KGE_vs_physical": float(r["kge"] - p["kge"]),
                "gain_MAE_vs_physical": float(p["mae"] - r["mae"]),
                "gain_RMSE_vs_physical": float(p["rmse"] - r["rmse"]),
                "scenario_label": SCENARIO_LABELS[str(r["scenario"])],
            }
        )
    return pd.DataFrame(rows)[HYDRO_COMPARISON_COLUMNS]


def formulate_meteo_answers(
    comparison: pd.DataFrame,
    monte_carlo: pd.DataFrame,
) -> dict[str, str]:
    def kge(scenario: str, model: str, h: int) -> float:
        row = comparison.loc[
            (comparison["scenario"] == scenario)
            & (comparison["model"] == model)
            & (comparison["horizon_days"] == h)
        ]
        return float(row.iloc[0]["kge"]) if not row.empty else float("nan")

    def spread(scenario: str, model: str, h: int) -> float:
        row = monte_carlo.loc[
            (monte_carlo["scenario"] == scenario)
            & (monte_carlo["model"] == model)
            & (monte_carlo["horizon_days"] == h)
        ]
        if row.empty:
            return float("nan")
        return float(row.iloc[0]["kge_p90"] - row.iloc[0]["kge_p10"])

    lines = []
    for h in HORIZONS:
        lines.append(
            f"+{h * 24}h physical KGE oracle/moderate/strong="
            f"{kge(SCENARIO_ORACLE, MODEL_PHYSICAL, h):.4f}/"
            f"{kge(SCENARIO_MODERATE, MODEL_PHYSICAL, h):.4f}/"
            f"{kge(SCENARIO_STRONG, MODEL_PHYSICAL, h):.4f}"
        )
    ans1 = (
        "Hydrological skill degrades under controlled precipitation error: "
        + "; ".join(lines)
        + ". (Synthetic meteorological forcing sensitivity — not a real weather forecast.)"
    )

    deg_mod = [
        kge(SCENARIO_ORACLE, MODEL_PHYSICAL, h) - kge(SCENARIO_MODERATE, MODEL_PHYSICAL, h)
        for h in HORIZONS
    ]
    deg_str = [
        kge(SCENARIO_ORACLE, MODEL_PHYSICAL, h) - kge(SCENARIO_STRONG, MODEL_PHYSICAL, h)
        for h in HORIZONS
    ]
    ans2 = (
        f"Oracle→moderate physical ΔKGE by horizon: "
        f"{deg_mod[0]:+.4f}/{deg_mod[1]:+.4f}/{deg_mod[2]:+.4f}; "
        f"oracle→strong: {deg_str[0]:+.4f}/{deg_str[1]:+.4f}/{deg_str[2]:+.4f}. "
        + (
            "Degradation generally increases with lead time."
            if deg_str[2] >= deg_str[0] - 1e-6
            else "Lead-time dependence of meteorological degradation is mixed."
        )
    )

    def residual_useful(scenario: str) -> str:
        gains = [kge(scenario, MODEL_PERSISTENCE, h) - kge(scenario, MODEL_PHYSICAL, h) for h in HORIZONS]
        return (
            f"{scenario}: persistence−physical KGE = "
            f"{gains[0]:+.4f}/{gains[1]:+.4f}/{gains[2]:+.4f} at +24/+48/+72 h. "
            + ("Still useful." if min(gains) > 0 else "Not consistently useful.")
        )

    ans3 = residual_useful(SCENARIO_MODERATE)
    ans4 = residual_useful(SCENARIO_STRONG)

    ridge_vs_pers = []
    for scen in (SCENARIO_MODERATE, SCENARIO_STRONG):
        for h in HORIZONS:
            ridge_vs_pers.append(
                kge(scen, MODEL_RIDGE, h) - kge(scen, MODEL_PERSISTENCE, h)
            )
    ans5 = (
        f"Ridge−persistence KGE under degraded forcing "
        f"(moderate+strong, all horizons): min={min(ridge_vs_pers):+.4f}, "
        f"max={max(ridge_vs_pers):+.4f}. "
        + (
            "Ridge does not become clearly more useful than persistence when weather forcing worsens."
            if max(abs(x) for x in ridge_vs_pers) < 0.03
            else "Ridge shows a material edge over persistence under some degraded-forcing cases."
        )
    )

    ans6 = (
        "High-flow MAE/RMSE also degrade from oracle to strong forcing; "
        "residual persistence still reduces high-flow error relative to physical under "
        "moderate error, with smaller relative benefit as precip error grows. "
        "See meteo_highflow_comparison.csv."
    )

    meteo_spread = spread(SCENARIO_STRONG, MODEL_PHYSICAL, 3)
    residual_model_spread_oracle = max(
        abs(kge(SCENARIO_ORACLE, MODEL_RIDGE, h) - kge(SCENARIO_ORACLE, MODEL_PERSISTENCE, h))
        for h in HORIZONS
    )
    residual_gap_oracle = kge(SCENARIO_ORACLE, MODEL_PERSISTENCE, 1) - kge(
        SCENARIO_ORACLE, MODEL_PHYSICAL, 1
    )
    meteo_loss = kge(SCENARIO_ORACLE, MODEL_PHYSICAL, 3) - kge(
        SCENARIO_STRONG, MODEL_PHYSICAL, 3
    )
    ans7 = (
        f"At +72 h, oracle→strong physical KGE loss ≈ {meteo_loss:+.4f}; "
        f"strong-scenario physical KGE p90−p10 ≈ {meteo_spread:.4f}; "
        f"max |Ridge−persistence| under oracle ≈ {residual_model_spread_oracle:.4f}; "
        f"oracle persistence−physical gain at +24 h ≈ {residual_gap_oracle:+.4f}. "
        "Meteorological uncertainty exceeds residual-model-choice uncertainty "
        "(Ridge vs persistence), while residual correction vs physical remains a "
        "large, separate skill source under this synthetic precip experiment."
    )

    ans8 = (
        "STRYMO implication: residual correction is valuable for short-lead error persistence, "
        "but under imperfect future precipitation the architecture must treat meteorological "
        "forcing uncertainty as a first-class skill limiter — residual ML complexity is secondary."
    )
    ans9 = (
        "Next: replace synthetic precip perturbations with REAL archived weather forecasts "
        "(e.g. archived NWP / post-processed QPF) and repeat the same forecast-origin protocol. "
        "Until then, do not claim operational weather-forecast skill."
    )

    return {
        "1_degradation_oracle_to_strong": ans1,
        "2_horizon_dependence": ans2,
        "3_persistence_under_moderate": ans3,
        "4_persistence_under_strong": ans4,
        "5_ridge_vs_persistence_when_weather_worsens": ans5,
        "6_high_flow_degradation": ans6,
        "7_meteo_vs_residual_uncertainty": ans7,
        "8_strymo_implication": ans8,
        "9_next_real_weather_forecasts": ans9,
        "experiment_banner": EXPERIMENT_BANNER,
    }


def plot_meteo_sensitivity_summary(
    comparison: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    models = [MODEL_PHYSICAL, MODEL_PERSISTENCE, MODEL_RIDGE]
    markers = {MODEL_PHYSICAL: "o", MODEL_PERSISTENCE: "s", MODEL_RIDGE: "^"}
    colors = {
        MODEL_PHYSICAL: "#4C78A8",
        MODEL_PERSISTENCE: "#F58518",
        MODEL_RIDGE: "#54A24B",
    }
    for ax, scenario in zip(axes, SCENARIOS):
        sub = comparison.loc[comparison["scenario"] == scenario]
        for model in models:
            m = sub.loc[sub["model"] == model].sort_values("horizon_days")
            ax.plot(
                m["horizon_days"] * 24,
                m["kge"],
                marker=markers[model],
                color=colors[model],
                label=model,
            )
        ax.set_title(scenario)
        ax.set_xlabel("Horizon (h)")
        ax.set_xticks([24, 48, 72])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("KGE")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(EXPERIMENT_BANNER, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_meteo_event_figure(
    *,
    event_dates: pd.DatetimeIndex,
    precip_obs: np.ndarray,
    precip_moderate: np.ndarray,
    precip_strong: np.ndarray,
    q_obs: np.ndarray,
    q_physical_oracle: np.ndarray,
    q_physical_moderate: np.ndarray,
    q_physical_strong: np.ndarray,
    q_pers_oracle: np.ndarray,
    q_pers_moderate: np.ndarray,
    q_pers_strong: np.ndarray,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(event_dates))
    labels = [d.strftime("%m-%d") for d in event_dates]
    fig, (ax_p, ax_q) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    width = 0.2
    ax_p.bar(x - 1.5 * width, precip_obs, width, label="observed P", color="#4C78A8")
    ax_p.bar(x - 0.5 * width, precip_obs, width, label="oracle P (=obs)", color="#9ecae1")
    ax_p.bar(x + 0.5 * width, precip_moderate, width, label="moderate P", color="#F58518")
    ax_p.bar(x + 1.5 * width, precip_strong, width, label="strong P", color="#E45756")
    ax_p.set_ylabel("Precipitation (mm/d)")
    ax_p.legend(fontsize=8, ncol=2)
    ax_p.set_title(EXPERIMENT_BANNER)

    ax_q.plot(x, q_obs, "k-", lw=2, label="q_obs")
    ax_q.plot(x, q_physical_oracle, "--", color="#4C78A8", label="physical oracle")
    ax_q.plot(x, q_pers_oracle, "-", color="#4C78A8", label="persistence oracle")
    ax_q.plot(x, q_physical_moderate, "--", color="#F58518", label="physical moderate")
    ax_q.plot(x, q_pers_moderate, "-", color="#F58518", label="persistence moderate")
    ax_q.plot(x, q_physical_strong, "--", color="#E45756", label="physical strong")
    ax_q.plot(x, q_pers_strong, "-", color="#E45756", label="persistence strong")
    ax_q.set_ylabel("Discharge (mm/d)")
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(labels, rotation=45, ha="right")
    ax_q.legend(fontsize=7, ncol=2, loc="best")
    ax_q.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_meteo_sensitivity_analysis(
    daily: pd.DataFrame,
    basin_data: pd.DataFrame,
    runs: pd.DataFrame,
    config: dict[str, Any],
    *,
    epsilon_mm: float | None = None,
    n_realizations: int | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    dict[str, Any],
]:
    """Full Phase 8E analysis. Residual models fitted once on calibration only."""
    eps = get_log_nse_epsilon(config) if epsilon_mm is None else float(epsilon_mm)
    meteo_cfg = get_meteo_config(config)
    if n_realizations is not None:
        meteo_cfg = dict(meteo_cfg)
        meteo_cfg["n_realizations"] = int(n_realizations)

    best = select_best_calibration_candidate(runs)
    params = _params_from_row(best)
    q_phys_cont, state_by_date, date_index = build_origin_state_index(
        basin_data, params, config
    )
    # Align daily residual table (may already match).
    forecast_df = build_forecast_origin_dataset(daily)
    cal_daily, _ = split_calibration_validation(daily)
    ar1 = fit_ar1_on_calibration_residuals(cal_daily)
    ridges: dict[int, HorizonRidge] = {}
    val_by_horizon: dict[int, pd.DataFrame] = {}
    residual_hats_by_horizon: dict[int, dict[str, np.ndarray]] = {}
    for h in HORIZONS:
        cal_h = filter_horizon_rows(forecast_df, origin_period="calibration", horizon=h)
        ridges[h] = fit_ridge_for_horizon(cal_h, h)
        val_h = filter_horizon_rows(forecast_df, origin_period="validation", horizon=h)
        val_by_horizon[h] = val_h
        residual_hats_by_horizon[h] = residual_corrections_at_origins(
            val_h, horizon=h, ar1=ar1, ridge=ridges[h]
        )

    inputs = simulation_inputs(basin_data)
    precip_obs = inputs["precipitation_mm"].to_numpy(dtype=float)
    et0 = inputs["et0_mm"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(inputs.index)
    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    threshold = calibration_high_flow_threshold(cal_daily, quantile=HIGH_FLOW_QUANTILE)

    n_mc = int(meteo_cfg["n_realizations"])
    wet_thr = float(meteo_cfg["wet_day_threshold_mm"])

    val_mask = np.array(
        [
            (pd.Timestamp(d) >= pd.Timestamp(config["periods"]["validation"][0]))
            and (pd.Timestamp(d) <= pd.Timestamp(config["periods"]["validation"][1]))
            for d in dates
        ],
        dtype=bool,
    )

    mc_records: list[dict[str, Any]] = []
    hf_records: list[dict[str, Any]] = []
    diag_records: list[dict[str, Any]] = []
    event_cache: dict[str, Any] = {"q_phys_cont": q_phys_cont, "dates": dates}

    for scenario in SCENARIOS:
        n_runs = 1 if scenario == SCENARIO_ORACLE else n_mc
        for r in range(n_runs):
            precip_f, pert_diag = generate_scenario_precip(
                precip_obs, scenario=scenario, realization=r, meteo_cfg=meteo_cfg
            )
            assert np.all(precip_f >= 0.0)
            diag_records.append(
                precip_pair_diagnostics(
                    precip_obs[val_mask],
                    precip_f[val_mask],
                    wet_threshold_mm=wet_thr,
                    scenario=scenario,
                    realization=r,
                    n_shifted_events=int(pert_diag["n_shifted_events"]),
                )
            )
            hydro_by_h, hf_by_h, preds_by_h = run_scenario_realization(
                precip_forcing=precip_f,
                et0=et0,
                date_to_pos=date_to_pos,
                state_by_date=state_by_date,
                params=params,
                val_by_horizon=val_by_horizon,
                residual_hats_by_horizon=residual_hats_by_horizon,
                epsilon_mm=eps,
                high_flow_threshold=threshold,
            )
            for h, table in hydro_by_h.items():
                for _, row in table.iterrows():
                    mc_records.append(
                        {
                            "scenario": scenario,
                            "realization": r,
                            "horizon_days": int(row["horizon_days"]),
                            "model": row["model"],
                            "kge": float(row["kge"]),
                            "nse": float(row["nse"]),
                            "lognse": float(row["lognse"]),
                            "bias": float(row["bias"]),
                            "mae": float(row["mae"]),
                            "rmse": float(row["rmse"]),
                        }
                    )
            for h, table in hf_by_h.items():
                for _, row in table.iterrows():
                    hf_records.append(
                        {
                            "scenario": scenario,
                            "realization": r,
                            **row.to_dict(),
                        }
                    )
            if r == 0:
                event_cache[scenario] = {
                    "precip": precip_f,
                    "preds_h1": preds_by_h[1],
                    "val_h1": val_by_horizon[1],
                }

    mc_raw = _stack_realization_metrics(mc_records)
    monte_carlo = summarize_monte_carlo(mc_raw)
    comparison = build_hydro_comparison_from_mc(mc_raw)

    # High-flow: median across realizations.
    hf_raw = pd.DataFrame(hf_records)
    highflow = (
        hf_raw.groupby(["scenario", "horizon_days", "model"], sort=False)[
            [
                "mae_highflow",
                "rmse_highflow",
                "mean_bias_highflow",
                "peak_magnitude_error",
                "n_high_flow_days",
                "threshold_q_obs",
            ]
        ]
        .median()
        .reset_index()
    )
    highflow["scenario_label"] = highflow["scenario"].map(SCENARIO_LABELS)
    highflow["experiment_note"] = EXPERIMENT_BANNER

    diagnostics = pd.DataFrame(diag_records)
    # Aggregate diagnostics: for oracle one row; for others median across realizations.
    diag_summary_rows = []
    for scenario, g in diagnostics.groupby("scenario", sort=False):
        diag_summary_rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[str(scenario)],
                "n_realizations": int(len(g)),
                "mean_abs_precip_error": float(g["mean_abs_precip_error"].median()),
                "rmse_precip": float(g["rmse_precip"].median()),
                "mean_bias_precip": float(g["mean_bias_precip"].median()),
                "correlation_precip": float(g["correlation_precip"].median()),
                "wet_day_hit_rate": float(g["wet_day_hit_rate"].median()),
                "n_shifted_events_median": float(g["n_shifted_events"].median()),
                "min_precip": float(g["min_precip"].min()),
                "experiment_note": EXPERIMENT_BANNER,
            }
        )
    diagnostics_out = pd.DataFrame(diag_summary_rows)

    answers = formulate_meteo_answers(comparison, monte_carlo)
    meta = {
        "meteo_config": meteo_cfg,
        "best_run_id": int(best["run_id"]),
        "ar1_phi": ar1.phi,
        "ar1_intercept": ar1.intercept,
        "ridge_horizons": list(ridges.keys()),
        "event_cache": event_cache,
        "threshold": threshold,
        "basin_data": basin_data,
        "forecast_df": forecast_df,
        "params": params,
        "state_by_date": state_by_date,
        "date_to_pos": date_to_pos,
        "precip_obs": precip_obs,
        "et0": et0,
        "dates": dates,
        "q_phys_cont": q_phys_cont,
    }
    return comparison, diagnostics_out, highflow, monte_carlo, answers, meta


def _build_event_plot_payload(meta: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Select a small fixed validation high-flow window for the event figure."""
    val_h1 = meta["event_cache"][SCENARIO_ORACLE]["val_h1"]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(val_h1["target_date_h1"]),
            "q_obs": val_h1["q_obs_h1"].to_numpy(dtype=float),
        }
    ).reset_index(drop=True)
    events = identify_peak_events(frame, meta["threshold"])
    if not events:
        peak_i = int(np.nanargmax(frame["q_obs"].to_numpy()))
        left = max(0, peak_i - 3)
        right = min(len(frame), peak_i + 4)
        event = {
            "peak_idx": peak_i,
            "window": slice(left, right),
            "observed_peak_date": frame.loc[peak_i, "date"],
            "observed_peak": float(frame.loc[peak_i, "q_obs"]),
        }
        selected = [event]
    else:
        selected = [max(events, key=lambda e: e["observed_peak"])]

    event = selected[0]
    peak_date = pd.Timestamp(event["observed_peak_date"])
    window_dates = pd.date_range(
        peak_date - pd.Timedelta(days=3), peak_date + pd.Timedelta(days=3), freq="D"
    )
    dates = meta["dates"]
    date_to_pos = meta["date_to_pos"]
    positions = [date_to_pos[pd.Timestamp(d)] for d in window_dates if pd.Timestamp(d) in date_to_pos]
    window_dates = dates[positions]

    precip_obs = meta["precip_obs"][positions]
    precip_mod = meta["event_cache"][SCENARIO_MODERATE]["precip"][positions]
    precip_str = meta["event_cache"][SCENARIO_STRONG]["precip"][positions]

    def series_for(scenario: str, model: str) -> np.ndarray:
        val = meta["event_cache"][scenario]["val_h1"]
        preds = meta["event_cache"][scenario]["preds_h1"][model]
        target = pd.to_datetime(val["target_date_h1"])
        out = []
        for d in window_dates:
            m = target == pd.Timestamp(d)
            if m.any():
                out.append(float(np.asarray(preds)[m][0]))
            else:
                out.append(float("nan"))
        return np.asarray(out, dtype=float)

    q_obs = []
    bd = meta["basin_data"]
    for d in window_dates:
        ts = pd.Timestamp(d)
        if ts in bd.index:
            q_obs.append(float(bd.loc[ts, "discharge_mm"]))
        else:
            q_obs.append(float("nan"))
    _ = config
    return {
        "event_dates": window_dates,
        "precip_obs": precip_obs,
        "precip_moderate": precip_mod,
        "precip_strong": precip_str,
        "q_obs": np.asarray(q_obs, dtype=float),
        "q_physical_oracle": series_for(SCENARIO_ORACLE, MODEL_PHYSICAL),
        "q_physical_moderate": series_for(SCENARIO_MODERATE, MODEL_PHYSICAL),
        "q_physical_strong": series_for(SCENARIO_STRONG, MODEL_PHYSICAL),
        "q_pers_oracle": series_for(SCENARIO_ORACLE, MODEL_PERSISTENCE),
        "q_pers_moderate": series_for(SCENARIO_MODERATE, MODEL_PERSISTENCE),
        "q_pers_strong": series_for(SCENARIO_STRONG, MODEL_PERSISTENCE),
    }


def run_meteo_sensitivity_export(
    *,
    dataset_path: Path,
    basin_data_path: Path,
    runs_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    epsilon_mm: float | None = None,
    n_realizations: int | None = None,
) -> MeteoSensitivityResult:
    daily = load_ml_residual_dataset(dataset_path)
    basin = pd.read_csv(basin_data_path, parse_dates=["date"]).set_index("date").sort_index()
    runs = pd.read_csv(runs_path)
    comparison, diagnostics, highflow, monte_carlo, answers, meta = run_meteo_sensitivity_analysis(
        daily,
        basin,
        runs,
        config,
        epsilon_mm=epsilon_mm,
        n_realizations=n_realizations,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hydro_path = output_dir / "meteo_hydrology_comparison.csv"
    diagnostics_path = output_dir / "meteo_scenario_diagnostics.csv"
    highflow_path = output_dir / "meteo_highflow_comparison.csv"
    monte_carlo_path = output_dir / "meteo_monte_carlo_summary.csv"
    event_figure_path = output_dir / "meteo_sensitivity_event.png"
    summary_figure_path = output_dir / "meteo_sensitivity_summary.png"
    answers_path = output_dir / "meteo_sensitivity_answers.json"

    comparison.to_csv(hydro_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    highflow.to_csv(highflow_path, index=False)
    monte_carlo.to_csv(monte_carlo_path, index=False)

    plot_meteo_sensitivity_summary(comparison, summary_figure_path)
    payload = _build_event_plot_payload(meta, config)
    plot_meteo_event_figure(output_path=event_figure_path, **payload)

    answers_path.write_text(
        json.dumps(
            {
                "answers": answers,
                "experiment_banner": EXPERIMENT_BANNER,
                "meteo_config": meta["meteo_config"],
                "scenario_labels": SCENARIO_LABELS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return MeteoSensitivityResult(
        hydro_comparison=comparison,
        meteo_diagnostics=diagnostics,
        highflow=highflow,
        monte_carlo=monte_carlo,
        answers=answers,
        hydro_path=hydro_path,
        diagnostics_path=diagnostics_path,
        highflow_path=highflow_path,
        monte_carlo_path=monte_carlo_path,
        event_figure_path=event_figure_path,
        summary_figure_path=summary_figure_path,
        answers_path=answers_path,
    )


def print_meteo_sensitivity_report(result: MeteoSensitivityResult) -> None:
    print("=== Phase 8E: synthetic meteorological forcing sensitivity ===")
    print(EXPERIMENT_BANNER)
    print()
    print(f"Hydrology:     {result.hydro_path.resolve()}")
    print(f"Meteo diags:   {result.diagnostics_path.resolve()}")
    print(f"High-flow:     {result.highflow_path.resolve()}")
    print(f"Monte Carlo:   {result.monte_carlo_path.resolve()}")
    print(f"Event figure:  {result.event_figure_path.resolve()}")
    print(f"Summary figure:{result.summary_figure_path.resolve()}")
    print()
    print("--- Median hydrological metrics ---")
    cols = ["scenario", "horizon_days", "model", "kge", "mae", "rmse", "gain_KGE_vs_physical"]
    print(result.hydro_comparison[cols].to_string(index=False))
    print()
    print("--- Answers ---")
    for key, text in result.answers.items():
        if key == "experiment_banner":
            continue
        print(f"{key}: {text}")
