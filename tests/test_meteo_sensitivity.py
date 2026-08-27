"""Tests for Phase 8E synthetic meteorological forcing sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.gr4j import GR4JParameters, run_gr4j
from src.meteo_sensitivity import (
    DEFAULT_METEO_CONFIG,
    EXPERIMENT_BANNER,
    ORACLE_TAG,
    SCENARIO_MODERATE,
    SCENARIO_ORACLE,
    SCENARIO_STRONG,
    forecast_discharge_from_origin,
    generate_scenario_precip,
    get_meteo_config,
    perturb_precipitation,
    realization_seed,
    run_meteo_sensitivity_export,
)
from src.ml_horizon_forecast import build_forecast_origin_dataset
from src.ml_residual_dataset import FEATURE_COLUMNS, TARGET_COLUMN


def _tiny_config() -> dict:
    return {
        "periods": {
            "warmup": ["2010-01-01", "2010-01-10"],
            "calibration": ["2010-01-11", "2010-02-20"],
            "validation": ["2010-02-21", "2010-03-31"],
        },
        "metrics": {"log_nse_epsilon_mm": 0.01},
        "meteo_sensitivity": {
            "n_realizations": 3,
            "seed": 7,
            "wet_day_threshold_mm": 1.0,
            "moderate": dict(DEFAULT_METEO_CONFIG["moderate"]),
            "strong": dict(DEFAULT_METEO_CONFIG["strong"]),
        },
    }


def _synthetic_basin(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    precip = rng.uniform(0.0, 8.0, n)
    precip[rng.random(n) < 0.4] = 0.0
    et0 = rng.uniform(0.5, 3.0, n)
    # Rough observed discharge correlated with precip for residual structure.
    q = 0.3 + 0.05 * pd.Series(precip).rolling(3, min_periods=1).mean().to_numpy()
    q = q + rng.normal(0.0, 0.05, n)
    q = np.maximum(q, 0.05)
    return pd.DataFrame(
        {
            "discharge_mm": q,
            "precipitation_mm": precip,
            "et0_mm": et0,
            "discharge_ls": q * 1000.0,
        },
        index=dates,
    )


def _daily_from_basin(basin: pd.DataFrame, config: dict, params: GR4JParameters) -> pd.DataFrame:
    inputs = basin[["precipitation_mm", "et0_mm"]]
    q_phys, _, states = run_gr4j(inputs, params, return_states=True)
    residual = basin["discharge_mm"] - q_phys
    index = basin.index
    doy = index.dayofyear.astype(float)
    angle = 2.0 * np.pi * (doy - 1.0) / 365.25
    periods = config["periods"]

    def label(ts: pd.Timestamp) -> str:
        for name, (start, end) in periods.items():
            if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
                return name
        return "unknown"

    frame = {
        "date": index.strftime("%Y-%m-%d"),
        "period": [label(ts) for ts in index],
        "q_obs": basin["discharge_mm"].to_numpy(dtype=float),
        "q_phys": q_phys.to_numpy(dtype=float),
        TARGET_COLUMN: residual.to_numpy(dtype=float),
        "precipitation_1d": basin["precipitation_mm"].to_numpy(dtype=float),
        "precipitation_3d": basin["precipitation_mm"].rolling(3, min_periods=3).sum().to_numpy(),
        "precipitation_7d": basin["precipitation_mm"].rolling(7, min_periods=7).sum().to_numpy(),
        "precipitation_30d": basin["precipitation_mm"].rolling(30, min_periods=30).sum().to_numpy(),
        "production_store": states["production_store"].to_numpy(dtype=float),
        "routing_store": states["routing_store"].to_numpy(dtype=float),
        "day_of_year_sin": np.sin(angle),
        "day_of_year_cos": np.cos(angle),
        "q_phys_change_1d": q_phys.diff(1).to_numpy(dtype=float),
        "q_obs_lag_1": basin["discharge_mm"].shift(1).to_numpy(dtype=float),
        "residual_lag_1": residual.shift(1).to_numpy(dtype=float),
        "et0_current": basin["et0_mm"].to_numpy(dtype=float),
        "et0_7d_mean": basin["et0_mm"].rolling(7, min_periods=7).mean().to_numpy(),
        "q_obs_change_1d": basin["discharge_mm"].diff(1).to_numpy(dtype=float),
    }
    for col in FEATURE_COLUMNS:
        if col not in frame:
            frame[col] = 0.0
    df = pd.DataFrame(frame)
    # Drop rows with incomplete trailing windows / lags.
    required = [
        TARGET_COLUMN,
        "q_obs",
        "q_phys",
        "precipitation_30d",
        "residual_lag_1",
        "q_obs_lag_1",
        "q_phys_change_1d",
        "et0_7d_mean",
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def test_oracle_forcing_is_identity() -> None:
    precip = np.array([0.0, 2.0, 5.0, 0.0, 1.5])
    out, diag = generate_scenario_precip(
        precip,
        scenario=SCENARIO_ORACLE,
        realization=0,
        meteo_cfg=DEFAULT_METEO_CONFIG,
    )
    np.testing.assert_allclose(out, precip)
    assert diag["n_shifted_events"] == 0
    assert ORACLE_TAG == "ORACLE METEOROLOGICAL FORCING"
    assert "not a real weather forecast" in EXPERIMENT_BANNER.lower()


def test_perturbations_use_precipitation_only_and_non_negative() -> None:
    precip = np.array([0.0, 3.0, 4.0, 0.0, 2.0, 0.0, 6.0, 0.0])
    discharge = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])  # must be ignored
    rng = np.random.default_rng(0)
    out, _ = perturb_precipitation(
        precip,
        rng=rng,
        magnitude_lognormal_sigma=0.2,
        timing_shift_probability=0.5,
        miss_probability=0.2,
        miss_factor=0.1,
        false_alarm_probability=0.2,
        false_alarm_mm=1.0,
        wet_day_threshold_mm=1.0,
    )
    assert np.all(out >= 0.0)
    # Re-run with different discharge-looking noise present only outside API —
    # function signature accepts precip alone, so discharge cannot influence output.
    rng2 = np.random.default_rng(0)
    out2, _ = perturb_precipitation(
        precip,
        rng=rng2,
        magnitude_lognormal_sigma=0.2,
        timing_shift_probability=0.5,
        miss_probability=0.2,
        miss_factor=0.1,
        false_alarm_probability=0.2,
        false_alarm_mm=1.0,
        wet_day_threshold_mm=1.0,
    )
    np.testing.assert_allclose(out, out2)
    _ = discharge


def test_fixed_seed_reproduces_perturbations() -> None:
    precip = np.linspace(0.0, 10.0, 40)
    precip[::3] = 0.0
    cfg = DEFAULT_METEO_CONFIG
    a, _ = generate_scenario_precip(precip, scenario=SCENARIO_MODERATE, realization=2, meteo_cfg=cfg)
    b, _ = generate_scenario_precip(precip, scenario=SCENARIO_MODERATE, realization=2, meteo_cfg=cfg)
    np.testing.assert_allclose(a, b)
    c, _ = generate_scenario_precip(precip, scenario=SCENARIO_MODERATE, realization=3, meteo_cfg=cfg)
    assert not np.allclose(a, c)


def test_moderate_and_strong_parameters_differ() -> None:
    cfg = get_meteo_config(_tiny_config())
    assert cfg["moderate"]["magnitude_lognormal_sigma"] < cfg["strong"]["magnitude_lognormal_sigma"]
    assert cfg["moderate"]["timing_shift_probability"] < cfg["strong"]["timing_shift_probability"]
    precip = np.array([0.0, 5.0, 8.0, 0.0, 3.0, 0.0, 7.0, 0.0, 4.0, 0.0] * 3)
    mod, _ = generate_scenario_precip(precip, scenario=SCENARIO_MODERATE, realization=1, meteo_cfg=cfg)
    strong, _ = generate_scenario_precip(precip, scenario=SCENARIO_STRONG, realization=1, meteo_cfg=cfg)
    # Same realization index but different scenario seeds => different fields.
    assert realization_seed(cfg["seed"], SCENARIO_MODERATE, 1) != realization_seed(
        cfg["seed"], SCENARIO_STRONG, 1
    )
    assert not np.allclose(mod, strong)


def test_forecast_origin_state_unchanged_across_scenarios() -> None:
    basin = _synthetic_basin(60)
    params = GR4JParameters(250.0, 0.0, 80.0, 1.5)
    inputs = basin[["precipitation_mm", "et0_mm"]]
    q_cont, _, history = run_gr4j(inputs, params, return_full_states=True)
    origin_i = 20
    state = history[origin_i]
    precip = inputs["precipitation_mm"].to_numpy()
    et0 = inputs["et0_mm"].to_numpy()
    future = slice(origin_i + 1, origin_i + 4)

    q_oracle = forecast_discharge_from_origin(precip[future], et0[future], params, state)
    # Perturbed precip but identical origin state object values.
    state2 = history[origin_i].copy()
    assert state2.production_store == pytest.approx(state.production_store)
    assert state2.routing_store == pytest.approx(state.routing_store)
    np.testing.assert_allclose(state2.uh1, state.uh1)
    precip_mod = precip.copy()
    precip_mod[future] = precip_mod[future] * 1.3
    q_mod = forecast_discharge_from_origin(precip_mod[future], et0[future], params, state2)
    # Continuous oracle path matches restart with observed precip.
    np.testing.assert_allclose(q_oracle, q_cont.iloc[future].to_numpy(), rtol=0, atol=1e-12)
    # Perturbed precip changes forecast, proving only forcing differs.
    assert not np.allclose(q_oracle, q_mod)


def test_monte_carlo_seeds_are_distinct() -> None:
    seeds = {
        realization_seed(42, scen, r)
        for scen in (SCENARIO_MODERATE, SCENARIO_STRONG)
        for r in range(5)
    }
    assert len(seeds) == 10


def test_export_smoke_and_high_flow_threshold_calibration(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    basin = _synthetic_basin(90)
    params = GR4JParameters(250.0, 0.0, 80.0, 1.5)
    daily = _daily_from_basin(basin, config, params)
    # Ensure enough cal/val rows after dropna.
    assert (daily["period"] == "calibration").sum() > 20
    assert (daily["period"] == "validation").sum() > 20

    dataset_path = tmp_path / "ml_residual_dataset.csv"
    basin_path = tmp_path / "basin_daily.csv"
    runs_path = tmp_path / "runs.csv"
    daily.to_csv(dataset_path, index=False)
    basin.reset_index().rename(columns={"index": "date"}).to_csv(basin_path, index=False)
    # Fix date column name if needed
    basin_out = basin.copy()
    basin_out.index.name = "date"
    basin_out.reset_index().to_csv(basin_path, index=False)

    pd.DataFrame(
        {
            "run_id": [1],
            "x1": [params.X1],
            "x2": [params.X2],
            "x3": [params.X3],
            "x4": [params.X4],
            "kge_cal": [0.9],
            "kge_val": [0.8],
        }
    ).to_csv(runs_path, index=False)

    result = run_meteo_sensitivity_export(
        dataset_path=dataset_path,
        basin_data_path=basin_path,
        runs_path=runs_path,
        output_dir=tmp_path / "out",
        config=config,
        n_realizations=2,
    )
    assert result.hydro_path.is_file()
    assert result.diagnostics_path.is_file()
    assert result.highflow_path.is_file()
    assert result.monte_carlo_path.is_file()
    assert result.event_figure_path.is_file()
    assert result.summary_figure_path.is_file()
    assert set(result.hydro_comparison["scenario"]) == {
        SCENARIO_ORACLE,
        SCENARIO_MODERATE,
        SCENARIO_STRONG,
    }
    # Residual models not retrained per scenario: oracle physical should be finite.
    oracle_phys = result.hydro_comparison.loc[
        (result.hydro_comparison["scenario"] == SCENARIO_ORACLE)
        & (result.hydro_comparison["model"] == "physical")
    ]
    assert oracle_phys["kge"].notna().all()
    # High-flow threshold constant across scenarios (calibration-derived).
    assert result.highflow["threshold_q_obs"].nunique() == 1
    assert "not a real weather forecast" in result.answers["experiment_banner"].lower()


def test_basin_yaml_documents_meteo_sensitivity(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert "meteo_sensitivity" in cfg
    assert cfg["meteo_sensitivity"]["n_realizations"] == 100
    assert cfg["meteo_sensitivity"]["moderate"]["magnitude_lognormal_sigma"] == 0.20
    assert cfg["meteo_sensitivity"]["strong"]["magnitude_lognormal_sigma"] == 0.40
