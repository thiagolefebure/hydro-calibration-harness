"""Tests for GLUE-inspired behavioral ensemble (Phase 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ensemble import (
    BEHAVIORAL_RUNS_COLUMNS,
    ENSEMBLE_TIMESERIES_COLUMNS,
    build_ensemble_timeseries,
    empirical_validation_coverage,
    ensemble_quantiles,
    envelope_width_diagnostics,
    envelope_width_series,
    get_behavioral_threshold,
    run_member_simulations,
    select_behavioral_members,
)
from src.evaluation import run_continuous_gr4j
from src.gr4j import GR4JParameters, run_gr4j_continuous_periods
from src.validation import period_masks


@pytest.fixture
def tiny_config() -> dict:
    return {
        "periods": {
            "warmup": ["2010-01-01", "2010-01-03"],
            "calibration": ["2010-01-04", "2010-01-07"],
            "validation": ["2010-01-08", "2010-01-10"],
        },
        "model": {
            "parameter_bounds": {
                "X1": [200.0, 400.0],
                "X2": [-1.0, 1.0],
                "X3": [50.0, 150.0],
                "X4": [1.2, 2.0],
            }
        },
        "calibration": {
            "behavioral_kge_threshold": 0.80,
        },
        "metrics": {"log_nse_epsilon_mm": 0.01},
    }


@pytest.fixture
def tiny_data() -> pd.DataFrame:
    index = pd.date_range("2010-01-01", "2010-01-10", freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "precipitation_mm": rng.uniform(0, 5, len(index)),
            "et0_mm": np.full(len(index), 0.5),
            "discharge_mm": rng.uniform(0.2, 2.0, len(index)),
        },
        index=index,
    )


@pytest.fixture
def sample_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 2, 3, 4],
            "x1": [250.0, 300.0, 350.0, 280.0],
            "x2": [0.0, -0.5, 0.5, -0.2],
            "x3": [80.0, 90.0, 100.0, 85.0],
            "x4": [1.5, 1.6, 1.7, 1.55],
            "kge_cal": [0.85, 0.75, 0.82, 0.60],
            "kge_val": [0.10, 0.99, 0.20, 0.95],
            "nse_cal": [0.8, 0.7, 0.75, 0.5],
            "r_cal": [0.9, 0.8, 0.85, 0.7],
            "alpha_cal": [1.0, 1.0, 1.0, 1.0],
            "beta_cal": [1.0, 1.0, 1.0, 1.0],
            "lognse_cal": [0.7, 0.6, 0.65, 0.4],
            "bias_cal": [0.0, 0.0, 0.0, 0.0],
            "nse_val": [0.5, 0.5, 0.5, 0.5],
            "r_val": [0.5, 0.5, 0.5, 0.5],
            "alpha_val": [1.0, 1.0, 1.0, 1.0],
            "beta_val": [1.0, 1.0, 1.0, 1.0],
            "lognse_val": [0.5, 0.5, 0.5, 0.5],
            "bias_val": [0.0, 0.0, 0.0, 0.0],
            "rank_kge_cal": [1, 3, 2, 4],
        }
    )


def test_threshold_from_config(tiny_config: dict) -> None:
    assert get_behavioral_threshold(tiny_config) == 0.80


def test_ensemble_membership_uses_kge_cal_only(sample_runs: pd.DataFrame) -> None:
    members = select_behavioral_members(sample_runs, threshold=0.80)
    assert set(members["run_id"]) == {1, 3}


def test_changing_kge_val_does_not_change_membership(sample_runs: pd.DataFrame) -> None:
    before = select_behavioral_members(sample_runs, threshold=0.80)
    modified = sample_runs.copy()
    modified["kge_val"] = 0.0
    after = select_behavioral_members(modified, threshold=0.80)
    assert set(before["run_id"]) == set(after["run_id"])


def test_quantile_ordering() -> None:
    rng = np.random.default_rng(1)
    sims = pd.DataFrame(
        rng.uniform(0.5, 2.0, (20, 15)),
        index=pd.date_range("2010-01-01", periods=20, freq="D"),
    )
    q = ensemble_quantiles(sims)
    assert (q["q05"] <= q["q50"]).all()
    assert (q["q50"] <= q["q95"]).all()


def test_empirical_coverage_synthetic_known_example() -> None:
    ts = pd.DataFrame(
        {
            "date": ["2010-01-08", "2010-01-09", "2010-01-10", "2010-01-11"],
            "q_obs": [1.0, 3.0, 2.0, np.nan],
            "q_best_cal": [1.0, 1.0, 1.0, 1.0],
            "q05": [0.5, 1.5, 1.0, 0.5],
            "q50": [1.0, 2.0, 1.5, 1.0],
            "q95": [1.5, 2.5, 2.0, 1.5],
            "period": ["validation"] * 4,
        }
    )
    assert empirical_validation_coverage(ts) == pytest.approx(2 / 3)


def test_envelope_width_calculation() -> None:
    ts = pd.DataFrame(
        {
            "q_obs": [1.0, 2.0],
            "q05": [0.0, 1.0],
            "q50": [0.5, 1.5],
            "q95": [1.0, 2.0],
            "period": ["validation", "validation"],
        }
    )
    width = envelope_width_series(ts)
    assert width.tolist() == [1.0, 1.0]
    stats = envelope_width_diagnostics(
        pd.DataFrame(
            {
                **ts.to_dict("list"),
                "date": ["2010-01-08", "2010-01-09"],
                "q_best_cal": [0.5, 1.5],
            }
        )
    )
    assert stats["mean"] == pytest.approx(1.0)


def test_behavioral_member_count(sample_runs: pd.DataFrame) -> None:
    assert len(select_behavioral_members(sample_runs, 0.80)) == 2


def test_ensemble_timeseries_schema(
    tiny_config: dict, tiny_data: pd.DataFrame, sample_runs: pd.DataFrame
) -> None:
    members = sample_runs.iloc[:2]
    sims = run_member_simulations(tiny_data, tiny_config, members)
    q = ensemble_quantiles(sims)
    observed = tiny_data.loc[q.index, "discharge_mm"]
    best = run_continuous_gr4j(tiny_data, tiny_config, GR4JParameters(250, 0, 80, 1.5))
    ts = build_ensemble_timeseries(observed, best, q, tiny_config)
    assert list(ts.columns) == ENSEMBLE_TIMESERIES_COLUMNS
    assert set(ts["period"].unique()) >= {"warmup", "calibration", "validation"}


def test_continuous_gr4j_state_in_ensemble(
    tiny_config: dict, tiny_data: pd.DataFrame, sample_runs: pd.DataFrame
) -> None:
    row = sample_runs.iloc[0]
    params = GR4JParameters(row.x1, row.x2, row.x3, row.x4)
    inputs = tiny_data[["precipitation_mm", "et0_mm"]]
    full, _ = run_gr4j_continuous_periods(inputs, params, period_bounds=tiny_config["periods"])
    split_idx = tiny_data.index.get_loc(pd.Timestamp(tiny_config["periods"]["calibration"][0]))
    first, mid = run_gr4j_continuous_periods(
        inputs.iloc[:split_idx], params, period_bounds=tiny_config["periods"]
    )
    second, _ = run_gr4j_continuous_periods(
        inputs.iloc[split_idx:],
        params,
        period_bounds=tiny_config["periods"],
        initial_state=mid,
    )
    pd.testing.assert_series_equal(full, pd.concat([first, second]))
    assert run_continuous_gr4j(tiny_data, tiny_config, params).equals(full)


def test_behavioral_runs_output_schema(sample_runs: pd.DataFrame) -> None:
    members = select_behavioral_members(sample_runs, 0.80)
    exported = members[BEHAVIORAL_RUNS_COLUMNS]
    assert list(exported.columns) == BEHAVIORAL_RUNS_COLUMNS


def test_run_ensemble_analysis_integration(
    tiny_config: dict, tiny_data: pd.DataFrame, sample_runs: pd.DataFrame, tmp_path: Path
) -> None:
    from src.ensemble import run_ensemble_analysis

    tiny_config["calibration"]["behavioral_kge_threshold"] = 0.80
    result = run_ensemble_analysis(tiny_config, tiny_data, sample_runs, tmp_path / "ens")
    assert result.n_members == 2
    assert (tmp_path / "ens" / "behavioral_runs.csv").is_file()
    assert (tmp_path / "ens" / "ensemble_timeseries.csv").is_file()
    assert (tmp_path / "ens" / "ensemble_validation.png").is_file()
