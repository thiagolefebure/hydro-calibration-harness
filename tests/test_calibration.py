"""Tests for calibration experiment orchestration (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import RUNS_CSV_COLUMNS, run_calibration_experiment
from src.validation import rank_by_kge_calibration, select_best_calibration_candidate, top_calibration_candidates


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
            "sampler": "latin_hypercube",
            "n_samples": 5,
            "seed": 123,
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


def test_calibration_experiment_one_row_per_sample(
    tiny_config: dict, tiny_data: pd.DataFrame, tmp_path: Path
) -> None:
    result = run_calibration_experiment(tiny_config, tiny_data, tmp_path)
    assert len(result.runs) == 5
    assert result.runs["run_id"].tolist() == [1, 2, 3, 4, 5]


def test_runs_csv_schema(tiny_config: dict, tiny_data: pd.DataFrame, tmp_path: Path) -> None:
    result = run_calibration_experiment(tiny_config, tiny_data, tmp_path)
    written = pd.read_csv(result.runs_csv_path)
    assert list(written.columns) == RUNS_CSV_COLUMNS


def test_calibration_ranking_uses_kge_cal_only(
    tiny_config: dict, tiny_data: pd.DataFrame, tmp_path: Path
) -> None:
    result = run_calibration_experiment(tiny_config, tiny_data, tmp_path)
    ranks = rank_by_kge_calibration(result.runs)
    best_id = int(select_best_calibration_candidate(result.runs)["run_id"])
    assert int(result.runs.loc[ranks.idxmin(), "run_id"]) == best_id


def test_changing_validation_values_does_not_change_calibration_ranking() -> None:
    runs = pd.DataFrame(
        {
            "run_id": [1, 2, 3],
            "kge_cal": [0.8, 0.6, 0.7],
            "kge_val": [0.1, 0.9, 0.2],
        }
    )
    ranks_before = rank_by_kge_calibration(runs)
    runs_modified = runs.copy()
    runs_modified["kge_val"] = [0.99, 0.99, 0.99]
    ranks_after = rank_by_kge_calibration(runs_modified)
    pd.testing.assert_series_equal(ranks_before, ranks_after)


def test_top20_export_has_twenty_or_fewer(
    tiny_config: dict, tiny_data: pd.DataFrame, tmp_path: Path
) -> None:
    result = run_calibration_experiment(tiny_config, tiny_data, tmp_path)
    top = pd.read_csv(result.top20_csv_path)
    expected = top_calibration_candidates(result.runs, n=20)
    assert len(top) == min(20, len(result.runs))
    assert list(top["run_id"]) == list(expected["run_id"])
    assert list(top.columns) == list(expected.columns)


def test_calibration_experiment_warmup_excluded_from_metrics(
    tiny_config: dict, tiny_data: pd.DataFrame, tmp_path: Path
) -> None:
    from src.evaluation import evaluate_temporal_performance, run_continuous_gr4j
    from src.gr4j import GR4JParameters
    from src.validation import period_masks

    result = run_calibration_experiment(tiny_config, tiny_data, tmp_path)
    row = result.runs.iloc[0]
    params = GR4JParameters(X1=row["x1"], X2=row["x2"], X3=row["x3"], X4=row["x4"])
    simulated = run_continuous_gr4j(tiny_data, tiny_config, params)
    observed = tiny_data.loc[simulated.index, "discharge_mm"]
    evaluation = evaluate_temporal_performance(observed, simulated, tiny_config)
    masks = period_masks(observed.index, tiny_config)

    assert evaluation.calibration.nse.n_valid == int(masks["calibration"].sum())
    assert evaluation.validation.nse.n_valid == int(masks["validation"].sum())
    assert evaluation.calibration.nse.n_valid != int(masks["warmup"].sum())


def test_calibration_experiment_preserves_continuous_state(
    tiny_config: dict, tiny_data: pd.DataFrame, tmp_path: Path
) -> None:
    from src.evaluation import evaluate_temporal_performance, run_continuous_gr4j
    from src.gr4j import GR4JParameters, run_gr4j_continuous_periods

    result = run_calibration_experiment(tiny_config, tiny_data, tmp_path)
    row = result.runs.iloc[0]
    params = GR4JParameters(X1=row["x1"], X2=row["x2"], X3=row["x3"], X4=row["x4"])
    inputs = tiny_data[["precipitation_mm", "et0_mm"]]

    full_sim, _ = run_gr4j_continuous_periods(
        inputs,
        params,
        period_bounds=tiny_config["periods"],
    )
    split_idx = tiny_data.index.get_loc(pd.Timestamp(tiny_config["periods"]["calibration"][0]))
    first_sim, mid_state = run_gr4j_continuous_periods(
        inputs.iloc[:split_idx],
        params,
        period_bounds=tiny_config["periods"],
    )
    second_sim, _ = run_gr4j_continuous_periods(
        inputs.iloc[split_idx:],
        params,
        period_bounds=tiny_config["periods"],
        initial_state=mid_state,
    )
    combined = pd.concat([first_sim, second_sim])
    pd.testing.assert_series_equal(full_sim, combined)

    observed = tiny_data.loc[full_sim.index, "discharge_mm"]
    eval_full = evaluate_temporal_performance(observed, full_sim, tiny_config)
    eval_split = evaluate_temporal_performance(observed, combined, tiny_config)
    assert eval_full.calibration.kge.value == pytest.approx(eval_split.calibration.kge.value)
    assert run_continuous_gr4j(tiny_data, tiny_config, params).equals(full_sim)
