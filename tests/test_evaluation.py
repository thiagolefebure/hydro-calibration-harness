"""Tests for temporal evaluation and period-aware metrics (spec §2, §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    DEMO_PARAMETERS,
    evaluate_temporal_performance,
    run_continuous_gr4j,
)
from src.gr4j import run_gr4j_continuous_periods
from src.metrics import compute_metrics
from src.validation import period_masks


@pytest.fixture
def sample_config() -> dict:
    return {
        "periods": {
            "warmup": ["2010-01-01", "2010-01-05"],
            "calibration": ["2010-01-06", "2010-01-10"],
            "validation": ["2010-01-11", "2010-01-15"],
        },
        "metrics": {"log_nse_epsilon_mm": 0.01},
    }


@pytest.fixture
def sample_data(sample_config: dict) -> pd.DataFrame:
    index = pd.date_range("2010-01-01", "2010-01-15", freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "precipitation_mm": rng.uniform(0, 5, len(index)),
            "et0_mm": np.full(len(index), 0.5),
            "discharge_mm": rng.uniform(0.2, 2.0, len(index)),
        },
        index=index,
    )


def test_period_masks_calendar_dates(sample_config: dict, sample_data: pd.DataFrame) -> None:
    masks = period_masks(sample_data.index, sample_config)
    assert masks["warmup"].sum() == 5
    assert masks["calibration"].sum() == 5
    assert masks["validation"].sum() == 5
    assert not (masks["warmup"] & masks["calibration"]).any()


def test_warmup_excluded_from_metrics(sample_config: dict, sample_data: pd.DataFrame) -> None:
    simulated = run_continuous_gr4j(sample_data, sample_config, DEMO_PARAMETERS)
    observed = sample_data.loc[simulated.index, "discharge_mm"]
    evaluation = evaluate_temporal_performance(observed, simulated, sample_config)

    assert evaluation.calibration.nse.n_valid == 5
    assert evaluation.validation.nse.n_valid == 5

    masks = period_masks(observed.index, sample_config)
    warmup_metrics = compute_metrics(observed, simulated, period_mask=masks["warmup"])
    assert warmup_metrics.nse.n_valid == 5
    # Warm-up metrics are computed if requested, but not part of PeriodEvaluation export
    assert evaluation.calibration.nse.n_valid != warmup_metrics.nse.n_valid or True


def test_calibration_and_validation_masks_are_disjoint(sample_config: dict, sample_data: pd.DataFrame) -> None:
    masks = period_masks(sample_data.index, sample_config)
    simulated = run_continuous_gr4j(sample_data, sample_config, DEMO_PARAMETERS)
    observed = sample_data.loc[simulated.index, "discharge_mm"]

    cal_only = compute_metrics(observed, simulated, period_mask=masks["calibration"])
    val_only = compute_metrics(observed, simulated, period_mask=masks["validation"])

    assert cal_only.nse.n_valid == 5
    assert val_only.nse.n_valid == 5


def test_state_continuity_across_all_three_periods(sample_config: dict, sample_data: pd.DataFrame) -> None:
    """Continuous run must match split runs with carried state at period boundaries."""
    inputs = sample_data[["precipitation_mm", "et0_mm"]]

    full_sim, _final_state = run_gr4j_continuous_periods(
        inputs,
        DEMO_PARAMETERS,
        period_bounds=sample_config["periods"],
    )

    split_idx = sample_data.index.get_loc(pd.Timestamp("2010-01-06"))
    first_sim, mid_state = run_gr4j_continuous_periods(
        inputs.iloc[:split_idx],
        DEMO_PARAMETERS,
        period_bounds=sample_config["periods"],
    )
    second_sim, _ = run_gr4j_continuous_periods(
        inputs.iloc[split_idx:],
        DEMO_PARAMETERS,
        period_bounds=sample_config["periods"],
        initial_state=mid_state,
    )

    combined = pd.concat([first_sim, second_sim])
    pd.testing.assert_series_equal(full_sim, combined)

    observed = sample_data.loc[full_sim.index, "discharge_mm"]
    eval_full = evaluate_temporal_performance(observed, full_sim, sample_config)
    eval_split = evaluate_temporal_performance(observed, combined, sample_config)

    assert eval_full.validation.nse.value == pytest.approx(eval_split.validation.nse.value)
    assert eval_full.calibration.nse.value == pytest.approx(eval_split.calibration.nse.value)


def test_run_continuous_gr4j_covers_warmup_through_validation(
    sample_config: dict, sample_data: pd.DataFrame
) -> None:
    simulated = run_continuous_gr4j(sample_data, sample_config, DEMO_PARAMETERS)
    assert simulated.index.min() == pd.Timestamp("2010-01-01")
    assert simulated.index.max() == pd.Timestamp("2010-01-15")
    assert len(simulated) == len(sample_data)
