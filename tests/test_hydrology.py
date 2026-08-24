"""Tests for hydrological period characterization (Phase 3B)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.hydrology import (
    annual_summaries,
    hydrological_summary_table,
    load_processed_daily,
    summarize_period,
)


@pytest.fixture
def sample_daily() -> pd.DataFrame:
    index = pd.date_range("2010-01-01", "2011-12-31", freq="D")
    return pd.DataFrame(
        {
            "precipitation_mm": [0.0, 2.0, 5.0] + [1.0] * (len(index) - 3),
            "discharge_mm": [0.5, 1.0, 2.0] + [0.8] * (len(index) - 3),
        },
        index=index,
    )


def test_summarize_period_wet_days_and_runoff_ratio(sample_daily: pd.DataFrame) -> None:
    row = summarize_period(sample_daily, label="2010", start="2010-01-01", end="2010-12-31")
    assert row["wet_days_p_gt_1mm"] >= 2
    assert row["runoff_ratio_qp"] == pytest.approx(
        row["annual_observed_discharge_depth_mm"] / row["annual_precipitation_mm"]
    )
    assert row["date_of_max_discharge"] is not None


def test_annual_summaries_one_row_per_year(sample_daily: pd.DataFrame) -> None:
    annual = annual_summaries(sample_daily, [2010, 2011])
    assert len(annual) == 2
    assert set(annual["period"]) == {"2010", "2011"}


def test_hydrological_summary_includes_calibration_validation(sample_daily: pd.DataFrame) -> None:
    config = {
        "periods": {
            "warmup": ["2010-01-01", "2010-12-31"],
            "calibration": ["2011-01-01", "2011-12-31"],
            "validation": ["2011-01-01", "2011-12-31"],
        }
    }
    summary = hydrological_summary_table(sample_daily, config)
    assert "calibration" in summary["period"].values
    assert "validation" in summary["period"].values


def test_load_processed_daily_roundtrip(tmp_path: Path, sample_daily: pd.DataFrame) -> None:
    path = tmp_path / "basin_daily.csv"
    out = sample_daily.copy()
    out.index.name = "date"
    out.reset_index().to_csv(path, index=False)
    loaded = load_processed_daily(path)
    assert len(loaded) == len(sample_daily)
