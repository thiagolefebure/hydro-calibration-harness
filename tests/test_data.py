"""Tests for data loading, conversion, alignment, and QA (spec §1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    HubEauDuplicateConflictError,
    align_timeseries,
    fetch_meteo,
    get_meteo_timezone,
    ls_to_mm_day,
    parse_hubeau_obs_elab,
    parse_open_meteo_daily,
    qa_summary,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def hubeau_records() -> list[dict]:
    payload = json.loads((FIXTURES / "hubeau_obs_elab_sample.json").read_text(encoding="utf-8"))
    return payload["data"]


@pytest.fixture
def openmeteo_payload() -> dict:
    return json.loads((FIXTURES / "openmeteo_sample.json").read_text(encoding="utf-8"))


def test_ls_to_mm_day_conversion() -> None:
    """Verify spec §1 conversion formula: Q_mm/d = Q_L/s × 0.0864 / A_km²."""
    q_ls = pd.Series([100.0])
    area_km2 = 500.0
    expected = 100.0 * 0.0864 / 500.0
    result = ls_to_mm_day(q_ls, area_km2)
    assert result.iloc[0] == pytest.approx(expected)


def test_ls_to_mm_day_rejects_invalid_area() -> None:
    q_ls = pd.Series([100.0])
    with pytest.raises(ValueError, match="basin_area_km2"):
        ls_to_mm_day(q_ls, 0)


def test_get_meteo_timezone_from_config(basin_config: dict) -> None:
    assert get_meteo_timezone(basin_config) == "Europe/Paris"


def test_get_meteo_timezone_defaults_to_utc() -> None:
    assert get_meteo_timezone({}) == "UTC"


def test_parse_hubeau_identical_duplicates_prefers_station_code(hubeau_records: list[dict]) -> None:
    df, audit = parse_hubeau_obs_elab(hubeau_records)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2011-01-01", "2011-01-02", "2011-01-04"]
    assert df.loc["2011-01-01", "discharge_ls"] == pytest.approx(1000.0)
    assert audit["identical_duplicate_dates_resolved"] == 1
    assert audit["conflicting_dates"] == []


def test_parse_hubeau_identical_duplicates_with_null_station_code() -> None:
    records = [
        {
            "code_station": None,
            "date_obs_elab": "2011-06-01",
            "resultat_obs_elab": 500.0,
        },
        {
            "code_station": "H020302002",
            "date_obs_elab": "2011-06-01",
            "resultat_obs_elab": 500.0,
        },
    ]
    df, audit = parse_hubeau_obs_elab(records)
    assert len(df) == 1
    assert df.loc["2011-06-01", "discharge_ls"] == pytest.approx(500.0)
    assert audit["identical_duplicate_dates_resolved"] == 1


def test_parse_hubeau_conflicting_duplicates_raises() -> None:
    records = [
        {
            "code_station": "H020302002",
            "date_obs_elab": "2011-06-01",
            "resultat_obs_elab": 500.0,
        },
        {
            "code_station": None,
            "date_obs_elab": "2011-06-01",
            "resultat_obs_elab": 600.0,
        },
    ]
    with pytest.raises(HubEauDuplicateConflictError) as exc_info:
        parse_hubeau_obs_elab(records)

    err = exc_info.value
    assert len(err.conflicts) == 1
    assert err.conflicts[0]["date"] == "2011-06-01"
    assert set(err.conflicts[0]["values"]) == {500.0, 600.0}


def test_parse_hubeau_obs_elab_empty() -> None:
    df, audit = parse_hubeau_obs_elab([])
    assert df.empty
    assert audit["identical_duplicate_dates_resolved"] == 0


def test_fetch_meteo_timezone_propagated_to_api_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_http_get_json(
        url: str,
        params: dict,
        cache_path: Path,
        *,
        force_refresh: bool = False,
    ) -> dict:
        captured["url"] = url
        captured["params"] = params
        return {
            "daily": {
                "time": ["2011-01-01"],
                "precipitation_sum": [0.0],
                "et0_fao_evapotranspiration": [0.5],
            }
        }

    monkeypatch.setattr("src.data._http_get_json", fake_http_get_json)

    fetch_meteo(
        lat=47.961780,
        lon=4.361000,
        start="2011-01-01",
        end="2011-01-01",
        cache_dir=tmp_path,
        timezone="Europe/Paris",
    )

    assert captured["params"]["timezone"] == "Europe/Paris"


def test_fetch_meteo_cache_key_includes_timezone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_paths: list[Path] = []

    def fake_http_get_json(
        url: str,
        params: dict,
        cache_path: Path,
        *,
        force_refresh: bool = False,
    ) -> dict:
        cache_paths.append(cache_path)
        return {
            "daily": {
                "time": ["2011-01-01"],
                "precipitation_sum": [0.0],
                "et0_fao_evapotranspiration": [0.5],
            }
        }

    monkeypatch.setattr("src.data._http_get_json", fake_http_get_json)

    fetch_meteo(47.0, 4.0, "2011-01-01", "2011-01-01", tmp_path, timezone="UTC")
    fetch_meteo(47.0, 4.0, "2011-01-01", "2011-01-01", tmp_path, timezone="Europe/Paris")

    assert len(cache_paths) == 2
    assert cache_paths[0] != cache_paths[1]


def test_parse_open_meteo_daily(openmeteo_payload: dict) -> None:
    df = parse_open_meteo_daily(openmeteo_payload)
    assert len(df) == 4
    assert df.loc["2011-01-02", "precipitation_mm"] == pytest.approx(5.2)
    assert pd.isna(df.loc["2011-01-03", "precipitation_mm"])
    assert pd.isna(df.loc["2011-01-04", "et0_mm"])


def test_align_timeseries_preserves_gaps(hubeau_records: list[dict], openmeteo_payload: dict) -> None:
    discharge, _ = parse_hubeau_obs_elab(hubeau_records)
    meteo = parse_open_meteo_daily(openmeteo_payload)
    aligned = align_timeseries(discharge, meteo, "2011-01-01", "2011-01-04")

    assert len(aligned) == 4
    assert pd.isna(aligned.loc["2011-01-03", "discharge_ls"])
    assert aligned.loc["2011-01-02", "precipitation_mm"] == pytest.approx(5.2)
    assert pd.isna(aligned.loc["2011-01-04", "et0_mm"])
    assert pd.isna(aligned.loc["2011-01-03", "discharge_ls"])


def test_align_timeseries_adds_missing_calendar_days(hubeau_records: list[dict], openmeteo_payload: dict) -> None:
    discharge, _ = parse_hubeau_obs_elab(hubeau_records)
    meteo = parse_open_meteo_daily(openmeteo_payload)
    aligned = align_timeseries(discharge, meteo, "2011-01-01", "2011-01-05")

    assert len(aligned) == 5
    assert pd.isna(aligned.loc["2011-01-05", "discharge_ls"])
    assert pd.isna(aligned.loc["2011-01-05", "precipitation_mm"])


def test_qa_summary_missing_counts(basin_config: dict) -> None:
    index = pd.date_range("2011-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "discharge_ls": [100.0, float("nan"), 120.0, 130.0],
            "discharge_mm": [0.1, float("nan"), 0.12, 0.13],
            "precipitation_mm": [0.0, 1.0, float("nan"), 2.0],
            "et0_mm": [0.5, 0.5, 0.5, 0.5],
        },
        index=index,
    )
    summary = qa_summary(df, basin_config, meteo_timezone="Europe/Paris")

    assert summary["meteo_timezone"] == "Europe/Paris"
    assert summary["variables"]["discharge_ls"]["n_missing"] == 1
    assert summary["variables"]["precipitation_mm"]["n_missing"] == 1
    assert summary["usable_period"]["n_days"] == 2


def test_qa_summary_usable_period_intersection(basin_config: dict) -> None:
    index = pd.date_range("2011-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "discharge_ls": [100.0, 100.0, 100.0],
            "discharge_mm": [0.1, float("nan"), 0.1],
            "precipitation_mm": [0.0, 1.0, 1.0],
            "et0_mm": [0.5, 0.5, 0.5],
        },
        index=index,
    )
    summary = qa_summary(df, basin_config, meteo_timezone="Europe/Paris")
    assert summary["usable_period"]["n_days"] == 2
