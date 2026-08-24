"""Smoke tests for configuration loading and scaffold integrity."""

from __future__ import annotations

from pathlib import Path


def test_basin_config_has_required_sections(basin_config: dict) -> None:
    assert "station" in basin_config
    assert "periods" in basin_config
    assert "model" in basin_config
    assert "calibration" in basin_config
    assert "data" in basin_config
    assert "meteo_timezone" in basin_config["data"]


def test_basin_config_meteo_timezone(basin_config: dict) -> None:
    assert basin_config["data"]["meteo_timezone"] == "Europe/Paris"


def test_basin_config_metrics_section(basin_config: dict) -> None:
    assert basin_config["metrics"]["log_nse_epsilon_mm"] == 0.01


def test_parameter_bounds_match_spec(basin_config: dict) -> None:
    bounds = basin_config["model"]["parameter_bounds"]
    assert bounds["X1"] == [100, 1200]
    assert bounds["X2"] == [-5, 3]
    assert bounds["X3"] == [20, 300]
    assert bounds["X4"] == [1.1, 2.9]


def test_run_py_exists() -> None:
    assert Path("run.py").is_file()
