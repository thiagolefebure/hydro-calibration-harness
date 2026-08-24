"""Shared pytest fixtures for the hydro-calibration harness."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "basin.yaml"


@pytest.fixture
def basin_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
