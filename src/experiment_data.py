"""Load experiment data for calibration runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data import get_meteo_timezone, load_basin_data
from src.hydrology import load_processed_daily


def load_experiment_data(
    config: dict[str, Any],
    cache_dir: Path,
    processed_path: Path,
    *,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Prefer processed daily CSV; fall back to live data pipeline if missing."""
    if processed_path.is_file():
        return load_processed_daily(processed_path)
    tz = timezone if timezone is not None else get_meteo_timezone(config)
    data, _qa = load_basin_data(config, cache_dir=cache_dir, timezone=tz)
    return data
