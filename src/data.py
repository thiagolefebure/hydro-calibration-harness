"""Data acquisition, alignment, unit conversion, and QA summary.

Responsibilities (per spec §1):
- Fetch observed discharge from Hub'Eau hydrometry API (obs_elab, QmnJ).
- Fetch daily precipitation and ET0 from Open-Meteo historical API at basin centroid.
- Disk-cache API responses for fast, network-independent reruns.
- Align rain / ET0 / discharge on a common daily index; leave gaps as NaN (no silent interpolation).
- Convert discharge L/s → mm/day using configured basin area.
- Produce a QA summary: raw period, usable period, gap counts and proportions per variable.

API note (spec vs live behaviour):
- The specification references Hub'Eau API v1 (`/api/v1/hydrometrie/obs_elab`).
- As of implementation, v1 returns HTTP 403; v2 (`/api/v2/hydrometrie/obs_elab`) is used instead.
- This discrepancy is surfaced in QA output (`api_issues`) rather than silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    import truststore

    truststore.inject_into_ssl()
    _SSL_BACKEND = "truststore"
except ImportError:
    _SSL_BACKEND = "certifi"

try:
    import certifi

    _SSL_VERIFY: bool | str = certifi.where()
except ImportError:
    _SSL_VERIFY = True

logger = logging.getLogger(__name__)

# Spec §1 endpoint (v1 — documented but currently returns 403)
HUBEAU_OBS_ELAB_URL_V1 = "https://hubeau.eaufrance.fr/api/v1/hydrometrie/obs_elab"
# Working endpoint (v2 — cursor pagination, depth limit 20 000 records per query)
HUBEAU_OBS_ELAB_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

HUBEAU_PAGE_SIZE = 20_000
HUBEAU_MAX_DEPTH = 20_000

VARIABLE_COLUMNS = {
    "discharge_ls": "discharge_ls",
    "discharge_mm": "discharge_mm",
    "precipitation_mm": "precipitation_mm",
    "et0_mm": "et0_mm",
}


class HubEauDuplicateConflictError(ValueError):
    """Raised when Hub'Eau returns conflicting discharge values for the same date."""

    def __init__(self, conflicts: list[dict[str, Any]]):
        self.conflicts = conflicts
        messages = [
            f"{c['date']}: values={c['values']} (stations={c['stations']})"
            for c in conflicts
        ]
        super().__init__(
            "Conflicting Hub'Eau discharge values for the same date: " + "; ".join(messages)
        )


def get_meteo_timezone(config: dict[str, Any]) -> str:
    """Return Open-Meteo daily aggregation timezone from basin configuration."""
    return config.get("data", {}).get("meteo_timezone", "UTC")


def ls_to_mm_day(q_ls: pd.Series, basin_area_km2: float) -> pd.Series:
    """Convert discharge from L/s to mm/day over a basin of given area (km²).

    Formula (spec §1): Q_mm/d = Q_L/s × 0.0864 / A_km²
    """
    if basin_area_km2 <= 0:
        raise ValueError("basin_area_km2 must be a positive configured value")
    return q_ls * 0.0864 / basin_area_km2


def _request_verify() -> bool | str:
    return _SSL_VERIFY


def _cache_key(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _raw_cache_path(cache_dir: Path, source: str, key: str) -> Path:
    return cache_dir / "raw" / source / f"{key}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _http_get_json(
    url: str,
    params: dict[str, Any],
    cache_path: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """GET JSON from URL with disk cache. Cached file stores URL, params, and response body."""
    if not force_refresh:
        cached = _read_cache(cache_path)
        if cached is not None and cached.get("response") is not None:
            logger.debug("Cache hit: %s", cache_path)
            return cached["response"]

    try:
        response = requests.get(
            url,
            params=params,
            timeout=120,
            verify=_request_verify(),
        )
        response.raise_for_status()
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(
            "SSL certificate verification failed when calling external API. "
            "Install/update certifi (`pip install certifi`) or fix the system CA store."
        ) from exc

    body = response.json()
    _write_cache(
        cache_path,
        {
            "url": response.url,
            "params": params,
            "response": body,
        },
    )
    return body


def _experiment_date_bounds(config: dict[str, Any]) -> tuple[str, str]:
    """Return inclusive ISO date bounds covering all configured experiment periods."""
    periods = config["periods"]
    starts = [periods["warmup"][0], periods["calibration"][0], periods["validation"][0]]
    ends = [periods["warmup"][1], periods["calibration"][1], periods["validation"][1]]
    return min(starts), max(ends)


def parse_hubeau_obs_elab(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse Hub'Eau obs_elab records into a daily discharge series (L/s).

    Hub'Eau v2 may return duplicate rows per date (with and without code_station).
    Before collapsing duplicates, resultat_obs_elab values are compared per date:
    - identical values: keep the row with an explicit code_station;
    - conflicting values: raise HubEauDuplicateConflictError (never silent selection).

    Returns the parsed series and an audit dict describing resolved identical duplicates.
    """
    audit: dict[str, Any] = {
        "identical_duplicate_dates_resolved": 0,
        "conflicting_dates": [],
    }

    if not records:
        empty = pd.DataFrame(columns=["discharge_ls"]).astype({"discharge_ls": float})
        return empty, audit

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date_obs_elab"], utc=False)
    df = df.rename(columns={"resultat_obs_elab": "discharge_ls"})

    selected_rows: list[pd.Series] = []
    conflicts: list[dict[str, Any]] = []

    for date, group in df.groupby("date", sort=True):
        if len(group) == 1:
            selected_rows.append(group.iloc[0])
            continue

        unique_values = group["discharge_ls"].dropna().unique()
        if len(unique_values) > 1:
            conflicts.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "values": [float(v) for v in unique_values],
                    "stations": [
                        None if pd.isna(s) else str(s) for s in group["code_station"].tolist()
                    ],
                }
            )
            continue

        audit["identical_duplicate_dates_resolved"] += 1
        group = group.copy()
        group["_has_station"] = group["code_station"].notna().astype(int)
        selected_rows.append(group.sort_values("_has_station", ascending=False).iloc[0])

    if conflicts:
        audit["conflicting_dates"] = [c["date"] for c in conflicts]
        raise HubEauDuplicateConflictError(conflicts)

    result = pd.DataFrame(selected_rows)
    series = result.set_index("date")[["discharge_ls"]].sort_index()
    series.index = series.index.normalize()
    return series, audit


def _fetch_hubeau_chunk(
    station_code: str,
    start: str,
    end: str,
    cache_dir: Path,
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    params_base = {
        "code_entite": station_code,
        "grandeur_hydro_elab": "QmnJ",
        "date_debut_obs_elab": start,
        "date_fin_obs_elab": end,
        "size": HUBEAU_PAGE_SIZE,
    }
    cache_key = _cache_key("hubeau", station_code, "QmnJ", start, end)
    cache_path = _raw_cache_path(cache_dir, "hubeau", cache_key)

    # Probe count with a lightweight first request (cached separately per cursor page).
    first = _http_get_json(
        HUBEAU_OBS_ELAB_URL,
        params_base,
        cache_path,
        force_refresh=force_refresh,
    )
    rows = list(first.get("data", []))
    cursor = first.get("next")
    total_count = first.get("count")

    while cursor:
        page_key = _cache_key("hubeau", station_code, "QmnJ", start, end, cursor)
        page_path = _raw_cache_path(cache_dir, "hubeau", page_key)
        page = _http_get_json(
            HUBEAU_OBS_ELAB_URL,
            {**params_base, "cursor": cursor},
            page_path,
            force_refresh=force_refresh,
        )
        rows.extend(page.get("data", []))
        cursor = page.get("next")

    if total_count is not None and len(rows) > HUBEAU_MAX_DEPTH:
        raise RuntimeError(
            f"Hub'Eau returned {len(rows)} rows for {start}–{end}; "
            f"exceeds API depth limit ({HUBEAU_MAX_DEPTH}). "
            "Use narrower date chunks."
        )

    return rows


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into calendar-year chunks."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks: list[tuple[str, str]] = []
    for year in range(start_ts.year, end_ts.year + 1):
        chunk_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
    return chunks


def fetch_discharge(
    station_code: str,
    start: str,
    end: str,
    cache_dir: Path,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch observed daily discharge (QmnJ, L/s) from Hub'Eau obs_elab."""
    all_records: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        chunk_records = _fetch_hubeau_chunk(
            station_code,
            chunk_start,
            chunk_end,
            cache_dir,
            force_refresh=force_refresh,
        )
        all_records.extend(chunk_records)

    series, audit = parse_hubeau_obs_elab(all_records)
    return series, audit


def parse_open_meteo_daily(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse Open-Meteo archive daily payload into precipitation and ET0 series."""
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        raise ValueError("Open-Meteo response missing daily.time")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"], utc=False),
            "precipitation_mm": daily.get("precipitation_sum"),
            "et0_mm": daily.get("et0_fao_evapotranspiration"),
        }
    )
    df["date"] = df["date"].dt.normalize()
    df = df.set_index("date").sort_index()
    return df


def fetch_meteo(
    lat: float,
    lon: float,
    start: str,
    end: str,
    cache_dir: Path,
    *,
    timezone: str = "UTC",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily precipitation and ET0 from Open-Meteo historical API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum,et0_fao_evapotranspiration",
        "timezone": timezone,
    }
    cache_key = _cache_key(
        "openmeteo",
        f"{lat:.6f}",
        f"{lon:.6f}",
        start,
        end,
        timezone,
    )
    cache_path = _raw_cache_path(cache_dir, "openmeteo", cache_key)
    payload = _http_get_json(
        OPEN_METEO_HISTORICAL_URL,
        params,
        cache_path,
        force_refresh=force_refresh,
    )
    return parse_open_meteo_daily(payload)


def align_timeseries(
    discharge: pd.DataFrame,
    meteo: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Align discharge, precipitation, and ET0 on a common daily DatetimeIndex.

    Missing calendar days within [start, end] are retained with NaN values.
    No interpolation is applied (spec §1).
    """
    index = pd.date_range(start=start, end=end, freq="D")
    aligned = pd.DataFrame(index=index)
    aligned.index.name = "date"

    if not discharge.empty:
        aligned["discharge_ls"] = discharge["discharge_ls"].reindex(index)
    else:
        aligned["discharge_ls"] = float("nan")

    if not meteo.empty:
        aligned["precipitation_mm"] = meteo["precipitation_mm"].reindex(index)
        aligned["et0_mm"] = meteo["et0_mm"].reindex(index)
    else:
        aligned["precipitation_mm"] = float("nan")
        aligned["et0_mm"] = float("nan")

    return aligned


def _variable_gap_stats(series: pd.Series) -> dict[str, Any]:
    total = len(series)
    valid = int(series.notna().sum())
    missing = total - valid
    proportion = (missing / total) if total else 0.0
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()
    return {
        "n_days": total,
        "n_valid": valid,
        "n_missing": missing,
        "missing_proportion": proportion,
        "first_valid": first_valid.strftime("%Y-%m-%d") if first_valid is not None else None,
        "last_valid": last_valid.strftime("%Y-%m-%d") if last_valid is not None else None,
    }


def qa_summary(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    meteo_timezone: str,
    hubeau_duplicate_audit: dict[str, Any] | None = None,
    api_issues: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize data coverage and gap statistics per variable."""
    start, end = _experiment_date_bounds(config)
    usable_mask = df[["discharge_mm", "precipitation_mm", "et0_mm"]].notna().all(axis=1)
    usable = df.loc[usable_mask]

    summary: dict[str, Any] = {
        "station_code": config["station"]["code"],
        "basin_area_km2": config["station"]["basin_area_km2"],
        "centroid_lat": config["station"]["centroid_lat"],
        "centroid_lon": config["station"]["centroid_lon"],
        "meteo_timezone": meteo_timezone,
        "hubeau_duplicate_audit": hubeau_duplicate_audit
        or {"identical_duplicate_dates_resolved": 0, "conflicting_dates": []},
        "requested_period": {"start": start, "end": end},
        "raw_period": {
            "start": start,
            "end": end,
            "n_days": len(df),
        },
        "variables": {
            "discharge_ls": _variable_gap_stats(df["discharge_ls"]),
            "discharge_mm": _variable_gap_stats(df["discharge_mm"]),
            "precipitation_mm": _variable_gap_stats(df["precipitation_mm"]),
            "et0_mm": _variable_gap_stats(df["et0_mm"]),
        },
        "usable_period": {
            "start": usable.index.min().strftime("%Y-%m-%d") if len(usable) else None,
            "end": usable.index.max().strftime("%Y-%m-%d") if len(usable) else None,
            "n_days": int(len(usable)),
            "missing_proportion": float(1 - len(usable) / len(df)) if len(df) else 0.0,
        },
        "units": {
            "discharge_raw": "L/s (Hub'Eau QmnJ)",
            "discharge_converted": "mm/day",
            "precipitation": "mm/day (Open-Meteo precipitation_sum)",
            "et0": "mm/day (Open-Meteo et0_fao_evapotranspiration, FAO-56)",
        },
        "conversion_formula": "Q_mm/d = Q_L/s × 0.0864 / basin_area_km2",
        "data_sources": {
            "discharge": HUBEAU_OBS_ELAB_URL,
            "meteo": OPEN_METEO_HISTORICAL_URL,
        },
        "limitations": [
            "Point precipitation at basin centroid — not basin-average rainfall (spec §1).",
            "Open-Meteo ET0 (FAO-56) may differ from operational GR4J ETP (spec §1).",
            "Daily time step only (spec §1).",
        ],
        "api_issues": api_issues or [],
    }
    return summary


def save_processed_data(df: pd.DataFrame, output_dir: Path) -> Path:
    """Persist aligned processed data separately from raw API cache."""
    path = output_dir / "data" / "basin_daily.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")
    return path


def load_basin_data(
    config: dict[str, Any],
    cache_dir: Path,
    *,
    output_dir: Path | None = None,
    timezone: str | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, align, convert, and QA-check all basin forcing and observation data."""
    station = config["station"]
    if station["basin_area_km2"] <= 0:
        raise ValueError(
            "station.basin_area_km2 must be configured with a positive value before running the pipeline"
        )

    meteo_timezone = timezone if timezone is not None else get_meteo_timezone(config)
    start, end = _experiment_date_bounds(config)
    api_issues: list[str] = []

    # Document spec vs live Hub'Eau API version behaviour.
    try:
        probe = requests.get(
            HUBEAU_OBS_ELAB_URL_V1,
            params={"code_entite": station["code"], "size": 1},
            timeout=30,
            verify=_request_verify(),
        )
        if probe.status_code == 403:
            api_issues.append(
                "Spec references Hub'Eau API v1 (/api/v1/hydrometrie/obs_elab) but it "
                f"returns HTTP 403; loader uses v2 ({HUBEAU_OBS_ELAB_URL}) instead."
            )
    except requests.RequestException as exc:
        api_issues.append(f"Could not probe Hub'Eau v1 endpoint: {exc}")

    discharge, hubeau_duplicate_audit = fetch_discharge(
        station_code=station["code"],
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    meteo = fetch_meteo(
        lat=station["centroid_lat"],
        lon=station["centroid_lon"],
        start=start,
        end=end,
        cache_dir=cache_dir,
        timezone=meteo_timezone,
        force_refresh=force_refresh,
    )

    aligned = align_timeseries(discharge, meteo, start, end)
    aligned["discharge_mm"] = ls_to_mm_day(aligned["discharge_ls"], station["basin_area_km2"])

    qa = qa_summary(
        aligned,
        config,
        meteo_timezone=meteo_timezone,
        hubeau_duplicate_audit=hubeau_duplicate_audit,
        api_issues=api_issues,
    )

    if output_dir is not None:
        qa["processed_data_path"] = str(save_processed_data(aligned, output_dir))

    return aligned, qa
