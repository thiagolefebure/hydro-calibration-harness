"""Tests for static interview demo export (presentation layer only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.demo_export import (
    DemoExportError,
    build_demo_data,
    export_static_demo,
    verify_required_artifacts,
)


def _write_min_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "output"
    demo = tmp_path / "demo"
    output.mkdir()
    (demo / "assets").mkdir(parents=True)

    runs = pd.DataFrame(
        {
            "run_id": [1, 2],
            "x1": [250.0, 280.0],
            "x2": [-3.0, -2.5],
            "x3": [90.0, 100.0],
            "x4": [2.0, 2.2],
            "nse_cal": [0.7, 0.8],
            "kge_cal": [0.81, 0.88],
            "r_cal": [0.88, 0.9],
            "alpha_cal": [1.0, 1.0],
            "beta_cal": [1.0, 1.0],
            "lognse_cal": [0.75, 0.8],
            "bias_cal": [0.1, -0.01],
            "nse_val": [0.65, 0.7],
            "kge_val": [0.2, 0.835],
            "r_val": [0.86, 0.88],
            "alpha_val": [1.0, 1.0],
            "beta_val": [1.0, 1.0],
            "lognse_val": [0.72, 0.74],
            "bias_val": [0.611, 0.06],
            "rank_kge_cal": [2, 1],
        }
    )
    runs.to_csv(output / "runs.csv", index=False)
    runs.iloc[[1]].to_csv(output / "behavioral_runs.csv", index=False)
    pd.DataFrame(
        {
            "metric": ["KGE", "NSE", "log-NSE", "Volume bias", "r", "alpha", "beta"],
            "calibration": [0.47, 0.44, 0.66, 0.44, 0.87, 1.2, 1.4],
            "validation": [0.203, 0.06, 0.58, 0.611, 0.89, 1.5, 1.6],
            "n_calibration": [10] * 7,
            "n_validation": [5] * 7,
        }
    ).to_csv(output / "metrics_uncalibrated.csv", index=False)

    dates = pd.date_range("2014-01-01", periods=5, freq="D")
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "q_obs": [1.0, 1.1, 0.9, 1.0, 1.2],
            "q_best_cal": [1.0] * 5,
            "q05": [0.8] * 5,
            "q50": [1.0] * 5,
            "q95": [1.2] * 5,
            "period": ["validation"] * 5,
        }
    ).to_csv(output / "ensemble_timeseries.csv", index=False)

    for name in (
        "demo_01_calibration_impact.png",
        "demo_02_validation.png",
        "demo_03_uncertainty.png",
    ):
        (output / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (output / "rapport_calage.md").write_text("# Report\ncontent\n", encoding="utf-8")
    (output / "experiment_metadata.json").write_text(
        '{"calibration_runtime_s": 100.0}',
        encoding="utf-8",
    )

    config = {
        "station": {
            "code": "H0203020",
            "name": "La Laignes à Molesme",
            "basin_area_km2": 615.0,
            "centroid_lat": 47.96,
            "centroid_lon": 4.36,
        },
        "periods": {
            "warmup": ["2010-01-01", "2010-12-31"],
            "calibration": ["2011-01-01", "2013-12-31"],
            "validation": ["2014-01-01", "2015-12-31"],
        },
        "model": {"name": "GR4J", "parameter_bounds": {"X1": [100, 1200]}},
        "calibration": {
            "sampler": "latin_hypercube",
            "n_samples": 5000,
            "seed": 42,
            "behavioral_kge_threshold": 0.80,
        },
        "demo": {"github_url": None},
    }
    config_path = tmp_path / "config" / "basin.yaml"
    config_path.parent.mkdir(parents=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return config_path, output


def test_export_fails_if_artifacts_missing(tmp_path: Path) -> None:
    with pytest.raises(DemoExportError):
        verify_required_artifacts(tmp_path / "empty")


def test_exported_kpis_match_artifacts(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    data = build_demo_data(config_path, output)
    assert data["kpis"]["n_samples"] == 2
    assert data["kpis"]["validation_kge_uncalibrated"] == pytest.approx(0.203)
    assert data["kpis"]["validation_kge_calibrated"] == pytest.approx(0.835)
    assert data["kpis"]["validation_bias_uncalibrated"] == pytest.approx(0.611)
    assert data["kpis"]["validation_bias_calibrated"] == pytest.approx(0.06)
    assert data["kpis"]["behavioral_members"] == 1
    assert data["kpis"]["behavioral_threshold"] == pytest.approx(0.80)
    assert "config_sha256" in data["reproducibility"]
    assert "git_commit" in data["reproducibility"]


def test_export_writes_assets(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    demo_dir = tmp_path / "demo"
    paths = export_static_demo(config_path, output, demo_dir)
    assert paths["demo_data"].is_file()
    payload = json.loads(paths["demo_data"].read_text(encoding="utf-8"))
    assert payload["reproducibility"]["config_sha256"]
    assert (demo_dir / "assets" / "demo_01_calibration_impact.png").is_file()
    assert (demo_dir / "assets" / "rapport_calage.html").is_file()


def test_frontend_has_no_scientific_calculations() -> None:
    root = Path(__file__).resolve().parents[1] / "demo"
    js = (root / "app.js").read_text(encoding="utf-8")
    html = (root / "index.html").read_text(encoding="utf-8")
    for token in (
        "0.2026",
        "0.8350",
        "59.9",
        "5000",
        "126",
        "latin_hypercube_sample",
        "run_continuous_gr4j",
    ):
        assert token not in js
    assert "0.203 → 0.835" not in html
    assert "5,000" not in html
    assert "126" not in html
    assert "+61.1%" not in html
    assert "PROTOTYPE — NOT FOR OPERATIONAL DECISION-MAKING" in html
    assert "Validation observations are never used for parameter fitting or ranking." in html
    assert "q05–q95 is NOT a calibrated 90% prediction interval" in html
    assert "HEC-HMS integration is a future step" in html
    assert "not implemented in this prototype" in html.lower()


def test_frontend_uncertainty_and_hecras_wording() -> None:
    html = (Path(__file__).resolve().parents[1] / "demo" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "parametric dispersion only" in html.lower() or "parametric dispersion" in html.lower()
    assert "HEC-HMS" in html
    assert "future" in html.lower()
    assert "not present capabilities" in html.lower() or "not implemented" in html.lower()
