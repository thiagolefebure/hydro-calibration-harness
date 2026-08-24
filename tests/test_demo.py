"""Tests for Phase 7 presentation-only demo figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.demo import (
    DEMO_01_FILENAME,
    DEMO_02_FILENAME,
    DEMO_03_FILENAME,
    UNCERTAINTY_ONLY_NOTE,
    VALIDATION_ISOLATION_NOTE,
    WINDOW_RULE_NOTE,
    demo_output_paths,
    generate_demo_figures,
)


def _write_demo_fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "output"
    (output_dir / "data").mkdir(parents=True)

    dates = pd.date_range("2010-01-01", "2015-12-31", freq="D")
    discharge = pd.Series(0.4, index=dates)
    discharge.loc["2015-05-05"] = 2.0
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "discharge_ls": discharge.values * 615.0 / 0.0864,
            "precipitation_mm": 1.0,
            "et0_mm": 0.5,
            "discharge_mm": discharge.values,
        }
    ).to_csv(output_dir / "data" / "basin_daily.csv", index=False)

    pd.DataFrame(
        {
            "metric": ["KGE", "NSE", "log-NSE", "Volume bias", "r", "alpha", "beta"],
            "calibration": [0.4, 0.3, 0.5, 0.2, 0.8, 1.2, 1.1],
            "validation": [0.2, 0.1, 0.4, 0.3, 0.85, 1.3, 1.2],
            "n_calibration": [100] * 7,
            "n_validation": [50] * 7,
        }
    ).to_csv(output_dir / "metrics_uncalibrated.csv", index=False)

    runs = pd.DataFrame(
        {
            "run_id": [1, 2],
            "x1": [250.0, 280.0],
            "x2": [-3.0, -2.5],
            "x3": [90.0, 100.0],
            "x4": [2.0, 2.2],
            "nse_cal": [0.7, 0.8],
            "kge_cal": [0.81, 0.88],
            "r_cal": [0.88, 0.90],
            "alpha_cal": [1.0, 1.0],
            "beta_cal": [1.0, 1.0],
            "lognse_cal": [0.75, 0.8],
            "bias_cal": [0.0, 0.0],
            "nse_val": [0.65, 0.7],
            "kge_val": [0.7, 0.77],
            "r_val": [0.86, 0.88],
            "alpha_val": [1.0, 1.0],
            "beta_val": [1.0, 1.0],
            "lognse_val": [0.72, 0.74],
            "bias_val": [0.02, 0.04],
            "rank_kge_cal": [2, 1],
        }
    )
    runs.to_csv(output_dir / "runs.csv", index=False)
    runs.iloc[[1]].to_csv(output_dir / "behavioral_runs.csv", index=False)

    ensemble_ts = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "q_obs": discharge.values,
            "q_best_cal": discharge.values * 0.95,
            "q05": discharge.values * 0.80,
            "q50": discharge.values * 0.90,
            "q95": discharge.values * 1.05,
            "period": [
                "warmup" if d.year == 2010 else "calibration" if d.year <= 2013 else "validation"
                for d in dates
            ],
        }
    )
    ensemble_ts.to_csv(output_dir / "ensemble_timeseries.csv", index=False)
    pd.DataFrame(
        {
            "threshold": [0.70, 0.75, 0.80, 0.85],
            "n_members": [2, 2, 2, 1],
            "validation_coverage": [0.8, 0.75, 0.7, 0.6],
        }
    ).to_csv(output_dir / "ensemble_threshold_sensitivity.csv", index=False)
    (output_dir / "experiment_metadata.json").write_text(
        '{"calibration_runtime_s": 12.34, "calibration_runtime_per_eval_s": 0.12}',
        encoding="utf-8",
    )

    config = {
        "station": {
            "code": "TEST001",
            "name": "Test Basin",
            "basin_area_km2": 615.0,
            "centroid_lat": 47.0,
            "centroid_lon": 4.0,
        },
        "data": {"meteo_timezone": "Europe/Paris"},
        "periods": {
            "warmup": ["2010-01-01", "2010-12-31"],
            "calibration": ["2011-01-01", "2013-12-31"],
            "validation": ["2014-01-01", "2015-12-31"],
        },
        "model": {
            "name": "GR4J",
            "parameter_bounds": {
                "X1": [100, 1200],
                "X2": [-5, 3],
                "X3": [20, 300],
                "X4": [1.1, 2.9],
            },
        },
        "calibration": {
            "sampler": "latin_hypercube",
            "n_samples": 5000,
            "seed": 42,
            "behavioral_kge_threshold": 0.80,
        },
        "metrics": {"log_nse_epsilon_mm": 0.01},
    }
    config_path = tmp_path / "config" / "basin.yaml"
    config_path.parent.mkdir(parents=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    return config_path, output_dir


def test_demo_output_paths_are_stable(tmp_path: Path) -> None:
    paths = demo_output_paths(tmp_path)
    assert paths["calibration_impact"].name == DEMO_01_FILENAME
    assert paths["validation_summary"].name == DEMO_02_FILENAME
    assert paths["uncertainty"].name == DEMO_03_FILENAME


def test_demo_wording_is_present_and_non_probabilistic() -> None:
    assert "parameter fitting or ranking" in VALIDATION_ISOLATION_NOTE.lower()
    assert "parametric uncertainty only" in UNCERTAINTY_ONLY_NOTE.lower()
    assert "confidence interval" not in UNCERTAINTY_ONLY_NOTE.lower()
    assert "prediction interval" not in UNCERTAINTY_ONLY_NOTE.lower()
    assert "probability interval" not in UNCERTAINTY_ONLY_NOTE.lower()
    assert "maximum observed validation discharge" in WINDOW_RULE_NOTE.lower()


def test_demo_source_has_no_hardcoded_pilot_metrics() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "demo.py").read_text(
        encoding="utf-8"
    )
    for token in ("0.2026", "0.8350", "59.9%", "126 behavioral", "5000 Latin Hypercube"):
        assert token not in source


def test_generate_demo_figures_from_artifacts(tmp_path: Path) -> None:
    config_path, output_dir = _write_demo_fixture_tree(tmp_path)
    paths = generate_demo_figures(config_path, output_dir)
    for path in paths.values():
        assert path.is_file()


def test_scientific_pipeline_modules_do_not_reference_demo_outputs() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    scientific_modules = [
        root / "data.py",
        root / "gr4j.py",
        root / "metrics.py",
        root / "calibration.py",
    ]
    for module in scientific_modules:
        source = module.read_text(encoding="utf-8")
        assert DEMO_01_FILENAME not in source
        assert DEMO_02_FILENAME not in source
        assert DEMO_03_FILENAME not in source
