"""Tests for automatic calibration report generation (Phase 6)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiment_metadata import config_sha256
from src.rapport_calage import (
    REPORT_FILENAME,
    STATUS_BANNER,
    UNCERTAINTY_ENVELOPE_STATEMENT,
    VALIDATION_ISOLATION_STATEMENT,
    generate_rapport_calage,
    load_report_inputs,
    render_rapport_calage,
)


@pytest.fixture
def report_fixture_dir(tmp_path: Path, basin_config: dict) -> Path:
    """Minimal output tree for report generation tests."""
    output = tmp_path / "output"
    data_dir = output / "data"
    data_dir.mkdir(parents=True)

    index = pd.date_range("2010-01-01", "2010-01-15", freq="D")
    pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "discharge_ls": 4000.0,
            "precipitation_mm": 1.0,
            "et0_mm": 0.5,
            "discharge_mm": 0.5,
        }
    ).to_csv(data_dir / "basin_daily.csv", index=False)

    runs = pd.DataFrame(
        {
            "run_id": [1, 2],
            "x1": [250.0, 300.0],
            "x2": [-3.0, -2.0],
            "x3": [90.0, 100.0],
            "x4": [2.0, 2.2],
            "nse_cal": [0.8, 0.85],
            "kge_cal": [0.82, 0.88],
            "r_cal": [0.9, 0.91],
            "alpha_cal": [1.0, 1.0],
            "beta_cal": [1.0, 1.0],
            "lognse_cal": [0.7, 0.75],
            "bias_cal": [0.0, 0.0],
            "nse_val": [0.7, 0.72],
            "kge_val": [0.75, 0.78],
            "r_val": [0.88, 0.89],
            "alpha_val": [1.0, 1.0],
            "beta_val": [1.0, 1.0],
            "lognse_val": [0.65, 0.68],
            "bias_val": [0.0, 0.0],
            "rank_kge_cal": [2, 1],
        }
    )
    runs.to_csv(output / "runs.csv", index=False)
    runs.iloc[[1]].to_csv(output / "behavioral_runs.csv", index=False)
    runs.head(1).to_csv(output / "top20_calibration.csv", index=False)

    pd.DataFrame(
        {
            "metric": ["NSE", "KGE", "r", "alpha", "beta", "log-NSE", "Volume bias"],
            "calibration": [0.4, 0.47, 0.87, 1.2, 1.4, 0.66, 0.44],
            "validation": [0.06, 0.20, 0.89, 1.5, 1.6, 0.58, 0.61],
            "n_calibration": [5, 5, 5, 5, 5, 5, 5],
            "n_validation": [5, 5, 5, 5, 5, 5, 5],
        }
    ).to_csv(output / "metrics_uncalibrated.csv", index=False)

    pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "q_obs": 0.5,
            "q_best_cal": 0.48,
            "q05": 0.40,
            "q50": 0.49,
            "q95": 0.60,
            "period": [
                "warmup",
                "warmup",
                "warmup",
                "warmup",
                "warmup",
                "calibration",
                "calibration",
                "calibration",
                "calibration",
                "calibration",
                "validation",
                "validation",
                "validation",
                "validation",
                "validation",
            ],
        }
    ).to_csv(output / "ensemble_timeseries.csv", index=False)

    pd.DataFrame(
        {
            "period": ["2010", "calibration", "validation"],
            "annual_precipitation_mm": [700.0, 1400.0, 800.0],
            "annual_observed_discharge_depth_mm": [100.0, 200.0, 120.0],
            "runoff_ratio_qp": [0.14, 0.14, 0.15],
            "mean_observed_discharge_mm_day": [0.27, 0.40, 0.33],
            "max_daily_observed_discharge_mm_day": [2.0, 3.0, 1.5],
            "date_of_max_discharge": ["2010-05-01", "2010-05-01", "2010-05-01"],
            "wet_days_p_gt_1mm": [100, 200, 120],
            "max_daily_precipitation_mm_day": [30.0, 40.0, 25.0],
        }
    ).to_csv(output / "hydrological_summary.csv", index=False)

    pd.DataFrame(
        {
            "threshold": [0.70, 0.75, 0.80, 0.85],
            "n_members": [2, 2, 1, 1],
            "validation_coverage": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(output / "ensemble_threshold_sensitivity.csv", index=False)

    cfg = basin_config.copy()
    cfg["calibration"]["n_samples"] = 2
    cfg["calibration"]["behavioral_kge_threshold"] = 0.80
    cfg_path = tmp_path / "config" / "basin.yaml"
    cfg_path.parent.mkdir(parents=True)
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)

    return output


@pytest.fixture
def report_config_path(report_fixture_dir: Path, tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config" / "basin.yaml"
    return cfg_path


def test_report_generated_from_outputs(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    path = generate_rapport_calage(report_config_path, report_fixture_dir)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "# Rapport de calage pluie–débit automatisé" in text
    assert "Rapport de calage" in text


def test_report_contains_required_sections(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    for heading in (
        "## 1. Périmètre du prototype",
        "## 2. Bassin versant et données",
        "## 3. Modèle hydrologique",
        "## 4. Expérience de calage",
        "## 12. Limites",
        "## 13. Reproductibilité",
        "## 14. Artéfacts",
        STATUS_BANNER,
    ):
        assert heading in text
    for phrase in (
        "Rapport de calage",
        "Période de calage",
        "Période de validation",
        "Ensemble comportemental",
        "Incertitude paramétrique",
        "Équifinalité",
        "Limites",
        "Reproductibilité",
    ):
        assert phrase in text


def test_english_section_headings_absent(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    for english_heading in (
        "# Automated Rainfall–Runoff Calibration Report",
        "## 1. Prototype scope",
        "## 2. Basin and data",
        "## 4. Calibration experiment",
        "## 9. Behavioral ensemble",
        "## 12. Limitations",
        "## 13. Reproducibility",
    ):
        assert english_heading not in text


def test_validation_isolation_statement_present(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    assert VALIDATION_ISOLATION_STATEMENT in text
    assert "ni pour l'échantillonnage" in text
    assert "ni pour le classement" in text
    assert "ni pour la sélection des paramètres" in text


def test_uncertainty_limitation_statement_present(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    assert UNCERTAINTY_ENVELOPE_STATEMENT in text
    assert "incertitude paramétrique" in text.lower()
    assert "intervalle de confiance à 90 %" in text
    assert "intervalle de prédiction à 90 %" in text
    assert "sous-couverture empirique" in text


def test_report_does_not_label_envelope_as_confidence_interval(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    assert not re.search(r"90%\s+(confidence|prediction)\s+interval", text, re.I)
    forbidden = [
        "calibrated 90% probability interval",
        "90% probability interval",
        "intervalle de confiance calibré à 90 %",
    ]
    for phrase in forbidden:
        assert phrase not in text
    # Explicit denial must remain present
    assert "ni d'un intervalle de confiance à 90 %" in text


def test_config_hash_reproducible(report_config_path: Path) -> None:
    first = config_sha256(report_config_path)
    second = config_sha256(report_config_path)
    assert first == second
    assert len(first) == 64


def test_report_values_derived_from_runs_not_hardcoded(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    assert "0.8987" not in text
    assert "0.9027" not in text
    assert "run_id=177" not in text.lower()
    inputs = load_report_inputs(report_config_path, report_fixture_dir)
    rendered = render_rapport_calage(inputs)
    assert str(int(inputs.runs["kge_cal"].max()))[:1] in rendered or "0.88" in rendered


def test_report_source_has_no_hardcoded_pilot_metrics() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "rapport_calage.py").read_text(
        encoding="utf-8"
    )
    for token in ("0.8987", "0.9027", "3058", "59.9", "126"):
        assert token not in source


def test_glue_inspired_french_wording(
    report_fixture_dir: Path, report_config_path: Path
) -> None:
    text = generate_rapport_calage(report_config_path, report_fixture_dir).read_text(
        encoding="utf-8"
    )
    assert "ensemble comportemental inspiré de l'approche GLUE" in text
    assert "implémentation complète de GLUE" in text
