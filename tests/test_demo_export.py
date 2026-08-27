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
    build_forecast_horizons_payload,
    build_forecast_uncertainty_payload,
    build_meteo_sensitivity_payload,
    build_ml_correction_payload,
    export_static_demo,
    verify_required_artifacts,
)


def _write_ml_artifacts(output: Path) -> None:
    pd.DataFrame(
        {
            "model": [
                "physical_gr4j",
                "dummy_mean_residual",
                "ridge",
                "hist_gradient_boosting",
            ],
            "kge_val": [0.835003, 0.829329, 0.985806, 0.973911],
            "nse_val": [0.79, 0.79, 0.98, 0.98],
            "lognse_val": [0.87, 0.88, 0.95, 0.98],
            "bias_val": [0.06, 0.07, 0.01, 0.0],
            "mae_val": [0.1, 0.1, 0.03, 0.03],
            "rmse_val": [0.18, 0.18, 0.05, 0.05],
        }
    ).to_csv(output / "ml_model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "group": ["baseline"] * 6,
            "model": [
                "physical",
                "mean_residual",
                "persistence",
                "ar1",
                "ridge_full",
                "hgb",
            ],
            "features": ["(none)"] * 6,
            "kge_val": [0.835003, 0.829329, 0.983784, 0.982402, 0.985806, 0.973911],
            "nse_val": [0.79] * 6,
            "lognse_val": [0.87] * 6,
            "bias_val": [0.0] * 6,
            "mae_val": [0.1] * 6,
            "rmse_val": [0.18] * 6,
        }
    ).to_csv(output / "ml_ablation_comparison.csv", index=False)
    (output / "ml_ablation_answers.json").write_text(
        json.dumps(
            {
                "answers": {"1_persistence_most_of_gain": "Yes"},
                "acf_summary": {
                    "acf_lag_1": 0.8936370025978588,
                    "acf_lag_2": 0.7764422479808538,
                    "acf_lag_3": 0.6874591819369,
                },
            }
        ),
        encoding="utf-8",
    )

    horizon_rows = []
    highflow_rows = []
    for h, phys, pers, ar1, ridge in (
        (1, 0.836747, 0.983846, 0.982265, 0.983989),
        (2, 0.837628, 0.965808, 0.964913, 0.963589),
        (3, 0.838346, 0.951670, 0.952505, 0.954514),
    ):
        for model, kge in (
            ("physical", phys),
            ("persistence_residual", pers),
            ("AR1_residual", ar1),
            ("ridge", ridge),
        ):
            horizon_rows.append(
                {
                    "horizon_days": h,
                    "model": model,
                    "kge": kge,
                    "nse": 0.9,
                    "lognse": 0.9,
                    "bias": 0.0,
                    "mae": 0.1,
                    "rmse": 0.1,
                }
            )
            highflow_rows.append(
                {
                    "horizon_days": h,
                    "model": model,
                    "n_high_flow_days": 20,
                    "n_peak_days": 4,
                    "threshold_q_obs": 1.345873,
                    "mae_highflow": 0.5 if model == "physical" else 0.2 + 0.05 * h,
                    "rmse_highflow": 0.6,
                    "mean_bias_highflow": -0.1,
                    "peak_magnitude_error": 0.0,
                    "experiment_tag": "ORACLE METEOROLOGICAL FORCING",
                }
            )
    pd.DataFrame(horizon_rows).to_csv(output / "ml_horizon_comparison.csv", index=False)
    pd.DataFrame(highflow_rows).to_csv(output / "ml_horizon_highflow.csv", index=False)
    pd.DataFrame(
        {
            "model": [
                "physical",
                "persistence_residual",
                "AR1_residual",
                "ridge",
            ],
            "kge_h1": [0.836747, 0.983846, 0.982265, 0.983989],
            "kge_h2": [0.837628, 0.965808, 0.964913, 0.963589],
            "kge_h3": [0.838346, 0.951670, 0.952505, 0.954514],
            "delta_KGE_1_to_3": [0.001599, -0.032176, -0.029760, -0.029475],
            "experiment_tag": ["ORACLE METEOROLOGICAL FORCING"] * 4,
        }
    ).to_csv(output / "ml_horizon_degradation.csv", index=False)
    (output / "ml_horizon_answers.json").write_text(
        json.dumps(
            {
                "experiment_tag": "ORACLE METEOROLOGICAL FORCING",
                "experiment_note": "Oracle forcing note",
                "answers": {"1_persistence_plus_24h": "Yes"},
            }
        ),
        encoding="utf-8",
    )

    # Phase 8E meteo sensitivity fixtures
    hydro_rows = []
    mc_rows = []
    hf_rows = []
    phys = {
        "oracle": {1: 0.836654, 2: 0.837583, 3: 0.838346},
        "moderate": {1: 0.836463, 2: 0.834012, 3: 0.824837},
        "strong": {1: 0.836009, 2: 0.822121, 3: 0.776091},
    }
    pers = {
        "oracle": {1: 0.983838, 2: 0.965796, 3: 0.951670},
        "moderate": {1: 0.983703, 2: 0.962230, 3: 0.938247},
        "strong": {1: 0.983365, 2: 0.950492, 3: 0.885109},
    }
    ridge = {
        "oracle": {1: 0.983944, 2: 0.963514, 3: 0.954514},
        "moderate": {1: 0.983913, 2: 0.962576, 3: 0.943060},
        "strong": {1: 0.983790, 2: 0.955554, 3: 0.900020},
    }
    labels = {
        "oracle": "ORACLE METEOROLOGICAL FORCING",
        "moderate": "SYNTHETIC METEOROLOGICAL FORCING — MODERATE ERROR",
        "strong": "SYNTHETIC METEOROLOGICAL FORCING — STRONG ERROR",
    }
    for scen in ("oracle", "moderate", "strong"):
        for h in (1, 2, 3):
            for model, store in (
                ("physical", phys),
                ("persistence_residual", pers),
                ("ridge", ridge),
            ):
                kge = store[scen][h]
                hydro_rows.append(
                    {
                        "scenario": scen,
                        "horizon_days": h,
                        "model": model,
                        "kge": kge,
                        "nse": 0.9,
                        "lognse": 0.9,
                        "bias": 0.0,
                        "mae": 0.1,
                        "rmse": 0.1,
                        "delta_KGE_vs_oracle": kge - phys["oracle"][h]
                        if model == "physical"
                        else 0.0,
                        "delta_NSE_vs_oracle": 0.0,
                        "delta_MAE_vs_oracle": 0.0,
                        "delta_RMSE_vs_oracle": 0.0,
                        "gain_KGE_vs_physical": kge - phys[scen][h],
                        "gain_MAE_vs_physical": 0.0,
                        "gain_RMSE_vs_physical": 0.0,
                        "scenario_label": labels[scen],
                    }
                )
                n_r = 1 if scen == "oracle" else 100
                p10 = kge - (0.07 if scen == "strong" and h == 3 else 0.001)
                p90 = kge + (0.05 if scen == "strong" and h == 3 else 0.001)
                if scen == "strong" and h == 3 and model == "physical":
                    p10, p90 = 0.704429, 0.827404
                mc_rows.append(
                    {
                        "scenario": scen,
                        "horizon_days": h,
                        "model": model,
                        "n_realizations": n_r,
                        "kge_median": kge,
                        "kge_p10": p10,
                        "kge_p90": p90,
                        "rmse_median": 0.1,
                        "rmse_p10": 0.1,
                        "rmse_p90": 0.1,
                        "mae_median": 0.1,
                        "scenario_label": labels[scen],
                        "experiment_note": "Synthetic meteorological forcing sensitivity — not a real weather forecast",
                    }
                )
                hf_rows.append(
                    {
                        "scenario": scen,
                        "horizon_days": h,
                        "model": model,
                        "mae_highflow": 0.5 if model == "physical" else 0.2,
                        "rmse_highflow": 0.6,
                        "mean_bias_highflow": -0.1,
                        "peak_magnitude_error": 0.0,
                        "n_high_flow_days": 20,
                        "threshold_q_obs": 1.345873,
                        "scenario_label": labels[scen],
                        "experiment_note": "Synthetic meteorological forcing sensitivity — not a real weather forecast",
                    }
                )
    pd.DataFrame(hydro_rows).to_csv(output / "meteo_hydrology_comparison.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(output / "meteo_monte_carlo_summary.csv", index=False)
    pd.DataFrame(hf_rows).to_csv(output / "meteo_highflow_comparison.csv", index=False)
    pd.DataFrame(
        {
            "scenario": ["oracle", "moderate", "strong"],
            "scenario_label": [labels[s] for s in ("oracle", "moderate", "strong")],
            "n_realizations": [1, 100, 100],
            "mean_abs_precip_error": [0.0, 0.7, 1.6],
            "rmse_precip": [0.0, 2.0, 3.8],
            "mean_bias_precip": [0.0, 0.0, 0.1],
            "correlation_precip": [1.0, 0.89, 0.64],
            "wet_day_hit_rate": [1.0, 0.89, 0.73],
            "n_shifted_events_median": [0, 35, 87],
            "min_precip": [0.0, 0.0, 0.0],
            "experiment_note": [
                "Synthetic meteorological forcing sensitivity — not a real weather forecast"
            ]
            * 3,
        }
    ).to_csv(output / "meteo_scenario_diagnostics.csv", index=False)
    (output / "meteo_sensitivity_answers.json").write_text(
        json.dumps(
            {
                "experiment_banner": (
                    "Synthetic meteorological forcing sensitivity — not a real weather forecast"
                ),
                "meteo_config": {
                    "n_realizations": 100,
                    "seed": 42,
                    "moderate": {"magnitude_lognormal_sigma": 0.2},
                    "strong": {"magnitude_lognormal_sigma": 0.45},
                },
                "answers": {},
            }
        ),
        encoding="utf-8",
    )

    # Phase 9 uncertainty fixtures
    unc_rows = []
    methods = {
        "behavioral_parametric": {1: 0.599451, 2: 0.600275, 3: 0.601100},
        "conditional_quantile": {1: 0.764060, 2: 0.776099, 3: 0.826685},
        "empirical_residual": {1: 0.935528, 2: 0.938187, 3: 0.946355},
        "split_conformal": {1: 0.971193, 2: 0.972527, 3: 0.975241},
    }
    widths = {
        "behavioral_parametric": 0.206,
        "conditional_quantile": 0.100,
        "empirical_residual": 0.256,
        "split_conformal": 0.347,
    }
    for scenario in ("oracle", "moderate"):
        for method, by_h in methods.items():
            for h, emp in by_h.items():
                for nom in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
                    # Scale empirical roughly with nominal for reliability curve
                    if nom == 0.90:
                        emp_n = emp if scenario == "oracle" else emp + (0.0 if h < 3 else 0.007)
                    else:
                        emp_n = min(0.99, emp * (nom / 0.90))
                    unc_rows.append(
                        {
                            "method": method,
                            "horizon_days": h,
                            "meteo_scenario": scenario,
                            "nominal_coverage": nom,
                            "empirical_coverage": emp_n,
                            "coverage_error": emp_n - nom,
                            "mean_width": widths[method] * (1.0 if nom == 0.90 else 0.8),
                            "median_width": widths[method],
                            "p90_width": widths[method] * 1.2,
                            "mean_width_norm": 0.5,
                            "mean_interval_score": 0.4,
                        }
                    )
    pd.DataFrame(unc_rows).to_csv(output / "uncertainty_coverage_summary.csv", index=False)

    regime_rows = []
    for regime, cov in (
        ("high_flow", 0.60),
        ("normal_flow", 0.928832),
        ("low_flow", 1.0),
    ):
        regime_rows.append(
            {
                "method": "empirical_residual",
                "horizon_days": 1,
                "meteo_scenario": "oracle",
                "nominal_coverage": 0.9,
                "regime": regime,
                "empirical_coverage": cov,
                "mean_width": 0.28,
                "n_days": 20 if regime == "high_flow" else 100,
                "threshold_high": 1.345873,
                "threshold_low": 0.085136,
            }
        )
    pd.DataFrame(regime_rows).to_csv(output / "uncertainty_regime_coverage.csv", index=False)

    pd.DataFrame(
        {
            "method": ["empirical_residual"] * 4,
            "horizon_days": [1] * 4,
            "meteo_scenario": ["oracle"] * 4,
            "event_id": [4, 3, 2, 1],
            "date": ["2015-05-05", "2015-02-02", "2015-01-17", "2014-12-20"],
            "q_obs": [2.754, 1.777, 1.459, 1.358],
            "q_point": [2.654, 1.811, 1.588, 1.369],
            "point_error": [0.1, -0.03, -0.13, -0.01],
            "lower_90": [2.50, 1.66, 1.44, 1.22],
            "upper_90": [2.79, 1.94, 1.72, 1.50],
            "inside_90": [True, True, True, True],
            "interval_width": [0.28, 0.28, 0.28, 0.28],
        }
    ).to_csv(output / "uncertainty_extreme_events.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2014-01-01"],
            "horizon_days": [1],
            "meteo_scenario": ["oracle"],
            "method": ["empirical_residual"],
            "q_obs": [1.0],
            "q_point": [1.0],
            "lower_80": [0.8],
            "upper_80": [1.2],
            "lower_90": [0.7],
            "upper_90": [1.3],
            "lower_95": [0.6],
            "upper_95": [1.4],
            "period": ["validation"],
        }
    ).to_csv(output / "uncertainty_forecasts.csv", index=False)

    (output / "uncertainty_answers.json").write_text(
        json.dumps(
            {
                "answers": {
                    "preferred_method": "empirical_residual",
                    "preferred_method_label": "preferred after validation comparison",
                    "10_needs_new_test_period": "A new untouched test period is required.",
                },
                "crps_note": "CRPS deferred — interval score used as primary probabilistic score.",
            }
        ),
        encoding="utf-8",
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
    (output / "rapport_calage.md").write_text(
        "# Rapport de calage\ncontenu\n", encoding="utf-8"
    )
    (output / "experiment_metadata.json").write_text(
        '{"calibration_runtime_s": 100.0}',
        encoding="utf-8",
    )
    _write_ml_artifacts(output)

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


def test_export_fails_if_ml_artifacts_missing(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    (output / "ml_model_comparison.csv").unlink()
    with pytest.raises(DemoExportError, match="ml_model_comparison|Artéfacts requis"):
        build_demo_data(config_path, output)


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
    assert data["title"] == "Calage automatisé d'un modèle pluie-débit"
    assert "NON DESTINÉ À LA DÉCISION HYDROLOGIQUE OPÉRATIONNELLE" in data["prototype_badge"]
    assert "calage ni pour le classement" in data["statements"]["validation_isolation"]


def test_ml_correction_values_match_artifacts(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    data = build_demo_data(config_path, output)
    ml = data["ml_correction"]
    comparison = pd.read_csv(output / "ml_model_comparison.csv")
    ablation = pd.read_csv(output / "ml_ablation_comparison.csv")
    answers = json.loads((output / "ml_ablation_answers.json").read_text(encoding="utf-8"))

    by_id = {m["id"]: m for m in ml["models"]}
    assert by_id["physical"]["kge_val"] == pytest.approx(
        float(comparison.loc[comparison["model"] == "physical_gr4j", "kge_val"].iloc[0])
    )
    assert by_id["persistence"]["kge_val"] == pytest.approx(
        float(ablation.loc[ablation["model"] == "persistence", "kge_val"].iloc[0])
    )
    assert by_id["ar1"]["kge_val"] == pytest.approx(
        float(ablation.loc[ablation["model"] == "ar1", "kge_val"].iloc[0])
    )
    assert by_id["ridge"]["kge_val"] == pytest.approx(
        float(comparison.loc[comparison["model"] == "ridge", "kge_val"].iloc[0])
    )
    assert by_id["hgb"]["kge_val"] == pytest.approx(
        float(
            comparison.loc[
                comparison["model"] == "hist_gradient_boosting", "kge_val"
            ].iloc[0]
        )
    )

    expected_pct = (
        100.0
        * (by_id["persistence"]["kge_val"] - by_id["physical"]["kge_val"])
        / (by_id["ridge"]["kge_val"] - by_id["physical"]["kge_val"])
    )
    assert ml["gain_captured_pct"] == pytest.approx(expected_pct)
    assert ml["gain_captured_display"] == f"~{expected_pct:.1f}%"
    assert ml["residual_acf"]["lag_1"] == pytest.approx(answers["acf_summary"]["acf_lag_1"])
    assert ml["residual_acf"]["lag_2"] == pytest.approx(answers["acf_summary"]["acf_lag_2"])
    assert ml["residual_acf"]["lag_3"] == pytest.approx(answers["acf_summary"]["acf_lag_3"])
    assert "Validation never used for fitting or ranking" not in ml["guardrails"]
    assert any("validation" in g.lower() and ("jamais" in g.lower() or "fitting" not in g) for g in ml["guardrails"])
    assert any("calage" in g.lower() or "période" in g.lower() for g in ml["guardrails"])
    assert "horizon" in ml["scope_note"].lower() or "forçage" in ml["scope_note"].lower()
    assert ml["section_title"] == "CORRECTION PILOTÉE PAR LES DONNÉES"
    assert "Machine Learning" in ml["section_subtitle"] or "données" in ml["section_subtitle"].lower()
    assert "Décision d'ingénierie" == ml["engineering_decision"]["heading"] or "ingénierie" in ml["engineering_decision"]["heading"]
    assert "98,7" not in ml["engineering_decision"]["body"]  # no wrong "ML improves by 98.7%" claim
    assert "justifiée" in ml["engineering_decision"]["body"].lower() or "complexité" in ml["engineering_decision"]["body"].lower()



def test_ml_correction_payload_fails_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(DemoExportError, match="Artéfact ML manquant|ml_model_comparison"):
        build_ml_correction_payload(tmp_path)


def test_forecast_horizons_values_match_artifacts(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    data = build_demo_data(config_path, output)
    fh = data["forecast_horizons"]
    comparison = pd.read_csv(output / "ml_horizon_comparison.csv")
    highflow = pd.read_csv(output / "ml_horizon_highflow.csv")
    degradation = pd.read_csv(output / "ml_horizon_degradation.csv")

    assert fh["forcing"] == "oracle"
    assert fh["operational_skill"] is False
    assert "ORACLE" in fh["forcing_label"].upper() or "FORÇAGE" in fh["forcing_label"].upper()
    assert "FORÇAGE MÉTÉOROLOGIQUE ORACLE" in fh["oracle_banner_title"]
    assert "opérationnelle" in fh["oracle_banner_emphasis"].lower()
    assert "PAS UNE RÉFÉRENCE" in fh["scope_note"] or "opérationnelle" in fh["scope_note"].lower()
    assert fh["section_title"] == "PRÉVISION À 24, 48 ET 72 HEURES"
    assert len(fh["horizons"]) == 3

    for row in fh["horizons"]:
        h = row["horizon_days"]
        for model_key, col in (
            ("physical", "physical_kge"),
            ("persistence_residual", "persistence_kge"),
            ("AR1_residual", "ar1_kge"),
            ("ridge", "ridge_kge"),
        ):
            expected = float(
                comparison.loc[
                    (comparison["model"] == model_key)
                    & (comparison["horizon_days"] == h),
                    "kge",
                ].iloc[0]
            )
            assert row[col] == pytest.approx(expected)
        assert row["persistence_gain_vs_physical"] == pytest.approx(
            row["persistence_kge"] - row["physical_kge"]
        )
        assert row["persistence_gain_display"] == (
            f"{row['persistence_gain_vs_physical']:+.3f}"
        )

    pers_deg = degradation.loc[
        degradation["model"] == "persistence_residual"
    ].iloc[0]
    expected_loss = float(pers_deg["kge_h1"]) - float(pers_deg["kge_h3"])
    assert fh["persistence_degradation_24_to_72"] == pytest.approx(expected_loss)
    assert fh["persistence_degradation_display"] == f"{expected_loss:.3f}"

    for row in fh["high_flow"]["rows"]:
        h = row["horizon_days"]
        for model_key, col in (
            ("physical", "physical_mae"),
            ("persistence_residual", "persistence_mae"),
        ):
            expected = float(
                highflow.loc[
                    (highflow["model"] == model_key)
                    & (highflow["horizon_days"] == h),
                    "mae_highflow",
                ].iloc[0]
            )
            assert row[col] == pytest.approx(expected)


def test_forecast_horizons_payload_fails_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(DemoExportError, match="Phase 8D|ml_horizon_comparison"):
        build_forecast_horizons_payload(tmp_path)


def test_export_fails_if_horizon_artifacts_missing(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    (output / "ml_horizon_comparison.csv").unlink()
    with pytest.raises(DemoExportError, match="ml_horizon_comparison|Artéfacts requis"):
        build_demo_data(config_path, output)


def test_meteo_sensitivity_values_match_artifacts(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    data = build_demo_data(config_path, output)
    ms = data["meteo_sensitivity"]
    hydro = pd.read_csv(output / "meteo_hydrology_comparison.csv")
    mc = pd.read_csv(output / "meteo_monte_carlo_summary.csv")
    hf = pd.read_csv(output / "meteo_highflow_comparison.csv")

    assert ms["experiment_type"] == "synthetic_sensitivity"
    assert ms["real_weather_benchmark"] is False
    assert ms["n_meteo_realizations"] == 100
    assert "NE CONSTITUE PAS UNE ÉVALUATION" in ms["banner_emphasis"]
    assert ms["section_title"] == "SENSIBILITÉ AU FORÇAGE MÉTÉOROLOGIQUE"
    assert "dégradation synthétique" in ms["skill_loss_label"].lower()
    assert "SOURCE D'INCERTITUDE MAJEURE" in ms["nwp_label"]
    assert "AROME" not in json.dumps(ms)
    assert "ECMWF" not in ms["banner_body"]
    assert "AROME-like" not in json.dumps(ms)

    for scen in ("oracle", "moderate", "strong"):
        for h in (1, 2, 3):
            expected = float(
                hydro.loc[
                    (hydro["scenario"] == scen)
                    & (hydro["model"] == "physical")
                    & (hydro["horizon_days"] == h),
                    "kge",
                ].iloc[0]
            )
            assert ms["physical_kge"][scen][str(h * 24)] == pytest.approx(expected)

    for h in (1, 2, 3):
        hours = str(h * 24)
        expected_loss = (
            ms["physical_kge"]["oracle"][hours] - ms["physical_kge"]["strong"][hours]
        )
        assert ms["strong_skill_loss"][hours] == pytest.approx(expected_loss)

    mc72 = mc.loc[
        (mc["scenario"] == "strong")
        & (mc["model"] == "physical")
        & (mc["horizon_days"] == 3)
    ].iloc[0]
    assert ms["monte_carlo"]["kge_median"] == pytest.approx(float(mc72["kge_median"]))
    assert ms["monte_carlo"]["kge_p10"] == pytest.approx(float(mc72["kge_p10"]))
    assert ms["monte_carlo"]["kge_p90"] == pytest.approx(float(mc72["kge_p90"]))
    assert ms["monte_carlo"]["n_realizations"] == 100

    for row in ms["persistence_comparison"]["rows"]:
        expected_p = float(
            hydro.loc[
                (hydro["scenario"] == row["scenario"])
                & (hydro["model"] == "physical")
                & (hydro["horizon_days"] == row["hours"] // 24),
                "kge",
            ].iloc[0]
        )
        assert row["physical_kge"] == pytest.approx(expected_p)

    for row in ms["high_flow"]["rows"]:
        expected = float(
            hf.loc[
                (hf["scenario"] == row["scenario"])
                & (hf["model"] == "physical")
                & (hf["horizon_days"] == row["hours"] // 24),
                "mae_highflow",
            ].iloc[0]
        )
        assert row["physical_mae"] == pytest.approx(expected)


def test_meteo_sensitivity_payload_fails_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(DemoExportError, match="Phase 8E|meteo_hydrology"):
        build_meteo_sensitivity_payload(tmp_path)


def test_export_fails_if_meteo_artifacts_missing(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    (output / "meteo_hydrology_comparison.csv").unlink()
    with pytest.raises(DemoExportError, match="meteo_hydrology|Artéfacts requis"):
        build_demo_data(config_path, output)


def test_forecast_uncertainty_values_match_artifacts(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    data = build_demo_data(config_path, output)
    fu = data["forecast_uncertainty"]
    coverage = pd.read_csv(output / "uncertainty_coverage_summary.csv")
    regime = pd.read_csv(output / "uncertainty_regime_coverage.csv")
    extremes = pd.read_csv(output / "uncertainty_extreme_events.csv")

    assert fu["preferred_method"] == "empirical_residual"
    assert fu["preferred_label"] == "préféré après comparaison en validation"
    assert "période de test" in fu["hero"]["caveat"].lower()
    assert "validation" in fu["calibration_note"].lower()
    assert fu["section_title"] == "INCERTITUDE DE PRÉVISION"
    assert "90 %" in fu["central_statement"] or "90%" in fu["central_statement"]
    assert "DÉPENDANCE AU RÉGIME" in fu["regime_coverage"]["warning_title"]
    assert "confiance" in fu["final_takeaway"].lower()

    for method in (
        "behavioral_parametric",
        "conditional_quantile",
        "empirical_residual",
        "split_conformal",
    ):
        for h in (1, 2, 3):
            expected = float(
                coverage.loc[
                    (coverage["method"] == method)
                    & (coverage["horizon_days"] == h)
                    & (coverage["meteo_scenario"] == "oracle")
                    & (coverage["nominal_coverage"] == 0.90),
                    "empirical_coverage",
                ].iloc[0]
            )
            cell = fu["coverage_90"][method][str(h * 24)]
            assert cell["empirical_coverage"] == pytest.approx(expected)
            assert cell["coverage_error"] == pytest.approx(expected - 0.90)

    hero = fu["hero"]
    assert hero["empirical"] == pytest.approx(
        float(
            coverage.loc[
                (coverage["method"] == "empirical_residual")
                & (coverage["horizon_days"] == 1)
                & (coverage["meteo_scenario"] == "oracle")
                & (coverage["nominal_coverage"] == 0.90),
                "empirical_coverage",
            ].iloc[0]
        )
    )
    assert hero["coverage_error"] == pytest.approx(hero["empirical"] - 0.90)

    for regime_name in ("high_flow", "normal_flow", "low_flow"):
        expected = float(
            regime.loc[regime["regime"] == regime_name, "empirical_coverage"].iloc[0]
        )
        assert fu["regime_coverage"][regime_name]["empirical_coverage"] == pytest.approx(
            expected
        )
    assert fu["regime_coverage"]["high_flow"]["threshold_high"] == pytest.approx(
        float(regime["threshold_high"].iloc[0])
    )

    assert len(fu["extreme_events"]) <= 5
    top = extremes.sort_values("q_obs", ascending=False).iloc[0]
    assert fu["extreme_events"][0]["date"] == str(top["date"])
    assert fu["extreme_events"][0]["q_obs"] == pytest.approx(float(top["q_obs"]))

    # Behavioral not rewritten as calibrated 90% CI
    assert "intervalle de prévision" in fu["method_cards"][0]["interpretation"].lower()
    assert "sémantique" in fu["method_cards"][0]["interpretation"].lower()


def test_forecast_uncertainty_payload_fails_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(DemoExportError, match="Phase 9|uncertainty_coverage"):
        build_forecast_uncertainty_payload(tmp_path)


def test_export_fails_if_uncertainty_artifacts_missing(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    (output / "uncertainty_coverage_summary.csv").unlink()
    with pytest.raises(DemoExportError, match="uncertainty_coverage|Artéfacts requis"):
        build_demo_data(config_path, output)


def test_export_writes_assets(tmp_path: Path) -> None:
    config_path, output = _write_min_artifacts(tmp_path)
    demo_dir = tmp_path / "demo"
    paths = export_static_demo(config_path, output, demo_dir)
    assert paths["demo_data"].is_file()
    payload = json.loads(paths["demo_data"].read_text(encoding="utf-8"))
    assert payload["reproducibility"]["config_sha256"]
    assert "Calage automatisé" in payload["title"]
    assert "ml_correction" in payload
    assert "forecast_horizons" in payload
    assert "meteo_sensitivity" in payload
    assert "forecast_uncertainty" in payload
    assert payload["forecast_horizons"]["operational_skill"] is False
    assert payload["meteo_sensitivity"]["real_weather_benchmark"] is False
    assert payload["forecast_uncertainty"]["preferred_method"] == "empirical_residual"
    assert (demo_dir / "assets" / "demo_01_calibration_impact.png").is_file()
    report_html = (demo_dir / "assets" / "rapport_calage.html").read_text(encoding="utf-8")
    assert "Rapport de calage" in report_html
    assert "Retour à la démo" in report_html


def test_frontend_has_no_scientific_calculations() -> None:
    root = Path(__file__).resolve().parents[1] / "demo"
    js = (root / "app.js").read_text(encoding="utf-8")
    html = (root / "index.html").read_text(encoding="utf-8")
    for token in (
        "0.2026",
        "0.8350",
        "0.835003",
        "0.985806",
        "0.983784",
        "0.836747",
        "0.032176",
        "0.776091",
        "0.062255",
        "0.935528",
        "0.256475",
        "59.9",
        "98.7",
        "5000",
        "126",
        "latin_hypercube_sample",
        "run_continuous_gr4j",
        "persistence_KGE",
        "ridge_KGE",
    ):
        assert token not in js
    assert "persistence_kge - physical" not in js
    assert "oracle_kge - strong" not in js
    assert "empirical - 0.90" not in js
    assert "kge_h1" not in js
    assert "delta_KGE" not in js
    assert "(persistence" not in js
    assert "0.203 → 0.835" not in html
    assert "5,000" not in html
    assert "126" not in html
    assert "+61.1%" not in html
    assert "PROTOTYPE — NON DESTINÉ À LA DÉCISION HYDROLOGIQUE OPÉRATIONNELLE" in html
    assert (
        "Les observations de validation ne sont jamais utilisées pour le calage "
        "ni pour le classement des paramètres."
    ) in html
    assert "q05–q95 N'EST PAS un intervalle de prédiction calibré à 90 %" in html
    assert "intégration HEC-HMS est une étape future" in html
    assert "non implémentée dans ce prototype" in html.lower()
    assert "Automated Rainfall–Runoff Calibration" not in html
    assert "NOT FOR OPERATIONAL DECISION-MAKING" not in html
    assert "Data-driven correction" not in html
    assert "Correction pilotée par les données" in html
    assert "panel-ml-correction" in html
    assert "Forecast 24–72 h" not in html
    assert "Prévision à 24–72 h" in html
    assert "panel-forecast-horizons" in html
    assert "FORÇAGE MÉTÉOROLOGIQUE ORACLE" in html or "forçage météorologique oracle" in html.lower()
    assert "Meteo sensitivity" not in html
    assert "Sensibilité au forçage météorologique" in html
    assert "panel-meteo-sensitivity" in html
    assert "NE CONSTITUE PAS UNE ÉVALUATION" in html or "système réel de prévision" in html.lower()
    assert "Forecast uncertainty" not in html
    assert "Incertitude de prévision" in html
    assert "panel-forecast-uncertainty" in html
    assert "AROME-like" not in html
    assert "ECMWF-like" not in html
    assert "More ML complexity is not justified" not in html
    assert "Engineering decision" not in html
    assert "Next question" not in html
    assert "Skill loss" not in html
    assert "High flow" not in html
    assert "98.7" not in html
    assert "0.147" not in html
    assert "93.6%" not in html  # comes from JSON


def test_french_presentation_layer_phase_10e(tmp_path: Path) -> None:
    """Phase 10E: user-visible demo strings are French; scientific values unchanged."""
    config_path, output = _write_min_artifacts(tmp_path)
    data = build_demo_data(config_path, output)
    root = Path(__file__).resolve().parents[1] / "demo"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    report = (root / "assets" / "rapport_calage.html").read_text(encoding="utf-8")

    assert data["ml_correction"]["section_title"] == "CORRECTION PILOTÉE PAR LES DONNÉES"
    assert data["forecast_horizons"]["section_title"] == "PRÉVISION À 24, 48 ET 72 HEURES"
    assert data["meteo_sensitivity"]["section_title"] == "SENSIBILITÉ AU FORÇAGE MÉTÉOROLOGIQUE"
    assert data["forecast_uncertainty"]["section_title"] == "INCERTITUDE DE PRÉVISION"

    assert "FORÇAGE MÉTÉOROLOGIQUE ORACLE" in data["forecast_horizons"]["oracle_banner_title"]
    assert "NE CONSTITUE PAS UNE ÉVALUATION" in data["meteo_sensitivity"]["banner_emphasis"]
    assert "90 %" in data["forecast_uncertainty"]["central_statement"] or (
        "90%" in data["forecast_uncertainty"]["central_statement"]
    )
    assert "confiance" in data["forecast_uncertainty"]["final_takeaway"].lower()
    assert "DÉPENDANCE AU RÉGIME" in data["forecast_uncertainty"]["regime_coverage"]["warning_title"]
    assert "HYDROLOGIQUE" in data["prototype_badge"]

    # No mixed-language presentation leftovers in HTML/JS sources
    for bad in (
        "Data-driven correction",
        "Forecast 24",
        "Meteo sensitivity",
        "Forecast uncertainty",
        "Engineering decision",
        "Next question",
        "Skill loss",
        "High flow",
        "Validation coverage",
        "ORACLE METEOROLOGICAL FORCING",
        "NOT A REAL WEATHER-FORECAST",
        "NOT FOR OPERATIONAL DECISION-MAKING",
        "preferred after validation comparison",
    ):
        assert bad not in html
        assert bad not in js

    # UTF-8 / no mojibake in report
    assert 'charset="utf-8"' in report.lower() or "charset=utf-8" in report.lower()
    for marker in ("Ã©", "Ã¨", "â€“", "â†’", "kmÂ²"):
        assert marker not in report
        assert marker not in html
    assert "é" in report or "è" in report or "à" in report


def test_frontend_uncertainty_and_hecras_wording() -> None:
    html = (Path(__file__).resolve().parents[1] / "demo" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "dispersion paramétrique" in html.lower()
    assert "HEC-HMS" in html
    assert "future" in html.lower()
    assert "non implémentée" in html.lower() or "travaux futurs" in html.lower()
