"""Export presentation-only demo assets from generated scientific artifacts.

No scientific calculations are redefined here: values are read from config,
CSV outputs, and experiment metadata written by the Python pipeline.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.calibration_diagnostics import kge_cal_val_correlation
from src.ensemble import empirical_validation_coverage, get_behavioral_threshold
from src.experiment_metadata import config_sha256, git_commit_hash, load_metadata
from src.validation import select_best_calibration_candidate

REQUIRED_ARTIFACTS = (
    "runs.csv",
    "metrics_uncalibrated.csv",
    "behavioral_runs.csv",
    "ensemble_timeseries.csv",
    "demo_01_calibration_impact.png",
    "demo_02_validation.png",
    "demo_03_uncertainty.png",
    "rapport_calage.md",
    "ml_model_comparison.csv",
    "ml_ablation_comparison.csv",
    "ml_ablation_answers.json",
    "ml_horizon_comparison.csv",
    "ml_horizon_highflow.csv",
    "ml_horizon_degradation.csv",
    "ml_horizon_answers.json",
    "meteo_hydrology_comparison.csv",
    "meteo_scenario_diagnostics.csv",
    "meteo_highflow_comparison.csv",
    "meteo_monte_carlo_summary.csv",
    "meteo_sensitivity_answers.json",
    "uncertainty_coverage_summary.csv",
    "uncertainty_regime_coverage.csv",
    "uncertainty_extreme_events.csv",
    "uncertainty_forecasts.csv",
    "uncertainty_answers.json",
)

DEMO_FIGURES = (
    "demo_01_calibration_impact.png",
    "demo_02_validation.png",
    "demo_03_uncertainty.png",
)

OPTIONAL_METEO_FIGURES = (
    "meteo_sensitivity_summary.png",
    "meteo_sensitivity_event.png",
)

OPTIONAL_UNCERTAINTY_FIGURES = (
    "uncertainty_reliability.png",
    "uncertainty_rolling_coverage.png",
    "uncertainty_calibrated_demo.png",
    "uncertainty_before_after.png",
)

# Phase 8B comparison + Phase 8C ablation model keys
ML_COMPARISON_PHYSICAL = "physical_gr4j"
ML_COMPARISON_RIDGE = "ridge"
ML_COMPARISON_HGB = "hist_gradient_boosting"
ML_ABLATION_PHYSICAL = "physical"
ML_ABLATION_PERSISTENCE = "persistence"
ML_ABLATION_AR1 = "ar1"
ML_ABLATION_RIDGE = "ridge_full"

# Phase 8D horizon model keys
HORIZON_PHYSICAL = "physical"
HORIZON_PERSISTENCE = "persistence_residual"
HORIZON_AR1 = "AR1_residual"
HORIZON_RIDGE = "ridge"
HORIZON_DAYS = (1, 2, 3)


class DemoExportError(FileNotFoundError):
    """Raised when required scientific artifacts are missing."""


def _baseline_metric(baseline: pd.DataFrame, name: str, column: str) -> float:
    row = baseline.loc[baseline["metric"] == name]
    if row.empty:
        raise KeyError(f"Metric {name!r} missing from metrics_uncalibrated.csv")
    return float(row.iloc[0][column])


def _require_model_kge(frame: pd.DataFrame, model: str, *, source: str) -> float:
    if "model" not in frame.columns or "kge_val" not in frame.columns:
        raise DemoExportError(f"{source} must contain columns model and kge_val")
    row = frame.loc[frame["model"] == model]
    if row.empty:
        raise DemoExportError(f"Model {model!r} missing from {source}")
    value = float(row.iloc[0]["kge_val"])
    if not pd.notna(value):
        raise DemoExportError(f"Model {model!r} has non-finite kge_val in {source}")
    return value


def build_ml_correction_payload(output_dir: Path) -> dict[str, Any]:
    """Assemble Phase 10A presentation block from Phase 8B/8C artifacts only."""
    comparison_path = output_dir / "ml_model_comparison.csv"
    ablation_path = output_dir / "ml_ablation_comparison.csv"
    answers_path = output_dir / "ml_ablation_answers.json"
    for path in (comparison_path, ablation_path, answers_path):
        if not path.is_file():
            raise DemoExportError(
                f"Artéfact ML manquant pour la section démo Phase 10A : {path.name}. "
                "Exécuter d'abord `--train-ml-baselines` et `--ml-ablation`."
            )

    comparison = pd.read_csv(comparison_path)
    ablation = pd.read_csv(ablation_path)
    answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))
    acf = answers_payload.get("acf_summary") or {}
    for key in ("acf_lag_1", "acf_lag_2", "acf_lag_3"):
        if key not in acf:
            raise DemoExportError(
                f"acf_summary.{key} missing from ml_ablation_answers.json"
            )

    physical = _require_model_kge(
        comparison, ML_COMPARISON_PHYSICAL, source="ml_model_comparison.csv"
    )
    ridge = _require_model_kge(
        comparison, ML_COMPARISON_RIDGE, source="ml_model_comparison.csv"
    )
    hgb = _require_model_kge(
        comparison, ML_COMPARISON_HGB, source="ml_model_comparison.csv"
    )
    persistence = _require_model_kge(
        ablation, ML_ABLATION_PERSISTENCE, source="ml_ablation_comparison.csv"
    )
    ar1 = _require_model_kge(
        ablation, ML_ABLATION_AR1, source="ml_ablation_comparison.csv"
    )
    physical_ablation = _require_model_kge(
        ablation, ML_ABLATION_PHYSICAL, source="ml_ablation_comparison.csv"
    )
    ridge_ablation = _require_model_kge(
        ablation, ML_ABLATION_RIDGE, source="ml_ablation_comparison.csv"
    )
    if abs(physical - physical_ablation) > 1e-9 or abs(ridge - ridge_ablation) > 1e-9:
        raise DemoExportError(
            "Incohérence Phase 8B/8C : KGE physical/ridge diffèrent entre "
            "ml_model_comparison.csv et ml_ablation_comparison.csv"
        )

    gain_total = ridge - physical
    if abs(gain_total) < 1e-12:
        raise DemoExportError(
            "Gain Physical→Ridge nul ou trop petit pour calculer le % capturé par "
            "la persistence"
        )
    gain_captured_pct = 100.0 * (persistence - physical) / gain_total

    models = [
        {
            "id": "physical",
            "label": "GR4J physique",
            "kge_val": physical,
            "source": "ml_model_comparison.csv",
        },
        {
            "id": "persistence",
            "label": "Persistance du résidu",
            "kge_val": persistence,
            "source": "ml_ablation_comparison.csv",
        },
        {
            "id": "ar1",
            "label": "AR(1)",
            "kge_val": ar1,
            "source": "ml_ablation_comparison.csv",
        },
        {
            "id": "ridge",
            "label": "Ridge",
            "kge_val": ridge,
            "source": "ml_model_comparison.csv",
        },
        {
            "id": "hgb",
            "label": "HistGradientBoosting",
            "kge_val": hgb,
            "source": "ml_model_comparison.csv",
        },
    ]

    return {
        "section_title": "CORRECTION PILOTÉE PAR LES DONNÉES",
        "section_subtitle": (
            "Une couche de Machine Learning peut-elle améliorer le modèle physique "
            "sans remplacer ses fondements hydrologiques ?"
        ),
        "methodology_label": "Période de validation · aucun réglage sur la validation",
        "pipeline_steps": [
            "PHYSIQUE D'ABORD",
            "CARACTÉRISER L'ERREUR RÉSIDUELLE",
            "TESTER UNE CORRECTION SIMPLE PILOTÉE PAR LES DONNÉES",
            "AJOUTER DE LA COMPLEXITÉ ML UNIQUEMENT SI ELLE APPORTE UNE VALEUR MESURABLE",
        ],
        "models": models,
        "kge_axis": {"min": 0.0, "max": 1.0, "label": "KGE de validation"},
        "gain_captured_pct": float(gain_captured_pct),
        "gain_captured_display": f"~{gain_captured_pct:.1f}%",
        "gain_captured_label": (
            "de l'amélioration de KGE Physique → Ridge est déjà capturée par la "
            "persistance du résidu"
        ),
        "residual_acf": {
            "lag_1": float(acf["acf_lag_1"]),
            "lag_2": float(acf["acf_lag_2"]),
            "lag_3": float(acf["acf_lag_3"]),
            "caption": "L'erreur du modèle physique a une mémoire temporelle.",
            "title": "Autocorrélation du résidu",
        },
        "explanation": {
            "physical": {
                "title": "1 — MODÈLE PHYSIQUE",
                "body": (
                    "GR4J représente la dynamique pluie–débit et demeure le cœur "
                    "hydrologique."
                ),
            },
            "residual": {
                "title": "2 — RÉSIDU",
                "equation": "Residual(t) = Qobserved(t) − Qphysical(t)",
                "body": (
                    "Les résidus de la période de calage présentent une forte "
                    "persistance temporelle."
                ),
            },
            "correction": {
                "title": "3 — CORRECTION PILOTÉE PAR LES DONNÉES",
                "body": (
                    "Un modèle statistique estime le résidu futur et corrige "
                    "la prévision physique :"
                ),
                "equation": "Qhybrid(t+h) = Qphysical(t+h) + residual_hat(t+h)",
            },
        },
        "result": {
            "heading": "Résultat",
            "body": (
                "Une correction pilotée par les données améliore sensiblement la "
                "référence physique. Mais un modèle simple de persistance du résidu "
                "capture presque toute l'amélioration obtenue par Ridge sur ce "
                "bassin pilote."
            ),
        },
        "engineering_decision": {
            "heading": "Décision d'ingénierie",
            "body": (
                "Dans ce bassin pilote, l'essentiel du gain provient d'une structure "
                "temporelle simple du résidu. Une complexité ML supplémentaire doit "
                "donc être justifiée par un gain mesurable."
            ),
        },
        "architecture_caption": (
            "Le modèle physique demeure le cœur hydrologique ; la couche pilotée "
            "par les données corrige le comportement résiduel systématique."
        ),
        "residual_layer_label": "Couche statistique / ML simple",
        "scope_note": (
            "Expérience de correction résiduelle — la performance opérationnelle "
            "de prévision dépend de l'horizon et du forçage météorologique."
        ),
        "guardrails": [
            "GR4J inchangé",
            "ML entraîné sur la période de calage uniquement",
            "Validation jamais utilisée pour l'ajustement ni le classement",
            "Même période de validation pour la comparaison des modèles",
            "Pas de fuite du débit observé du jour même",
            "Correction résiduelle évaluée face à des références simples",
        ],
        "artifacts": {
            "model_comparison": "ml_model_comparison.csv",
            "ablation_comparison": "ml_ablation_comparison.csv",
            "ablation_answers": "ml_ablation_answers.json",
        },
    }


def _horizon_metric(
    frame: pd.DataFrame,
    *,
    model: str,
    horizon_days: int,
    column: str,
    source: str,
) -> float:
    row = frame.loc[
        (frame["model"] == model) & (frame["horizon_days"] == horizon_days)
    ]
    if row.empty:
        raise DemoExportError(
            f"Missing {model!r} at horizon_days={horizon_days} in {source}"
        )
    value = float(row.iloc[0][column])
    if not pd.notna(value):
        raise DemoExportError(
            f"Non-finite {column} for {model!r} h={horizon_days} in {source}"
        )
    return value


def _format_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _svg_polyline_points(
    values: list[float],
    *,
    axis_min: float,
    axis_max: float,
    xs: list[float],
    y_top: float,
    y_bottom: float,
) -> str:
    """Map KGE values to SVG polyline points (presentation geometry only)."""
    span = axis_max - axis_min
    if span <= 0:
        raise DemoExportError("KGE chart axis span must be positive")
    pts: list[str] = []
    for x, value in zip(xs, values):
        y_frac = (float(value) - axis_min) / span
        y = y_bottom - y_frac * (y_bottom - y_top)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def build_forecast_horizons_payload(output_dir: Path) -> dict[str, Any]:
    """Assemble Phase 10B presentation block from Phase 8D artifacts only."""
    comparison_path = output_dir / "ml_horizon_comparison.csv"
    highflow_path = output_dir / "ml_horizon_highflow.csv"
    degradation_path = output_dir / "ml_horizon_degradation.csv"
    answers_path = output_dir / "ml_horizon_answers.json"
    acf_path = output_dir / "ml_ablation_answers.json"
    for path in (
        comparison_path,
        highflow_path,
        degradation_path,
        answers_path,
        acf_path,
    ):
        if not path.is_file():
            raise DemoExportError(
                f"Artéfact Phase 8D/8C manquant pour la section démo Phase 10B : "
                f"{path.name}. Exécuter d'abord `--ml-horizon-forecast` "
                "(et `--ml-ablation` pour l'ACF)."
            )

    comparison = pd.read_csv(comparison_path)
    highflow = pd.read_csv(highflow_path)
    degradation = pd.read_csv(degradation_path)
    answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))
    acf_payload = json.loads(acf_path.read_text(encoding="utf-8"))
    acf = acf_payload.get("acf_summary") or {}
    for key in ("acf_lag_1", "acf_lag_2", "acf_lag_3"):
        if key not in acf:
            raise DemoExportError(
                f"acf_summary.{key} missing from ml_ablation_answers.json"
            )

    experiment_tag = answers_payload.get("experiment_tag") or (
        (answers_payload.get("answers") or {}).get("experiment_tag")
    )
    if not experiment_tag:
        experiment_tag = "ORACLE METEOROLOGICAL FORCING"
    # Presentation-only: map English oracle tag to French display string.
    _tag_upper = str(experiment_tag).strip().upper()
    if _tag_upper == "ORACLE METEOROLOGICAL FORCING" or (
        "ORACLE" in _tag_upper and "FORCING" in _tag_upper
    ):
        experiment_tag = "FORÇAGE MÉTÉOROLOGIQUE ORACLE"

    horizons: list[dict[str, Any]] = []
    series_values: dict[str, list[float]] = {
        "physical": [],
        "persistence": [],
        "ar1": [],
        "ridge": [],
    }
    for h in HORIZON_DAYS:
        physical = _horizon_metric(
            comparison,
            model=HORIZON_PHYSICAL,
            horizon_days=h,
            column="kge",
            source="ml_horizon_comparison.csv",
        )
        persistence = _horizon_metric(
            comparison,
            model=HORIZON_PERSISTENCE,
            horizon_days=h,
            column="kge",
            source="ml_horizon_comparison.csv",
        )
        ar1 = _horizon_metric(
            comparison,
            model=HORIZON_AR1,
            horizon_days=h,
            column="kge",
            source="ml_horizon_comparison.csv",
        )
        ridge = _horizon_metric(
            comparison,
            model=HORIZON_RIDGE,
            horizon_days=h,
            column="kge",
            source="ml_horizon_comparison.csv",
        )
        gain = persistence - physical
        hours = int(h * 24)
        horizons.append(
            {
                "horizon_days": h,
                "hours": hours,
                "label": f"+{hours} h",
                "physical_kge": physical,
                "persistence_kge": persistence,
                "ar1_kge": ar1,
                "ridge_kge": ridge,
                "persistence_gain_vs_physical": gain,
                "physical_kge_display": f"{physical:.3f}",
                "persistence_kge_display": f"{persistence:.3f}",
                "ar1_kge_display": f"{ar1:.3f}",
                "ridge_kge_display": f"{ridge:.3f}",
                "persistence_gain_display": _format_signed(gain, 3),
            }
        )
        series_values["physical"].append(physical)
        series_values["persistence"].append(persistence)
        series_values["ar1"].append(ar1)
        series_values["ridge"].append(ridge)

    deg_row = degradation.loc[degradation["model"] == HORIZON_PERSISTENCE]
    if deg_row.empty:
        raise DemoExportError(
            "persistence_residual missing from ml_horizon_degradation.csv"
        )
    delta_1_to_3 = float(deg_row.iloc[0]["delta_KGE_1_to_3"])
    kge_loss = float(deg_row.iloc[0]["kge_h1"]) - float(deg_row.iloc[0]["kge_h3"])

    high_flow_rows: list[dict[str, Any]] = []
    threshold = None
    for h in HORIZON_DAYS:
        phys_mae = _horizon_metric(
            highflow,
            model=HORIZON_PHYSICAL,
            horizon_days=h,
            column="mae_highflow",
            source="ml_horizon_highflow.csv",
        )
        pers_mae = _horizon_metric(
            highflow,
            model=HORIZON_PERSISTENCE,
            horizon_days=h,
            column="mae_highflow",
            source="ml_horizon_highflow.csv",
        )
        thr = _horizon_metric(
            highflow,
            model=HORIZON_PHYSICAL,
            horizon_days=h,
            column="threshold_q_obs",
            source="ml_horizon_highflow.csv",
        )
        threshold = thr
        high_flow_rows.append(
            {
                "horizon_days": h,
                "hours": int(h * 24),
                "label": f"+{h * 24} h",
                "physical_mae": phys_mae,
                "persistence_mae": pers_mae,
                "physical_mae_display": f"{phys_mae:.3f}",
                "persistence_mae_display": f"{pers_mae:.3f}",
            }
        )

    # Honest KGE axis 0–1 (does not exaggerate Persistence vs Ridge).
    axis_min, axis_max = 0.0, 1.0
    xs = [40.0, 160.0, 280.0]
    y_top, y_bottom = 12.0, 118.0
    chart_series = [
        {
            "id": "physical",
            "label": "GR4J physique",
            "kge": series_values["physical"],
            "polyline": _svg_polyline_points(
                series_values["physical"],
                axis_min=axis_min,
                axis_max=axis_max,
                xs=xs,
                y_top=y_top,
                y_bottom=y_bottom,
            ),
        },
        {
            "id": "persistence",
            "label": "Persistance",
            "kge": series_values["persistence"],
            "polyline": _svg_polyline_points(
                series_values["persistence"],
                axis_min=axis_min,
                axis_max=axis_max,
                xs=xs,
                y_top=y_top,
                y_bottom=y_bottom,
            ),
        },
        {
            "id": "ar1",
            "label": "AR(1)",
            "kge": series_values["ar1"],
            "polyline": _svg_polyline_points(
                series_values["ar1"],
                axis_min=axis_min,
                axis_max=axis_max,
                xs=xs,
                y_top=y_top,
                y_bottom=y_bottom,
            ),
        },
        {
            "id": "ridge",
            "label": "Ridge",
            "kge": series_values["ridge"],
            "polyline": _svg_polyline_points(
                series_values["ridge"],
                axis_min=axis_min,
                axis_max=axis_max,
                xs=xs,
                y_top=y_top,
                y_bottom=y_bottom,
            ),
        },
    ]

    return {
        "section_title": "PRÉVISION À 24, 48 ET 72 HEURES",
        "section_subtitle": (
            "Comment évolue la performance de prévision de +24 h à +72 h ?"
        ),
        "forcing": "oracle",
        "forcing_label": experiment_tag,
        "operational_skill": False,
        "oracle_banner_title": "FORÇAGE MÉTÉOROLOGIQUE ORACLE",
        "oracle_banner_body": (
            "Les précipitations futures observées sont utilisées comme forçage. "
            "Il ne s'agit pas d'une performance météorologique opérationnelle. "
            "Cette expérience utilise les précipitations futures réellement "
            "observées afin d'isoler la composante hydrologique de la prévision."
        ),
        "oracle_banner_emphasis": (
            "Ces résultats ne constituent pas une mesure de performance "
            "opérationnelle de prévision."
        ),
        "scope_note": (
            "CECI N'EST PAS UNE RÉFÉRENCE DE PRÉVISION MÉTÉOROLOGIQUE OPÉRATIONNELLE. "
            "Qphysical(t+h) utilise les précipitations futures historiquement observées. "
            "L'expérience mesure donc la composante hydrologique / résiduelle de la "
            "performance de prévision sous forçage météorologique parfait. "
            "La performance opérationnelle dépendra en outre de la qualité des "
            "prévisions de précipitations."
        ),
        "horizons": horizons,
        "chart": {
            "title": "KGE de validation",
            "axis_min": axis_min,
            "axis_max": axis_max,
            "x_labels": [f"+{h * 24} h" for h in HORIZON_DAYS],
            "x_positions": xs,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "view_box": "0 0 320 140",
            "series": chart_series,
            "axis_note": "Axe KGE 0–1 (sans zoom sur le haut de l'échelle).",
        },
        "persistence_degradation_24_to_72": float(kge_loss),
        "persistence_delta_kge_1_to_3": float(delta_1_to_3),
        "persistence_degradation_display": f"{kge_loss:.3f}",
        "degradation_caption": (
            "La correction résiduelle reste utile à plus long horizon, mais sa "
            "valeur prédictive décroît progressivement."
        ),
        "residual_acf": {
            "title": "Autocorrélation du résidu",
            "lag_1": float(acf["acf_lag_1"]),
            "lag_2": float(acf["acf_lag_2"]),
            "lag_3": float(acf["acf_lag_3"]),
            "lag_1_display": f"{float(acf['acf_lag_1']):.2f}",
            "lag_2_display": f"{float(acf['acf_lag_2']):.2f}",
            "lag_3_display": f"{float(acf['acf_lag_3']):.2f}",
            "explanation": (
                "L'erreur actuelle du modèle contient une mémoire à court terme. "
                "Un résidu récent porte donc de l'information sur l'erreur de "
                "modèle à court terme."
            ),
        },
        "equation": {
            "hybrid": "Q_hybrid(t+h | t) = Q_physical(t+h | t) + ê_h(t)",
            "persistence": "ê_h(t) = e(t)",
            "residual_def": "e(t) = Q_observed(t) − Q_physical(t)",
            "note": (
                "Pour la référence de persistance, le résidu de modèle observé "
                "le plus récent est reporté à +24 / +48 / +72 h."
            ),
        },
        "architecture_caption": (
            "Précipitations futures étiquetées ORACLE P dans cette expérience."
        ),
        "high_flow": {
            "threshold_q_obs": float(threshold) if threshold is not None else None,
            "threshold_note": (
                "Seuil de haut débit dérivé des données de calage."
            ),
            "rows": high_flow_rows,
            "interpretation": (
                "La correction par persistance aide la prévision en haut débit ; "
                "son avantage diminue avec l'horizon."
            ),
        },
        "result": {
            "heading": "Résultat",
            "body": (
                "La correction résiduelle reste utile de +24 h à +72 h dans "
                "cette expérience contrôlée."
            ),
        },
        "model_complexity": {
            "heading": "Complexité du modèle",
            "body": (
                "Ridge ne surpasse pas sensiblement la persistance du résidu "
                "sur les trois horizons."
            ),
        },
        "next_question": {
            "heading": "Question suivante",
            "body": (
                "Quelle part de cette performance survit lorsque les précipitations "
                "futures ne sont plus parfaitement connues ?"
            ),
        },
        "artifacts": {
            "comparison": "ml_horizon_comparison.csv",
            "highflow": "ml_horizon_highflow.csv",
            "degradation": "ml_horizon_degradation.csv",
            "answers": "ml_horizon_answers.json",
        },
    }


def _meteo_cell(
    frame: pd.DataFrame,
    *,
    scenario: str,
    model: str,
    horizon_days: int,
    column: str,
    source: str,
) -> float:
    row = frame.loc[
        (frame["scenario"] == scenario)
        & (frame["model"] == model)
        & (frame["horizon_days"] == horizon_days)
    ]
    if row.empty:
        raise DemoExportError(
            f"Missing {scenario!r}/{model!r}/h={horizon_days} in {source}"
        )
    value = float(row.iloc[0][column])
    if not pd.notna(value):
        raise DemoExportError(
            f"Non-finite {column} for {scenario}/{model}/h={horizon_days} in {source}"
        )
    return value


def build_meteo_sensitivity_payload(output_dir: Path) -> dict[str, Any]:
    """Assemble Phase 10C presentation block from Phase 8E artifacts only."""
    hydro_path = output_dir / "meteo_hydrology_comparison.csv"
    diag_path = output_dir / "meteo_scenario_diagnostics.csv"
    highflow_path = output_dir / "meteo_highflow_comparison.csv"
    mc_path = output_dir / "meteo_monte_carlo_summary.csv"
    answers_path = output_dir / "meteo_sensitivity_answers.json"
    for path in (hydro_path, diag_path, highflow_path, mc_path, answers_path):
        if not path.is_file():
            raise DemoExportError(
                f"Artéfact Phase 8E manquant pour la section démo Phase 10C : "
                f"{path.name}. Exécuter d'abord `--meteo-sensitivity`."
            )

    hydro = pd.read_csv(hydro_path)
    diagnostics = pd.read_csv(diag_path)
    highflow = pd.read_csv(highflow_path)
    monte = pd.read_csv(mc_path)
    answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))

    scenarios = ("oracle", "moderate", "strong")
    scenario_label_fr = {
        "oracle": "Oracle",
        "moderate": "Modéré",
        "strong": "Fort",
    }
    scenario_cards = []
    for scen in scenarios:
        diag = diagnostics.loc[diagnostics["scenario"] == scen]
        if diag.empty:
            raise DemoExportError(f"Scenario {scen!r} missing from meteo_scenario_diagnostics.csv")
        drow = diag.iloc[0]
        purpose = {
            "oracle": "Expérience hydrologique de borne supérieure.",
            "moderate": "Tester la sensibilité à un forçage imparfait.",
            "strong": "Mettre à l'épreuve la robustesse de la prévision.",
        }[scen]
        precip_desc = {
            "oracle": "Précipitations futures observées.",
            "moderate": "Perturbation synthétique contrôlée des précipitations.",
            "strong": "Perturbation synthétique renforcée des précipitations.",
        }[scen]
        scenario_cards.append(
            {
                "id": scen,
                "label": scenario_label_fr[scen],
                "scenario_label": scenario_label_fr[scen],
                "description": precip_desc,
                "purpose": purpose,
                "n_realizations": int(drow["n_realizations"]),
            }
        )

    n_meteo = int(
        diagnostics.loc[diagnostics["scenario"] == "strong", "n_realizations"].iloc[0]
    )
    meteo_cfg = answers_payload.get("meteo_config") or {}
    if "n_realizations" in meteo_cfg:
        n_meteo = int(meteo_cfg["n_realizations"])

    physical_kge: dict[str, dict[str, Any]] = {}
    series_values: dict[str, list[float]] = {s: [] for s in scenarios}
    for scen in scenarios:
        by_h: dict[str, float] = {}
        for h in HORIZON_DAYS:
            kge = _meteo_cell(
                hydro,
                scenario=scen,
                model=HORIZON_PHYSICAL,
                horizon_days=h,
                column="kge",
                source="meteo_hydrology_comparison.csv",
            )
            by_h[str(h * 24)] = kge
            series_values[scen].append(kge)
        physical_kge[scen] = by_h

    strong_skill_loss: dict[str, Any] = {}
    loss_cards = []
    for h in HORIZON_DAYS:
        hours = str(h * 24)
        loss = physical_kge["oracle"][hours] - physical_kge["strong"][hours]
        strong_skill_loss[hours] = float(loss)
        loss_cards.append(
            {
                "hours": int(h * 24),
                "label": f"+{h * 24} h",
                "oracle_kge": physical_kge["oracle"][hours],
                "strong_kge": physical_kge["strong"][hours],
                "skill_loss": float(loss),
                "skill_loss_display": f"{loss:.3f}",
                "oracle_display": f"{physical_kge['oracle'][hours]:.3f}",
                "strong_display": f"{physical_kge['strong'][hours]:.3f}",
            }
        )

    hero_oracle = physical_kge["oracle"]["72"]
    hero_strong = physical_kge["strong"]["72"]
    hero_delta = hero_strong - hero_oracle  # signed for display as ΔKGE
    hero_loss = hero_oracle - hero_strong

    # Monte Carlo +72 h strong physical
    mc72 = monte.loc[
        (monte["scenario"] == "strong")
        & (monte["model"] == HORIZON_PHYSICAL)
        & (monte["horizon_days"] == 3)
    ]
    if mc72.empty:
        raise DemoExportError("Missing strong/physical/+72h in meteo_monte_carlo_summary.csv")
    mc_row = mc72.iloc[0]
    mc_median = float(mc_row["kge_median"])
    mc_p10 = float(mc_row["kge_p10"])
    mc_p90 = float(mc_row["kge_p90"])
    mc_spread = mc_p90 - mc_p10

    # Persistence vs physical under imperfect forcing
    persistence_rows = []
    for scen in ("moderate", "strong"):
        for h in HORIZON_DAYS:
            phys = _meteo_cell(
                hydro,
                scenario=scen,
                model=HORIZON_PHYSICAL,
                horizon_days=h,
                column="kge",
                source="meteo_hydrology_comparison.csv",
            )
            pers = _meteo_cell(
                hydro,
                scenario=scen,
                model=HORIZON_PERSISTENCE,
                horizon_days=h,
                column="kge",
                source="meteo_hydrology_comparison.csv",
            )
            persistence_rows.append(
                {
                    "scenario": scen,
                    "hours": int(h * 24),
                    "label": f"+{h * 24} h",
                    "physical_kge": phys,
                    "persistence_kge": pers,
                    "gain": pers - phys,
                    "physical_display": f"{phys:.3f}",
                    "persistence_display": f"{pers:.3f}",
                    "gain_display": _format_signed(pers - phys, 3),
                }
            )

    # Ridge vs persistence under imperfect forcing
    ridge_diffs = []
    for scen in ("moderate", "strong"):
        for h in HORIZON_DAYS:
            pers = _meteo_cell(
                hydro,
                scenario=scen,
                model=HORIZON_PERSISTENCE,
                horizon_days=h,
                column="kge",
                source="meteo_hydrology_comparison.csv",
            )
            ridge = _meteo_cell(
                hydro,
                scenario=scen,
                model=HORIZON_RIDGE,
                horizon_days=h,
                column="kge",
                source="meteo_hydrology_comparison.csv",
            )
            ridge_diffs.append(abs(ridge - pers))
    ridge_vs_pers_max = float(max(ridge_diffs)) if ridge_diffs else float("nan")
    ridge_vs_pers_mean = float(sum(ridge_diffs) / len(ridge_diffs)) if ridge_diffs else float("nan")

    # Scale comparison bar widths (presentation geometry)
    scale_max = max(mc_spread, ridge_vs_pers_max, 1e-9)
    meteo_bar_pct = 100.0 * mc_spread / scale_max
    ridge_bar_pct = 100.0 * ridge_vs_pers_max / scale_max

    # High-flow oracle vs strong
    hf_rows = []
    hf_threshold = None
    for scen in ("oracle", "strong"):
        for h in HORIZON_DAYS:
            phys_mae = _meteo_cell(
                highflow,
                scenario=scen,
                model=HORIZON_PHYSICAL,
                horizon_days=h,
                column="mae_highflow",
                source="meteo_highflow_comparison.csv",
            )
            pers_mae = _meteo_cell(
                highflow,
                scenario=scen,
                model=HORIZON_PERSISTENCE,
                horizon_days=h,
                column="mae_highflow",
                source="meteo_highflow_comparison.csv",
            )
            thr = _meteo_cell(
                highflow,
                scenario=scen,
                model=HORIZON_PHYSICAL,
                horizon_days=h,
                column="threshold_q_obs",
                source="meteo_highflow_comparison.csv",
            )
            hf_threshold = thr
            hf_rows.append(
                {
                    "scenario": scen,
                    "hours": int(h * 24),
                    "label": f"+{h * 24} h",
                    "physical_mae": phys_mae,
                    "persistence_mae": pers_mae,
                    "physical_mae_display": f"{phys_mae:.3f}",
                    "persistence_mae_display": f"{pers_mae:.3f}",
                }
            )

    axis_min, axis_max = 0.0, 1.0
    xs = [40.0, 160.0, 280.0]
    y_top, y_bottom = 12.0, 118.0
    chart_series = []
    for scen, label in (
        ("oracle", "Oracle"),
        ("moderate", "Modéré"),
        ("strong", "Fort"),
    ):
        chart_series.append(
            {
                "id": scen,
                "label": label,
                "kge": series_values[scen],
                "polyline": _svg_polyline_points(
                    series_values[scen],
                    axis_min=axis_min,
                    axis_max=axis_max,
                    xs=xs,
                    y_top=y_top,
                    y_bottom=y_bottom,
                ),
            }
        )

    # MC band geometry for +72h strong (display on 0–1 axis as a 1D bar)
    def _mc_pos(v: float) -> float:
        return 100.0 * (float(v) - axis_min) / (axis_max - axis_min)

    figures = {}
    for name in OPTIONAL_METEO_FIGURES:
        if (output_dir / name).is_file():
            figures[name.replace(".png", "").replace("meteo_sensitivity_", "")] = (
                f"assets/{name}"
            )

    return {
        "section_title": "SENSIBILITÉ AU FORÇAGE MÉTÉOROLOGIQUE",
        "section_subtitle": (
            "Que devient la prévision lorsque les précipitations futures ne sont plus "
            "connues parfaitement ?"
        ),
        "experiment_type": "synthetic_sensitivity",
        "real_weather_benchmark": False,
        "n_meteo_realizations": n_meteo,
        "banner_title": "ANALYSE DE SENSIBILITÉ SYNTHÉTIQUE AU FORÇAGE MÉTÉOROLOGIQUE",
        "banner_body": (
            f"Perturbations contrôlées des précipitations · N = {n_meteo} "
            "réalisations météorologiques"
        ),
        "banner_emphasis": (
            "NE CONSTITUE PAS UNE ÉVALUATION D'UN SYSTÈME RÉEL DE PRÉVISION MÉTÉOROLOGIQUE"
        ),
        "disclaimer": (
            "Ces scénarios sont des perturbations synthétiques des précipitations "
            "observées. Ils ne mesurent PAS la performance historique d'un système "
            "opérationnel de prévision météorologique. La prochaine étape de "
            "validation opérationnelle nécessiterait des archives de prévisions "
            "telles qu'elles étaient disponibles à chaque origine historique."
        ),
        "scenario_cards": scenario_cards,
        "methodology_details": {
            "n_realizations": n_meteo,
            "seed": meteo_cfg.get("seed"),
            "moderate": meteo_cfg.get("moderate"),
            "strong": meteo_cfg.get("strong"),
            "note": (
                "Les paramètres de perturbation décrivent une erreur synthétique "
                "contrôlée — et non un produit opérationnel de prévision numérique "
                "du temps."
            ),
        },
        "physical_kge": physical_kge,
        "chart": {
            "title": "GR4J physique — KGE de validation",
            "axis_min": axis_min,
            "axis_max": axis_max,
            "x_labels": [f"+{h * 24} h" for h in HORIZON_DAYS],
            "x_positions": xs,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "view_box": "0 0 320 140",
            "series": chart_series,
            "axis_note": "Axe KGE 0–1 (échelle honnête ; sans zoom sur le haut).",
        },
        "hero_72h": {
            "label": "Impact de la dégradation synthétique des précipitations à +72 h",
            "oracle_kge": hero_oracle,
            "strong_kge": hero_strong,
            "delta_kge": float(hero_delta),
            "skill_loss": float(hero_loss),
            "oracle_display": f"{hero_oracle:.3f}",
            "strong_display": f"{hero_strong:.3f}",
            "delta_display": _format_signed(hero_delta, 3),
            "statement": (
                f"Sous le scénario de forçage synthétique fort, le KGE du modèle "
                f"physique diminue d'environ {hero_loss:.3f} à +72 h."
            ),
        },
        "strong_skill_loss": strong_skill_loss,
        "skill_loss_cards": loss_cards,
        "skill_loss_label": (
            "Perte de performance liée à la dégradation synthétique des précipitations"
        ),
        "monte_carlo": {
            "scenario": "strong",
            "horizon_hours": 72,
            "model": "physical",
            "n_realizations": int(mc_row["n_realizations"]),
            "kge_median": mc_median,
            "kge_p10": mc_p10,
            "kge_p90": mc_p90,
            "spread_p90_p10": float(mc_spread),
            "median_display": f"{mc_median:.3f}",
            "p10_display": f"{mc_p10:.3f}",
            "p90_display": f"{mc_p90:.3f}",
            "spread_display": f"{mc_spread:.3f}",
            "p10_pct": _mc_pos(mc_p10),
            "median_pct": _mc_pos(mc_median),
            "p90_pct": _mc_pos(mc_p90),
            "caption_n": (
                f"N = {int(mc_row['n_realizations'])} réalisations météorologiques "
                "avec graine fixée"
            ),
            "caption_spread": (
                "La dispersion représente la sensibilité aux perturbations "
                "synthétiques des précipitations, et non une distribution "
                "météorologique probabiliste calibrée."
            ),
        },
        "persistence_comparison": {
            "rows": persistence_rows,
            "interpretation": (
                "La persistance du résidu reste utile même lorsque le forçage "
                "précipitation est dégradé. Son bénéfice ne supprime pas "
                "l'incertitude météorologique."
            ),
        },
        "model_complexity": {
            "heading": "Complexité du modèle",
            "ridge_vs_persistence_max_abs": ridge_vs_pers_max,
            "ridge_vs_persistence_mean_abs": ridge_vs_pers_mean,
            "max_abs_display": f"{ridge_vs_pers_max:.3f}",
            "mean_abs_display": f"{ridge_vs_pers_mean:.3f}",
            "body": (
                "Ridge ne montre pas d'avantage matériel sur la persistance du "
                "résidu dans cette expérience."
            ),
            "footnote": (
                "La complexité additionnelle du modèle de résidu est secondaire "
                "face à l'incertitude météorologique dans cette expérience. "
                "L'affirmation ne concerne que cette expérience pilote."
            ),
        },
        "scale_comparison": {
            "title": "SOURCE DE VARIABILITÉ",
            "subtitle": "Comparaison d'échelle — diagnostics différents",
            "meteo_label": "Forçage météo synthétique (fort +72 h p10–p90)",
            "meteo_value": float(mc_spread),
            "meteo_display": f"{mc_spread:.3f}",
            "meteo_bar_pct": float(meteo_bar_pct),
            "residual_label": "Choix du modèle de résidu (max |Ridge − Persistance|)",
            "residual_value": ridge_vs_pers_max,
            "residual_display": f"{ridge_vs_pers_max:.3f}",
            "residual_bar_pct": float(ridge_bar_pct),
            "note": (
                "Incertitude météo ≫ différence Ridge vs Persistance "
                "(diagnostics différents ; quantités statistiques non identiques)."
            ),
        },
        "high_flow": {
            "threshold_q_obs": float(hf_threshold) if hf_threshold is not None else None,
            "threshold_note": (
                "Seuil de haut débit dérivé des observations de calage uniquement."
            ),
            "rows": hf_rows,
            "interpretation": (
                "Les erreurs en haut débit augmentent lorsque le forçage "
                "précipitation se dégrade et que l'horizon s'allonge. La "
                "correction résiduelle aide encore, mais n'élimine pas le "
                "problème de forçage."
            ),
        },
        "architecture_caption": (
            "L'architecture opérationnelle doit propager l'incertitude "
            "météorologique plutôt que de traiter les prévisions de "
            "précipitations comme une vérité déterministe."
        ),
        "nwp_label": "SOURCE D'INCERTITUDE MAJEURE",
        "result": {
            "heading": "Résultat",
            "body": (
                "La dégradation synthétique des précipitations a peu d'impact "
                "à +24 h dans ce pilote, mais affecte de plus en plus la "
                "performance aux horizons plus longs. À +72 h, le scénario fort "
                "dégrade sensiblement la prévision physique."
            ),
        },
        "engineering_decision": {
            "heading": "Décision d'ingénierie",
            "body": (
                "L'incertitude météorologique doit être représentée explicitement "
                "dans l'architecture de prévision de STRYMO."
            ),
        },
        "ml_decision": {
            "heading": "Décision ML",
            "body": (
                "Avant d'ajouter un ML résiduel plus complexe, tester le système "
                "avec de vraies archives de prévisions météorologiques."
            ),
        },
        "next_question": {
            "heading": "Question suivante",
            "body": (
                "Si les entrées de prévision et les erreurs de modèle sont "
                "incertaines, peut-on produire un intervalle de prévision dont "
                "la couverture annoncée est réellement observée ?"
            ),
            "lead_in": "→ Calage de l'incertitude de prévision",
        },
        "figures": figures,
        "artifacts": {
            "hydrology": "meteo_hydrology_comparison.csv",
            "diagnostics": "meteo_scenario_diagnostics.csv",
            "highflow": "meteo_highflow_comparison.csv",
            "monte_carlo": "meteo_monte_carlo_summary.csv",
            "answers": "meteo_sensitivity_answers.json",
        },
    }


def _unc_row(
    frame: pd.DataFrame,
    *,
    method: str,
    horizon_days: int,
    scenario: str,
    nominal: float,
) -> pd.Series:
    row = frame.loc[
        (frame["method"] == method)
        & (frame["horizon_days"] == horizon_days)
        & (frame["meteo_scenario"] == scenario)
        & (frame["nominal_coverage"] == nominal)
    ]
    if row.empty:
        raise DemoExportError(
            f"Missing uncertainty row method={method!r} h={horizon_days} "
            f"scenario={scenario!r} nominal={nominal}"
        )
    return row.iloc[0]


def build_forecast_uncertainty_payload(output_dir: Path) -> dict[str, Any]:
    """Assemble Phase 10D presentation block from Phase 9 artifacts only."""
    coverage_path = output_dir / "uncertainty_coverage_summary.csv"
    regime_path = output_dir / "uncertainty_regime_coverage.csv"
    extreme_path = output_dir / "uncertainty_extreme_events.csv"
    forecasts_path = output_dir / "uncertainty_forecasts.csv"
    answers_path = output_dir / "uncertainty_answers.json"
    for path in (
        coverage_path,
        regime_path,
        extreme_path,
        forecasts_path,
        answers_path,
    ):
        if not path.is_file():
            raise DemoExportError(
                f"Artéfact Phase 9 manquant pour la section démo Phase 10D : "
                f"{path.name}. Exécuter d'abord `--uncertainty-calibration`."
            )

    coverage = pd.read_csv(coverage_path)
    regime = pd.read_csv(regime_path)
    extremes = pd.read_csv(extreme_path)
    # forecasts file required for presence; values come from summary/regime/extremes
    _ = pd.read_csv(forecasts_path, nrows=5)
    answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))
    answers = answers_payload.get("answers") or {}

    preferred = str(
        answers.get("preferred_method")
        or "empirical_residual"
    )
    preferred_label = str(
        answers.get("preferred_method_label")
        or "preferred after validation comparison"
    )
    # Presentation-only: map English answers-JSON label to French display string.
    if preferred_label.strip().lower() == "preferred after validation comparison":
        preferred_label = "préféré après comparaison en validation"

    methods_90 = (
        "behavioral_parametric",
        "conditional_quantile",
        "empirical_residual",
        "split_conformal",
    )
    coverage_90: dict[str, dict[str, Any]] = {}
    coverage_bars: list[dict[str, Any]] = []
    for method in methods_90:
        by_h: dict[str, Any] = {}
        for h in HORIZON_DAYS:
            row = _unc_row(
                coverage,
                method=method,
                horizon_days=h,
                scenario="oracle",
                nominal=0.90,
            )
            emp = float(row["empirical_coverage"])
            err = float(row["coverage_error"])
            mean_w = float(row["mean_width"])
            by_h[str(h * 24)] = {
                "empirical_coverage": emp,
                "coverage_error": err,
                "mean_width": mean_w,
                "median_width": float(row["median_width"]),
                "mean_interval_score": float(row["mean_interval_score"]),
                "empirical_pct_display": f"{100.0 * emp:.1f}%",
                "coverage_error_pp_display": f"{100.0 * err:+.1f} pp",
                "mean_width_display": f"{mean_w:.3f}",
                "bar_pct": 100.0 * emp,  # 0–1 coverage on 0–100% axis
            }
            coverage_bars.append(
                {
                    "method": method,
                    "hours": int(h * 24),
                    "empirical_coverage": emp,
                    "bar_pct": 100.0 * emp,
                    "label": f"{method} +{h * 24}h",
                }
            )
        coverage_90[method] = by_h

    method_cards = [
        {
            "id": "behavioral_parametric",
            "title": "PARAMÉTRIQUE COMPORTEMENTAL",
            "purpose": "Incertitude paramétrique.",
            "coverage_display": coverage_90["behavioral_parametric"]["24"][
                "empirical_pct_display"
            ],
            "interpretation": (
                "Trop étroit pour servir d'intervalle de prévision calibré à 90 %. "
                "Sa sémantique d'origine diffère d'un intervalle de prévision."
            ),
        },
        {
            "id": "conditional_quantile",
            "title": "QUANTILE CONDITIONNEL",
            "purpose": "Intervalle conditionnel piloté par les données.",
            "coverage_display": (
                f"{coverage_90['conditional_quantile']['24']['empirical_pct_display']} / "
                f"{coverage_90['conditional_quantile']['48']['empirical_pct_display']} / "
                f"{coverage_90['conditional_quantile']['72']['empirical_pct_display']}"
            ),
            "interpretation": "Précis, mais sous-couvre de façon matérielle.",
        },
        {
            "id": "empirical_residual",
            "title": "RÉSIDU EMPIRIQUE",
            "purpose": "Distribution historique des erreurs de prévision.",
            "coverage_display": coverage_90["empirical_residual"]["24"][
                "empirical_pct_display"
            ],
            "interpretation": (
                "Meilleur équilibre pratique entre fiabilité et largeur dans "
                "cette comparaison pilote."
            ),
        },
        {
            "id": "split_conformal",
            "title": "CALIBRATION CONFORME (SPLIT)",
            "purpose": "Calibration orientée couverture.",
            "coverage_display": coverage_90["split_conformal"]["24"][
                "empirical_pct_display"
            ],
            "interpretation": (
                "Garde-fou conservateur ; plus fiable mais plus large."
            ),
        },
    ]

    preferred_by_h: dict[str, Any] = {}
    for h in HORIZON_DAYS:
        row = _unc_row(
            coverage,
            method=preferred,
            horizon_days=h,
            scenario="oracle",
            nominal=0.90,
        )
        emp = float(row["empirical_coverage"])
        err = float(row["coverage_error"])
        preferred_by_h[str(h * 24)] = {
            "nominal": 0.90,
            "empirical": emp,
            "coverage_error": err,
            "mean_width": float(row["mean_width"]),
            "median_width": float(row["median_width"]),
            "interval_score": float(row["mean_interval_score"]),
            "nominal_display": "90%",
            "empirical_display": f"{100.0 * emp:.1f}%",
            "coverage_error_pp_display": (
                f"{100.0 * err:+.1f} points de pourcentage"
            ),
            "mean_width_display": f"{float(row['mean_width']):.3f} mm/jour",
        }

    hero = preferred_by_h["24"]
    behavioral_24 = coverage_90["behavioral_parametric"]["24"]["empirical_coverage"]
    preferred_24 = hero["empirical"]

    # Sharpness at +24 h
    sharpness_rows = []
    for method in ("conditional_quantile", "empirical_residual", "split_conformal"):
        cell = coverage_90[method]["24"]
        sharpness_rows.append(
            {
                "method": method,
                "label": {
                    "conditional_quantile": "Quantile conditionnel",
                    "empirical_residual": "Résidu empirique",
                    "split_conformal": "Calibration conforme (split)",
                }[method],
                "coverage": cell["empirical_coverage"],
                "coverage_display": cell["empirical_pct_display"],
                "mean_width": cell["mean_width"],
                "mean_width_display": f"{cell['mean_width']:.3f}",
            }
        )

    # Regime coverage preferred +24h oracle 90%
    regime_out: dict[str, Any] = {}
    for regime_name in ("high_flow", "normal_flow", "low_flow"):
        rrow = regime.loc[
            (regime["method"] == preferred)
            & (regime["horizon_days"] == 1)
            & (regime["meteo_scenario"] == "oracle")
            & (regime["nominal_coverage"] == 0.90)
            & (regime["regime"] == regime_name)
        ]
        if rrow.empty:
            raise DemoExportError(
                f"Missing regime coverage {regime_name!r} for preferred method"
            )
        rr = rrow.iloc[0]
        regime_out[regime_name] = {
            "empirical_coverage": float(rr["empirical_coverage"]),
            "mean_width": float(rr["mean_width"]),
            "n_days": int(rr["n_days"]),
            "display": f"{100.0 * float(rr['empirical_coverage']):.1f}%",
            "threshold_high": float(rr["threshold_high"]),
            "threshold_low": float(rr["threshold_low"]),
        }
    overall_24 = hero["empirical"]

    # Extreme events — preferred, oracle, +24h, top by q_obs
    ext = extremes.loc[
        (extremes["method"] == preferred)
        & (extremes["meteo_scenario"] == "oracle")
        & (extremes["horizon_days"] == 1)
    ].copy()
    if ext.empty:
        raise DemoExportError("No extreme events for preferred method under oracle/+24h")
    ext = ext.sort_values("q_obs", ascending=False).head(4)
    extreme_rows = []
    for er in ext.itertuples(index=False):
        inside = bool(er.inside_90)
        extreme_rows.append(
            {
                "date": str(er.date),
                "q_obs": float(er.q_obs),
                "q_point": float(er.q_point),
                "lower_90": float(er.lower_90),
                "upper_90": float(er.upper_90),
                "inside_90": inside,
                "inside_display": "OUI" if inside else "NON",
                "q_obs_display": f"{float(er.q_obs):.3f}",
                "q_point_display": f"{float(er.q_point):.3f}",
                "lower_display": f"{float(er.lower_90):.3f}",
                "upper_display": f"{float(er.upper_90):.3f}",
            }
        )

    # Meteo interaction for preferred
    meteo_rows = []
    for h in HORIZON_DAYS:
        o = _unc_row(
            coverage, method=preferred, horizon_days=h, scenario="oracle", nominal=0.90
        )
        m = _unc_row(
            coverage,
            method=preferred,
            horizon_days=h,
            scenario="moderate",
            nominal=0.90,
        )
        meteo_rows.append(
            {
                "hours": int(h * 24),
                "label": f"+{h * 24} h",
                "oracle_coverage": float(o["empirical_coverage"]),
                "moderate_coverage": float(m["empirical_coverage"]),
                "oracle_display": f"{100.0 * float(o['empirical_coverage']):.1f}%",
                "moderate_display": f"{100.0 * float(m['empirical_coverage']):.1f}%",
                "oracle_width": float(o["mean_width"]),
                "moderate_width": float(m["mean_width"]),
            }
        )

    # Reliability curve points for preferred, all horizons
    reliability = {}
    for h in HORIZON_DAYS:
        pts = coverage.loc[
            (coverage["method"] == preferred)
            & (coverage["meteo_scenario"] == "oracle")
            & (coverage["horizon_days"] == h)
        ].sort_values("nominal_coverage")
        reliability[str(h * 24)] = [
            {
                "nominal": float(r.nominal_coverage),
                "empirical": float(r.empirical_coverage),
                "nominal_display": f"{100.0 * float(r.nominal_coverage):.0f}%",
                "empirical_display": f"{100.0 * float(r.empirical_coverage):.1f}%",
            }
            for r in pts.itertuples(index=False)
        ]

    figures: dict[str, str] = {}
    for name in OPTIONAL_UNCERTAINTY_FIGURES:
        if (output_dir / name).is_file():
            key = name.replace("uncertainty_", "").replace(".png", "")
            figures[key] = f"assets/{name}"

    return {
        "section_title": "INCERTITUDE DE PRÉVISION",
        "section_subtitle": (
            "Un intervalle annoncé à 90 % couvre-t-il réellement environ 90 % "
            "des observations ?"
        ),
        "preferred_method": preferred,
        "preferred_label": preferred_label,
        "nominal_levels": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        "target_nominal": 0.90,
        "target_display": "90%",
        "calibration_note": (
            "La couverture est calibrée à partir des données antérieures à la "
            "validation uniquement. Les observations de validation servent à "
            "l'évaluation, pas à l'ajustement des intervalles."
        ),
        "central_statement": (
            "Qualifier un intervalle de « 90 % » ne garantit pas qu'il couvre "
            "réellement 90 % des observations. Il n'est utile que si sa couverture "
            "observée est cohérente avec cette affirmation tout en restant "
            "suffisamment précis."
        ),
        "reliability_def": (
            "Fiabilité : couverture empirique proche de la couverture nominale."
        ),
        "sharpness_def": (
            "Précision de l'intervalle (sharpness) : intervalles aussi étroits "
            "que possible, sous réserve d'une fiabilité adéquate."
        ),
        "table_headers": {
            "method": "Méthode",
            "coverage": "Couverture",
            "mean_width": "Largeur moyenne",
        },
        "coverage_90": coverage_90,
        "coverage_bars": coverage_bars,
        "method_cards": method_cards,
        "preferred": preferred_by_h,
        "hero": {
            **hero,
            "method": preferred,
            "preferred_label": preferred_label,
            "caveat": (
                "Une nouvelle période de test intacte est requise avant toute "
                "affirmation de performance opérationnelle."
            ),
        },
        "before_after": {
            "behavioral_coverage": behavioral_24,
            "behavioral_display": f"{100.0 * behavioral_24:.1f}%",
            "calibrated_coverage": preferred_24,
            "calibrated_display": f"{100.0 * preferred_24:.1f}%",
            "calibrated_method": preferred,
            "caption": (
                "De la dispersion de jeux de paramètres plausibles à un "
                "intervalle de prévision évalué."
            ),
            "semantics_note": (
                "Sémantiques d'incertitude différentes — comparaison présentée "
                "pour la progression méthodologique, non comme une performance "
                "de modèles strictement comparable."
            ),
        },
        "reliability": {
            "explanation": (
                "Un modèle d'incertitude bien calibré doit rester raisonnablement "
                "proche de la diagonale."
            ),
            "curves": reliability,
            "figure": figures.get("reliability"),
        },
        "sharpness": {
            "horizon_hours": 24,
            "rows": sharpness_rows,
            "note": (
                "Plus étroit n'est préférable que si la fiabilité reste acceptable."
            ),
        },
        "regime_coverage": {
            "method": preferred,
            "horizon_hours": 24,
            "nominal": 0.90,
            "overall": overall_24,
            "overall_display": f"{100.0 * overall_24:.1f}%",
            "high_flow": regime_out["high_flow"],
            "normal_flow": regime_out["normal_flow"],
            "low_flow": regime_out["low_flow"],
            "threshold_note": (
                "Seuil de haut débit : Qobs > 90e percentile de la période de calage."
            ),
            "warning_title": "DÉPENDANCE AU RÉGIME HYDROLOGIQUE",
            "warning_body": (
                "Une bonne couverture globale peut masquer une sous-couverture "
                "importante dans les conditions de haut débit — précisément celles "
                "qui sont les plus importantes pour un système d'alerte."
            ),
            "why_it_matters": (
                "Supposons que 90 % des jours ordinaires soient très bien couverts "
                "mais que les événements extrêmes / de haut débit soient "
                "systématiquement manqués. La couverture globale peut alors "
                "paraître excellente alors que le système échoue sur son cas "
                "d'usage à plus fort enjeu. Pour STRYMO, la couverture doit donc "
                "être évaluée conditionnellement par régime hydrologique, et non "
                "seulement globalement."
            ),
        },
        "extreme_events": extreme_rows,
        "meteo_comparison": {
            "method": preferred,
            "rows": meteo_rows,
            "message": (
                "Le calage des intervalles et l'incertitude météorologique "
                "ne peuvent pas être traités comme indépendants indéfiniment."
            ),
            "follow_up": (
                "La validation opérationnelle doit être répétée avec le produit "
                "météorologique de prévision réellement utilisé."
            ),
        },
        "uncertainty_sources": {
            "heading": "QUE DOIT ENTRER À TERME DANS L'INTERVALLE ?",
            "items": [
                {
                    "title": "1. INCERTITUDE PARAMÉTRIQUE",
                    "body": "Différents jeux de paramètres hydrologiques plausibles.",
                },
                {
                    "title": "2. INCERTITUDE MÉTÉOROLOGIQUE",
                    "body": "Différentes trajectoires possibles de précipitations futures.",
                },
                {
                    "title": "3. RÉSIDU / ERREUR DE MODÈLE",
                    "body": "Ce qui reste inexpliqué par la prévision physique.",
                },
            ],
            "future": (
                "Composantes futures : incertitude d'observation et incertitude "
                "structurelle du modèle. La Phase 9 ne propage pas encore "
                "entièrement chaque source."
            ),
        },
        "architecture_caption": (
            "Surveiller la couverture globale, la couverture en haut débit, la "
            "largeur d'intervalle et l'erreur de prévision. Recalibrer après "
            "suffisamment de nouvelles observations / événements — pas à l'aveugle."
        ),
        "result": {
            "heading": "Résultat",
            "body": (
                "Sur cette période de validation pilote, un intervalle simple de "
                "résidu empirique autour de la prévision physique + persistance "
                "atteint approximativement la couverture nominale de 90 % au "
                "niveau global."
            ),
        },
        "decision": {
            "heading": "Décision",
            "body": (
                "Utiliser des références d'incertitude transparentes avant "
                "d'introduire un ML probabiliste plus complexe."
            ),
        },
        "limit": {
            "heading": "Limite",
            "body": (
                "La couverture globale ne suffit pas : la couverture en haut débit "
                "reste nettement plus faible et doit être validée sur une période "
                "indépendante avant un usage opérationnel."
            ),
        },
        "final_takeaway": (
            "L'objectif n'est pas de produire l'intervalle le plus étroit. "
            "L'objectif est de produire l'intervalle le plus étroit auquel on "
            "puisse faire confiance."
        ),
        "figures": figures,
        "crps_note": answers_payload.get("crps_note")
        or answers.get("crps_note")
        or (
            "CRPS différé — score d'intervalle utilisé comme score probabiliste "
            "principal."
        ),
        "artifacts": {
            "coverage": "uncertainty_coverage_summary.csv",
            "regime": "uncertainty_regime_coverage.csv",
            "extremes": "uncertainty_extreme_events.csv",
            "forecasts": "uncertainty_forecasts.csv",
            "answers": "uncertainty_answers.json",
        },
    }


def git_remote_url(repo_root: Path | None = None) -> str | None:
    """Return origin remote URL if available."""
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    url = result.stdout.strip()
    if url.endswith(".git"):
        url = url[:-4]
    return url or None


def verify_required_artifacts(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        raise DemoExportError(
            "Artéfacts requis manquants pour l'export de la démo : "
            + ", ".join(missing)
            + ". Exécuter d'abord le pipeline scientifique / `python run.py --demo`."
        )


def build_demo_data(
    config_path: Path,
    output_dir: Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble presentation JSON from artifacts. Fails if artifacts are missing."""
    verify_required_artifacts(output_dir)
    if config is None:
        with config_path.open(encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

    station = config["station"]
    periods = config["periods"]
    model = config["model"]
    cal = config["calibration"]
    metadata = load_metadata(output_dir)

    runs = pd.read_csv(output_dir / "runs.csv")
    baseline = pd.read_csv(output_dir / "metrics_uncalibrated.csv")
    behavioral = pd.read_csv(output_dir / "behavioral_runs.csv")
    timeseries = pd.read_csv(output_dir / "ensemble_timeseries.csv")
    best = select_best_calibration_candidate(runs)

    threshold = get_behavioral_threshold(config)
    coverage = empirical_validation_coverage(timeseries)
    corr = kge_cal_val_correlation(runs)

    github_url = git_remote_url()
    if not github_url:
        github_url = config.get("demo", {}).get("github_url") or None

    analysis_start = periods["warmup"][0][:4]
    analysis_end = periods["validation"][1][:4]

    return {
        "title": "Calage automatisé d'un modèle pluie-débit",
        "subtitle": (
            "Prototype d'un harnais reproductible de calage de modèles hydrologiques"
        ),
        "secondary_line": (
            "Données hydrométriques réelles · exploration des paramètres sous "
            "contraintes · validation indépendante · incertitude explicite"
        ),
        "prototype_badge": (
            "PROTOTYPE — NON DESTINÉ À LA DÉCISION HYDROLOGIQUE OPÉRATIONNELLE"
        ),
        "station": {
            "code": station["code"],
            "name": station.get("name", station["code"]),
            "basin_area_km2": float(station["basin_area_km2"]),
            "centroid_lat": float(station["centroid_lat"]),
            "centroid_lon": float(station["centroid_lon"]),
        },
        "periods": {
            "analysis_label": f"{analysis_start}–{analysis_end}",
            "warmup": periods["warmup"],
            "calibration": periods["calibration"],
            "validation": periods["validation"],
            "temporal_resolution": "Journalière",
        },
        "model": {
            "name": model["name"],
            "label": f"{model['name']} — modèle conceptuel pluie–débit",
        },
        "kpis": {
            "n_samples": int(len(runs)),
            "n_samples_configured": int(cal["n_samples"]),
            "validation_kge_uncalibrated": _baseline_metric(baseline, "KGE", "validation"),
            "validation_kge_calibrated": float(best["kge_val"]),
            "validation_bias_uncalibrated": _baseline_metric(baseline, "Volume bias", "validation"),
            "validation_bias_calibrated": float(best["bias_val"]),
            "behavioral_members": int(len(behavioral)),
            "behavioral_threshold": float(threshold),
        },
        "metrics": {
            "uncalibrated": {
                "kge_cal": _baseline_metric(baseline, "KGE", "calibration"),
                "kge_val": _baseline_metric(baseline, "KGE", "validation"),
                "nse_cal": _baseline_metric(baseline, "NSE", "calibration"),
                "nse_val": _baseline_metric(baseline, "NSE", "validation"),
                "lognse_cal": _baseline_metric(baseline, "log-NSE", "calibration"),
                "lognse_val": _baseline_metric(baseline, "log-NSE", "validation"),
                "bias_cal": _baseline_metric(baseline, "Volume bias", "calibration"),
                "bias_val": _baseline_metric(baseline, "Volume bias", "validation"),
            },
            "calibrated": {
                "kge_cal": float(best["kge_cal"]),
                "kge_val": float(best["kge_val"]),
                "nse_cal": float(best["nse_cal"]),
                "nse_val": float(best["nse_val"]),
                "lognse_cal": float(best["lognse_cal"]),
                "lognse_val": float(best["lognse_val"]),
                "bias_cal": float(best["bias_cal"]),
                "bias_val": float(best["bias_val"]),
            },
        },
        "uncertainty": {
            "empirical_validation_coverage": float(coverage),
            "corr_kge_cal_val": float(corr),
        },
        "runtime": {
            "calibration_runtime_s": metadata.get("calibration_runtime_s"),
            "calibration_runtime_per_eval_s": metadata.get("calibration_runtime_per_eval_s"),
        },
        "reproducibility": {
            "config_sha256": config_sha256(config_path),
            "git_commit": git_commit_hash(),
            "random_seed": cal.get("seed"),
            "sampler": cal.get("sampler", "latin_hypercube"),
        },
        "github_url": github_url,
        "figures": {
            "calibration": "assets/demo_01_calibration_impact.png",
            "validation": "assets/demo_02_validation.png",
            "uncertainty": "assets/demo_03_uncertainty.png",
        },
        "report_html": "assets/rapport_calage.html",
        "statements": {
            "validation_isolation": (
                "Les observations de validation ne sont jamais utilisées pour le "
                "calage ni pour le classement des paramètres."
            ),
            "parametric_only": (
                "N'inclut pas les incertitudes liées aux précipitations, aux "
                "observations, à l'état initial ni à la structure du modèle."
            ),
            "envelope_disclaimer": (
                "q05–q95 N'EST PAS un intervalle de prédiction calibré à 90 %"
            ),
            "behavioral_envelope": (
                "L'enveloppe comportementale représente uniquement la dispersion "
                "paramétrique."
            ),
        },
        "ui": {
            "kpi_samples_caption": "Échantillonnage Latin Hypercube (LHS)",
            "kpi_kge_caption": "non calé → calé automatiquement",
            "kpi_bias_caption": "non calé → calé",
            "git_commit_unavailable": "non disponible",
        },
        "ml_correction": build_ml_correction_payload(output_dir),
        "forecast_horizons": build_forecast_horizons_payload(output_dir),
        "meteo_sensitivity": build_meteo_sensitivity_payload(output_dir),
        "forecast_uncertainty": build_forecast_uncertainty_payload(output_dir),
    }


def write_report_html(markdown_path: Path, html_path: Path) -> Path:
    """Wrap the existing report Markdown in a simple HTML page (content unchanged)."""
    content = markdown_path.read_text(encoding="utf-8")
    escaped = html.escape(content)
    page = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rapport de calage</title>
  <style>
    body {{
      margin: 0;
      background: #f8fafc;
      color: #111827;
      font-family: "Segoe UI", Calibri, Arial, sans-serif;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 20px 48px;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      padding: 20px;
      line-height: 1.45;
      font-size: 13px;
    }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <main>
    <p><a href="../index.html">← Retour à la démo</a></p>
    <pre>{escaped}</pre>
  </main>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return html_path


def export_static_demo(
    config_path: Path,
    output_dir: Path,
    demo_dir: Path,
) -> dict[str, Path]:
    """Verify artifacts, copy figures, write demo_data.json and report HTML."""
    verify_required_artifacts(output_dir)
    assets = demo_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}
    for name in DEMO_FIGURES:
        dest = assets / name
        shutil.copy2(output_dir / name, dest)
        figure_paths[name] = dest

    for name in OPTIONAL_METEO_FIGURES:
        src = output_dir / name
        if src.is_file():
            dest = assets / name
            shutil.copy2(src, dest)
            figure_paths[name] = dest

    for name in OPTIONAL_UNCERTAINTY_FIGURES:
        src = output_dir / name
        if src.is_file():
            dest = assets / name
            shutil.copy2(src, dest)
            figure_paths[name] = dest

    data = build_demo_data(config_path, output_dir)
    data_path = assets / "demo_data.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report_html = write_report_html(
        output_dir / "rapport_calage.md",
        assets / "rapport_calage.html",
    )

    return {
        "demo_data": data_path,
        "report_html": report_html,
        **figure_paths,
    }
