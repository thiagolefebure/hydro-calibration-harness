"""Génération automatique du rapport de calage auditable (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.calibration_diagnostics import (
    kge_cal_distribution,
    kge_cal_val_correlation,
    parameters_close_to_bounds,
    threshold_counts,
)
from src.ensemble import (
    empirical_validation_coverage,
    envelope_width_diagnostics,
    get_behavioral_threshold,
    metrics_to_dict,
    parameter_range_diagnostics,
    validation_metrics_for_series,
)
from src.evaluation import DEMO_PARAMETERS
from src.experiment_metadata import build_reproducibility_block
from src.hydrology import load_processed_daily
from src.validation import select_best_calibration_candidate

REPORT_FILENAME = "rapport_calage.md"

GR4J_PARAMETER_DOCS = {
    "X1": ("Capacité du réservoir de production", "mm"),
    "X2": ("Flux d'échange souterrain", "mm"),
    "X3": ("Capacité du réservoir de routage", "mm"),
    "X4": ("Base temporelle des hydrogrammes unitaires (UH1/UH2)", "jours"),
}

VALIDATION_ISOLATION_STATEMENT = (
    "Les données de validation ne sont utilisées ni pour l'échantillonnage, "
    "ni pour le classement, ni pour la sélection des paramètres."
)

UNCERTAINTY_ENVELOPE_STATEMENT = (
    "L'enveloppe q05–q95 est une enveloppe représentant uniquement l'incertitude "
    "paramétrique (dispersion des simulations comportementales retenues). "
    "Il ne s'agit ni d'un intervalle de confiance à 90 %, ni d'un intervalle "
    "de prédiction à 90 %, ni d'une probabilité de 90 %."
)

UNDER_COVERAGE_STATEMENT = (
    "La sous-couverture empirique observée sur la période de validation est "
    "attendue, car les incertitudes sur les précipitations, sur les observations "
    "et sur la structure du modèle ne sont pas propagées dans ce prototype : "
    "la dispersion paramétrique seule ne suffit pas à représenter l'incertitude "
    "prédictive totale."
)

EQUIFINALITY_STATEMENT = (
    "**Équifinalité.** Plusieurs jeux de paramètres distincts atteignent des "
    "performances de calage similaires ; le jeu de plus haut score ne doit donc "
    "pas être interprété comme une vérité physique unique."
)

STATUS_BANNER = (
    "**STATUT : PROTOTYPE / NON DESTINÉ À LA DÉCISION HYDROLOGIQUE OPÉRATIONNELLE**"
)


@dataclass(frozen=True)
class ReportInputs:
    config: dict[str, Any]
    config_path: Path
    output_dir: Path
    runs: pd.DataFrame
    baseline_metrics: pd.DataFrame
    behavioral_runs: pd.DataFrame
    ensemble_timeseries: pd.DataFrame
    hydrological_summary: pd.DataFrame
    sensitivity: pd.DataFrame | None
    processed_data: pd.DataFrame
    reproducibility: dict[str, Any]


def _fmt(value: float | int | None, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _format_bound_proximity_fr(close: dict[str, str | None]) -> str:
    flagged = {name: side for name, side in close.items() if side is not None}
    if not flagged:
        return "aucun (tous les paramètres à plus de 2 % des bornes configurées)"
    side_fr = {"lower": "inférieure", "upper": "supérieure"}
    parts = [
        f"{name} proche de la borne {side_fr.get(side, side)}"
        for name, side in flagged.items()
    ]
    return "; ".join(parts)


def summarize_processed_data(df: pd.DataFrame) -> dict[str, Any]:
    """Compute missing-data and usable-observation counts from processed CSV."""
    variables = ("precipitation_mm", "et0_mm", "discharge_mm")
    n_days = len(df)
    missing = {var: int(df[var].isna().sum()) for var in variables}
    usable_mask = df[list(variables)].notna().all(axis=1)
    return {
        "n_calendar_days": n_days,
        "missing": missing,
        "usable_observations": int(usable_mask.sum()),
        "analysis_start": df.index.min().strftime("%Y-%m-%d"),
        "analysis_end": df.index.max().strftime("%Y-%m-%d"),
    }


def load_report_inputs(
    config_path: Path,
    output_dir: Path,
) -> ReportInputs:
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    runs = pd.read_csv(output_dir / "runs.csv")
    baseline_metrics = pd.read_csv(output_dir / "metrics_uncalibrated.csv")
    behavioral_runs = pd.read_csv(output_dir / "behavioral_runs.csv")
    ensemble_timeseries = pd.read_csv(output_dir / "ensemble_timeseries.csv")
    hydrological_summary = pd.read_csv(output_dir / "hydrological_summary.csv")
    processed_data = load_processed_daily(output_dir / "data" / "basin_daily.csv")

    sensitivity_path = output_dir / "ensemble_threshold_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.is_file() else None

    return ReportInputs(
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        runs=runs,
        baseline_metrics=baseline_metrics,
        behavioral_runs=behavioral_runs,
        ensemble_timeseries=ensemble_timeseries,
        hydrological_summary=hydrological_summary,
        sensitivity=sensitivity,
        processed_data=processed_data,
        reproducibility=build_reproducibility_block(
            config_path, config, output_dir=output_dir
        ),
    )


def _metrics_row(df: pd.DataFrame, metric_name: str) -> pd.Series:
    row = df.loc[df["metric"] == metric_name]
    if row.empty:
        raise KeyError(f"Metric {metric_name!r} not found")
    return row.iloc[0]


def _comparison_metrics_table(
    baseline: pd.DataFrame,
    best: pd.Series,
) -> str:
    rows = [
        ("NSE", "NSE", "nse_cal", "nse_val"),
        ("KGE", "KGE", "kge_cal", "kge_val"),
        ("r", "r", "r_cal", "r_val"),
        ("alpha", "alpha", "alpha_cal", "alpha_val"),
        ("beta", "beta", "beta_cal", "beta_val"),
        ("log-NSE", "log-NSE", "lognse_cal", "lognse_val"),
        ("Biais volumique", "Volume bias", "bias_cal", "bias_val"),
    ]
    lines = [
        "| Métrique | Non calé (calage) | Non calé (validation) | "
        "Meilleur calage (calage) | Meilleur calage (validation) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, artifact_name, cal_col, val_col in rows:
        base_cal = _metrics_row(baseline, artifact_name)["calibration"]
        base_val = _metrics_row(baseline, artifact_name)["validation"]
        lines.append(
            f"| {label} | {_fmt(base_cal)} | {_fmt(base_val)} | "
            f"{_fmt(best[cal_col])} | {_fmt(best[val_col])} |"
        )
    return "\n".join(lines)


def _hydrological_table(summary: pd.DataFrame) -> str:
    annual = summary[summary["period"].str.fullmatch(r"\d{4}")]
    lines = [
        "| Année | Précipitations (mm) | Lame d'eau Q (mm) | "
        "Coefficient d'écoulement | Q moyen (mm/j) | Q max (mm/j) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in annual.iterrows():
        lines.append(
            f"| {row['period']} | {_fmt(row['annual_precipitation_mm'], 1)} | "
            f"{_fmt(row['annual_observed_discharge_depth_mm'], 1)} | "
            f"{_fmt(row['runoff_ratio_qp'])} | "
            f"{_fmt(row['mean_observed_discharge_mm_day'])} | "
            f"{_fmt(row['max_daily_observed_discharge_mm_day'])} |"
        )
    return "\n".join(lines)


def _calibration_validation_aggregate(summary: pd.DataFrame) -> str:
    label_fr = {
        "calibration": "calage",
        "validation": "validation",
    }
    lines = [
        "| Agrégat de période | Précipitations (mm) | Lame d'eau Q (mm) | "
        "Coefficient d'écoulement | Q moyen (mm/j) | Q max (mm/j) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ("calibration", "validation"):
        row = summary.loc[summary["period"] == label].iloc[0]
        lines.append(
            f"| {label_fr[label]} | {_fmt(row['annual_precipitation_mm'], 1)} | "
            f"{_fmt(row['annual_observed_discharge_depth_mm'], 1)} | "
            f"{_fmt(row['runoff_ratio_qp'])} | "
            f"{_fmt(row['mean_observed_discharge_mm_day'])} | "
            f"{_fmt(row['max_daily_observed_discharge_mm_day'])} |"
        )
    return "\n".join(lines)


def _generalization_paragraph(best: pd.Series) -> str:
    kge_cal = float(best["kge_cal"])
    kge_val = float(best["kge_val"])
    delta = kge_val - kge_cal
    if delta >= -0.08:
        return (
            "Le meilleur jeu de paramètres issu du calage conserve l'essentiel "
            "de sa performance de calage en validation "
            f"(KGE_cal = {_fmt(kge_cal)}, KGE_val = {_fmt(kge_val)}). "
            "Cela suggère une généralisation split-sample raisonnable pour ce "
            "bassin pilote, sans revendiquer une robustesse opérationnelle."
        )
    return (
        "La performance en calage est nettement supérieure à celle en validation "
        f"(KGE_cal = {_fmt(kge_cal)}, KGE_val = {_fmt(kge_val)}). "
        "Une partie de la performance du modèle ne se transfère pas pleinement "
        "à la période de validation ; ce comportement est typique des expériences "
        "split-sample et ne doit pas servir à retuner les paramètres."
    )


def render_rapport_calage(inputs: ReportInputs) -> str:
    """Rendre le rapport Markdown complet à partir des entrées chargées."""
    cfg = inputs.config
    station = cfg["station"]
    station_name = station.get("name", station["code"])
    periods = cfg["periods"]
    bounds = cfg["model"]["parameter_bounds"]
    cal_cfg = cfg["calibration"]
    data_summary = summarize_processed_data(inputs.processed_data)
    tz = cfg.get("data", {}).get("meteo_timezone", "UTC")

    best = select_best_calibration_candidate(inputs.runs)
    close = parameters_close_to_bounds(best.to_dict(), bounds)
    dist = kge_cal_distribution(inputs.runs)
    counts = threshold_counts(inputs.runs)
    corr = kge_cal_val_correlation(inputs.runs)
    param_ranges, weakly = parameter_range_diagnostics(
        inputs.behavioral_runs, bounds
    )
    threshold = get_behavioral_threshold(cfg)
    coverage = empirical_validation_coverage(inputs.ensemble_timeseries)
    width = envelope_width_diagnostics(inputs.ensemble_timeseries)

    val_ts = inputs.ensemble_timeseries[inputs.ensemble_timeseries["period"] == "validation"]
    observed = pd.Series(
        val_ts["q_obs"].values,
        index=pd.to_datetime(val_ts["date"]),
    )
    q50 = pd.Series(val_ts["q50"].values, index=pd.to_datetime(val_ts["date"]))
    q50_metrics = validation_metrics_for_series(observed, q50, cfg)
    q50_dict = metrics_to_dict(q50_metrics)
    best_val_dict = {
        "nse": float(best["nse_val"]),
        "kge": float(best["kge_val"]),
        "r": float(best["r_val"]),
        "alpha": float(best["alpha_val"]),
        "beta": float(best["beta_val"]),
        "lognse": float(best["lognse_val"]),
        "bias": float(best["bias_val"]),
    }

    repro = inputs.reproducibility
    runtime_total = repro.get("calibration_runtime_s")
    runtime_per = repro.get("calibration_runtime_per_eval_s")

    sections: list[str] = [
        STATUS_BANNER,
        "",
        "# Rapport de calage pluie–débit automatisé",
        "",
        "## 1. Périmètre du prototype",
        "",
        "Ce document décrit un **prototype d'ingénierie du calage** construit autour "
        "du modèle conceptuel pluie–débit **GR4J**. L'objectif est de démontrer "
        "l'exploration automatique de l'espace des paramètres, la séparation "
        "explicite calage / validation, un ensemble comportemental inspiré de "
        "l'approche GLUE, la communication transparente de l'incertitude et la "
        "reproductibilité complète.",
        "",
        "Il ne s'agit **pas** d'un système opérationnel de prévision de crue ni "
        "d'un outil réglementaire d'aide à la décision.",
        "",
        "## 2. Bassin versant et données",
        "",
        f"- **Code station :** {station['code']}",
        f"- **Nom de la station :** {station_name}",
        f"- **Surface du bassin versant :** {station['basin_area_km2']} km²",
        f"- **Centroïde :** ({station['centroid_lat']}, {station['centroid_lon']})",
        f"- **Période d'analyse :** {data_summary['analysis_start']} → {data_summary['analysis_end']}",
        f"- **Période de mise en route (warm-up) :** {periods['warmup'][0]} → {periods['warmup'][1]}",
        f"- **Période de calage :** {periods['calibration'][0]} → {periods['calibration'][1]}",
        f"- **Période de validation :** {periods['validation'][0]} → {periods['validation'][1]}",
        "- **Source de débit :** API Hub'Eau hydrométrie v2 (`obs_elab`, `QmnJ`, L/s)",
        "- **Source précipitations / ET0 :** Open-Meteo Historical Weather API (journalière)",
        "- **Résolution temporelle :** journalière",
        f"- **Fuseau horaire (agrégation météo) :** {tz}",
        f"- **Précipitations manquantes :** {data_summary['missing']['precipitation_mm']} jours",
        f"- **ET0 manquante :** {data_summary['missing']['et0_mm']} jours",
        f"- **Débit manquant :** {data_summary['missing']['discharge_mm']} jours",
        f"- **Observations exploitables (toutes variables présentes) :** {data_summary['usable_observations']} jours",
        "",
        "Les précipitations sont représentées par un point Open-Meteo unique au "
        "centroïde du bassin versant ; il ne s'agit pas d'une précipitation "
        "moyenne de bassin.",
        "",
        "Conversion du débit : Q_mm/j = Q_L/s × 0.0864 / basin_area_km².",
        "",
        "## 3. Modèle hydrologique",
        "",
        "**GR4J** (Perrin, Michel & Andréassian, 2003) — modèle conceptuel à quatre paramètres.",
        "",
        "GR4J est un modèle conceptuel. Les paramètres calés ne doivent pas être "
        "interprétés automatiquement comme des mesures physiques directes des "
        "propriétés du bassin versant.",
        "",
        "| Paramètre | Signification | Unité | Bornes de démonstration |",
        "| --- | --- | --- | --- |",
    ]

    for name, (meaning, unit) in GR4J_PARAMETER_DOCS.items():
        lo, hi = bounds[name]
        sections.append(f"| {name} | {meaning} | {unit} | [{lo}, {hi}] |")

    warmup_days = (
        pd.Timestamp(periods["warmup"][1]) - pd.Timestamp(periods["warmup"][0])
    ).days + 1

    sections.extend(
        [
            "",
            "**Convention d'état initial :** réservoir de production à 30 % de X1, "
            "réservoir de routage à 50 % de X3, stocks d'hydrogrammes unitaires vides "
            "(fractions par défaut airGR).",
            f"**Durée de la période de mise en route (warm-up) :** {warmup_days} jours "
            f"({periods['warmup'][0]} → {periods['warmup'][1]}), exclue de toutes "
            "les métriques reportées.",
            "**Continuité des états :** GR4J s'exécute en continu du début de la "
            "période de mise en route (warm-up) jusqu'à la fin de la validation, "
            "sans réinitialisation des états aux frontières de période.",
            "",
            "## 4. Expérience de calage",
            "",
            f"- **Méthode d'échantillonnage :** Latin Hypercube Sampling ({cal_cfg.get('sampler', 'latin_hypercube')})",
            f"- **N :** {cal_cfg['n_samples']}",
            f"- **Graine aléatoire :** {cal_cfg['seed']}",
            "- **Bornes des paramètres :** voir section 3",
            "- **Objectif de classement :** KGE_cal (période de calage uniquement)",
            "- **Règle d'isolement de la validation :** les métriques de validation sont purement diagnostiques",
            f"- **Temps de calcul total :** {_fmt(runtime_total, 2)} s"
            + (
                " (non enregistré — relancer `python run.py --calibrate` pour persister)"
                if runtime_total is None
                else ""
            ),
            f"- **Temps moyen par évaluation :** {_fmt(runtime_per, 4)} s"
            + (" (non enregistré)" if runtime_per is None else ""),
            "",
            VALIDATION_ISOLATION_STATEMENT,
            "",
            "## 5. Référence non calée",
            "",
            "Paramètres de démonstration fixes (non ajustés manuellement aux observations) : "
            f"X1={DEMO_PARAMETERS.X1}, X2={DEMO_PARAMETERS.X2}, "
            f"X3={DEMO_PARAMETERS.X3}, X4={DEMO_PARAMETERS.X4}.",
            "",
            "| Métrique | Calage | Validation |",
            "| --- | ---: | ---: |",
        ]
    )

    metric_label_fr = {
        "NSE": "NSE",
        "KGE": "KGE",
        "r": "r",
        "alpha": "alpha",
        "beta": "beta",
        "log-NSE": "log-NSE",
        "Volume bias": "Biais volumique",
    }
    for _, row in inputs.baseline_metrics.iterrows():
        label = metric_label_fr.get(str(row["metric"]), str(row["metric"]))
        sections.append(
            f"| {label} | {_fmt(row['calibration'])} | {_fmt(row['validation'])} |"
        )

    sections.extend(
        [
            "",
            "## 6. Meilleur jeu de paramètres issu du calage",
            "",
            f"- **Identifiant d'exécution (run_id) :** {int(best['run_id'])}",
            f"- **X1–X4 :** {_fmt(best['x1'], 3)}, {_fmt(best['x2'], 3)}, "
            f"{_fmt(best['x3'], 3)}, {_fmt(best['x4'], 3)}",
            "",
            "**Métriques de calage :** "
            f"NSE={_fmt(best['nse_cal'])}, KGE={_fmt(best['kge_cal'])}, "
            f"r={_fmt(best['r_cal'])}, alpha={_fmt(best['alpha_cal'])}, "
            f"beta={_fmt(best['beta_cal'])}, log-NSE={_fmt(best['lognse_cal'])}, "
            f"biais={_fmt(best['bias_cal'])}.",
            "",
            "**Métriques de validation (diagnostiques uniquement) :** "
            f"NSE={_fmt(best['nse_val'])}, KGE={_fmt(best['kge_val'])}, "
            f"r={_fmt(best['r_val'])}, alpha={_fmt(best['alpha_val'])}, "
            f"beta={_fmt(best['beta_val'])}, log-NSE={_fmt(best['lognse_val'])}, "
            f"biais={_fmt(best['bias_val'])}.",
            "",
            f"**Proximité des bornes (à moins de 2 % de l'amplitude configurée) :** "
            f"{_format_bound_proximity_fr(close)}.",
            "",
            "## 7. Calage vs validation",
            "",
            _comparison_metrics_table(inputs.baseline_metrics, best),
            "",
            _generalization_paragraph(best),
            "",
            "## 8. Diagnostics de l'espace des paramètres",
            "",
            "**Distribution de KGE_cal (N = "
            f"{len(inputs.runs)}) :** "
            f"min={_fmt(dist['min'])}, médiane={_fmt(dist['median'])}, "
            f"p90={_fmt(dist['p90'])}, p95={_fmt(dist['p95'])}, "
            f"p99={_fmt(dist['p99'])}, max={_fmt(dist['max'])}.",
            "",
            "**Effectifs par seuil :**",
        ]
    )
    for key, value in counts.items():
        sections.append(f"- {key}: {value}")

    sections.extend(
        [
            "",
            f"**corr(KGE_cal, KGE_val) :** {_fmt(corr)}",
            "",
            "**Plages des paramètres comportementaux (KGE_cal > seuil officiel) :**",
            "",
            "| Paramètre | min | médiane | max |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in param_ranges.iterrows():
        sections.append(
            f"| {row['parameter']} | {_fmt(row['min'], 3)} | "
            f"{_fmt(row['median'], 3)} | {_fmt(row['max'], 3)} |"
        )

    weakly_text = ", ".join(weakly) if weakly else "aucun identifié"
    sections.extend(
        [
            "",
            EQUIFINALITY_STATEMENT,
            "",
            f"**Paramètres faiblement contraints dans l'ensemble comportemental :** {weakly_text}.",
            "",
            "## 9. Ensemble comportemental",
            "",
            "- **Méthode :** ensemble comportemental inspiré de l'approche GLUE "
            "(et non une implémentation complète de GLUE)",
            f"- **Critère :** KGE_cal > {threshold:g}",
            f"- **Taille de l'ensemble :** {len(inputs.behavioral_runs)} membres",
            "- **Le critère est configurable** et **n'est pas** un seuil "
            "d'acceptabilité hydrologique universel",
            "- **La validation n'intervient pas dans l'appartenance à l'ensemble**",
            "",
            "**Métriques de validation de q50 (diagnostiques) :** "
            f"NSE={_fmt(q50_dict['nse'])}, KGE={_fmt(q50_dict['kge'])}, "
            f"r={_fmt(q50_dict['r'])}, alpha={_fmt(q50_dict['alpha'])}, "
            f"beta={_fmt(q50_dict['beta'])}, log-NSE={_fmt(q50_dict['lognse'])}, "
            f"biais={_fmt(q50_dict['bias'])}.",
            "",
            "## 10. Diagnostics d'incertitude",
            "",
            f"- **Couverture empirique de validation de l'enveloppe comportementale (q05–q95) :** {_pct(coverage)}",
            f"- **Largeur moyenne de l'enveloppe :** {_fmt(width['mean'])} mm/j",
            f"- **Largeur médiane de l'enveloppe :** {_fmt(width['median'])} mm/j",
            f"- **Largeur p90 de l'enveloppe :** {_fmt(width['p90'])} mm/j",
            f"- **Largeur moyenne / débit observé moyen en validation :** {_fmt(width['relative_to_obs_mean'])}",
            "",
            UNCERTAINTY_ENVELOPE_STATEMENT,
            "",
            UNDER_COVERAGE_STATEMENT,
            "",
            "L'enveloppe exclut explicitement :",
            "- l'incertitude sur les précipitations ;",
            "- l'incertitude observationnelle ;",
            "- l'incertitude de structure du modèle ;",
            "- l'incertitude d'état initial.",
            "",
            "**Tableau de sensibilité au seuil (diagnostique ; le seuil n'est pas "
            "choisi à partir de la validation) :**",
            "",
            "| KGE_cal > | Membres | Couverture empirique de validation |",
            "| --- | ---: | ---: |",
        ]
    )

    if inputs.sensitivity is not None:
        for _, row in inputs.sensitivity.iterrows():
            sections.append(
                f"| {row['threshold']:.2f} | {int(row['n_members'])} | "
                f"{_pct(row['validation_coverage'])} |"
            )
    else:
        for threshold_val in (0.70, 0.75, 0.80, 0.85):
            n = int((inputs.runs["kge_cal"] > threshold_val).sum())
            cov = (
                _pct(coverage)
                if threshold_val == threshold
                else "n/a (relancer `python run.py --ensemble`)"
            )
            sections.append(f"| {threshold_val:.2f} | {n} | {cov} |")

    sections.extend(
        [
            "",
            "**Comparaison q50 vs meilleur calage en validation (diagnostique) :**",
            "",
            "| Métrique | q50 | Meilleur calage |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in ("nse", "kge", "r", "alpha", "beta", "lognse", "bias"):
        sections.append(
            f"| {key} | {_fmt(q50_dict[key])} | {_fmt(best_val_dict[key])} |"
        )

    sections.extend(
        [
            "",
            "## 11. Caractérisation hydrologique des périodes",
            "",
            "### Synthèse annuelle (2010–2015)",
            "",
            _hydrological_table(inputs.hydrological_summary),
            "",
            "### Agrégats calage vs validation",
            "",
            _calibration_validation_aggregate(inputs.hydrological_summary),
            "",
            "Note : l'année 2014 présente un maximum journalier de débit observé "
            "plus bas que les années voisines dans ce jeu de données ; cette "
            "observation est rapportée à titre diagnostique et nécessite "
            "investigation, sans être qualifiée d'anomalie définitive.",
            "",
            "## 12. Limites",
            "",
            "- Résolution temporelle journalière uniquement",
            "- Précipitations au centroïde plutôt qu'une précipitation moyenne de bassin",
            "- Modèle conceptuel global plutôt qu'une représentation physique distribuée",
            "- Pas d'ensemble de précipitations",
            "- Pas de propagation d'incertitude de courbe de tarage",
            "- Pas d'assimilation d'état",
            "- Incertitude paramétrique seule dans l'enveloppe reportée",
            "- Seuil comportemental spécifique au prototype et configurable",
            "- Un seul bassin versant pilote (une configuration de station)",
            "",
            "## 13. Reproductibilité",
            "",
            f"- **Généré le (UTC) :** {repro['generated_at_utc']}",
            f"- **SHA256 de la configuration :** `{repro['config_sha256']}`",
            f"- **Commit Git :** {repro['git_commit'] or 'non disponible'}",
            f"- **Version Python :** {repro['python_version']}",
            f"- **Version modèle / prototype :** {repro['model_version']}",
            f"- **Graine aléatoire :** {repro['random_seed']}",
            f"- **N simulations :** {repro['n_samples']}",
            "",
            "**Versions des paquets :**",
        ]
    )
    for pkg, ver in repro["package_versions"].items():
        sections.append(f"- {pkg}: {ver}")

    artifact_lines = [
        f"- `{inputs.output_dir / 'data' / 'basin_daily.csv'}` — données journalières traitées",
        f"- `{inputs.output_dir / 'runs.csv'}` — {len(inputs.runs)} expériences de calage",
        f"- `{inputs.output_dir / 'top20_calibration.csv'}` — meilleurs candidats de calage",
        f"- `{inputs.output_dir / 'behavioral_runs.csv'}` — jeux de paramètres comportementaux",
        f"- `{inputs.output_dir / 'ensemble_timeseries.csv'}` — séries temporelles des quantiles d'ensemble",
        f"- `{inputs.output_dir / 'ensemble_validation.png'}` — figure d'incertitude en validation",
        f"- `{inputs.output_dir / 'ensemble_full_validation.png'}` — diagnostic de validation complète",
        f"- `{inputs.output_dir / 'parameter_space_diagnostic.png'}` — diagnostic de l'espace des paramètres",
        f"- `{inputs.output_dir / 'hydrological_years.png'}` — figure de caractérisation hydrologique",
    ]

    sections.extend(
        [
            "",
            "## 14. Artéfacts",
            "",
            *artifact_lines,
            "",
            "## 15. Bannière de décision",
            "",
            STATUS_BANNER,
            "",
            "Ce prototype ne doit pas être interprété comme une validation "
            "réglementaire, une prévision ou une validation hydrologique opérationnelle.",
            "",
        ]
    )

    return "\n".join(sections)


def generate_rapport_calage(
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Charger les sorties d'expérience et écrire rapport_calage.md."""
    inputs = load_report_inputs(config_path, output_dir)
    content = render_rapport_calage(inputs)
    output_path = output_dir / REPORT_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
