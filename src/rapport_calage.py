"""Automatic auditable calibration report generation (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.calibration_diagnostics import (
    format_bound_proximity,
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
    "X1": ("Production store capacity", "mm"),
    "X2": ("Groundwater exchange flux", "mm"),
    "X3": ("Routing store capacity", "mm"),
    "X4": ("Unit-hydrograph time base (UH1/UH2)", "days"),
}

VALIDATION_ISOLATION_STATEMENT = (
    "Validation observations were not used for parameter sampling, ranking, "
    "parameter selection, stopping criteria, or behavioral-ensemble membership."
)

UNCERTAINTY_ENVELOPE_STATEMENT = (
    "The q05–q95 envelope is the dispersion of the selected behavioral "
    "parameter simulations. It is not a calibrated 90% confidence or "
    "prediction interval."
)

UNDER_COVERAGE_STATEMENT = (
    "The observed under-coverage demonstrates that parametric dispersion alone "
    "is insufficient to represent total predictive uncertainty."
)

EQUIFINALITY_STATEMENT = (
    "Multiple distinct parameter sets achieve similar calibration performance; "
    "therefore the highest-scoring parameter set should not be interpreted as "
    "a uniquely identified physical truth."
)

STATUS_BANNER = (
    "**STATUS: PROTOTYPE / NOT FOR OPERATIONAL HYDROLOGICAL DECISION-MAKING**"
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
        ("NSE", "nse_cal", "nse_val"),
        ("KGE", "kge_cal", "kge_val"),
        ("r", "r_cal", "r_val"),
        ("alpha", "alpha_cal", "alpha_val"),
        ("beta", "beta_cal", "beta_val"),
        ("log-NSE", "lognse_cal", "lognse_val"),
        ("volumetric bias", "bias_cal", "bias_val"),
    ]
    lines = [
        "| Metric | Uncalibrated (cal) | Uncalibrated (val) | "
        "Best cal (cal) | Best cal (val) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, cal_col, val_col in rows:
        base_cal = _metrics_row(baseline, label.replace("volumetric ", "Volume "))["calibration"]
        base_val = _metrics_row(baseline, label.replace("volumetric ", "Volume "))["validation"]
        lines.append(
            f"| {label} | {_fmt(base_cal)} | {_fmt(base_val)} | "
            f"{_fmt(best[cal_col])} | {_fmt(best[val_col])} |"
        )
    return "\n".join(lines)


def _hydrological_table(summary: pd.DataFrame) -> str:
    annual = summary[summary["period"].str.fullmatch(r"\d{4}")]
    lines = [
        "| Year | Precip (mm) | Q depth (mm) | Runoff ratio | Mean Q (mm/d) | Max Q (mm/d) |",
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
    lines = [
        "| Period aggregate | Precip (mm) | Q depth (mm) | Runoff ratio | Mean Q (mm/d) | Max Q (mm/d) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ("calibration", "validation"):
        row = summary.loc[summary["period"] == label].iloc[0]
        lines.append(
            f"| {label} | {_fmt(row['annual_precipitation_mm'], 1)} | "
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
            "The best calibration candidate retains most of its calibration-period "
            f"skill in validation (KGE_cal = {_fmt(kge_cal)}, KGE_val = {_fmt(kge_val)}). "
            "This suggests reasonable split-sample generalization for this pilot basin, "
            "without claiming operational robustness."
        )
    return (
        "Calibration performance is materially higher than validation performance "
        f"(KGE_cal = {_fmt(kge_cal)}, KGE_val = {_fmt(kge_val)}). "
        "Some calibration skill does not fully transfer to the validation period, "
        "which is expected in split-sample experiments and must not be used to retune parameters."
    )


def render_rapport_calage(inputs: ReportInputs) -> str:
    """Render the full Markdown report from loaded inputs."""
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
        "# Automated Rainfall–Runoff Calibration Report",
        "",
        "## 1. Prototype scope",
        "",
        "This document describes a **calibration-engineering prototype** built around "
        "the conceptual **GR4J** rainfall–runoff model. The goal is to demonstrate "
        "automated parameter exploration, explicit calibration/validation separation, "
        "GLUE-inspired behavioral-ensemble diagnostics, transparent uncertainty "
        "communication, and full reproducibility.",
        "",
        "This is **not** an operational flood-forecasting or regulatory decision-support system.",
        "",
        "## 2. Basin and data",
        "",
        f"- **Station code:** {station['code']}",
        f"- **Station name:** {station_name}",
        f"- **Basin area:** {station['basin_area_km2']} km²",
        f"- **Centroid:** ({station['centroid_lat']}, {station['centroid_lon']})",
        f"- **Analysis period:** {data_summary['analysis_start']} → {data_summary['analysis_end']}",
        f"- **Warm-up period:** {periods['warmup'][0]} → {periods['warmup'][1]}",
        f"- **Calibration period:** {periods['calibration'][0]} → {periods['calibration'][1]}",
        f"- **Validation period:** {periods['validation'][0]} → {periods['validation'][1]}",
        "- **Discharge source:** Hub'Eau hydrometry API v2 (`obs_elab`, `QmnJ`, L/s)",
        "- **Precipitation / ET0 source:** Open-Meteo Historical Weather API (daily)",
        "- **Temporal resolution:** daily",
        f"- **Timezone (meteo aggregation):** {tz}",
        f"- **Missing precipitation:** {data_summary['missing']['precipitation_mm']} days",
        f"- **Missing ET0:** {data_summary['missing']['et0_mm']} days",
        f"- **Missing discharge:** {data_summary['missing']['discharge_mm']} days",
        f"- **Usable observations (all variables present):** {data_summary['usable_observations']} days",
        "",
        "Precipitation is represented by a single Open-Meteo point at the basin "
        "centroid and is not basin-averaged precipitation.",
        "",
        "Discharge conversion: Q_mm/day = Q_L/s × 0.0864 / basin_area_km².",
        "",
        "## 3. Hydrological model",
        "",
        "**GR4J** (Perrin, Michel & Andréassian, 2003) — four-parameter conceptual model.",
        "",
        "GR4J is a conceptual model. Calibrated parameters must not automatically "
        "be interpreted as direct physical measurements of catchment properties.",
        "",
        "| Parameter | Meaning | Unit | Demonstration bounds |",
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
            "**Initial-state convention:** production store at 30% of X1, routing store at 50% of X3, empty unit-hydrograph stores (airGR default fractions).",
            f"**Warm-up duration:** {warmup_days} days ({periods['warmup'][0]} → {periods['warmup'][1]}), excluded from all reported metrics.",
            "**Continuous-state behavior:** GR4J runs continuously from warm-up start through validation end without resetting states at period boundaries.",
            "",
            "## 4. Calibration experiment",
            "",
            f"- **Sampling method:** Latin Hypercube Sampling ({cal_cfg.get('sampler', 'latin_hypercube')})",
            f"- **N:** {cal_cfg['n_samples']}",
            f"- **Random seed:** {cal_cfg['seed']}",
            "- **Parameter bounds:** as in Section 3",
            "- **Ranking objective:** KGE_cal (calibration period only)",
            "- **Validation isolation rule:** validation metrics are diagnostic only",
            f"- **Total runtime:** {_fmt(runtime_total, 2)} s"
            + (" (not recorded — re-run `--calibrate` to persist)" if runtime_total is None else ""),
            f"- **Mean runtime per evaluation:** {_fmt(runtime_per, 4)} s"
            + (" (not recorded)" if runtime_per is None else ""),
            "",
            VALIDATION_ISOLATION_STATEMENT,
            "",
            "## 5. Uncalibrated baseline",
            "",
            "Fixed demonstration parameters (not manually tuned to observations): "
            f"X1={DEMO_PARAMETERS.X1}, X2={DEMO_PARAMETERS.X2}, "
            f"X3={DEMO_PARAMETERS.X3}, X4={DEMO_PARAMETERS.X4}.",
            "",
            "| Metric | Calibration | Validation |",
            "| --- | ---: | ---: |",
        ]
    )

    for _, row in inputs.baseline_metrics.iterrows():
        sections.append(
            f"| {row['metric']} | {_fmt(row['calibration'])} | {_fmt(row['validation'])} |"
        )

    sections.extend(
        [
            "",
            "## 6. Best calibration candidate",
            "",
            f"- **Run ID:** {int(best['run_id'])}",
            f"- **X1–X4:** {_fmt(best['x1'], 3)}, {_fmt(best['x2'], 3)}, "
            f"{_fmt(best['x3'], 3)}, {_fmt(best['x4'], 3)}",
            "",
            "**Calibration metrics:** "
            f"NSE={_fmt(best['nse_cal'])}, KGE={_fmt(best['kge_cal'])}, "
            f"r={_fmt(best['r_cal'])}, alpha={_fmt(best['alpha_cal'])}, "
            f"beta={_fmt(best['beta_cal'])}, log-NSE={_fmt(best['lognse_cal'])}, "
            f"bias={_fmt(best['bias_cal'])}.",
            "",
            "**Validation metrics (diagnostic only):** "
            f"NSE={_fmt(best['nse_val'])}, KGE={_fmt(best['kge_val'])}, "
            f"r={_fmt(best['r_val'])}, alpha={_fmt(best['alpha_val'])}, "
            f"beta={_fmt(best['beta_val'])}, log-NSE={_fmt(best['lognse_val'])}, "
            f"bias={_fmt(best['bias_val'])}.",
            "",
            f"**Bound proximity (within 2% of configured range):** {format_bound_proximity(close)}.",
            "",
            "## 7. Calibration vs validation",
            "",
            _comparison_metrics_table(inputs.baseline_metrics, best),
            "",
            _generalization_paragraph(best),
            "",
            "## 8. Parameter-space diagnostics",
            "",
            "**KGE_cal distribution (N = "
            f"{len(inputs.runs)}):** "
            f"min={_fmt(dist['min'])}, median={_fmt(dist['median'])}, "
            f"p90={_fmt(dist['p90'])}, p95={_fmt(dist['p95'])}, "
            f"p99={_fmt(dist['p99'])}, max={_fmt(dist['max'])}.",
            "",
            "**Threshold counts:**",
        ]
    )
    for key, value in counts.items():
        sections.append(f"- {key}: {value}")

    sections.extend(
        [
            "",
            f"**corr(KGE_cal, KGE_val):** {_fmt(corr)}",
            "",
            "**Behavioral parameter ranges (KGE_cal > official threshold):**",
            "",
            "| Parameter | min | median | max |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in param_ranges.iterrows():
        sections.append(
            f"| {row['parameter']} | {_fmt(row['min'], 3)} | "
            f"{_fmt(row['median'], 3)} | {_fmt(row['max'], 3)} |"
        )

    weakly_text = ", ".join(weakly) if weakly else "none identified"
    sections.extend(
        [
            "",
            EQUIFINALITY_STATEMENT,
            "",
            f"**Weakly constrained parameters in the behavioral set:** {weakly_text}.",
            "",
            "## 9. Behavioral ensemble",
            "",
            "- **Method:** GLUE-inspired behavioral ensemble (not a complete GLUE implementation)",
            f"- **Criterion:** KGE_cal > {threshold:g}",
            f"- **Ensemble size:** {len(inputs.behavioral_runs)} members",
            "- **Criterion is configurable** and is **not** a universal hydrological acceptability threshold",
            "- **Validation does not affect membership**",
            "",
            "**q50 validation metrics (diagnostic):** "
            f"NSE={_fmt(q50_dict['nse'])}, KGE={_fmt(q50_dict['kge'])}, "
            f"r={_fmt(q50_dict['r'])}, alpha={_fmt(q50_dict['alpha'])}, "
            f"beta={_fmt(q50_dict['beta'])}, log-NSE={_fmt(q50_dict['lognse'])}, "
            f"bias={_fmt(q50_dict['bias'])}.",
            "",
            "## 10. Uncertainty diagnostics",
            "",
            f"- **Empirical validation coverage of the behavioral envelope (q05–q95):** {_pct(coverage)}",
            f"- **Mean envelope width:** {_fmt(width['mean'])} mm/d",
            f"- **Median envelope width:** {_fmt(width['median'])} mm/d",
            f"- **p90 envelope width:** {_fmt(width['p90'])} mm/d",
            f"- **Mean width / mean observed validation discharge:** {_fmt(width['relative_to_obs_mean'])}",
            "",
            UNCERTAINTY_ENVELOPE_STATEMENT,
            "",
            UNDER_COVERAGE_STATEMENT,
            "",
            "The envelope explicitly excludes:",
            "- precipitation uncertainty",
            "- observation uncertainty",
            "- model-structure uncertainty",
            "- initial-state uncertainty",
            "",
            "**Threshold-sensitivity table (diagnostic; threshold not selected from validation):**",
            "",
            "| KGE_cal > | Members | Validation envelope coverage |",
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
            cov = _pct(coverage) if threshold_val == threshold else "n/a (re-run `--ensemble`)"
            sections.append(f"| {threshold_val:.2f} | {n} | {cov} |")

    sections.extend(
        [
            "",
            "**q50 vs best-calibration validation comparison (diagnostic):**",
            "",
            "| Metric | q50 | Best calibration |",
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
            "## 11. Hydrological-period characterization",
            "",
            "### Annual summary (2010–2015)",
            "",
            _hydrological_table(inputs.hydrological_summary),
            "",
            "### Calibration vs validation aggregates",
            "",
            _calibration_validation_aggregate(inputs.hydrological_summary),
            "",
            "Note: 2014 shows a lower annual maximum daily discharge than neighbouring years "
            "in this dataset; this is reported as a diagnostic observation requiring "
            "investigation, not as a definitive anomaly label.",
            "",
            "## 12. Limitations",
            "",
            "- Daily temporal resolution only",
            "- Centroid precipitation instead of basin-average precipitation",
            "- Conceptual lumped model rather than a physically distributed representation",
            "- No precipitation ensemble",
            "- No rating-curve uncertainty propagation",
            "- No state assimilation",
            "- Parametric uncertainty only in the reported envelope",
            "- Behavioral threshold is prototype-specific and configurable",
            "- Single pilot basin (one station configuration)",
            "",
            "## 13. Reproducibility",
            "",
            f"- **Generated at (UTC):** {repro['generated_at_utc']}",
            f"- **Configuration SHA256:** `{repro['config_sha256']}`",
            f"- **Git commit:** {repro['git_commit'] or 'not available'}",
            f"- **Python version:** {repro['python_version']}",
            f"- **Model / prototype version:** {repro['model_version']}",
            f"- **Random seed:** {repro['random_seed']}",
            f"- **N simulations:** {repro['n_samples']}",
            "",
            "**Package versions:**",
        ]
    )
    for pkg, ver in repro["package_versions"].items():
        sections.append(f"- {pkg}: {ver}")

    artifact_lines = [
        f"- `{inputs.output_dir / 'data' / 'basin_daily.csv'}` — processed daily data",
        f"- `{inputs.output_dir / 'runs.csv'}` — {len(inputs.runs)} calibration experiments",
        f"- `{inputs.output_dir / 'top20_calibration.csv'}` — top calibration candidates",
        f"- `{inputs.output_dir / 'behavioral_runs.csv'}` — behavioral ensemble members",
        f"- `{inputs.output_dir / 'ensemble_timeseries.csv'}` — ensemble quantile time series",
        f"- `{inputs.output_dir / 'ensemble_validation.png'}` — validation uncertainty figure",
        f"- `{inputs.output_dir / 'ensemble_full_validation.png'}` — full validation diagnostic",
        f"- `{inputs.output_dir / 'parameter_space_diagnostic.png'}` — parameter-space diagnostic",
        f"- `{inputs.output_dir / 'hydrological_years.png'}` — hydrological characterization figure",
    ]

    sections.extend(
        [
            "",
            "## 14. Artifacts",
            "",
            *artifact_lines,
            "",
            "## 15. Decision banner",
            "",
            STATUS_BANNER,
            "",
            "This prototype must not be interpreted as regulatory, forecasting, or "
            "operational hydrological validation.",
            "",
        ]
    )

    return "\n".join(sections)


def generate_rapport_calage(
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Load experiment outputs and write rapport_calage.md."""
    inputs = load_report_inputs(config_path, output_dir)
    content = render_rapport_calage(inputs)
    output_path = output_dir / REPORT_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
