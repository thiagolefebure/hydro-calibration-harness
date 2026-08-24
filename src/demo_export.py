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
)

DEMO_FIGURES = (
    "demo_01_calibration_impact.png",
    "demo_02_validation.png",
    "demo_03_uncertainty.png",
)


class DemoExportError(FileNotFoundError):
    """Raised when required scientific artifacts are missing."""


def _baseline_metric(baseline: pd.DataFrame, name: str, column: str) -> float:
    row = baseline.loc[baseline["metric"] == name]
    if row.empty:
        raise KeyError(f"Metric {name!r} missing from metrics_uncalibrated.csv")
    return float(row.iloc[0][column])


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
            "Missing required artifacts for demo export: "
            + ", ".join(missing)
            + ". Run the scientific pipeline / `python run.py --demo` first."
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
        "title": "Automated Rainfall–Runoff Calibration",
        "subtitle": "Prototype of a reproducible calibration harness for hydrological models",
        "secondary_line": (
            "Real hydrometric data · constrained parameter exploration · "
            "independent validation · explicit uncertainty"
        ),
        "prototype_badge": "PROTOTYPE — NOT FOR OPERATIONAL DECISION-MAKING",
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
            "temporal_resolution": "Daily",
        },
        "model": {
            "name": model["name"],
            "label": f"{model['name']} — conceptual rainfall–runoff model",
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
                "Validation observations are never used for parameter fitting or ranking."
            ),
            "parametric_only": (
                "Precipitation, observation, initial-state and model-structure "
                "uncertainty are not propagated in this prototype."
            ),
            "envelope_disclaimer": (
                "q05–q95 is NOT a calibrated 90% prediction interval"
            ),
        },
    }


def write_report_html(markdown_path: Path, html_path: Path) -> Path:
    """Wrap the existing report Markdown in a simple HTML page (content unchanged)."""
    content = markdown_path.read_text(encoding="utf-8")
    escaped = html.escape(content)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Calibration report</title>
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
    <p><a href="../index.html">← Back to demo</a></p>
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

    data = build_demo_data(config_path, output_dir)
    data_path = assets / "demo_data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    report_html = write_report_html(
        output_dir / "rapport_calage.md",
        assets / "rapport_calage.html",
    )

    return {
        "demo_data": data_path,
        "report_html": report_html,
        **figure_paths,
    }
