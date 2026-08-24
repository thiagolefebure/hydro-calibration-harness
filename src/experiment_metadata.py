"""Experiment metadata for reproducibility blocks in the calibration report."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src import MODEL_VERSION

METADATA_FILENAME = "experiment_metadata.json"


def config_sha256(config_path: Path) -> str:
    """Return SHA256 hex digest of the raw configuration file bytes."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def git_commit_hash(repo_root: Path | None = None) -> str | None:
    """Return current Git commit hash if available."""
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit or None


def package_versions() -> dict[str, str]:
    """Collect versions of core runtime dependencies."""
    packages = ("numpy", "pandas", "matplotlib", "PyYAML", "pytest")
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / METADATA_FILENAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / METADATA_FILENAME
    existing = load_metadata(output_dir)
    existing.update(metadata)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    return path


def build_reproducibility_block(
    config_path: Path,
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble reproducibility metadata from config and environment."""
    meta = load_metadata(output_dir) if output_dir is not None else {}
    return {
        "generated_at_utc": utc_timestamp(),
        "config_sha256": config_sha256(config_path),
        "git_commit": git_commit_hash(),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "model_version": MODEL_VERSION,
        "random_seed": config.get("calibration", {}).get("seed"),
        "n_samples": config.get("calibration", {}).get("n_samples"),
        "calibration_runtime_s": meta.get("calibration_runtime_s"),
        "calibration_runtime_per_eval_s": meta.get("calibration_runtime_per_eval_s"),
        "ensemble_runtime_s": meta.get("ensemble_runtime_s"),
    }
