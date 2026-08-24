"""Parameter-space sampling for calibration experiments.

Responsibilities (per spec §4):
- Latin Hypercube sampling over configurable parameter bounds.
- Fixed random seed for reproducibility.
- Return N parameter sets within documented demonstration bounds (not universal physical bounds).

Bounds are never hard-coded here; they are read from configuration only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARAMETER_ORDER = ("X1", "X2", "X3", "X4")


def latin_hypercube_sample(
    parameter_bounds: dict[str, list[float]],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Draw N parameter sets via Latin Hypercube sampling within configured bounds.

    Uses a stratified unit-hypercube design scaled independently per parameter.
    Columns match GR4J parameter names (X1, X2, X3, X4).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    names = [name for name in PARAMETER_ORDER if name in parameter_bounds]
    if not names:
        raise ValueError("parameter_bounds must include GR4J parameters")

    rng = np.random.default_rng(seed)
    n_dim = len(names)
    unit = np.zeros((n_samples, n_dim), dtype=float)

    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        unit[:, j] = (perm + rng.random(n_samples)) / n_samples

    rows = {}
    for j, name in enumerate(names):
        lower, upper = parameter_bounds[name]
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {name}: [{lower}, {upper}]")
        rows[name] = lower + unit[:, j] * (upper - lower)

    return pd.DataFrame(rows)


def sample_parameters(config: dict[str, Any]) -> pd.DataFrame:
    """Sample parameter sets according to calibration section of basin config."""
    cal = config["calibration"]
    bounds = config["model"]["parameter_bounds"]
    sampler = cal.get("sampler", "latin_hypercube")

    if sampler != "latin_hypercube":
        raise ValueError(f"Unsupported sampler: {sampler!r}")

    return latin_hypercube_sample(
        parameter_bounds=bounds,
        n_samples=cal["n_samples"],
        seed=cal["seed"],
    )


def save_parameter_samples(samples: pd.DataFrame, output_path: Path) -> Path:
    """Persist the sampled parameter matrix for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False)
    return output_path
