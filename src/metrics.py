"""Objective functions for calibration and validation.

Responsibilities (spec §4):
- NSE (Nash-Sutcliffe efficiency)
- KGE with components r, alpha, beta (Kling-Gupta efficiency)
- log-NSE (behaviour on low flows)
- Volume bias

All metrics evaluate only timestamps where both observed and simulated values
are finite. Missing values are never interpolated or replaced.

log-NSE zero-flow convention (explicit, configurable, uniform):
    Transform discharge as ln(Q + epsilon) before applying the NSE formula.
    Default epsilon = 0.01 mm/day (``metrics.log_nse_epsilon_mm`` in config).
    The same epsilon is used for every evaluation; it is not tuned per basin.
    Values with Q + epsilon <= 0 after transformation would be invalid; with
    epsilon > 0 and non-negative discharge this does not occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LOG_NSE_EPSILON_MM = 0.01


@dataclass(frozen=True)
class MetricValue:
    """Result of a single metric evaluation."""

    value: float
    n_valid: int
    is_defined: bool
    undefined_reason: str | None = None

    def formatted(self, precision: int = 4) -> str:
        if self.is_defined:
            return f"{self.value:.{precision}f}"
        return f"undefined ({self.undefined_reason})"


@dataclass(frozen=True)
class MetricResult:
    """Container for all metrics on one period."""

    nse: MetricValue
    kge: MetricValue
    kge_r: MetricValue
    kge_alpha: MetricValue
    kge_beta: MetricValue
    lognse: MetricValue
    bias: MetricValue


def get_log_nse_epsilon(config: dict[str, Any] | None = None) -> float:
    """Return configured log-NSE epsilon [mm/day]."""
    if config is None:
        return DEFAULT_LOG_NSE_EPSILON_MM
    return float(config.get("metrics", {}).get("log_nse_epsilon_mm", DEFAULT_LOG_NSE_EPSILON_MM))


def _as_arrays(
    observed: np.ndarray | pd.Series,
    simulated: np.ndarray | pd.Series,
    period_mask: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError("observed and simulated must have the same shape")

    valid = np.isfinite(obs) & np.isfinite(sim)
    if period_mask is not None:
        mask = np.asarray(period_mask, dtype=bool)
        if mask.shape != obs.shape:
            raise ValueError("period_mask must match observed/simulated shape")
        valid &= mask

    return obs[valid], sim[valid], int(valid.sum())


def _undefined(n_valid: int, reason: str) -> MetricValue:
    return MetricValue(value=float("nan"), n_valid=n_valid, is_defined=False, undefined_reason=reason)


def nse(
    observed: np.ndarray | pd.Series,
    simulated: np.ndarray | pd.Series,
    period_mask: np.ndarray | pd.Series | None = None,
) -> MetricValue:
    """Nash-Sutcliffe efficiency."""
    obs, sim, n_valid = _as_arrays(observed, simulated, period_mask)
    if n_valid == 0:
        return _undefined(0, "no valid observation pairs")
    obs_mean = np.mean(obs)
    denominator = np.sum((obs - obs_mean) ** 2)
    if denominator == 0.0:
        return _undefined(n_valid, "zero observed variance")
    value = 1.0 - np.sum((obs - sim) ** 2) / denominator
    return MetricValue(value=float(value), n_valid=n_valid, is_defined=True)


def kge_components(
    observed: np.ndarray | pd.Series,
    simulated: np.ndarray | pd.Series,
    period_mask: np.ndarray | pd.Series | None = None,
) -> tuple[MetricValue, MetricValue, MetricValue]:
    """KGE components r (correlation), alpha (variability ratio), beta (bias ratio)."""
    obs, sim, n_valid = _as_arrays(observed, simulated, period_mask)
    if n_valid < 2:
        return (
            _undefined(n_valid, "fewer than 2 valid pairs for correlation"),
            _undefined(n_valid, "fewer than 2 valid pairs for alpha"),
            _undefined(n_valid, "fewer than 2 valid pairs for beta"),
        )

    obs_std = np.std(obs, ddof=1)
    sim_std = np.std(sim, ddof=1)
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)

    if obs_std == 0.0:
        r_val = _undefined(n_valid, "zero observed standard deviation")
        alpha_val = _undefined(n_valid, "zero observed standard deviation")
    else:
        if sim_std == 0.0:
            alpha_val = _undefined(n_valid, "zero simulated standard deviation")
        else:
            alpha_val = MetricValue(value=float(sim_std / obs_std), n_valid=n_valid, is_defined=True)
        r = np.corrcoef(obs, sim)[0, 1]
        if not np.isfinite(r):
            r_val = _undefined(n_valid, "correlation undefined")
        else:
            r_val = MetricValue(value=float(r), n_valid=n_valid, is_defined=True)

    if obs_mean == 0.0:
        beta_val = _undefined(n_valid, "zero observed mean")
    else:
        beta_val = MetricValue(value=float(sim_mean / obs_mean), n_valid=n_valid, is_defined=True)

    return r_val, alpha_val, beta_val


def kge(
    observed: np.ndarray | pd.Series,
    simulated: np.ndarray | pd.Series,
    period_mask: np.ndarray | pd.Series | None = None,
) -> tuple[MetricValue, MetricValue, MetricValue, MetricValue]:
    """Kling-Gupta efficiency and components (r, alpha, beta)."""
    r_val, alpha_val, beta_val = kge_components(observed, simulated, period_mask)
    n_valid = r_val.n_valid

    if not (r_val.is_defined and alpha_val.is_defined and beta_val.is_defined):
        reasons = [
            mv.undefined_reason
            for mv in (r_val, alpha_val, beta_val)
            if not mv.is_defined and mv.undefined_reason
        ]
        reason = "; ".join(reasons) if reasons else "component undefined"
        return (
            _undefined(n_valid, reason),
            r_val,
            alpha_val,
            beta_val,
        )

    value = 1.0 - np.sqrt(
        (r_val.value - 1.0) ** 2
        + (alpha_val.value - 1.0) ** 2
        + (beta_val.value - 1.0) ** 2
    )
    kge_val = MetricValue(value=float(value), n_valid=n_valid, is_defined=True)
    return kge_val, r_val, alpha_val, beta_val


def log_nse(
    observed: np.ndarray | pd.Series,
    simulated: np.ndarray | pd.Series,
    period_mask: np.ndarray | pd.Series | None = None,
    *,
    epsilon_mm: float = DEFAULT_LOG_NSE_EPSILON_MM,
) -> MetricValue:
    """Log-transformed Nash-Sutcliffe efficiency using ln(Q + epsilon)."""
    obs, sim, n_valid = _as_arrays(observed, simulated, period_mask)
    if n_valid == 0:
        return _undefined(0, "no valid observation pairs")
    if epsilon_mm <= 0.0:
        raise ValueError("log-NSE epsilon must be positive")

    log_obs = np.log(obs + epsilon_mm)
    log_sim = np.log(sim + epsilon_mm)
    if not (np.all(np.isfinite(log_obs)) and np.all(np.isfinite(log_sim))):
        return _undefined(n_valid, "log transform produced non-finite values")

    log_mean = np.mean(log_obs)
    denominator = np.sum((log_obs - log_mean) ** 2)
    if denominator == 0.0:
        return _undefined(n_valid, "zero variance of log-transformed observations")

    value = 1.0 - np.sum((log_obs - log_sim) ** 2) / denominator
    return MetricValue(value=float(value), n_valid=n_valid, is_defined=True)


def volume_bias(
    observed: np.ndarray | pd.Series,
    simulated: np.ndarray | pd.Series,
    period_mask: np.ndarray | pd.Series | None = None,
) -> MetricValue:
    """Volumetric bias: (sum(sim) - sum(obs)) / sum(obs)."""
    obs, sim, n_valid = _as_arrays(observed, simulated, period_mask)
    if n_valid == 0:
        return _undefined(0, "no valid observation pairs")
    obs_sum = np.sum(obs)
    if obs_sum == 0.0:
        return _undefined(n_valid, "zero observed volume")
    value = (np.sum(sim) - obs_sum) / obs_sum
    return MetricValue(value=float(value), n_valid=n_valid, is_defined=True)


def compute_metrics(
    observed: pd.Series,
    simulated: pd.Series,
    period_mask: pd.Series | None = None,
    *,
    epsilon_mm: float = DEFAULT_LOG_NSE_EPSILON_MM,
) -> MetricResult:
    """Compute all objective metrics for a given period mask."""
    mask = np.asarray(period_mask, dtype=bool) if period_mask is not None else None
    nse_val = nse(observed, simulated, mask)
    kge_val, r_val, alpha_val, beta_val = kge(observed, simulated, mask)
    lognse_val = log_nse(observed, simulated, mask, epsilon_mm=epsilon_mm)
    bias_val = volume_bias(observed, simulated, mask)
    return MetricResult(
        nse=nse_val,
        kge=kge_val,
        kge_r=r_val,
        kge_alpha=alpha_val,
        kge_beta=beta_val,
        lognse=lognse_val,
        bias=bias_val,
    )
