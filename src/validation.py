"""Split-sample validation and candidate ranking.

Responsibilities (per spec §4):
- Enforce explicit calendar-based warmup / calibration / validation periods.
- Exclude warm-up from all metrics.
- Select best calibration candidate (max KGE_cal) without using validation for optimization.
- Retain Top-N calibration candidates and report their validation performance (no retuning).
- Make overfitting / robustness trade-offs visible in outputs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Spec §4 example: inspect Top 20 calibration candidates in validation.
DEFAULT_TOP_N_CANDIDATES = 20

CALIBRATION_RANK_COLUMN = "rank_kge_cal"


def period_masks(dates: pd.DatetimeIndex, config: dict[str, Any]) -> dict[str, pd.Series]:
    """Build boolean masks for warmup, calibration, and validation periods."""
    periods = config["periods"]

    def _mask(start: str, end: str) -> pd.Series:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return (dates >= start_ts) & (dates <= end_ts)

    return {
        "warmup": _mask(*periods["warmup"]),
        "calibration": _mask(*periods["calibration"]),
        "validation": _mask(*periods["validation"]),
    }


def rank_by_kge_calibration(runs: pd.DataFrame) -> pd.Series:
    """Rank runs by KGE_cal only (1 = best). Undefined KGE_cal ranks last."""
    if "kge_cal" not in runs.columns:
        raise ValueError("runs DataFrame missing kge_cal column")
    return runs["kge_cal"].rank(method="first", ascending=False).astype(int)


def select_best_calibration_candidate(runs: pd.DataFrame) -> pd.Series:
    """Return the run with maximum KGE on the calibration period."""
    ranked = runs.sort_values(["kge_cal", "run_id"], ascending=[False, True], na_position="last")
    if ranked.empty or ranked["kge_cal"].notna().sum() == 0:
        raise ValueError("No runs with defined KGE_cal available for selection")
    return ranked.iloc[0]


def top_calibration_candidates(
    runs: pd.DataFrame,
    n: int = DEFAULT_TOP_N_CANDIDATES,
) -> pd.DataFrame:
    """Return Top-N runs ranked exclusively by KGE_calibration."""
    ranked = runs.sort_values(["kge_cal", "run_id"], ascending=[False, True], na_position="last")
    return ranked.head(n).copy()
