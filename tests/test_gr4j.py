"""Tests for the GR4J hydrological model (spec §2).

Reference case validated against grsuite (Python port of airGR 1.7.9 Fortran).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.gr4j import (
    GR4JParameters,
    default_initial_state,
    run_gr4j,
    run_gr4j_continuous_periods,
    unit_hydrograph_ordinates,
)

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "gr4j_grsuite_reference.json").read_text(encoding="utf-8"))


def _reference_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precipitation_mm": REFERENCE["precipitation_mm"],
            "et0_mm": REFERENCE["et0_mm"],
        }
    )


def _reference_params() -> GR4JParameters:
    return GR4JParameters(**REFERENCE["params"])


def test_deterministic_reproducibility() -> None:
    inputs = _reference_inputs()
    params = _reference_params()
    q1, s1 = run_gr4j(inputs, params)
    q2, s2 = run_gr4j(inputs, params)
    pd.testing.assert_series_equal(q1, q2)
    assert s1.production_store == pytest.approx(s2.production_store)
    assert s1.routing_store == pytest.approx(s2.routing_store)
    assert np.array_equal(s1.uh1, s2.uh1)
    assert np.array_equal(s1.uh2, s2.uh2)


def test_matches_grsuite_reference_case() -> None:
    """Numerical comparison with trusted grsuite/airGR reference outputs."""
    inputs = _reference_inputs()
    params = _reference_params()
    simulated, _ = run_gr4j(inputs, params)
    expected = np.array(REFERENCE["discharge_mm"])
    np.testing.assert_allclose(simulated.values, expected, rtol=0.0, atol=1e-12)


def test_zero_rain_storage_decay() -> None:
    """Under zero precipitation, stores decline and discharge tends toward zero."""
    params = GR4JParameters(X1=300, X2=0, X3=100, X4=1.5)
    n = 30
    inputs = pd.DataFrame(
        {
            "precipitation_mm": np.zeros(n),
            "et0_mm": np.ones(n),
        }
    )
    simulated, final_state = run_gr4j(inputs, params)
    initial = default_initial_state(params)

    assert final_state.production_store < initial.production_store
    assert final_state.routing_store <= initial.routing_store
    assert simulated.iloc[-1] < simulated.iloc[0]
    assert simulated.min() >= 0.0


def test_impulse_response() -> None:
    """Single-day rainfall impulse produces a bounded, delayed hydrograph peak."""
    params = GR4JParameters(X1=300, X2=0, X3=100, X4=1.5)
    n = 20
    precip = np.zeros(n)
    precip[5] = 30.0
    inputs = pd.DataFrame({"precipitation_mm": precip, "et0_mm": np.ones(n)})

    simulated, _ = run_gr4j(inputs, params)

    assert simulated.max() > 0.0
    peak_idx = int(simulated.values.argmax())
    assert peak_idx > 5
    assert simulated.max() < 30.0


def test_simulated_discharge_non_negative() -> None:
    """Simulated discharge must remain non-negative for varied forcing."""
    params = GR4JParameters(X1=350, X2=1.0, X3=90, X4=2.0)
    n = 40
    rng = np.random.default_rng(42)
    inputs = pd.DataFrame(
        {
            "precipitation_mm": rng.uniform(0, 25, n),
            "et0_mm": rng.uniform(0.5, 4.0, n),
        }
    )
    simulated, _ = run_gr4j(inputs, params)
    assert (simulated >= 0.0).all()


def test_state_continuity_across_period_boundary() -> None:
    """Continuous simulation must match split run with carried-over state."""
    params = _reference_params()
    inputs = _reference_inputs()
    split = 25

    full_q, _ = run_gr4j(inputs, params)
    first_q, mid_state = run_gr4j(inputs.iloc[:split], params)
    second_q, _ = run_gr4j(inputs.iloc[split:], params, initial_state=mid_state)

    combined = pd.concat([first_q, second_q])
    pd.testing.assert_series_equal(full_q, combined)


def test_run_gr4j_continuous_periods_does_not_reset_state() -> None:
    """Period metadata must not trigger internal state reinitialization."""
    config = {
        "periods": {
            "warmup": ["2000-01-01", "2000-01-10"],
            "calibration": ["2000-01-11", "2000-01-20"],
            "validation": ["2000-01-21", "2000-01-30"],
        }
    }
    inputs = _reference_inputs()
    params = _reference_params()

    q_continuous, _ = run_gr4j_continuous_periods(
        inputs,
        params,
        period_bounds=config["periods"],
    )
    q_direct, _ = run_gr4j(inputs, params)
    pd.testing.assert_series_equal(q_continuous, q_direct)


def test_step_gr4j_rejects_nan_inputs_via_run_gr4j() -> None:
    inputs = _reference_inputs()
    inputs.loc[inputs.index[0], "precipitation_mm"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_gr4j(inputs, _reference_params())


def test_default_initial_state_fractions() -> None:
    params = GR4JParameters(X1=400, X2=0, X3=200, X4=2.0)
    state = default_initial_state(params)
    assert state.production_store == pytest.approx(0.30 * 400)
    assert state.routing_store == pytest.approx(0.50 * 200)
    assert state.uh1.sum() == 0.0
    assert state.uh2.sum() == 0.0


def test_unit_hydrograph_ordinates_sum_to_one() -> None:
    ord1, ord2 = unit_hydrograph_ordinates(1.4)
    assert ord1.sum() == pytest.approx(1.0)
    assert ord2.sum() == pytest.approx(1.0)
