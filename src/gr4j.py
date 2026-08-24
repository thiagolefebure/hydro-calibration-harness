"""GR4J conceptual rainfall-runoff model (daily, 4 parameters).

Global lumped conceptual model after Perrin, Michel & Andréassian (2003).
Implementation follows the airGR Fortran reference (`frun_GR4J.f90`, `utils_D.f90`).

Parameter definitions (not bounds — bounds live in configuration):
- X1 : production store capacity [mm]
- X2 : intercatchment groundwater exchange coefficient [mm/day]
- X3 : routing store capacity [mm]
- X4 : unit hydrograph time base for UH1 [days] (UH2 base = 2*X4)

Internal states (carried continuously across time steps and period boundaries):
- production_store : level of the production store S [mm]
- routing_store    : level of the routing store R [mm]
- uh1              : UH1 convolution store (20 daily ordinates) [mm]
- uh2              : UH2 convolution store (40 daily ordinates) [mm]
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# airGR constants (frun_GR4J.f90)
UH1_SIZE = 20
UH2_SIZE = 40
UH_EXPONENT = 2.5
SPLIT_RATIO = float(np.float32(0.9))  # airGR uses REAL(4) constant B=0.9
STORED_VAL = 25.62890625  # (9/4)^4, used in percolation and routing outflow
WS_MAX = 13.0  # cap on Pn/X1 and En/X1 ratios

# airGR default initialisation (CreateIniStates): 30 % of X1, 50 % of X3
DEFAULT_PRODUCTION_STORE_FRACTION = 0.30
DEFAULT_ROUTING_STORE_FRACTION = 0.50


@dataclass(frozen=True)
class GR4JParameters:
    """GR4J model parameters.

    Units follow INRAE/airGR documentation (Perrin et al., 2003):
    - X1 : production store capacity [mm]
    - X2 : catchment water exchange coefficient [mm/day]
    - X3 : routing store one-day-ahead capacity [mm]
    - X4 : UH1 time base [days]
    """

    X1: float
    X2: float
    X3: float
    X4: float


@dataclass
class GR4JState:
    """Explicit GR4J internal states at a given time step."""

    production_store: float
    routing_store: float
    uh1: np.ndarray
    uh2: np.ndarray

    def copy(self) -> GR4JState:
        return replace(
            self,
            uh1=self.uh1.copy(),
            uh2=self.uh2.copy(),
        )


def default_initial_state(params: GR4JParameters) -> GR4JState:
    """Return airGR-default initial states (30 % X1, 50 % X3, empty UH stores)."""
    return GR4JState(
        production_store=DEFAULT_PRODUCTION_STORE_FRACTION * params.X1,
        routing_store=DEFAULT_ROUTING_STORE_FRACTION * params.X3,
        uh1=np.zeros(UH1_SIZE, dtype=float),
        uh2=np.zeros(UH2_SIZE, dtype=float),
    )


def _tanh_scaled(ws: float) -> float:
    """Scaled hyperbolic tangent used in production-store equations (tanHyp in airGR)."""
    if ws > WS_MAX:
        ws = WS_MAX
    exp_ws = np.exp(2.0 * ws)
    return (exp_ws - 1.0) / (exp_ws + 1.0)


def _ss1(i: int, x4: float) -> float:
    """S-curve SS1 for UH1 (utils_D.f90)."""
    if i <= 0:
        return 0.0
    if i < x4:
        return (i / x4) ** UH_EXPONENT
    return 1.0


def _ss2(i: int, x4: float) -> float:
    """S-curve SS2 for UH2 (utils_D.f90)."""
    if i <= 0:
        return 0.0
    if i <= x4:
        return 0.5 * (i / x4) ** UH_EXPONENT
    if i < 2.0 * x4:
        return 1.0 - 0.5 * (2.0 - i / x4) ** UH_EXPONENT
    return 1.0


def unit_hydrograph_ordinates(x4: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute discrete UH1 and UH2 ordinates from X4."""
    ord_uh1 = np.array([_ss1(i, x4) - _ss1(i - 1, x4) for i in range(1, UH1_SIZE + 1)], dtype=float)
    ord_uh2 = np.array([_ss2(i, x4) - _ss2(i - 1, x4) for i in range(1, UH2_SIZE + 1)], dtype=float)
    return ord_uh1, ord_uh2


def _percolation_from_production_store(level: float, x1: float) -> float:
    """Percolation from production store (uses STORED_VAL speed-up, airGR MOD_GR4J)."""
    if level <= 0.0 or x1 <= 0.0:
        return 0.0
    sr = level / x1
    sr4 = sr * sr * sr * sr
    return level * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + sr4 / STORED_VAL)))


def _outflow_from_routing_store(level: float, x3: float) -> float:
    """Outflow from routing store (no STORED_VAL divisor, airGR MOD_GR4J)."""
    if level <= 0.0 or x3 <= 0.0:
        return 0.0
    rr = level / x3
    rr4 = rr * rr * rr * rr
    return level * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + rr4)))


def _exchange(routing_level: float, x2: float, x3: float) -> float:
    """Potential intercatchment exchange term (EXCH in airGR)."""
    if x3 <= 0.0:
        return 0.0
    rr = routing_level / x3
    return x2 * (rr ** 3.5)


def _convolve_uh1(state_uh1: np.ndarray, ord_uh1: np.ndarray, prhu1: float, x4: float) -> None:
    limit = max(1, min(UH1_SIZE - 1, int(x4 + 1.0)))
    for k in range(limit):
        state_uh1[k] = state_uh1[k + 1] + ord_uh1[k] * prhu1
    state_uh1[UH1_SIZE - 1] = ord_uh1[UH1_SIZE - 1] * prhu1


def _convolve_uh2(state_uh2: np.ndarray, ord_uh2: np.ndarray, prhu2: float, x4: float) -> None:
    limit = max(1, min(UH2_SIZE - 1, 2 * int(x4 + 1.0)))
    for k in range(limit):
        state_uh2[k] = state_uh2[k + 1] + ord_uh2[k] * prhu2
    state_uh2[UH2_SIZE - 1] = ord_uh2[UH2_SIZE - 1] * prhu2


def step_gr4j(
    precipitation_mm: float,
    pet_mm: float,
    params: GR4JParameters,
    state: GR4JState,
    ord_uh1: np.ndarray,
    ord_uh2: np.ndarray,
) -> tuple[float, GR4JState]:
    """Advance GR4J by one daily time step.

    Returns simulated discharge [mm/day] and updated state.
    """
    x1, x2, x3, x4 = params.X1, params.X2, params.X3, params.X4
    st = state.copy()

    p1 = float(precipitation_mm)
    e = float(pet_mm)

    # Production store (MOD_GR4J in frun_GR4J.f90)
    if p1 <= e:
        en = e - p1
        pn = 0.0
        ws = en / x1
        t_ws = _tanh_scaled(ws)
        sr = st.production_store / x1
        er = st.production_store * (2.0 - sr) * t_ws / (1.0 + (1.0 - sr) * t_ws)
        st.production_store -= er
        ps = 0.0
        pr = 0.0
    else:
        pn = p1 - e
        ws = pn / x1
        t_ws = _tanh_scaled(ws)
        sr = st.production_store / x1
        ps = x1 * (1.0 - sr * sr) * t_ws / (1.0 + sr * t_ws)
        pr = pn - ps
        st.production_store += ps

    if st.production_store < 0.0:
        st.production_store = 0.0

    perc = _percolation_from_production_store(st.production_store, x1)
    st.production_store -= perc
    pr += perc

    prhu1 = pr * SPLIT_RATIO
    prhu2 = pr * (1.0 - SPLIT_RATIO)

    _convolve_uh1(st.uh1, ord_uh1, prhu1, x4)
    _convolve_uh2(st.uh2, ord_uh2, prhu2, x4)

    exch = _exchange(st.routing_store, x2, x3)

    q9 = st.uh1[0]
    q1 = st.uh2[0]

    aexch1 = exch
    if st.routing_store + q9 + exch < 0.0:
        aexch1 = -st.routing_store - q9
    st.routing_store = st.routing_store + q9 + exch
    if st.routing_store < 0.0:
        st.routing_store = 0.0

    qr = _outflow_from_routing_store(st.routing_store, x3)
    st.routing_store -= qr

    aexch2 = exch
    if q1 + exch < 0.0:
        aexch2 = -q1
    qd = max(0.0, q1 + exch)

    discharge = qr + qd
    if discharge < 0.0:
        discharge = 0.0

    return discharge, st


def run_gr4j(
    inputs: pd.DataFrame,
    params: GR4JParameters,
    initial_state: GR4JState | None = None,
) -> tuple[pd.Series, GR4JState]:
    """Run GR4J forward over daily precipitation and PET inputs.

    Parameters
    ----------
    inputs:
        DataFrame indexed by date with columns ``precipitation_mm`` and ``et0_mm``.
        Must not contain NaN values (gaps must be handled upstream).
    params:
        GR4J parameter set. Bounds are not enforced here.
    initial_state:
        Optional starting state. Defaults to airGR fractions (30 % X1, 50 % X3).

    Returns
    -------
    discharge_mm:
        Simulated daily discharge [mm/day], indexed like ``inputs``.
    final_state:
        Model state after the last time step (for continuous simulations).
    """
    required = {"precipitation_mm", "et0_mm"}
    missing = required - set(inputs.columns)
    if missing:
        raise ValueError(f"inputs missing required columns: {sorted(missing)}")

    if inputs[["precipitation_mm", "et0_mm"]].isna().any().any():
        raise ValueError("inputs contain NaN; resolve gaps before running GR4J")

    state = default_initial_state(params) if initial_state is None else initial_state.copy()
    ord_uh1, ord_uh2 = unit_hydrograph_ordinates(params.X4)

    discharge = np.empty(len(inputs), dtype=float)
    for i, (_, row) in enumerate(inputs.iterrows()):
        q, state = step_gr4j(
            precipitation_mm=row["precipitation_mm"],
            pet_mm=row["et0_mm"],
            params=params,
            state=state,
            ord_uh1=ord_uh1,
            ord_uh2=ord_uh2,
        )
        discharge[i] = q

    series = pd.Series(discharge, index=inputs.index, name="discharge_mm")
    return series, state


def run_gr4j_continuous_periods(
    inputs: pd.DataFrame,
    params: GR4JParameters,
    period_bounds: dict[str, tuple[str, str]] | None = None,
    initial_state: GR4JState | None = None,
) -> tuple[pd.Series, GR4JState]:
    """Run one continuous GR4J simulation over all inputs.

    Period boundaries (warmup / calibration / validation) do not reset model
    states. The ``period_bounds`` argument is accepted for API clarity but does
    not alter the continuous simulation (spec §2).
    """
    _ = period_bounds
    return run_gr4j(inputs, params, initial_state=initial_state)
