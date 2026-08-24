"""Generate a basic observed-vs-simulated GR4J hydrograph (Phase 2 demo)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data import get_meteo_timezone, load_basin_data
from src.gr4j import GR4JParameters, run_gr4j_continuous_periods

# Demonstration parameters (within spec bounds; not calibrated).
DEMO_PARAMETERS = GR4JParameters(X1=350.0, X2=0.0, X3=90.0, X4=1.4)


def _period_mask(index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return (index >= start_ts) & (index <= end_ts)


def plot_gr4j_hydrograph(
    data: pd.DataFrame,
    simulated: pd.Series,
    config: dict,
    output_path: Path,
) -> None:
    """Plot observed vs simulated discharge with inverted precipitation (spec §6 style)."""
    periods = config["periods"]
    index = data.index

    fig, (ax_rain, ax_q) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 3]},
    )

    ax_rain.bar(
        index,
        data["precipitation_mm"],
        width=1.0,
        color="#6baed6",
        alpha=0.7,
        label="Precipitation",
    )
    ax_rain.invert_yaxis()
    ax_rain.set_ylabel("Precip (mm/d)")
    ax_rain.legend(loc="lower right")

    ax_q.plot(index, data["discharge_mm"], color="black", linewidth=1.0, label="Observed")
    ax_q.plot(index, simulated, color="#e6550d", linewidth=1.0, label="Simulated (GR4J demo)")

    for name, color in [("warmup", "#f0f0f0"), ("calibration", "#e8f4ea"), ("validation", "#fde8e8")]:
        start, end = periods[name]
        mask = _period_mask(index, start, end)
        if mask.any():
            ax_q.axvspan(index[mask][0], index[mask][-1], color=color, alpha=0.5, label=name)

    ax_q.set_ylabel("Discharge (mm/d)")
    ax_q.set_xlabel("Date")
    ax_q.legend(loc="upper right")
    ax_q.set_title("GR4J demo hydrograph — observed vs simulated")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_demo(
    config_path: Path = Path("config/basin.yaml"),
    cache_dir: Path = Path("cache"),
    output_path: Path = Path("output/hydrograph_gr4j_demo.png"),
) -> tuple[pd.Series, Path]:
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    data, _qa = load_basin_data(
        config,
        cache_dir=cache_dir,
        timezone=get_meteo_timezone(config),
    )
    model_inputs = data[["precipitation_mm", "et0_mm"]].dropna()
    simulated, _ = run_gr4j_continuous_periods(
        model_inputs,
        DEMO_PARAMETERS,
        period_bounds=config["periods"],
    )

    aligned_obs = data.loc[simulated.index, "discharge_mm"]
    plot_df = data.loc[simulated.index].copy()
    plot_gr4j_hydrograph(plot_df, simulated, config, output_path)

    return simulated, output_path


if __name__ == "__main__":
    series, path = run_demo()
    print(f"GR4J demo hydrograph written to {path.resolve()}")
    print(f"Simulated discharge range: {series.min():.3f} - {series.max():.3f} mm/d")
