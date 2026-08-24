**STATUS: PROTOTYPE / NOT FOR OPERATIONAL HYDROLOGICAL DECISION-MAKING**

# Automated Rainfall–Runoff Calibration Report

## 1. Prototype scope

This document describes a **calibration-engineering prototype** built around the conceptual **GR4J** rainfall–runoff model. The goal is to demonstrate automated parameter exploration, explicit calibration/validation separation, GLUE-inspired behavioral-ensemble diagnostics, transparent uncertainty communication, and full reproducibility.

This is **not** an operational flood-forecasting or regulatory decision-support system.

## 2. Basin and data

- **Station code:** H0203020
- **Station name:** La Laignes à Molesme
- **Basin area:** 615.0 km²
- **Centroid:** (47.96178, 4.361)
- **Analysis period:** 2010-01-01 → 2015-12-31
- **Warm-up period:** 2010-01-01 → 2010-12-31
- **Calibration period:** 2011-01-01 → 2013-12-31
- **Validation period:** 2014-01-01 → 2015-12-31
- **Discharge source:** Hub'Eau hydrometry API v2 (`obs_elab`, `QmnJ`, L/s)
- **Precipitation / ET0 source:** Open-Meteo Historical Weather API (daily)
- **Temporal resolution:** daily
- **Timezone (meteo aggregation):** Europe/Paris
- **Missing precipitation:** 0 days
- **Missing ET0:** 0 days
- **Missing discharge:** 0 days
- **Usable observations (all variables present):** 2191 days

Precipitation is represented by a single Open-Meteo point at the basin centroid and is not basin-averaged precipitation.

Discharge conversion: Q_mm/day = Q_L/s × 0.0864 / basin_area_km².

## 3. Hydrological model

**GR4J** (Perrin, Michel & Andréassian, 2003) — four-parameter conceptual model.

GR4J is a conceptual model. Calibrated parameters must not automatically be interpreted as direct physical measurements of catchment properties.

| Parameter | Meaning | Unit | Demonstration bounds |
| --- | --- | --- | --- |
| X1 | Production store capacity | mm | [100, 1200] |
| X2 | Groundwater exchange flux | mm | [-5, 3] |
| X3 | Routing store capacity | mm | [20, 300] |
| X4 | Unit-hydrograph time base (UH1/UH2) | days | [1.1, 2.9] |

**Initial-state convention:** production store at 30% of X1, routing store at 50% of X3, empty unit-hydrograph stores (airGR default fractions).
**Warm-up duration:** 365 days (2010-01-01 → 2010-12-31), excluded from all reported metrics.
**Continuous-state behavior:** GR4J runs continuously from warm-up start through validation end without resetting states at period boundaries.

## 4. Calibration experiment

- **Sampling method:** Latin Hypercube Sampling (latin_hypercube)
- **N:** 5000
- **Random seed:** 42
- **Parameter bounds:** as in Section 3
- **Ranking objective:** KGE_cal (calibration period only)
- **Validation isolation rule:** validation metrics are diagnostic only
- **Total runtime:** 898.59 s
- **Mean runtime per evaluation:** 0.1797 s

Validation observations were not used for parameter sampling, ranking, parameter selection, stopping criteria, or behavioral-ensemble membership.

## 5. Uncalibrated baseline

Fixed demonstration parameters (not manually tuned to observations): X1=350.0, X2=0.0, X3=90.0, X4=1.4.

| Metric | Calibration | Validation |
| --- | ---: | ---: |
| NSE | 0.4402 | 0.0600 |
| KGE | 0.4731 | 0.2026 |
| r | 0.8714 | 0.8936 |
| alpha | 1.2594 | 1.5013 |
| beta | 1.4403 | 1.6109 |
| log-NSE | 0.6633 | 0.5791 |
| Volume bias | 0.4403 | 0.6109 |

## 6. Best calibration candidate

- **Run ID:** 3058
- **X1–X4:** 248.360, -3.368, 86.318, 2.699

**Calibration metrics:** NSE=0.8083, KGE=0.9027, r=0.9071, alpha=1.0269, beta=0.9897, log-NSE=0.7695, bias=-0.0103.

**Validation metrics (diagnostic only):** NSE=0.7923, KGE=0.8350, r=0.9171, alpha=1.1295, beta=1.0597, log-NSE=0.8705, bias=0.0597.

**Bound proximity (within 2% of configured range):** none (all parameters >2% from configured bounds).

## 7. Calibration vs validation

| Metric | Uncalibrated (cal) | Uncalibrated (val) | Best cal (cal) | Best cal (val) |
| --- | ---: | ---: | ---: | ---: |
| NSE | 0.4402 | 0.0600 | 0.8083 | 0.7923 |
| KGE | 0.4731 | 0.2026 | 0.9027 | 0.8350 |
| r | 0.8714 | 0.8936 | 0.9071 | 0.9171 |
| alpha | 1.2594 | 1.5013 | 1.0269 | 1.1295 |
| beta | 1.4403 | 1.6109 | 0.9897 | 1.0597 |
| log-NSE | 0.6633 | 0.5791 | 0.7695 | 0.8705 |
| volumetric bias | 0.4403 | 0.6109 | -0.0103 | 0.0597 |

The best calibration candidate retains most of its calibration-period skill in validation (KGE_cal = 0.9027, KGE_val = 0.8350). This suggests reasonable split-sample generalization for this pilot basin, without claiming operational robustness.

## 8. Parameter-space diagnostics

**KGE_cal distribution (N = 5000):** min=-3.9547, median=0.4862, p90=0.7080, p95=0.7609, p99=0.8450, max=0.9027.

**Threshold counts:**
- kge_cal_gt_0.5: 2342
- kge_cal_gt_0.6: 1271
- kge_cal_gt_0.7: 540
- kge_cal_gt_0.75: 280
- kge_cal_gt_0.8: 126
- kge_cal_gt_0.85: 45

**corr(KGE_cal, KGE_val):** 0.9529

**Behavioral parameter ranges (KGE_cal > official threshold):**

| Parameter | min | median | max |
| --- | ---: | ---: | ---: |
| X1 | 180.700 | 291.893 | 688.265 |
| X2 | -4.997 | -3.104 | -0.571 |
| X3 | 31.514 | 95.396 | 209.448 |
| X4 | 1.117 | 1.996 | 2.897 |

Multiple distinct parameter sets achieve similar calibration performance; therefore the highest-scoring parameter set should not be interpreted as a uniquely identified physical truth.

**Weakly constrained parameters in the behavioral set:** X2, X3, X4.

## 9. Behavioral ensemble

- **Method:** GLUE-inspired behavioral ensemble (not a complete GLUE implementation)
- **Criterion:** KGE_cal > 0.8
- **Ensemble size:** 126 members
- **Criterion is configurable** and is **not** a universal hydrological acceptability threshold
- **Validation does not affect membership**

**q50 validation metrics (diagnostic):** NSE=0.7893, KGE=0.8342, r=0.9120, alpha=1.0860, beta=1.1111, log-NSE=0.8754, bias=0.1111.

## 10. Uncertainty diagnostics

- **Empirical validation coverage of the behavioral envelope (q05–q95):** 59.9%
- **Mean envelope width:** 0.2063 mm/d
- **Median envelope width:** 0.1586 mm/d
- **p90 envelope width:** 0.4023 mm/d
- **Mean width / mean observed validation discharge:** 0.5086

The q05–q95 envelope is the dispersion of the selected behavioral parameter simulations. It is not a calibrated 90% confidence or prediction interval.

The observed under-coverage demonstrates that parametric dispersion alone is insufficient to represent total predictive uncertainty.

The envelope explicitly excludes:
- precipitation uncertainty
- observation uncertainty
- model-structure uncertainty
- initial-state uncertainty

**Threshold-sensitivity table (diagnostic; threshold not selected from validation):**

| KGE_cal > | Members | Validation envelope coverage |
| --- | ---: | ---: |
| 0.70 | 540 | 71.8% |
| 0.75 | 280 | 67.1% |
| 0.80 | 126 | 59.9% |
| 0.85 | 45 | 47.1% |

**q50 vs best-calibration validation comparison (diagnostic):**

| Metric | q50 | Best calibration |
| --- | ---: | ---: |
| nse | 0.7893 | 0.7923 |
| kge | 0.8342 | 0.8350 |
| r | 0.9120 | 0.9171 |
| alpha | 1.0860 | 1.1295 |
| beta | 1.1111 | 1.0597 |
| lognse | 0.8754 | 0.8705 |
| bias | 0.1111 | 0.0597 |

## 11. Hydrological-period characterization

### Annual summary (2010–2015)

| Year | Precip (mm) | Q depth (mm) | Runoff ratio | Mean Q (mm/d) | Max Q (mm/d) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2010 | 903.5 | 142.0 | 0.1572 | 0.3891 | 3.4588 |
| 2011 | 712.3 | 113.7 | 0.1596 | 0.3115 | 3.1396 |
| 2012 | 920.4 | 188.4 | 0.2047 | 0.5147 | 3.3464 |
| 2013 | 980.3 | 310.8 | 0.3171 | 0.8516 | 3.5026 |
| 2014 | 840.6 | 147.7 | 0.1757 | 0.4046 | 1.3578 |
| 2015 | 724.6 | 148.4 | 0.2048 | 0.4065 | 2.7543 |

### Calibration vs validation aggregates

| Period aggregate | Precip (mm) | Q depth (mm) | Runoff ratio | Mean Q (mm/d) | Max Q (mm/d) |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration | 2613.0 | 612.9 | 0.2346 | 0.5592 | 3.5026 |
| validation | 1565.2 | 296.1 | 0.1892 | 0.4056 | 2.7543 |

Note: 2014 shows a lower annual maximum daily discharge than neighbouring years in this dataset; this is reported as a diagnostic observation requiring investigation, not as a definitive anomaly label.

## 12. Limitations

- Daily temporal resolution only
- Centroid precipitation instead of basin-average precipitation
- Conceptual lumped model rather than a physically distributed representation
- No precipitation ensemble
- No rating-curve uncertainty propagation
- No state assimilation
- Parametric uncertainty only in the reported envelope
- Behavioral threshold is prototype-specific and configurable
- Single pilot basin (one station configuration)

## 13. Reproducibility

- **Generated at (UTC):** 2026-08-23T16:42:06Z
- **Configuration SHA256:** `8b0f60d562d633a7b8a23eec414e91d022b665f1d3c9f774c5f6fc13c39b1645`
- **Git commit:** not available
- **Python version:** 3.14.0
- **Model / prototype version:** gr4j-prototype-v0.1
- **Random seed:** 42
- **N simulations:** 5000

**Package versions:**
- python: 3.14.0
- numpy: 2.3.5
- pandas: 2.3.3
- matplotlib: 3.11.0
- PyYAML: 6.0.3
- pytest: 8.4.2

## 14. Artifacts

- `output\data\basin_daily.csv` — processed daily data
- `output\runs.csv` — 5000 calibration experiments
- `output\top20_calibration.csv` — top calibration candidates
- `output\behavioral_runs.csv` — behavioral ensemble members
- `output\ensemble_timeseries.csv` — ensemble quantile time series
- `output\ensemble_validation.png` — validation uncertainty figure
- `output\ensemble_full_validation.png` — full validation diagnostic
- `output\parameter_space_diagnostic.png` — parameter-space diagnostic
- `output\hydrological_years.png` — hydrological characterization figure

## 15. Decision banner

**STATUS: PROTOTYPE / NOT FOR OPERATIONAL HYDROLOGICAL DECISION-MAKING**

This prototype must not be interpreted as regulatory, forecasting, or operational hydrological validation.
