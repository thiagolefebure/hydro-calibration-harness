# hydro-calibration-harness

Prototype for reproducible rainfall-runoff calibration engineering around a documented, replaceable GR4J model.

The repository demonstrates:

- scripted data acquisition and QA;
- deterministic Latin Hypercube parameter exploration;
- strict calibration / validation separation;
- a GLUE-inspired behavioral ensemble for parametric uncertainty diagnostics;
- an automatically generated auditable calibration report.

It is a prototype, not an operational hydrological forecasting system.

## 4-minute demo

Suggested presentation order:

1. Problem and architecture: show `config/basin.yaml`, the frozen pipeline stages, and the reproducible outputs in `output/`.
2. Uncalibrated vs automatically calibrated model: show `output/demo_01_calibration_impact.png`.
3. Calibration vs independent validation: show `output/demo_02_validation.png`.
4. Behavioral ensemble and parametric uncertainty: show `output/demo_03_uncertainty.png`.
5. Automatically generated auditable report: show `output/rapport_calage.md`.

Regenerate the demo with one command:

```bash
python run.py --all
```

Or step by step:

```bash
python run.py --data-only
python run.py --metrics-demo
python run.py --hydro-summary
python run.py --calibrate
python run.py --ensemble
python run.py --report
python run.py --demo
```
