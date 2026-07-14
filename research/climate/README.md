# research/climate — climate-change exposure paper

Paper 3 (planned): residential energy demand and vulnerability under climate
change — running the calibrated model against future climate scenarios.

## Notebooks (`notebooks/`)

| notebook | does |
|---|---|
| `01-climate-prep.ipynb` | prepares the climate inputs (ERA5 / future-scenario timeseries) |
| `paper3_climate_exposure.ipynb` | climate-exposure analysis (existing draft) |

## Status — starting point only
These are the **pre-v7 notebooks**, relocated here unchanged. They have **not**
been updated to the current (v7) calibration, nor re-run. Before using them:

- point them at the v7 config (`results/calibration_v7_cohort/calibrated_config.yaml`)
  and the de-fudged per-cohort anchors / monthly profiles;
- confirm the climate timeseries (note the 2026 file is the validation default; the
  2039 file is the future scenario — don't mix them).

This is the climate paper's home; the demand engine and calibration it should build
on live in `research/applied/`.
