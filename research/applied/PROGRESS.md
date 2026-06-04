# Paper 1 — Applied Energy Methodology Paper

**Working title:** *A spatially explicit agent-based model of household energy
demand: calibration, transfer, and uncertainty quantification at LSOA scale.*

**Target journal:** Applied Energy

**Status:** Plumbing phase — started 2026-05-21

---

## Paper framing (decided 2026-05-21)

Reframed from the original policy targeting paper into a **pure methodology
contribution**. The 7.58 GWh/yr social-rent vs top-user targeting finding
moves to Paper 2 (PSJ). This paper makes one claim:

> A national-panel-calibrated household energy ABM predicts LSOA-scale
> annual demand within X% across UK local authorities of varying climatic
> and socioeconomic character, with rigorously propagated parameter
> uncertainty and forward temporal validation.

The Newcastle stakeholder focus becomes an *application* of the model
discussed in the conclusion, not the calibration target.

---

## Validation hierarchy

| Tier | Test | Year(s) | Status |
|---|---|---|---|
| 1 | In-sample fit (Newcastle vs DESNZ) | 2020–2022 | existing, to be re-run |
| 2 | Held-out SERL year (Newcastle DESNZ) | 2023 | new |
| 3 | Spatial transfer (Sunderland, Waltham Forest) | 2023 | new |
| 4 | Forward temporal validation | 2024 | new |
| 5 | Multi-city extension (Manchester, Cornwall) | 2023 | pending data acquisition |

---

## Methodology stack

| Element | Approach | Compute |
|---|---|---|
| Calibration | SERL 2020–2022, OLS on baseline anchors + HDD slopes | minutes |
| Calibration uncertainty | Bootstrap N=200 row resamples | ~1–2 hours |
| Parameter SA | Morris elementary effects, 12 params, ~20k dwelling subsample | overnight |
| Headline UQ | MC propagation (bootstrap × literature priors), full 97k stock | overnight per city |
| Naive comparator | Regression baseline (kWh ~ floor + EPC + HDD + tenure) | minutes |

---

## Decisions log

- **2026-05-21 — Reframe.** Paper 1 is now methodology only. Policy targeting
  comparison moves to Paper 2 (PSJ).
- **2026-05-21 — Parameter rename.** `heating_setpoint_C` →
  `heating_trigger_temp_C`. Verified at `agent.py:712`: parameter compares
  against ambient (outdoor) temperature, not indoor comfort. The 13°C
  calibrated value is the outdoor temp at which heating engages, not a
  comfort thermostat reading. Old name was misleading.
- **2026-05-21 — Calibration window.** SERL covers 2020–2023. Calibrate on
  2020–2022, hold out 2023 as SERL+DESNZ cross-check year. 2020 is
  COVID-anomalous; drop if residuals look bad after first run.
- **2026-05-21 — Temporal validation.** DESNZ available through 2024.
  2025 backcast skipped (no ground truth). Forward validation runs to 2024.
- **2026-05-21 — Morris vs MC scope.** Morris on ~20k stratified subsample
  (rank-only, defensible). MC propagation on full 97k stock.
- **2026-05-21 — Bootstrap N=200.** Above AE comparator papers (Xu &
  Qadrdan use N=30; Sridhar uses N=1000 but with much smaller model).
- **2026-05-21 — Calibration framing.** SERL is a national panel.
  Calibration is therefore *national*, not Newcastle-specific. Multi-city
  validation is the natural generalisability test, not a defensive add-on.
- **2026-05-22 — Silent calibration bypass in batch runner.**
  `household_energy/run_lsoa_batch.RunConfig` had `config_path` removed in
  a refactor without updating callers. The batch runner `_run_single_lsoa`
  stopped threading `config_path=` into `EnergyModel(...)`, so every
  previous "calibrated" notebook batch run silently fell back to
  `config_defaults.yaml`. Restored `config_path` (and `save_model_timeseries`
  as a no-op for API back-compat) and threaded through to `EnergyModel`.
  Implication: prior results under `notebooks/results_lsoa/` are NOT
  calibration runs; they are default-config runs. Re-running the batch
  now produces actually-calibrated outputs for the first time.
- **2026-05-22 — Multi-year `params.yaml` is incomplete as model input.**
  The fresh notebook-side calibration (`results/calibration_20260522_092130/
  calibrated_config.yaml`) writes two fields my multi-year pipeline does
  not: `heat_slope_max: 5.0` (uncaps the slope; default 0.10 would silently
  cap the calibrated 0.21+ value) and `property_type_mult_base_gas/electric`
  (building-type multipliers in dict-of-dwelling-type form). Until task #17
  lands, my multi-year params.yaml is **not** a drop-in replacement —
  using it would under-predict heating. Validation batch should keep using
  the notebook's calibrated_config.yaml for now.
- **2026-06-01 — Calibration v2 delivered (#20 done, Option A).**
  Added SAP-band and building-age heating multipliers to the model:
    Gas SAP multipliers (D=1.0): A&B 0.69, C 0.76, E 1.20, F&G 1.27
    Gas age multipliers (overall=1.0): pre-1900 1.18, ..., 2003+ 0.74
  Built `research/applied/scripts/calibrate_serl_v2.py` + `model.py`/`agent.py`
  edits. Property-type heating multiplier computed for diagnostic
  completeness but NOT wired into the model — v1 already differentiates
  heating by floor area via `heat_slope_area_exp=0.6`, so applying a fresh
  SERL-derived pt multiplier on top would double-count. Defer to a v3
  refit of `heat_slope_area_exp` jointly with property-type if validation
  shows it's needed.
  Backwards compat: missing keys default to 1.0; older configs unchanged.
- **2026-06-01 — Calibration diagnostic findings (#18 done).**
  Direct decomposition of where the per-dwelling over-prediction comes from:
  1. Gas heating slope is ~22% over-calibrated. ABM gas/elec ratio = 4.99
     vs DESNZ 4.11. The 33pp gas-over-elec gap traces to the slope, not
     the baseload.
  2. Baseline gas anchor (0.289 kWh/h = 2530 kWh/yr/dwelling) is fine —
     ABM baseline share 13% vs implied DESNZ 21%.
  3. Property_type multipliers are TOO aggressive for gas. ABM exaggerates
     the detached/flat spread (r=+0.54 / -0.70 vs DESNZ +0.13 / -0.11).
  4. Model has WRONG SIGN on dwelling age effect (ABM +0.13, DESNZ -0.22)
     and weak SAP-band signal (ABM -0.17, DESNZ -0.37). Missing the
     insulation signal DESNZ shows.
  Task #20 added: recalibrate gas slope, soften property mults, add
  SAP/age coefficients. Expected to take gas MAPE 49% → ~25-30% per-house.
- **2026-06-01 — Paper claim refocused: per-house, not per-LSOA.**
  After running the 185-LSOA batch and seeing 36% per-dwelling MAPE
  reframed as "publishable with limitations," Alex called the framing
  question explicitly: the methodology paper claims per-house demand
  prediction; DESNZ is supporting evidence, not the primary KPI.
  Critical consequence: the 37% per-dwelling MAPE is the DESNZ-meter
  vs EPC-dwelling denominator artifact (97k / 134k ≈ 0.72; 1/0.72 = 1.39
  ≈ observed 1.37 median ratio). Under per-house claim:
    Layer 1: citywide totals (ABM 1.99 TWh vs DESNZ 1.97 TWh, +2% bias) — PRIMARY aggregate evidence
    Layer 2: per-LSOA totals shape correlation (r=0.79 total) — supporting
    Layer 3: per-meter rate MAPE (37%) — denominator-mismatch sensitivity, NOT a per-house error
  Open questions: heating-vs-baseload over-prediction (gas 49% vs elec 16%
  suggests heating slope may be over-calibrated), Newcastle dwellings vs
  SERL national panel representativeness, and whether SERL respondent-level
  data is accessible for direct per-house validation. Tasks #18 + #19
  added to pursue these.
- **2026-06-01 — Multi-year `params.yaml` is now a drop-in (#17 done).**
  Updated `calibrate_serl.py` to emit the full schema the model reads:
  `heat_slope_max=5.0`, `heating_months=[10,11,12,1,2,3,4]`,
  `heating_trigger_temp_C=13.0` (+ legacy `heating_setpoint_C` alias),
  and the property-type multiplier dicts under the correct keys
  (`property_type_mult_base_gas/electric` — not `building_type_multipliers_*`
  as before). Re-ran calibration; schema now matches the notebook reference
  with no missing keys. Multi-year vs single-year value differences are
  data-driven (multi-year captures pre-crisis demand): gas anchor 0.368
  vs 0.289 (+27%), gas slope 0.238 vs 0.210 (+13%), detached-house gas
  multiplier 1.272 vs 1.018 (+25%, COVID-era WFH effect). Both calibrations
  are now valid drop-ins; the choice between them is methodological.
- **2026-05-22 — `_run_single_lsoa` output schema was reduced to 5 columns.**
  The function had been gutted to emit only `[year, abm_kwh, lsoa_code,
  run_dwellings, abm_kwh_per_dw]`, but the validation notebook cell 18
  requires the full 22-column schema (per-fuel kWh + gas-connection
  counters + per-dwelling rate columns). Restored the full output;
  dwelling-counter columns now computed from EPC stock flags
  (`is_gas`, `is_off_gas`, `main_fuel_type`). Combined with the prior bug,
  this means the validation notebook has not run end-to-end against
  calibrated parameters with the right schema since the refactor.
- **2026-05-21 — Bootstrap result + scope note.** N=200 row-level
  bootstrap on SERL 2020–2022 aggregate targets (53,118 base rows).
  Result 95% percentile CIs:
    gas anchor [0.338, 0.397], elec anchor [0.324, 0.347],
    gas slope  [0.231, 0.245], elec slope  [0.061, 0.073].
  CIs are narrow (±3–9%) because the base set is large — the law of
  large numbers compresses sampling variance. The methods section
  should disclose that this captures **aggregation-level** sampling
  variance, not respondent-level uncertainty (we don't have SERL
  microdata in scope). Implication for MC propagation (#11): the
  calibrated parameters will contribute a small share of the total
  output-variance band; most of the spread will come from literature
  priors on uncalibrated parameters (occupancy, EPC band, age, etc.).
  This is the honest story — and likely the right one for AE.
- **2026-05-21 — Model checkpoint/resume.** `run_model()` pickles the
  EnergyModel every 720 simulated hours to `<output_dir>/_model_checkpoint.pkl`
  with an `.json` signature sidecar (hash of params + year + window + climate +
  n_dwellings). A re-run with the same args picks up where it left off; any
  signature change invalidates the checkpoint. Clean exit deletes the file.
  Bootstrap/Morris/MC will use iteration-level resume (skip rows already in
  output parquet) when they're built.
- **2026-05-21 — Keep 2020 SERL.** Multi-year calibration on 2020–2022 produced
  gas R² = 0.969 and electric R² = 0.945. The COVID-era 2020 data did not
  degrade the fit, so the calibration window stays at [2020, 2021, 2022].
  Headline parameters (multi-year vs single-year 2023):
    gas anchor 0.368 vs 0.289 kWh/h, gas slope 0.238 vs 0.210 kWh/h/°C,
    elec anchor 0.335 vs 0.290 kWh/h, elec slope 0.067 (new).
  Higher multi-year values reflect COVID-era home-occupancy uplift and
  pre-energy-crisis demand levels — closer to the long-run mean than the
  post-crisis 2023 single year.
- **2026-06-04 — Electricity overshoot root cause + v4 fix.**
  The Paper 1 "open mystery" (model produced ~20% more per-dwelling
  electricity than SERL/DESNZ, concentrated in all-electric LSOAs) was NOT
  broken heating physics. Root cause: the model had a SINGLE shared base
  heating slope `heating_slope_kWh_per_deg` set to the gas fit (0.20967),
  and electric-heated homes inherited it. SERL fits the electric-heated
  slope at **0.05606 — 3.7× lower**. Duty saturation (`duty=loss/(loss+K)`,
  K=3) compressed the 3.7× slope error into the observed ~1.7× energy
  overshoot. **Fix:** added a fuel-specific `heating_slope_kWh_per_deg_electric`
  param (`model.py:130`, used via `agent.py _model_base_heat_slope` at
  `agent.py:356` + `model.py:481`); electric SAP/age multipliers set to the
  gas shapes (envelope heat loss is fuel-agnostic). The legacy
  `systems.electric_heating.heating_slope_mult` is neutralized to 1.0 to
  avoid double-dampening (0.056 ≈ 0.21 × 0.267). Config:
  `results/calibration_v4_elecslope_20260604_094157/calibrated_config.yaml`.
  Full 185-LSOA 2023 re-run (`results_lsoa/abm_year_all_v4_2023.csv`):
  city dwelling-wt elec 10.39 → **8.85** kWh/day; all-electric tail gone
  (E01033543 1.69× → 0.90× DESNZ); gas untouched. Wired into
  `serl_calibration_clean.ipynb` (emits the new param + neutralized mult)
  and `energy-model-validation.ipynb` (auto-picks newest calibrated config;
  loader now accepts the v4 csv).
- **2026-06-04 — Narrative flip: model now UNDERSHOOTS DESNZ (by design).**
  v3's totals "matched" DESNZ (0.97×) only because two errors cancelled —
  the electric over-ramp inflated per-home energy while the EPC-dwelling
  (97,714) vs DESNZ-meter (132,253 = 1.353× gap) denominator deflated it.
  With the level fixed, v4 city electricity total is **315.7 GWh vs DESNZ
  ~381 = 0.829×**: on a like-for-like EPC-dwelling denominator the model
  runs ~17% UNDER DESNZ. The residual per-meter 1.122× = 0.829 (energy) ×
  1.353 (denominator). This is the intended direction — the model should
  not conjure energy for dwellings with no EPC stock record. HTMLs
  `serl_model_overlay_tempband.html` and `serl_desnz_blend_reconciliation.html`
  refreshed with v4 data and corrected narrative (warm +17% person-spike
  baseload persists; cold over-ramp removed, now under the SERL blend).
- **2026-06-04 — DESNZ is a benchmark, not a calibration target; built a
  coverage-aware confidence layer.** Decision: do NOT calibrate/correct the
  model to match DESNZ LSOA electricity — it is an independent benchmark
  with known coverage contamination (empties, second meters, non-EPC stock
  the SERL params don't describe). SERL stays the calibration anchor (clean
  per-household panel, carries the fuel split). Instead we quantify *where*
  the model is trustworthy. Per-LSOA `tot_ratio = model_total / DESNZ_total`
  regressed on EPC-to-meter coverage gives **R²=0.75** (0.80 adding electric
  share); fit ≈ 0.13 + 0.96·coverage (slope ≈ 1 ⇒ the LSOA-total gap is
  almost entirely "we model fewer units than there are meters", not a
  per-dwelling energy error). **Lesson:** compare totals per LSOA — defining
  the ratio as model-per-EPC-dwelling ÷ DESNZ-per-meter makes coverage
  cancel and hides the effect (R²=0.02–0.26).
  Confidence tier (score 0–8 from 3 computable covariates): coverage
  (weighted 2×), electric-heated share (SERL sample thinness), meter count
  (small-LSOA noise). High ≥6 / Medium 4–5 / Low ≤3 → **138 High / 43 Medium
  / 4 Low**. The flag cleanly separates trustworthy from untrustworthy LSOAs
  (High: corr 0.88; Medium 0.32; Low −0.45). This is the deployable Paper 1
  uncertainty contribution: a per-LSOA reliability map, not a citywide
  pass/fail. Artifacts: `research/applied/lsoa_confidence_2023.html`
  (self-contained — coverage scatter + centroid confidence map + tier
  tables) and `lsoa_confidence_2023.csv`; the validation notebook reproduces
  the analysis and exports `lsoa_confidence_2023_notebook.csv`.

---

## Folder layout

```
research/applied/
├── PROGRESS.md              ← this file
├── scripts/                 ← CLI-runnable long jobs
│   ├── utils.py             ← validate(city, year, params) shared
│   ├── calibrate_serl.py    ← multi-year SERL calibration
│   ├── bootstrap_serl.py    ← N=200 calibration draws
│   ├── morris_sa.py         ← Morris elementary effects
│   ├── mc_propagate.py      ← MC uncertainty propagation
│   └── validate_run.py      ← per-(city, year) validation
├── notebooks/               ← analysis & figure generation only
│   └── methodology_paper_figures.ipynb
└── results/                 ← cached outputs from script runs
    ├── calibration/
    ├── bootstrap/
    ├── transfer/{city}_{year}/
    ├── temporal/{city}_{year}/
    ├── morris/
    └── mc/{city}_{year}/
```

**Rule:** scripts produce data, notebooks visualize. No model logic in notebooks.

---

## Task status

Tracked in session TaskList (#1–#15). Summary regenerated below as work
progresses.

| # | Task | Status | Output location |
|---|---|---|---|
| 1 | Set up scripts/ + utilities | **done** | `scripts/utils.py` |
| 2 | Rename `heating_setpoint_C` → `heating_trigger_temp_C` | **done** | model.py, config_defaults.yaml (shim warns on old key) |
| 3 | Multi-year SERL calibration script | **done** (script built; not yet executed) | `scripts/calibrate_serl.py` → `results/calibration/<years_label>/` |
| 4 | Bootstrap SERL N=200 | **done** (script built; not yet executed) | `scripts/bootstrap_serl.py` → `results/bootstrap/<years_label>/` |
| 5 | Revise Morris param table | **done** | `research/applied/docs/morris_param_table.md` (13 params) |
| 6 | `validate(city, year, params)` pipeline | **done** | `scripts/utils.py` (run_model + validate) + `scripts/validate_run.py` (CLI) |
| 17 | `calibrate_serl.py` produces a model drop-in | **done** | Re-run produced full-schema `params.yaml` matching notebook reference |
| 7 | Sunderland transfer 2023 | pending | `results/transfer/sunderland_2023/` |
| 8 | Waltham Forest transfer 2023 | pending | `results/transfer/waltham_forest_2023/` |
| 9 | Temporal validation 2024 (Newcastle) | pending | `results/temporal/newcastle_2024/` |
| 10 | Morris SA (12 params, subsample) | pending | `results/morris/` |
| 11 | MC propagation (headline + bands) | pending | `results/mc/` |
| 12 | Figures notebook | pending | `notebooks/methodology_paper_figures.ipynb` |
| 13 | Regression baseline comparator | **done** | `scripts/regression_baseline.py` → `results/baseline/<train>_<year>/` |
| 14 | Appendix + methods write-up | pending | `docs/paper1_methodology/` |
| 15 | Manchester + Cornwall transfers | pending | `results/transfer/{city}_2023/` |

---

## Open questions

- **2020 SERL drop decision** — defer until first multi-year calibration
  finishes. Check residual pattern by year; if 2020 is >2σ off 2021–2022 means,
  drop it and rerun with 2021–2022.
- **Manchester + Cornwall data acquisition timeline** — Alex to fetch EPC +
  synthpop. Pipeline is ready once data is in place.
- **2025 projection** — skipped for AE (no ground truth). Could be included as
  a forward demand estimate (not validation) to demo the Nature paper's
  machinery. Defer decision.

---

## Outputs index

(populated as runs complete)
