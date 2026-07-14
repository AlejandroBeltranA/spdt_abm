# Paper 1 Methodology Pipeline — Runbook

End-to-end CLI instructions for the Applied Energy methodology paper.

This is a **living document**. Each command is marked:

- ✅ **Ready** — script exists, tested syntactically
- 🚧 **Planned** — listed for reference; script not yet written
- ⏳ **Depends on** — needs an upstream step's output before it runs

For the design rationale behind each step, see [`PROGRESS.md`](./PROGRESS.md).
For the master task list, see the in-session TaskList.

---

## 0. Prerequisites

### 0.1 Environment

```bash
# Repo root
cd /Users/abeltran/Documents/GitHub/spdt_abm

# Python 3.11+ required (per pyproject.toml)
python3 --version

# If you don't already have a working env, create one:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below assume you are in the repo root and `.venv` is activated.

### 0.2 Data files

Required, all in `data/`:

| File | Used by | Notes |
|---|---|---|
| `serl_8963_targets/daily_targets.csv` | calibration | SERL national panel, 2020–2023 |
| `serl_8963_targets/diurnal_targets_hourly_mean.csv` | calibration | hourly profiles |
| `epc_abm_{city}.geojson` | per-city runs | newcastle, sunderland, waltham_forest |
| `hidp_uprn_matches_tiered.csv` | per-city runs | national HIDP synthpop |
| `wf_hidp_uprn_matches_tiered.csv` | waltham_forest | city-specific HIDP |
| `ncc_2t_timeseries_2010_2026.parquet` | newcastle, sunderland | ERA5 climate forcing |
| `waltham_forest_2t_timeseries_2010_2026.parquet` | waltham_forest | ERA5 climate forcing |
| `LSOA_domestic_elec_2010-2024.xlsx` | validation | DESNZ ground truth |
| `LSOA_domestic_gas_2010-2024.xlsx` | validation | DESNZ ground truth |

Verify with:

```bash
ls data/serl_8963_targets/daily_targets.csv \
   data/epc_abm_newcastle.geojson \
   data/hidp_uprn_matches_tiered.csv \
   data/ncc_2t_timeseries_2010_2026.parquet \
   data/LSOA_domestic_elec_2010-2024.xlsx \
   data/LSOA_domestic_gas_2010-2024.xlsx
```

### 0.3 Checking progress on long runs

**In-terminal (no second shell needed):** every script prints a live progress
line that updates in place every ~5 seconds, e.g.:

```
validate_newcastle_2023_sanity: 1234/8760  (14.1%) elapsed=00:05:23 eta=00:32:47 h
```

Milestone events (start/section boundaries/finish) print on their own line.
You see continuous progress in the same terminal you launched the job from.

**Persistent log files** still get written under the job's output dir:

- `_progress.log` — human-readable, append-only, one line per event
- `_progress.json` — machine-readable snapshot (count, pct, ETA, last message)

If you want to monitor from a second terminal anyway (e.g. you backgrounded
the job with `nohup … &`), use:

```bash
tail -f research/applied/results/<kind>/<city>_<year>/_progress.log

# One-shot status check (pretty):
cat research/applied/results/<kind>/<city>_<year>/_progress.json | python -m json.tool
```

Note: when stderr is not a TTY (e.g. `nohup`, redirection), the in-place
progress line is suppressed — only the persistent log file is written. This
avoids `\r` characters polluting `nohup.out`.

### 0.4 Crash recovery / resume

The model loop in `validate_run.py` (the only single-shot long job, ~30–60 min
on full Newcastle stock) pickles the model state every 720 hours of simulated
time (≈ ~30 days, ~3–5 min wall, depending on machine). If the machine reboots
or the process is killed mid-run, simply **re-run the same command** — the
script detects `_model_checkpoint.pkl` in the output directory, verifies its
signature matches the current invocation (same params, year, window, stock),
and resumes the loop from the saved hour.

```bash
# First run: starts fresh from hour 0
python research/applied/scripts/validate_run.py --city newcastle --year 2023 \
    --params .../params.yaml --kind sanity

# … crash / kill / reboot at hour ~5000 …

# Re-run the same command — resumes from the last checkpoint (≈ hour 4320 or 5040
# depending on when the last save happened):
python research/applied/scripts/validate_run.py --city newcastle --year 2023 \
    --params .../params.yaml --kind sanity
```

**Signature validation:** the checkpoint is keyed by a hash of `(params, year,
window_hours, climate_path, n_dwellings)`. Changing any of those silently
discards the checkpoint and starts fresh — no silent stale-data hazard.

**On clean exit** the checkpoint file is automatically deleted, so the next
fresh run starts clean.

**Flags:**

- `--checkpoint-every-hours 720` — default cadence; lower for safer (more
  disk), raise for faster (less disk). `--checkpoint-every-hours 0` disables.
- `--no-resume` — force a fresh start even if a checkpoint exists.

**Checkpoint size:** depends on stock size — typically 50–150 MB for full
Newcastle (97k agents). One file overwritten in place (atomic via tmp+rename),
so disk usage stays bounded.

**If pickling fails** (some Mesa internals don't pickle on every version),
the script logs a warning and runs without checkpointing rather than crashing
— so you never lose the ability to run.

**Multi-iteration jobs** (bootstrap, Morris, MC) will use a cheaper pattern:
each iteration writes its result to the output parquet immediately, and on
restart the script skips iterations already present. Bootstrap is fast enough
(~5 min for N=200) that re-running it is cheap, so it currently does not have
iteration-level resume. Morris and MC will get this when they're built (#10,
#11) — those are the overnight runs where it matters.

### 0.4 Output location

All script outputs land under `research/applied/results/`. Folder layout:

```
research/applied/results/
├── calibration/<years_label>/        ← step 1
├── bootstrap/<years_label>/          ← step 2
├── transfer/<city>_<year>/           ← step 4
├── temporal/<city>_<year>/           ← step 5
├── morris/                           ← step 6
├── mc/<city>_<year>/                 ← step 7
└── baseline/                         ← step 8
```

Each contains `metrics.json` (scalar metrics) + `*.parquet` (per-LSOA tables) +
diagnostic PNGs.

---

## 0.5 ✅ Current calibration (v7) and how it ships

The published calibration is **v7_cohort** (calibration year 2023), promoted
2026-06-24 as the canonical shipped config. It builds on the v5_phase5b base fits
— analytic anchors, gas HDD slope, building-type multipliers, and the seven
`fit_*.py` fits (setpoint, area, SAP, age, electricity baseline, presence, baseline
electricity diurnal profile) — then **replaces the electricity-anchor recentring with
per-cohort baseline anchors** (`fit_cohort_params.py`, no correction multiplier) and
adds mean-1.0 **monthly profiles** (`fit_monthly_profile.py`). The full v7 chain is
reproduced end to end by `research/applied/notebooks/1_calibration.ipynb`. The base
fits come from one command:

```bash
python research/applied/scripts/calibrate_serl.py --years 2023 --label v5_phase5b
# → results/calibration_v5_phase5b/calibrated_config.yaml
```

**Promote it into the engine** so every run picks it up by default (this is the
only artifact the engine needs; the engine layer is `household_energy/`). The
promote script ships the `model:` block only, dropping the calibration's `meta`
provenance and the deprecated `heating_setpoint_C` alias:

```bash
python research/applied/scripts/promote_config.py
# or for a different calibration:
#   python research/applied/scripts/promote_config.py \
#       --source results/calibration_<label>/calibrated_config.yaml --label <label>
```

After that, the ABM runs accurately on just the data — `energy-run`,
`energy-run-lsoa`, and `transfer.py` all default to
`household_energy/calibrated_config.yaml`. No `--config` needed (pass
`--config defaults` to run uncalibrated). The validation below (`transfer.py`)
also defaults to the shipped config, so steps 3–4 need no params path.

**Diurnal shape (added post-v5).** `fit_diurnal_profile.py` emits a mean-1.0
24-hour `base_profile_24h_electric` from the SERL population-aggregate hourly
electricity curve (`seg3_var='none'`). `model._reset_base_loads` multiplies it
onto the per-hour baseline, restoring the SERL diurnal shape (≈2.5× trough→peak
swing) that the flat baseline anchor flattened to ≈1.2×. Because the profile is
normalised to mean 1.0 it is **mean-preserving**: annual totals and every other
fitted parameter are unchanged (verified: ΔkWh/yr ≈ 0.000%). Gas baseline is
left flat — total-gas diurnal is space-heating-dominated and belongs in a refit
of `heating_profile_24h` as a residual over the climate signal (not yet done).

**Schedule-retired SERL-profile architecture (experimental, flag-gated).**
`fit_serl_profiles.py` → `results/calibration_serl_fits/serl_profiles.yaml`
emits the full diurnal "profile pack" (electric baseline + per-occupant slope,
gas cooking baseline, and gas heating profile by outdoor-temperature band). With
`use_serl_profiles: true` in the config, the model stops simulating per-person
schedules and drives the hourly SHAPE straight from SERL (occupancy = static
count; gas heating interpolated by ambient temperature). This fixes the gas
diurnal (shape RMSE roughly halved across temperature bands and reproduces the
cold-flat / mild-peaky HDD interaction). Status: **shape validated, annual
preservation still open** (gas +~8% from the relaxed cap + non-HD-weighted
per-band normalisation; electric −~5% from the per-person deviation reference) —
so the flag stays OFF in the shipped config. The electric baseline diurnal
profile alone (mean-1.0, annual-exact) IS shipped and active. See
research/applied/scripts/fit_serl_profiles.py.

Sections 1–8 below document the broader/older multi-year harness; §0.5 above
is the canonical v7 path.

---

> ## ⚠️ Sections 1–8 are historical (pre-v5)
>
> They describe the earlier multi-year harness (`--years 2020 2021 2022`,
> `validate_run.py`, `research/applied/results/...`) and several steps still
> marked 🚧 planned. The published Paper 1 results use the **v7 path in §0.5**
> (`notebooks/1_calibration.ipynb` → `promote_config.py` → `transfer.py`; the
> v5_phase5b command provides only the base fits the v7 chain builds on).
> Read §1–8 for design rationale and the bootstrap/Morris/MC harness, not as the
> current run recipe. Where they conflict with §0.5, §0.5 wins.

---

## 1. ✅ Multi-year SERL calibration

Pool SERL 2020–2022 → baseline anchors + HDD slopes + building-type multipliers.
2023 is held out for the in-sample cross-check.

```bash
python research/applied/scripts/calibrate_serl.py \
    --years 2020 2021 2022 \
    --label 2020_2022
```

**Runtime:** seconds (it's OLS on a few hundred rows).

**Output:** `research/applied/results/calibration/2020_2022/`
- `params.yaml` — model-config-compatible parameter file (this is the input
  to every downstream run)
- `diagnostics.json` — R², n_points per fuel, monthly fit table
- `hdd_regression.png` — visual fit check

**What to look at:**

- `diagnostics.json["fit"]["gas"]["r_squared"]` should be > 0.9.
- `diagnostics.json["anchors"]` should give gas and elec in roughly 0.25–0.35 kWh/h.
- `diagnostics.json["hdd_slopes"]` should give gas ~0.20, elec lower.

**If 2020 COVID is contaminating the fit** (R² drops noticeably vs prior single-year runs):

```bash
python research/applied/scripts/calibrate_serl.py \
    --years 2021 2022 \
    --label 2021_2022
```

Document the decision in [`PROGRESS.md`](./PROGRESS.md) decisions log.

---

## 2. ✅ Bootstrap calibration (N=200 parameter draws)

Resample SERL respondents with replacement, refit anchors + slopes per draw,
save the full distribution.

```bash
python research/applied/scripts/bootstrap_serl.py \
    --years 2020 2021 2022 \
    --n-draws 200 \
    --seed 7 \
    --label 2020_2022
```

**Runtime:** ~1–5 minutes (200 OLS refits — calibration is fast).

**Output:** `research/applied/results/bootstrap/2020_2022/`
- `draws.parquet` — one row per draw, columns: anchors (gas/elec), slopes (gas/elec), R², n_points
- `building_type_multipliers.parquet` — one row per (draw, fuel, building_type)
- `summary.json` — mean, std, 95% percentile CI per parameter

This is the input distribution for the MC propagation (step 7).

**What to look at:**
- `summary.json` 95% CIs should bracket the point estimates from step 1
- Per-parameter `std` shows which parameters are well-constrained vs noisy
- If a non-trivial number of draws failed (skipped in log), the SERL filter is too tight; lower `--min-n`

---

## 3. ✅ Validation pipeline — sanity check

Run the calibrated parameters on Newcastle 2023 and compare to DESNZ. This
catches any plumbing bugs before launching the long jobs.

```bash
python research/applied/scripts/validate_run.py \
    --city newcastle \
    --year 2023 \
    --params research/applied/results/calibration/2020_2022/params.yaml
```

**Runtime:** ~30–60 min for full 97k Newcastle stock at 8760h.

**Output:** `research/applied/results/transfer/newcastle_2023/`
- `metrics.json` — MAPE (elec, gas, total), bias, n_lsoas
- `residuals.parquet` — per-LSOA ABM vs DESNZ table

**Expected ballpark:** MAPE_total < 20% if calibration is working. If much
higher, debug before proceeding.

---

## 4. ✅ Spatial transfer validation

Apply Newcastle-calibrated parameters to other LAs without recalibration.
This is the headline validation of the paper.

⏳ **Depends on:** step 1 (params.yaml)

### 4a. Sunderland (closest neighbour)

```bash
python research/applied/scripts/validate_run.py \
    --city sunderland \
    --year 2023 \
    --params research/applied/results/calibration/2020_2022/params.yaml
```

### 4b. Waltham Forest (London, different climate + stock)

```bash
python research/applied/scripts/validate_run.py \
    --city waltham_forest \
    --year 2023 \
    --params research/applied/results/calibration/2020_2022/params.yaml
```

### 4c. (Future) Manchester + Cornwall

Once stock data lands, add to `CITY_CONVENTIONS` in `scripts/utils.py` and run:

```bash
python research/applied/scripts/validate_run.py --city manchester --year 2023 ...
python research/applied/scripts/validate_run.py --city cornwall --year 2023 ...
```

**Runtime:** ~30–60 min per city.

**Output per city:** `research/applied/results/transfer/<city>_<year>/`
(same structure as step 3).

---

## 5. ✅ Temporal forward validation

Same machinery, future year. Bridge to the Nature paper.

⏳ **Depends on:** step 1

```bash
python research/applied/scripts/validate_run.py \
    --city newcastle \
    --year 2024 \
    --params research/applied/results/calibration/2020_2022/params.yaml
```

**Runtime:** ~30–60 min.

**Output:** `research/applied/results/temporal/newcastle_2024/`

DESNZ 2024 is the most recent year published; 2025 not yet released, skipped.

---

## 6. 🚧 Morris elementary effects sensitivity analysis

Variance-based ranking of which parameters drive LSOA-scale prediction error.
Runs on a stratified ~20k dwelling subsample for tractability.

⏳ **Depends on:** step 1, step 3 (sanity check passed)

```bash
python research/applied/scripts/morris_sa.py \
    --city newcastle \
    --year 2023 \
    --base-params research/applied/results/calibration/2020_2022/params.yaml \
    --n-trajectories 10 \
    --subsample-size 20000 \
    --subsample-seed 7
```

**Runtime:** ~25–50 wall hours on 8 workers. **Launch overnight.**

**Output:** `research/applied/results/morris/`
- `elementary_effects.parquet`
- `mu_star_sigma.png` — the ranking figure

---

## 7. 🚧 Monte Carlo uncertainty propagation

Headline numbers with confidence bands. Full 97k stock; combines bootstrap
calibration draws with literature priors for non-calibrated parameters.

⏳ **Depends on:** step 2 (bootstrap draws), step 6 (knows which params matter)

```bash
python research/applied/scripts/mc_propagate.py \
    --city newcastle \
    --year 2023 \
    --bootstrap research/applied/results/bootstrap/2020_2022/draws.parquet \
    --priors docs/research/morris_param_table.md \
    --n-draws 200
```

**Runtime:** ~24–48 hours per (city, year). Launch on a free machine and walk away.

**Output:** `research/applied/results/mc/newcastle_2023/`
- `draws.parquet` — per-draw citywide totals + LSOA aggregates
- `summary.json` — mean ± 95% CI on citywide demand

Repeat per (city, year) combination needed for the paper.

---

## 8. ✅ Naive regression baseline

What the ABM beats. `kWh ~ floor_area + EPC_band + HDD + tenure` on Newcastle
DESNZ, applied to Sunderland and Waltham Forest.

⏳ **Depends on:** step 4 outputs (so we can compare)

```bash
python research/applied/scripts/regression_baseline.py \
    --train-city newcastle \
    --train-year 2023 \
    --test-cities sunderland waltham_forest \
    --test-year 2023
```

**Runtime:** seconds.

**Output:** `research/applied/results/baseline/`
- `regression_coefs.json`
- `test_mape.csv` — regression vs ABM MAPE side by side

---

## 9. Analysis notebook → paper figures

Once all runs above complete, regenerate the paper figures from cached results:

```bash
jupyter notebook research/applied/notebooks/methodology_paper_figures.ipynb
```

The notebook does **no model runs** — it loads parquet from `results/` and produces
the five paper figures. If you change a parameter, rerun the relevant CLI step,
then restart the notebook kernel.

---

## Typical end-to-end run order

For a clean run from scratch on a freshly-cloned repo:

```bash
# Day 1 — fast
python research/applied/scripts/calibrate_serl.py --years 2020 2021 2022  # step 1
python research/applied/scripts/bootstrap_serl.py --years 2020 2021 2022 --n-draws 200  # step 2
python research/applied/scripts/validate_run.py --city newcastle --year 2023 \
       --params research/applied/results/calibration/2020_2022/params.yaml  # step 3

# Day 1 evening — kick off overnight runs in parallel
python research/applied/scripts/validate_run.py --city sunderland     --year 2023 ... &  # step 4a
python research/applied/scripts/validate_run.py --city waltham_forest --year 2023 ... &  # step 4b
python research/applied/scripts/validate_run.py --city newcastle      --year 2024 ... &  # step 5
python research/applied/scripts/morris_sa.py --city newcastle --year 2023 ...           # step 6
wait

# Day 2 — kick off MC, regression baseline runs in seconds
python research/applied/scripts/mc_propagate.py --city newcastle --year 2023 ...        # step 7
python research/applied/scripts/regression_baseline.py ...                              # step 8

# Day 3+ — analysis notebook
jupyter notebook research/applied/notebooks/methodology_paper_figures.ipynb
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `DeprecationWarning: heating_setpoint_C` | Old config still in use | Update config key to `heating_trigger_temp_C`; old one still works but emits warning |
| `KeyError: 'Unknown city'` | City not in `CITY_CONVENTIONS` | Add an entry to `scripts/utils.py` |
| MAPE >> 50% on validation | Likely calibration didn't load, or wrong year filter | Check `params.yaml` actually loaded; check climate parquet covers target year |
| `FileNotFoundError: epc_abm_<city>.geojson` | Stock data missing | Acquire stock, place in `data/` |
| Memory blowup on 97k full-stock runs | LSOA-level chunking not engaged | Reduce `--n-workers` or `--max-lsoas-per-batch` |
| Morris run never finishes | Forgot `--subsample-size` | Re-run with subsample; full stock Morris is 100+ hours |
| "Is it still running?" anxiety | Lost the launching terminal | `tail -f <job>/_progress.log`, or `cat <job>/_progress.json` |

---

## Where to log issues / decisions

- **Code issues, plumbing bugs** → fix on `main`, note in PROGRESS.md decisions log.
- **Calibration anomalies** (e.g. 2020 COVID effect) → document in PROGRESS.md
  with the decision (drop / keep / weight).
- **Reviewer-facing methods text** → `docs/paper1_methodology/`.
