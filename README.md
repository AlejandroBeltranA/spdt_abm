# Household Energy ABM (Mesa + mesa-geo)
Simulates hour-by-hour residential energy demand for every dwelling in a GeoJSON, with validation against DESNZ subnational electricity & gas (2020–2023) and built-in policy levers (heat-pumps, retrofits, socio-demographic targeting).

---

## What’s new (Jan 2026)
- **Meter-anchored baseline:** fixed 0.40 kWh/h with mild area/property-type scaling; structural multipliers only affect heating. Baseline capped at 1.5 kWh/h.
- **Duty-cycle heating + caps:** sublinear area scaling, bounded property-type multipliers, 20 kWh/h total cap with per-component clip diagnostics.
- **Scenario testing v2:** ready-to-wire policy selectors (income/education, top users, kids vs elderly, social rent) documented in `docs/scenario_testing_v2_guide.md`.
- **DESNZ calibration notebook:** `notebooks/energy-model-calibration.ipynb` compares ABM vs DESNZ per LSOA and exports batch runs.

---

## Directory map
```
household_energy/            # Core package
├── agent.py                 # Household/Person agents (baseline/heating logic)
├── model.py                 # EnergyModel (hourly step, caps, collectors)
├── run.py                   # CLI headless runner
├── analyze.py               # Post-run plots/maps
└── climate.py               # ClimateField helpers

notebooks/
├── energy-model-calibration.ipynb   # DESNZ alignment, LSOA batch runs
├── household_energy_abm_tutorial.ipynb
├── scenario_testing_v2.ipynb        # Policy scenarios (mirrors docs guide)
└── enrichment_check.py              # HIDP/EPC enrichment sanity

docs/
└── scenario_testing_v2_guide.md     # Dashboard-ready scenario guide

scripts/
└── run_lsoa_batch.py                # Batch LSOA runner (per-dwelling outputs)
```

---

## Setup
```bash
python -m venv esa_mesa
source esa_mesa/bin/activate
pip install -r requirements.txt
# dev mode for CLI imports
pip install -e .
```

Inputs (git-ignored):
- Household GeoJSON with UPRN + attributes (`data/abm_households_*.geojson`)
- Hourly climate parquet (`data/hourly_climate.parquet`)
- Optional HIDP enrichment CSV (`data/hidp_uprn_matches_tiered.csv`)
- DESNZ Excel workbooks for validation (electricity + gas)

---

## Running the model (CLI)
7-day smoke:
```bash
energy-run data/abm_households.geojson \
  --climate data/hourly_climate.parquet \
  --days 7 \
  --outdir results_smoke \
  --no-agent-level
```

Windowed (validation, 2020–2024):
```bash
energy-run data/abm_households.geojson \
  --climate data/hourly_climate.parquet \
  --start-utc 2020-01-01T00:00:00Z \
  --end-utc   2025-01-01T00:00:00Z \
  --outdir results_window
```
Flags of interest:
- `--hidp-csv` to merge socio-demographics
- `--agent-level/--no-agent-level` for per-dwelling collection
- `--print-every-hours` progress cadence

Outputs:
- `model_timeseries.parquet` (UTC hourly, model-level)
- `agent_timeseries.parquet` (if enabled; includes socio fields)
- `model_daily.parquet`, `model_hourly.parquet`
- `energy_timeseries.csv` (compat)
- `energy_model.pkl` (unless `--no-pickle`)

### LSOA batch (per-dwelling annuals, DESNZ-ready)
```bash
energy-run-lsoa \
  --geojson data/abm_households.geojson \
  --climate data/hourly_climate.parquet \
  --start-utc 2020-01-01T00:00:00Z \
  --end-utc   2025-01-01T00:00:00Z \
  --outdir results_lsoa
```
Outputs live under `results_lsoa/<LSOA>/run_<stamp>/` with `abm_year_<LSOA>_<stamp>.parquet/.csv` plus hourly model timeseries; a combined rollup is saved as `abm_year_all_<stamp>.*`.

### Visual teaser
![ABM animation](notebooks/abm_animation_synthpop.gif)

---

## Key notebooks
- **Calibration:** `notebooks/energy-model-calibration.ipynb` — loads LSOA batch outputs, computes ABM vs DESNZ per-dwelling kWh, plots ratios, and can rerun batches via `run_lsoa_batch.py`.
- **Policy scenarios:** `notebooks/scenario_testing_v2.ipynb` — smoke/perf harness for income/education HP grants, top users, kids vs elderly, social rent; aligns with `docs/scenario_testing_v2_guide.md`.
- **Tutorial:** `notebooks/household_energy_abm_tutorial.ipynb` — end-to-end single-run demo and plotting utilities.

---

## Baseline & heating (model summary)
- Baseline: `0.40 kWh/h * (area/70)^0.20` clipped [0.85, 1.25] * property-type baseline multiplier (detached 1.20, semi 1.10, end 1.05, mid 1.00, flats 0.85–0.90); cap 1.5 kWh/h. No SAP/envelope/fuel scaling.
- Heating: duty-cycle on heat-loss signal; property-type heat multipliers (detached 1.20, semi 1.10, flats 0.85–0.90), sublinear area scaling, capped by capacity and global 20 kWh/h total cap. Diagnostics per dwelling: `base_kwh`, `heat_kwh`, `spike_kwh`, `cap_clip_*`.

---

## Policy levers (ready for UI)
- Heat-pump eligibility/class (`is_heatpump_candidate`, `heatpump_candidate_class`, adoption rate)
- Cohort masks: income band, education, tenure, children flag, schedule_type, dwelling_bucket, top-X% energy users
- Retrofit flags: `loft_ins_flag`, `glazing_flag`, etc.
- Setpoint tweak (e.g., elderly setback)

Reference implementations live in `notebooks/scenario_testing_v2.ipynb` and mirrored in `docs/scenario_testing_v2_guide.md`.

---

## Validation vs DESNZ
Use `energy-model-calibration.ipynb`:
- Load DESNZ electricity + gas, compute per-dwelling DESNZ kWh (elec meters as denominator).
- Load ABM batch parquet(s) from `results_lsoa/*/run_*/abm_year_*.parquet`.
- Compare per LSOA, per year: ABM kWh, DESNZ kWh, ratios; plot and export summaries.

---

## Support scripts
- `scripts/run_lsoa_batch.py` — run per-LSOA batches (agent-level optional) and save annual kWh by LSOA/year.
- `household_energy/run_lsoa_batch.py` — package entry variant used by notebooks.
- `make_animation.py` — helper for quick visuals (optional).

---

## Security
`data/` is git-ignored. Keep local inputs there; do not commit sensitive files.

---

## License
Apache-2.0 (unless stated otherwise in subcomponents).
