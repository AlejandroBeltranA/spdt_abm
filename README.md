# Household-Energy ABM  
*Mesa 3 · mesa-geo · Solara dashboard*

Agent-based model that simulates **hour-by-hour residential energy demand** for every dwelling in a neighbourhood GeoJSON. Outputs feed both an **interactive Solara dashboard** and an **offline analysis pipeline**. A validation workflow compares model outputs to **DESNZ subnational electricity & gas** statistics (2020–2023).

---

## Directory layout

```
household_energy/            # Python package (entry points below)
├── agent.py                 # HouseholdAgent & PersonAgent
├── model.py                 # EnergyModel (ABM core)
├── climate.py               # ClimateField helpers
├── run.py                   # Headless run (CLI)
├── analyze.py               # Post-run plots & maps
├── server.py                # Live dashboard
└── __init__.py

notebooks/
├── 01-climate-prep.ipynb    # Prepare hourly climate parquet
└── 02-energy-model-tests.ipynb
    # Windowed run validation (2020–2023) vs DESNZ

data/                        # 🔒 Git-ignored sensitive inputs
└── .gitkeep                 # placeholder

results/                     # Model outputs (git-ignored)
```

---

## Security: data handling

All contents of `data/` are ignored by Git (keep placeholders only):

```
/data/**
!/data/.gitkeep
!/data/README.md
```

Put **local inputs** here and do **not** commit them:
- Neighbourhood GeoJSON (`data/abm_households_XXXXX.geojson`)
- Hourly climate parquet (`data/xxx_2t_TIME.parquet`)
- DESNZ workbooks  
  - `Subnational_electricity_consumption_statistics_2005-2023.xlsx`  
  - `Subnational_gas_consumption_statistics_2005-2023.xlsx`



---

## Setup

```bash
python -m venv esa_mesa
source esa_mesa/bin/activate
pip install -r requirements.txt
# optional (dev mode so CLI works as module):
pip install -e .
```

Python ≥3.10 recommended.

---

## Prepare climate parquet (coming soon!)

Use `notebooks/01-climate-prep.ipynb` to create an **hourly** climate parquet the model can index. Save to e.g.:

```
data/hourly_climate.parquet
```


---

## Run the model (headless)

`run.py` can simulate either a fixed number of days **or** an explicit UTC window aligned to the climate grid.

### Quick 7-day run

```bash
energy-run \
  data/abm_households_newcastle.geojson \
  --climate data/ncc_2t_timeseries_2010_2039.parquet \
  --days 7 \
  --outdir results \
  --no-agent-level
```

### Windowed run (recommended for validation)

```bash
energy-run
 data/abm_households_newcastle.geojson \
  --climate data/hourly_climate.parquet \
  --start-utc 2020-01-01T00:00:00Z 
  --end-utc 2025-01-01T00:00:00Z \
  --outdir results_2020_2024 
  --no-agent-level
```

Flags:
- `--climate` (required): hourly parquet prepared above
- `--start-utc` / `--end-utc` (exclusive): align steps to climate indices
- `--days`: alternative to start/end; runs `days × 24` steps
- `--local-tz`: default `Europe/London`
- `--no-agent-level`: skip agent-level DataCollector (faster)
- `--no-pickle`: skip pickling the full model
- `--print-every-hours`: progress cadence (default weekly)
- `--hidp-csv`: optional enrichment CSV (see below) to add HIDP, household size, tenure, income band, dwelling bucket, and schedule_type per UPRN

### Outputs

Written to `--outdir`:

| File                       | Description |
|---------------------------|-------------|
| `energy_timeseries.csv`   | Simple per-hour totals & average (compat CSV) |
| `model_timeseries.parquet`| Model-level DataCollector (indexed by UTC hour) |
| `agent_timeseries.parquet`| Agent-level DataCollector (if not `--no-agent-level`) |
| `model_hourly.parquet`    | Model-level hourly with UTC index (convenience) |
| `model_daily.parquet`     | Daily aggregates: `total_energy_kWh`, carrier splits, avg ambient |
| `energy_model.pkl`        | Full pickled model (unless `--no-pickle`) |
| `agent_timeseries.parquet` (if enabled) | Now includes HIDP, hh_n_people, hh_children, hh_income_band, hh_edu_detail, dwelling_bucket, tenure, size_band, schedule_type/schedule_profile |

---

## Interactive dashboard
Still troubleshooting this for the full sample. 

```bash
GEOJSON_PATH=data/abm_households_newcastle.geojson \
CLIMATE_PATH=data/ncc_2t_timeseries_2010_2039.parquet \
solara run household_energy/server.py
```

Includes:
- Leaflet map (energy-coloured outlines)
- Load by property type & wealth group
- Cumulative energy time-series


---

## Analyze results (static)

```bash
energy-analyze 
--geojson data/abm_households_newcastle.geojson 
--outdir results
```
or
```bash
python -m household_energy.analyze   
--geojson data/abm_households_newcastle.geojson   
--outdir results

```

Creates:
- `plot_hexbin.png` – spatial hex-bin heat-map
- `plot_prop_type.png` – avg daily kWh by dwelling type
- `plot_wealth.png` – avg daily kWh by wealth group
- `plot_day_hour.png` – demand matrix (day × hour)
- `high_usage_map.html` – interactive Leaflet map

---

## Per-house math (plain text)

Each dwelling, each hour:

1) **Baseline** (optionally scaled):
   - Start from calibrated annual `E_ann`.
   - If `apply_structural_multipliers=True`, multiply by SAP/ptype/area/envelope/control/system nudges, then divide by 365*24:
     `E_base_h = (E_ann * M_struct) / (365*24)`
   - If False, use `E_ann/(365*24)` directly.

2) **Climate load**:
   - Heating degree: `HD = max(0, (T_set - T_out) - db)`
   - Cooling degree: `CD = max(0, (T_out - T_cool) - db)`
   - Per-dwelling heating slope `slope_h` from archetype/SAP/area/retro/noise.
   - Heat-pump factor `M_HP = boiler_efficiency / COP` (flat COP today).
   - `E_climate_h = HD * slope_h * M_HP + CD * slope_cool` (dampened if empty).

3) **Occupancy/appliance spikes**:
   - `E_spike_h = sum_per_person(E_home_or_away)` with per-person spikes adjusted by wealth/SAP.

4) **Total per hour**:
   - `E_hh_h = E_base_h + E_climate_h + E_spike_h`

Annual per dwelling = sum over 8,760 hours; area totals = sum over dwellings. Scenario deltas (e.g., heat-pump adoption): `DeltaE = E_scenario - E_default`; £ savings = `-DeltaE * tariff`.

---

## Validation vs DESNZ (2020–2023) (COMING SOON)

Use `notebooks/02-energy-model-tests.ipynb`:

---

## Optional household enrichment (HIDP + socio-demographics)

You can enrich the GeoJSON with a synthetic household table (e.g., `data/hidp_uprn_matches_tiered.csv`) by passing `--hidp-csv` to `run.py`/`energy-run`. The join is left-on UPRN (`UPRN` in GeoJSON by default, configurable) to `uprn_chr` in the CSV. No rows are dropped; unmatched rows keep legacy behaviour.

Columns consumed (all optional, defaults preserve old behaviour):
- `hidp` (household id; falls back to UPRN)
- `hh_n_people` (household size; capped by config)
- `hh_children` (bool flag)
- `dwelling_bucket`, `tenure`, `size_band` (bedrooms 1–4)
- `hh_income_band` (q1_lowest … q5_highest), `hh_edu_detail`
- `schedule_type` (controls daily schedules; see below)

Config knobs (in `household_energy/config_defaults.yaml`, override as needed):
- `households.hidp_csv`: default CSV path (otherwise pass via CLI)
- `households.merge_on`, `households.geojson_uprn_field`: join keys
- `households.resident_cap`: max people per household
- `households.bedroom_multiplier`: kWh scaling by bedrooms
- `households.n_residents_default`: legacy fallback resident count

### Schedule_type mapping (with jitter)

If `schedule_type` is present, residents get leave/return times from archetypes with ±1h jitter for diversity:
- `retired_household` → all home all day
- `unemployed_or_inactive` → mostly home; some part‑time PM
- `working_adult_household` → standard work hours
- `dual_earner_household` → mix of standard + early/late shifts
- `student_household` → student hours
- `family_with_children` → one adult on school run, others work; children added
- `single_parent_with_children` → school run / part‑time AM
- Unknown/missing → legacy Parent/Worker/Homebody profiles

Children are inferred from `hh_children` (bool) and schedule_type; counts are bounded by `resident_cap`. Energy baseline optionally scales by bedrooms via `households.bedroom_multiplier`. All new fields are exported to `agent_timeseries.parquet` for targeting and analysis.

### Quick enrichment check

After a run with agent-level collection enabled, you can summarize coverage and schedules:

```bash
python notebooks/enrichment_check.py --results results_smoke_epc
```

If `agent_timeseries.parquet` is missing, rerun without `--no-agent-level` (or set a short window) to generate it.

---

## Configs & overrides (how to create/use them)

The ABM reads a YAML config (defaults in `household_energy/config_defaults.yaml`). You can supply a minimal override and only the keys you change will overwrite defaults (deep merge).

**Minimal example (`my_scenario.yaml`):**
```yaml
meta:
  name: heat_pump_push
  date: 2026-01-26
  notes: 40% HP adoption, cooler setpoint
model:
  heating_setpoint_C: 18.0
  heatpump_adoption_rate: 0.4
households:
  hidp_csv: data/hidp_uprn_matches_tiered.csv
  merge_on: uprn_chr
  geojson_uprn_field: UPRN
schedules:
  wfh_share: 0.30
```

**Run with it:**
- CLI: `python -m household_energy.run data/geo.json --climate data/climate.parquet --config-path my_scenario.yaml --days 7`
- Notebook/animation: pass `config_path="my_scenario.yaml"` to `EnergyModel`.

**Common knobs:**
- `model`: setpoints, heating/cooling slopes, per-person spikes, heatpump adoption and class weights.
- `households`: HIDP CSV path, join keys, resident cap, bedroom multipliers, default resident count.
- `schedules`: default profiles (leave/return), `wfh_share`.
- `envelope_levers`: multipliers for CWI/SWI/loft/floor/glazing.
- `systems`: level/slope multipliers for heat pumps, electric/gas, storage heaters.

Keep overrides minimal and dated in `meta` for traceability.

1) Run the **Windowed run (2020–2024)** section to generate:
   - `results_2020_2024/model_hourly.parquet`
   - `results_2020_2024/model_daily.parquet`

2) Run the **Multi-year validation** block:
   - Loads DESNZ **Electricity**, **Gas**, and **Elec+Gas** for your Local Authority.
   - Compares **ABM 2020–2023** totals from `model_daily.parquet` to DESNZ.
   - Reports per-year totals, per-home means, and **ABM ÷ DESNZ** ratios (% diff).

---

## Requirements (high-level)

- `mesa`, `mesa-geo`, `pandas`, `pyarrow`
- `geopandas`, `shapely`, `folium`, `solara`
- `openpyxl` 
- See `requirements.txt` / `pyproject.toml` for exact versions.

---
