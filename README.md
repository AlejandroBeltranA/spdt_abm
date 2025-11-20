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
