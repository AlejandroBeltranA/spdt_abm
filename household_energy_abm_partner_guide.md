# Household‑Energy ABM — Partner Operations Guide 

> **Purpose.** A shareable, partner‑facing software document explaining how to install, configure, run, and validate the Household‑Energy ABM; it also records key assumptions, data requirements, and outputs.

> **Audience.** Data/analytics teams at local authorities, utilities, and research partners. We assume basic familiarity with Python and GIS data.


## 1) What this software does

An agent‑based model (ABM) that simulates **hour‑by‑hour residential energy demand** for every dwelling in a neighbourhood GeoJSON. The model can be run headless for batch output, explored via an interactive **Solara** dashboard, and validated against **DESNZ subnational electricity & gas** statistics (2020–2023).

**Core stack:** Mesa 3, mesa‑geo, pandas/pyarrow, GeoPandas/Shapely, Folium, Solara.

**Repository layout (high‑level):**

```
household_energy/            # Python package (entry points)
├── agent.py                 # HouseholdAgent & PersonAgent
├── model.py                 # EnergyModel core
├── climate.py               # ClimateField helpers
├── run.py                   # Headless run (CLI)
├── analyze.py               # Post‑run plots & maps
└── server.py                # Live dashboard

notebooks/
├── 01-climate-prep.ipynb    # Prepare hourly climate parquet
└── 02-energy-model-tests.ipynb  # Validation vs DESNZ (2020–2023)

data/                        # 🔒 Git‑ignored sensitive inputs
results/                     # Model outputs (git‑ignored)
```


## 2) Assumptions & scope

This section captures **what the model expects** and **what it does not cover**. When onboarding a new geography, verify each item.

### 2.1 Model scope

* Residential demand only; non‑domestic loads are out of scope.
* Simulation clock operates at **1‑hour resolution** aligned to the climate data index (UTC) and can be aggregated to daily.
* Default local timezone: `Europe/London` (configurable via CLI flag).

### 2.2 Data model assumptions

* **Dwelling geography:** A neighbourhood‑level **GeoJSON** with one feature per dwelling. Minimum fields typically include a unique dwelling ID and geometry (polygons or points with buffers); additional attributes (e.g., property type, wealth/IMD group) improve realism.
* **Climate driver:** An **hourly parquet** with ambient temperature (and any other climate fields, if provided) indexed by UTC datetimes and spatially joinable/look‑upable by dwelling (e.g., single series applied to all homes or a grid‑cell lookup).
* **Validation stats:** DESNZ subnational **electricity** and **gas** workbooks for 2020–2023, at Local Authority scale.

### 2.3 Security & governance assumptions

* All sensitive inputs live under `data/` and are **git‑ignored**. Partners should maintain their own secure storage and not commit regulated data.
* Outputs in `results/` are safe to share unless they breach partner‑specific disclosure rules.



## 3) System requirements

* **Python:** 3.10 or newer.
* **OS:** Linux/macOS/Windows.
* **RAM:** 8–16 GB recommended for neighbourhood‑scale runs; increase for city‑wide.
* **Disk:** Allow several GB for multi‑year hourly outputs.



## 4) Installation

Create an isolated environment and install dependencies:

```bash
python -m venv esa_mesa
source esa_mesa/bin/activate   # Windows: esa_mesa\Scripts\activate
pip install -r requirements.txt
pip install -e .
```



## 5) Data preparation

### 5.1 Secure data layout

The repository expects:

```
/data/**            # ignored by git
!/data/.gitkeep
!/data/README.md
```

Place local inputs here (do not commit them):

* Neighbourhood **GeoJSON**, e.g. `data/abm_households_XXXXX.geojson`.
* Hourly **climate parquet**, e.g. `data/hourly_climate.parquet`.
* DESNZ workbooks:

  * `Subnational_electricity_consumption_statistics_2005-2023.xlsx`
  * `Subnational_gas_consumption_statistics_2005-2023.xlsx`

### 5.2 Climate parquet (hourly) 

COMING SOON

Use `notebooks/01-climate-prep.ipynb` to build an **hourly** parquet with at least ambient temperature. Save to `data/hourly_climate.parquet` (or configure another path when running).


## 6) Running the model (headless)

The command‑line runner can simulate either (a) a fixed number of days or (b) an explicit UTC window.

### 6.1 Quick 7‑day smoke test

```bash
energy-run \
  data/abm_households_newcastle.geojson \
  --climate data/ncc_2t_timeseries_2010_2039.parquet \
  --days 7 \
  --outdir results \
  --no-agent-level
```

### 6.2 Windowed run (recommended for validation)

```bash
energy-run \
  data/abm_households_newcastle.geojson \
  --climate data/hourly_climate.parquet \
  --start-utc 2020-01-01T00:00:00Z \
  --end-utc   2025-01-01T00:00:00Z \
  --outdir results_2020_2024 \
  --no-agent-level
```

**Key flags:**

* `--climate` *(required)*: hourly parquet prepared above.
* `--start-utc` / `--end-utc` (exclusive): align steps to the climate index.
* `--days`: alternative to start/end; runs `days × 24` steps.
* `--local-tz`: default `Europe/London`.
* `--no-agent-level`: skip agent‑level DataCollector for faster runs.
* `--no-pickle`: skip pickling the full model state.
* `--print-every-hours`: progress cadence (default weekly).

**Outputs in `--outdir`:**

| File                       | Description                                                       |
| -------------------------- | ----------------------------------------------------------------- |
| `energy_timeseries.csv`    | Per‑hour totals & average (CSV for simple tools)                  |
| `model_timeseries.parquet` | Model‑level DataCollector (UTC hourly index)                      |
| `agent_timeseries.parquet` | Agent‑level DataCollector (if not `--no-agent-level`)             |
| `model_hourly.parquet`     | Hourly with UTC index (convenience)                               |
| `model_daily.parquet`      | Daily aggregates: `total_energy_kWh`, carrier splits, avg ambient |
| `energy_model.pkl`         | Full pickled model (unless `--no-pickle`)                         |

---

## 7) Interactive dashboard (Solara)

```bash
GEOJSON_PATH=data/abm_households_newcastle.geojson \
CLIMATE_PATH=data/ncc_2t_timeseries_2010_2039.parquet \
solara run household_energy/server.py
```

**Includes:**

* Leaflet map with energy‑coloured building outlines.
* Load breakdowns by property type and wealth group.
* Cumulative energy time‑series.

> Note: The dashboard for the full sample is still being stabilised. If you encounter errors, see Troubleshooting.


## 8) Offline analysis (static reports)

Generate plots and maps from a completed model run:

```bash
energy-analyze \
  --geojson data/abm_households_newcastle.geojson \
  --outdir   results
```

or equivalently:

```bash
python -m household_energy.analyze \
  --geojson data/abm_households_newcastle.geojson \
  --outdir  results
```

**Artifacts created:**

* `plot_hexbin.png` – spatial hex‑bin heatmap of energy intensity.
* `plot_prop_type.png` – average daily kWh by dwelling type.
* `plot_wealth.png` – average daily kWh by wealth group.
* `plot_day_hour.png` – demand matrix (day × hour).
* `high_usage_map.html` – interactive Leaflet map of highest‑usage dwellings.


## 9) Validation vs DESNZ (2020–2023)

COMING SOON

Open `notebooks/02-energy-model-tests.ipynb` and:

1. **Run the windowed model** for 2020–2024 to produce:

   * `results_2020_2024/model_hourly.parquet`
   * `results_2020_2024/model_daily.parquet`
2. **Run the multi‑year validation** block:

   * Loads DESNZ **Electricity**, **Gas**, and **Elec+Gas** for your Local Authority.
   * Compares **ABM 2020–2023** totals from `model_daily.parquet` to DESNZ.
   * Reports per‑year totals, per‑home means, and **ABM ÷ DESNZ** ratios (% diff).




## 10) Data dictionary (DRAFT)

|           Field | File    | Type          | Description                                | Required |
| --------------: | :------ | :------------ | :----------------------------------------- | :------: |
|   `dwelling_id` | GeoJSON | string/int    | Unique dwelling identifier                 |     ✅    |
|      `geometry` | GeoJSON | polygon/point | Dwelling footprint or centroid             |     ✅    |
| `property_type` | GeoJSON | string        | e.g., Detached, Semi, Flat                 |     —    |
|  `wealth_group` | GeoJSON | string/int    | Local wealth/IMD group                     |     —    |
|          `temp` | parquet | float         | Ambient temperature (°C), hourly UTC index |     ✅    |
|               … | …       | …             | …                                          |     …    |



## 11) Reproducibility & performance

* **Reproducible envs:** Commit `requirements.txt`/`pyproject.toml`. 
* **Chunked runs:** For multi‑year periods, prefer the UTC windowed mode; consider running per‑year to bound memory and then concatenating outputs.



## 12) Troubleshooting

* **`FileNotFoundError` for inputs** → Confirm paths under `data/` and that files are not accidentally committed to repo history.
* **Timezone misalignment** → Ensure `--start-utc`/`--end-utc` align to the climate parquet’s hourly index; set `--local-tz` if you need local‑time reporting.
* **Dashboard errors** → Verify `solara` is installed and environment variables `GEOJSON_PATH`/`CLIMATE_PATH` point to readable files.
* **Out‑of‑memory on agent outputs** → Use `--no-agent-level` for large runs.




## 13) Governance & licensing




## 14) Change log (excerpt)

* **v0.1.0 (Draft):** Initial partner guide; adds assumptions, data dictionary template, troubleshooting.


## 15) Appendix — Quickstart checklist

* [ ] Python 3.10+ installed.
* [ ] Virtual environment created & dependencies installed.
* [ ] GeoJSON and hourly climate parquet placed under `data/`.
* [ ] 7‑day smoke test completes with outputs in `results/`.
* [ ] Multi‑year window run completes and validation notebook runs.
* [ ] Dashboard opens and renders sample views.
* [ ] Local data dictionary filled (Section 10).
