# Household-Energy ABM  
*Mesa 3 · mesa-geo · Solara dashboard*

Agent-based model that simulates **hour-by-hour electricity demand** for every
dwelling in a neighbourhood GeoJSON.  
Outputs feed both an **interactive Solara dashboard** and an **offline
analysis pipeline** (static plots + Leaflet heat-map).

---

## Directory layout

```
mesa_model/
└── modules/
    └── agent.py # HouseholdAgent & PersonAgent
    └── model.py # EnergyModel (ABM core)
    └── analyze.py # Plots and maps
├── run.py # Headless run
├── server.py # Live dashboard
├── requirements.txt 
└── data/
    └── neighborhood.geojson
```

## Setup

```bash
python3 -m venv esa_mesa
source esa_mesa/bin/activate  
pip install -r requirements.txt
```
`
## Running Simulation

```bash
energy-run data/ncc_neighborhood.geojson --days 7 --outdir results
```

Creates the following files in results/:

| File                       | Description                         |
| -------------------------- | ----------------------------------- |
| `energy_timeseries.csv`    | hourly totals (kWh)                 |
| `model_timeseries.parquet` | model-level DataCollector variables |
| `agent_timeseries.parquet` | agent-level DataCollector variables |
| `energy_model.pkl`         | full pickled model (cloudpickle)    |

## Interactive Dashboard

```bash
solara run server.py 
```
Dashboard contains: 
* Leaflet map (energy-coloured outlines)

* Bar charts (load by property type & wealth group)

* Cumulative-energy line plot

## Analyze Results

```bash
energy-analyze --geojson data/ncc_neighborhood.geojson --outdir results
```
Creates in results/:

* `plot_hexbin.png` – spatial hex-bin heat-map (with CartoDB basemap)

* `plot_prop_type.png` – average daily kWh by dwelling type

* `plot_wealth.png` – average daily kWh by wealth group

* `plot_day_hour.png` – demand matrix (day × hour)

* `high_usage_map.html` – interactive Leaflet map of top-quartile homes