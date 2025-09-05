#!/usr/bin/env python
"""
server.py – Solara dashboard
=============================
Run:
    solara run server.py

Env overrides (optional):
    GEOJSON_PATH=/path/to/households.geojson
    CLIMATE_PATH=/path/to/hourly_climate.parquet
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import solara
from mesa.visualization import SolaraViz, make_plot_component
from mesa_geo.visualization import make_geospace_component

from household_energy.model import EnergyModel
from household_energy.agent import PROPERTY_TYPES

# ─── Configurable data sources (env vars override defaults) ──────
GEOJSON_PATH = Path(os.environ.get("GEOJSON_PATH", "../data/abm_households_newcastle.geojson"))
CLIMATE_PATH = os.environ.get("CLIMATE_PATH", None)  # optional

if not GEOJSON_PATH.exists():
    raise FileNotFoundError(f"GeoJSON not found: {GEOJSON_PATH.resolve()}")

# ─── Colour ramp utilities ────────────────────────────────────────
# Prebuild a pleasant 5-stop ramp
_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "energy_ramp", ["#e0f3f8", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
)

def _rgba(value: float, vmin: float, vmax: float, alpha: float = 0.85) -> str:
    """Map value→CSS rgba using linear ramp; robust to vmin==vmax."""
    if not (vmax > vmin):
        scale = 0.5  # flat field → middle of ramp
    else:
        scale = (value - vmin) / (vmax - vmin)
        scale = max(0.0, min(1.0, float(scale)))
    r, g, b, _ = _COLORMAP(scale)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"

def energy_draw(agent):
    """Portrayal: fill + outline by the household's *current* hourly kWh."""
    if not hasattr(agent, "energy_consumption"):
        return {}

    m = agent.model

    # Cache min/max once per tick for speed
    step_key = getattr(m, "current_hour", 0)
    if getattr(m, "_viz_step", None) != step_key:
        vals = [getattr(h, "energy_consumption", 0.0) for h in m.household_agents] or [0.0]
        m._viz_min = min(vals)
        m._viz_max = max(vals)
        m._viz_step = step_key

    vmin = getattr(m, "_viz_min", 0.0)
    vmax = getattr(m, "_viz_max", 1.0)

    # robust color
    def _rgba(value, vmin, vmax, alpha=0.85):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "energy_ramp", ["#e0f3f8", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        )
        if not (vmax > vmin):
            scale = 0.5
        else:
            scale = (value - vmin) / (vmax - vmin)
            scale = max(0.0, min(1.0, float(scale)))
        r, g, b, _ = cmap(scale)
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"

    color = _rgba(getattr(agent, "energy_consumption", 0.0), vmin, vmax)

    return {
        "color": color,         # stroke color
        "weight": 1,            # <-- int, not float
        "opacity": 0.9,         # stroke opacity
        "fill": True,
        "fill_color": color,    # <-- snake_case
        "fill_opacity": 0.85,   # <-- snake_case
        "radius": 6,            # used for Point geometries (Circle marker)
    }


# ─── Solara components ───────────────────────────────────────────
geo_component = make_geospace_component(
    energy_draw,
    portrayal_method="dynamic",   # colours update every tick
    zoom=14,
    scroll_wheel_zoom=True,
)

energy_type_plot = make_plot_component(PROPERTY_TYPES)
wealth_plot      = make_plot_component(["high", "medium", "low"])
cumulative_plot  = make_plot_component(["cumulative_energy"])

# ─── Build model + app ───────────────────────────────────────────
gdf = gpd.read_file(GEOJSON_PATH)

# Instantiate with optional climate (if provided via env)
model = EnergyModel(
    gdf=gdf,
    climate_parquet=CLIMATE_PATH,
    # For live runs we start at the climate file's first timestamp (if provided)
    climate_start=None,
    local_tz="Europe/London",
    collect_agent_level=False,  # keep UI light; enable if you need it
)

app = SolaraViz(
    model,
    components=[
        geo_component,
        energy_type_plot,
        wealth_plot,
        cumulative_plot,
    ],
    name="Household Energy ABM",
)

app
