#!/usr/bin/env python
"""
server.py – Solara dashboard (hard-coded GeoJSON)
=================================================
Run:
    solara run server.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import solara
from mesa.visualization import SolaraViz, make_plot_component
from mesa_geo.visualization import make_geospace_component

from household_energy.model import EnergyModel
from household_energy.agent import PROPERTY_TYPES

# ─── Hard-coded data source ───────────────────────────────────────
GEOJSON_PATH = Path("data/ncc_neighborhood.geojson")  # <- adjust if you move the file
if not GEOJSON_PATH.exists():
    raise FileNotFoundError(GEOJSON_PATH.resolve())

#gdf = gpd.read_file("data/ncc_neighborhood.geojson")

# ─── Colour ramp + portrayal function ─────────────────────────────
def _rgba(value, vmin, vmax):
    """Map *value*→CSS rgba string using a 5-colour linear ramp."""
    scale = (value - vmin) / (vmax - vmin) if vmax > vmin else 0
    r, g, b = mcolors.LinearSegmentedColormap.from_list(
        "", ["#e0f3f8", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
    )(scale)[:3]
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)"

def energy_draw(agent):
    """Portrayal: colour outline by current kWh for each household."""
    if not hasattr(agent, "energy_consumption"):
        return {}
    vals = [h.energy_consumption for h in agent.model.household_agents]
    return {
        "color": _rgba(agent.energy_consumption, min(vals), max(vals)),
        "weight": 2,
        "fillOpacity": 0.9,
        # "fillColor": same rgba  # enable to fill polygons
    }

# ─── Solara components ────────────────────────────────────────────
geo_component = make_geospace_component(
    energy_draw,
    portrayal_method="dynamic",   # colours update every tick
    zoom=14,
    scroll_wheel_zoom=True,
)
energy_type_plot = make_plot_component(PROPERTY_TYPES)
wealth_plot      = make_plot_component(["high", "medium", "low"])
cumulative_plot  = make_plot_component(["cumulative_energy"])

# ─── Build dashboard (pass *class* + constructor params) ──────────
gdf   = gpd.read_file(GEOJSON_PATH)
model = EnergyModel(gdf)         

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