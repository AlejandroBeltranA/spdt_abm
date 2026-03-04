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
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import solara
from mesa.visualization import SolaraViz, make_plot_component
from mesa_geo.visualization import make_geospace_component

from household_energy.model import EnergyModel
from household_energy.agent import PROPERTY_TYPES
from household_energy.model import WEALTH_BUCKETS

# ─── Configurable data sources (env vars override defaults) ──────
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_DEFAULT_GEOJSON = _REPO_ROOT / "data" / "abm_households_newcastle.geojson"

GEOJSON_PATH = Path(os.environ.get("GEOJSON_PATH", str(_DEFAULT_GEOJSON)))
CLIMATE_PATH = os.environ.get("CLIMATE_PATH", None)  # optional


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
wealth_plot      = make_plot_component(WEALTH_BUCKETS)
cumulative_plot  = make_plot_component(["cumulative_energy"])

def _build_app() -> object:
    if not GEOJSON_PATH.exists():
        @solara.component
        def Page():
            solara.Markdown(
                "\n".join(
                    [
                        "# Household Energy ABM",
                        "",
                        "Missing input GeoJSON.",
                        "",
                        f"- Looked for: `{GEOJSON_PATH.resolve()}`",
                        "- Set `GEOJSON_PATH` env var to a valid file and restart.",
                    ]
                )
            )

        return Page

    gdf = gpd.read_file(GEOJSON_PATH)
    model = EnergyModel(
        gdf=gdf,
        climate_parquet=CLIMATE_PATH,
        climate_start=None,
        local_tz="Europe/London",
        collect_agent_level=False,
    )

    return SolaraViz(
        model,
        components=[
            geo_component,
            energy_type_plot,
            wealth_plot,
            cumulative_plot,
        ],
        name="Household Energy ABM",
    )


app = _build_app()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ.copy()
    if "MPLCONFIGDIR" not in env:
        env["MPLCONFIGDIR"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / "mplconfig")
    if "XDG_CACHE_HOME" not in env:
        env["XDG_CACHE_HOME"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdgcache")

    cmd = [sys.executable, "-m", "solara", "run", "household_energy.server", *argv]
    return subprocess.call(cmd, env=env)
