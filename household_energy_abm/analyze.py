#!/usr/bin/env python
"""
analyze.py
==========

Offline post-processing of a Household-Energy ABM run.

Inputs (produced by *run.py* in the same ``--outdir``):

* ``energy_timeseries.csv``          – hourly model totals
* ``model_timeseries.parquet``       – DataCollector (model-level)
* ``agent_timeseries.parquet``       – DataCollector (agent-level)

Optional static file:

* ``--geojson`` (required)           – building footprints with *fid* + geometry

Outputs (written next to the inputs):

* ``plot_hexbin.png``                – hex-binned spatial heat-map
* ``plot_prop_type.png``             – avg daily kWh by property type
* ``plot_wealth.png``                – avg daily kWh by wealth group
* ``plot_day_hour.png``              – day × hour temporal heat-map
* ``high_usage_map.html``            – interactive Leaflet map (opt-out with --no-map)

Usage::

    python analyze.py --geojson data/ncc_neighborhood.geojson --outdir results
"""

# ───────────────────────── imports ──────────────────────────────
import argparse
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from shapely.geometry import Point
import contextily as cx  

# ────────────────────── CLI parser helper ───────────────────────
def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    p = argparse.ArgumentParser(description="Make plots + Leaflet map from ABM outputs")
    p.add_argument("--outdir", default=".",
                   help="Folder containing energy_timeseries.csv etc. (default: .)")
    p.add_argument("--geojson", required=True,
                   help="Neighbourhood GeoJSON used by the ABM (required)")
    p.add_argument("--jitter", type=float, default=25,
                   help="Privacy jitter radius in metres (default: 25)")
    p.add_argument("--no-map", action="store_true",
                   help="Skip generating high_usage_map.html")
    return p.parse_args()

# ────────────────────── geometry helpers ────────────────────────
def jitter(geom, r: float) -> Point:
    """Return a geometry shifted randomly inside ±r metres (privacy masking)."""
    if geom.geom_type != "Point":
        geom = geom.centroid
    return Point(
        geom.x + random.uniform(-r, r),
        geom.y + random.uniform(-r, r),
    )

def reset_agent_index(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the MultiIndex from Mesa’s Parquet -> simple columns.

    Mesa v3 writes (step, AgentID) as a two-level index; we rename the second
    level to *agent_id* for clarity and downstream joins.
    """
    df = df.reset_index()
    return df.rename(
        columns={c: "agent_id" for c in df.columns
                 if "Agent" in c or c.endswith("_1")}
    )

# ────────────────────────── main ────────────────────────────────
def main() -> None:
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ── 1. Load simulation outputs ───────────────────────────────
    hourly   = pd.read_csv(outdir / "energy_timeseries.csv")
    model_ts = pd.read_parquet(outdir / "model_timeseries.parquet")
    agent_ts = reset_agent_index(pd.read_parquet(outdir / "agent_timeseries.parquet"))

    # ── 2. Build *high-usage* household slice ────────────────────
    # a. keep only household rows (energy == 0 means PersonAgent)
    houses = agent_ts[
        (agent_ts["energy"] == 0) &
        (agent_ts["energy_consumption"] > 0)
    ]

    # b. total kWh per household across entire run
    totals = (houses.groupby("agent_id", as_index=False)
                     ["energy_consumption"].sum()
                     .rename(columns={"energy_consumption": "total_energy"}))

    # c. attach geometry + property_type from GeoJSON
    gdf = gpd.read_file(args.geojson)[["fid", "geometry", "property_type"]]
    gdf["fid"]         = gdf["fid"].astype(str)      # unify dtype with totals
    totals["agent_id"] = totals["agent_id"].astype(str)
    gdf = gdf.rename(columns={"fid": "agent_id"}).merge(totals, on="agent_id")

    # d. take top quartile (fallback to top-half for tiny samples)
    q75 = gdf["total_energy"].quantile(0.75)
    hi  = gdf[gdf["total_energy"] >= q75]
    if hi.empty:
        hi = gdf.nlargest(max(3, len(gdf)//2), "total_energy")

    # e. jitter coordinates for privacy and keep dual CRS
    hi        = hi.to_crs(3857)                 # metres for hexbin
    hi["geometry"] = hi["geometry"].apply(lambda g: jitter(g, args.jitter))
    hi_latlon = hi.to_crs(4326)                 # WGS-84 for Leaflet pop-ups

    print("Sample of high-usage homes\n", hi.head()[["agent_id", "total_energy"]])

    # ── 3. Prepare time-series for bar plots ─────────────────────
    wealth_cols = [c for c in ("high", "medium", "low") if c in model_ts.columns]
    prop_cols   = [c for c in model_ts.columns
                   if c not in ["total_energy", "cumulative_energy", *wealth_cols]]

    # force to numeric (handles strings or missing)
    model_ts[prop_cols + wealth_cols] = model_ts[prop_cols + wealth_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    model_ts["day"] = model_ts.index // 24    # index == step

    # ── 4. Plot 1 – spatial hex-bin with basemap ─────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 6))

    hb = ax1.hexbin(
        hi.geometry.x,
        hi.geometry.y,
        C=hi["total_energy"],
        reduce_C_function=sum,
        gridsize=40,
        mincnt=1,
    )

    # ▼  add an OSM/CartoDB background  ▼
    cx.add_basemap(
        ax1,
        crs="EPSG:3857",
        source=cx.providers.CartoDB.Positron,   # light-grey background
        attribution=False,                      # omit tiny © text
    )

    ax1.set_axis_off()
    fig1.colorbar(hb, label="aggregated kWh")
    ax1.set_title("High-usage homes (jittered)")
    fig1.tight_layout()
    fig1.savefig(outdir / "plot_hexbin.png", dpi=150)

    # ── 5. Plot 2 – average daily kWh by property type ───────────
    daily_type = model_ts.groupby("day")[prop_cols].sum()
    fig2 = plt.figure()
    daily_type.mean().sort_values(ascending=False).plot.bar()
    plt.ylabel("avg kWh / day")
    plt.title("Daily average by property type")
    plt.xticks(rotation=45, ha="right")
    fig2.tight_layout()
    fig2.savefig(outdir / "plot_prop_type.png", dpi=150)

    # ── 6. Plot 3 – average daily kWh by wealth group ────────────
    if wealth_cols:
        daily_w = model_ts.groupby("day")[wealth_cols].sum()
        avg_w   = daily_w.mean().loc[wealth_cols]    # preserve ordering
        fig3 = plt.figure()
        ax = avg_w.plot.bar(color=["#d73027", "#fc8d59", "#91bfdb"][: len(avg_w)])
        ax.set_ylabel("avg kWh / day")
        ax.set_title("Daily average by wealth group")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        plt.xticks(rotation=0)
        fig3.tight_layout()
        fig3.savefig(outdir / "plot_wealth.png", dpi=150)
    else:
        print("⚠️  No wealth columns – skipping wealth plot")

    # ── 7. Plot 4 – temporal heat-map (day × hour) ───────────────
    fig4 = plt.figure(figsize=(7, 3))
    pivot = hourly.pivot_table(
        index="day", columns="hour", values="total_energy", aggfunc="sum"
    )
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="kWh")
    plt.xlabel("hour"); plt.ylabel("day")
    plt.title("Total demand • day × hour")
    fig4.tight_layout()
    fig4.savefig(outdir / "plot_day_hour.png", dpi=150)

    plt.show()   

    # ── 8. Optional interactive Leaflet map ──────────────────────
    if not args.no_map:
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            print("Install *folium* for interactive map support"); return

        centre = [hi_latlon.geometry.y.mean(), hi_latlon.geometry.x.mean()]
        fmap   = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

        # add heat layer (weight = total kWh)
        heat_data = [[p.geometry.y, p.geometry.x, p.total_energy]
                     for p in hi_latlon.itertuples()]
        HeatMap(heat_data, radius=15, blur=10).add_to(fmap)

        # add circle markers with property type tooltip
        for p in hi_latlon.itertuples():
            folium.CircleMarker(
                [p.geometry.y, p.geometry.x],
                radius=3, color="#ff6e54", fill=True, fill_opacity=0.7,
                popup=f"{p.property_type.title()}<br>{p.total_energy:.1f} kWh"
            ).add_to(fmap)

        html = outdir / "high_usage_map.html"
        fmap.save(html)
        print("Saved Leaflet map →", html)

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
