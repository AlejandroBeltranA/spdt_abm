#!/usr/bin/env python
"""
analyze.py
==========

Offline post-processing of a Household-Energy ABM run.

Inputs (produced by *run.py* in the same ``--outdir``):
* ``energy_timeseries.csv``          – hourly model totals (CSV)
* ``model_timeseries.parquet``       – DataCollector (model-level, hourly UTC)
* ``agent_timeseries.parquet``       – DataCollector (agent-level, optional)

Static input:
* ``--geojson`` (required) – building footprints with an ID key (default: UPRN)

Outputs (written next to the inputs):
* ``plot_hexbin.png``       – hex-binned spatial heat-map (if agent data)
* ``plot_prop_type.png``    – avg daily kWh by property type (if identifiable)
* ``plot_wealth.png``       – avg daily kWh by wealth group (if columns exist)
* ``plot_day_hour.png``     – day × hour temporal heat-map (from CSV)
* ``high_usage_map.html``   – interactive Leaflet map (opt-out with --no-map)

Usage:
    energy-analyze --geojson data/abm_households_newcastle.geojson --outdir results
"""

from __future__ import annotations

# ───────────────────────── imports ──────────────────────────────
import argparse
import random
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from shapely.geometry import Point
import contextily as cx  # basemap

# ────────────────────── CLI parser helper ───────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make plots + Leaflet map from ABM outputs")
    p.add_argument("--outdir", default=".", help="Folder with model outputs (default: .)")
    p.add_argument("--geojson", required=True, help="Neighbourhood GeoJSON used by the ABM")
    p.add_argument("--id-field", default="UPRN",
                   help="GeoJSON column to join with agent_id (default: UPRN). Fallbacks: fid,id,index")
    p.add_argument("--jitter", type=float, default=25.0,
                   help="Privacy jitter radius in metres (default: 25)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible jitter")
    p.add_argument("--no-map", action="store_true", help="Skip generating high_usage_map.html")
    return p.parse_args()

# ────────────────────── geometry helpers ────────────────────────
def jitter_point(geom, r: float) -> Point:
    """Return a geometry shifted randomly inside ±r metres (privacy masking)."""
    if geom.geom_type != "Point":
        geom = geom.centroid
    return Point(geom.x + random.uniform(-r, r), geom.y + random.uniform(-r, r))

def reset_agent_index(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten Mesa DataCollector agent MultiIndex to regular columns."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    # Standardize agent id column name
    for cand in ("AgentID", "AgentID_1", "agent_id"):
        if cand in df.columns:
            df = df.rename(columns={cand: "agent_id"})
            break
    if "agent_id" not in df.columns:
        # Last-resort: if a column looks like agent ID
        for c in df.columns:
            if "agent" in c.lower() and "id" in c.lower():
                df = df.rename(columns={c: "agent_id"})
                break
    return df

# ────────────────────── inference helpers ───────────────────────
def find_wealth_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ("high", "medium", "low") if c in df.columns]

def infer_property_energy_cols(model_ts: pd.DataFrame) -> List[str]:
    """
    Heuristically pick columns that represent energy by property type at the model level.
    Rule: numeric columns excluding known metrics/wealth, whose row-wise sum ≈ total_energy.
    """
    known_metrics = {"total_energy", "cumulative_energy", "ambient_mean_tempC"}
    wealth = set(find_wealth_cols(model_ts))
    numeric_cols = [c for c in model_ts.columns if pd.api.types.is_numeric_dtype(model_ts[c])]
    candidates = [c for c in numeric_cols if c not in known_metrics | wealth]

    if "total_energy" not in model_ts.columns or not candidates:
        return []

    # Accept if sum of all candidates reconstructs total_energy to within tolerance most of the time.
    s = model_ts[candidates].sum(axis=1)
    te = model_ts["total_energy"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (np.isfinite(te) & (np.abs(s - te) <= np.maximum(1e-6, 1e-3 * np.abs(te))))
    if ok.mean() >= 0.95:
        return candidates

    # Fallback: common property labels
    common = [c for c in candidates if c.lower() in {
        "detached", "semi_detached", "semi-detached", "terraced", "flat",
        "apartment", "bungalow", "maisonette"
    }]
    return common

# ────────────────────────── main ────────────────────────────────
def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ── 1. Load simulation outputs (robustly) ────────────────────
    hourly_csv = outdir / "energy_timeseries.csv"
    model_parq = outdir / "model_timeseries.parquet"
    agent_parq = outdir / "agent_timeseries.parquet"

    if not model_parq.exists():
        raise FileNotFoundError(f"Missing required file: {model_parq}")
    if not hourly_csv.exists():
        raise FileNotFoundError(f"Missing required file: {hourly_csv}")

    hourly = pd.read_csv(hourly_csv)
    model_ts = pd.read_parquet(model_parq)

    agent_ts: Optional[pd.DataFrame]
    if agent_parq.exists():
        try:
            agent_ts = reset_agent_index(pd.read_parquet(agent_parq))
            if "agent_id" not in agent_ts.columns:
                print("⚠️  agent_timeseries present but no agent_id column after reset; disabling agent analyses.")
                agent_ts = None
        except Exception as e:
            print(f"⚠️  Could not read agent_timeseries.parquet ({e}); disabling agent analyses.")
            agent_ts = None
    else:
        print("ℹ️  No agent_timeseries.parquet found; skipping household hexbin and Leaflet map.")
        agent_ts = None

    # ── 2. High-usage household slice (if agent data available) ──
    hi_latlon = None
    if agent_ts is not None:
        # Keep agents with positive energy consumption across the run (households only).
        # This avoids relying on any model-specific 'energy==0 means PersonAgent' heuristic.
        if "energy_consumption" not in agent_ts.columns:
            print("⚠️  agent_timeseries missing 'energy_consumption'; skipping agent analyses.")
        else:
            totals = (agent_ts.groupby("agent_id", as_index=False)["energy_consumption"]
                               .sum()
                               .rename(columns={"energy_consumption": "total_energy"}))

            # Attach geometry + property_type; join key is configurable
            gdf = gpd.read_file(args.geojson)
            join_key = args.id_field
            if join_key not in gdf.columns:
                # fallbacks
                for alt in ("fid", "id"):
                    if alt in gdf.columns:
                        join_key = alt
                        break
                else:
                    # use index as last resort
                    gdf = gdf.reset_index().rename(columns={"index": "index_id"})
                    join_key = "index_id"

            gdf[join_key] = gdf[join_key].astype(str)
            totals["agent_id"] = totals["agent_id"].astype(str)
            cols_to_keep = [join_key, "geometry"] + ([c for c in ("property_type",) if c in gdf.columns])
            gdf = gdf[cols_to_keep].rename(columns={join_key: "agent_id"})
            merged = gdf.merge(totals, on="agent_id", how="inner")

            if merged.empty:
                print("⚠️  Join produced no households; skipping hexbin/Leaflet.")
            else:
                # Top quartile (fallback to top-half for tiny samples)
                q75 = merged["total_energy"].quantile(0.75)
                hi = merged[merged["total_energy"] >= q75]
                if hi.empty:
                    hi = merged.nlargest(max(3, len(merged) // 2), "total_energy")

                # Jitter in metres (Web Mercator), keep WGS84 copy for Leaflet
                hi = hi.to_crs(3857)
                hi["geometry"] = hi["geometry"].apply(lambda g: jitter_point(g, args.jitter))
                hi_latlon = hi.to_crs(4326)

                print("Sample of high-usage homes\n", hi[["agent_id", "total_energy"]].head())

                # ── 3. Plot 1 – spatial hex-bin with basemap ───────────
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                hb = ax1.hexbin(
                    hi.geometry.x, hi.geometry.y,
                    C=hi["total_energy"], reduce_C_function=np.sum,
                    gridsize=40, mincnt=1,
                )
                cx.add_basemap(ax1, crs="EPSG:3857",
                               source=cx.providers.CartoDB.Positron,
                               attribution=False)
                ax1.set_axis_off()
                fig1.colorbar(hb, label="aggregated kWh")
                ax1.set_title("High-usage homes (jittered)")
                fig1.tight_layout()
                fig1.savefig(outdir / "plot_hexbin.png", dpi=150)

    # ── 4. Time-series prep for bar plots ────────────────────────
    # Force numeric where possible
    model_ts = model_ts.copy()
    wealth_cols = find_wealth_cols(model_ts)
    prop_cols   = infer_property_energy_cols(model_ts)

    num_cols = wealth_cols + prop_cols + [c for c in ("total_energy",) if c in model_ts.columns]
    for c in num_cols:
        model_ts[c] = pd.to_numeric(model_ts[c], errors="coerce")

    # In run.py, model_ts index is time-like; but sometimes it's the step (int).
    # Create a 'day' index from rows since it’s hourly.
    if "day" not in model_ts.columns:
        # Assume hourly cadence with a contiguous index
        model_ts["day"] = np.arange(len(model_ts)) // 24

    # ── 5. Plot 2 – average daily kWh by property type ───────────
    if prop_cols:
        daily_type = model_ts.groupby("day")[prop_cols].sum()
        fig2 = plt.figure()
        daily_type.mean().sort_values(ascending=False).plot.bar()
        plt.ylabel("avg kWh / day")
        plt.title("Daily average by property type")
        plt.xticks(rotation=45, ha="right")
        fig2.tight_layout()
        fig2.savefig(outdir / "plot_prop_type.png", dpi=150)
    else:
        print("ℹ️  Could not identify property-type columns; skipping property-type plot.")

    # ── 6. Plot 3 – average daily kWh by wealth group ────────────
    if wealth_cols:
        daily_w = model_ts.groupby("day")[wealth_cols].sum()
        avg_w   = daily_w.mean().loc[wealth_cols]    # keep declared order
        fig3 = plt.figure()
        ax = avg_w.plot.bar()
        ax.set_ylabel("avg kWh / day")
        ax.set_title("Daily average by wealth group")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        plt.xticks(rotation=0)
        fig3.tight_layout()
        fig3.savefig(outdir / "plot_wealth.png", dpi=150)
    else:
        print("ℹ️  No wealth columns – skipping wealth plot.")

    # ── 7. Plot 4 – temporal heat-map (day × hour) ───────────────
    if {"day", "hour", "total_energy"}.issubset(hourly.columns):
        fig4 = plt.figure(figsize=(7, 3))
        pivot = hourly.pivot_table(index="day", columns="hour",
                                   values="total_energy", aggfunc="sum")
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="kWh")
        plt.xlabel("hour"); plt.ylabel("day")
        plt.title("Total demand • day × hour")
        fig4.tight_layout()
        fig4.savefig(outdir / "plot_day_hour.png", dpi=150)
    else:
        print("⚠️  energy_timeseries.csv missing expected columns; skipping day×hour plot.")

    plt.show()

    # ── 8. Optional interactive Leaflet map ──────────────────────
    if (not args.no_map) and (hi_latlon is not None) and (len(hi_latlon) > 0):
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            print("Install *folium* for interactive map support")
            return

        centre = [hi_latlon.geometry.y.mean(), hi_latlon.geometry.x.mean()]
        fmap   = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

        # Heat layer (weight = total kWh)
        heat_data = [[p.geometry.y, p.geometry.x, p.total_energy] for p in hi_latlon.itertuples()]
        HeatMap(heat_data, radius=15, blur=10).add_to(fmap)

        # Circle markers with property type tooltip if available
        for p in hi_latlon.itertuples():
            tip = getattr(p, "property_type", None)
            lbl = (tip.title() if isinstance(tip, str) else "household")
            folium.CircleMarker(
                [p.geometry.y, p.geometry.x],
                radius=3, color="#ff6e54", fill=True, fill_opacity=0.7,
                popup=f"{lbl}<br>{p.total_energy:.1f} kWh"
            ).add_to(fmap)

        html = outdir / "high_usage_map.html"
        fmap.save(html)
        print("Saved Leaflet map →", html)
    elif not args.no_map:
        print("ℹ️  No household slice available; skipping Leaflet map.")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
