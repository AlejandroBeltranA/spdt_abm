#!/usr/bin/env python
"""
make_animation.py
=================

Utility script that runs the Household Energy ABM for a one-week window over a
single LSOA and turns the simulation output into a side-by-side animation:

• Left: map of the top-N households ranked by weekly kWh, coloured by their
        per-hour energy consumption with a hazy temperature overlay.
• Right: scatter/line view of total model energy vs. ambient temperature.

Usage
-----
python make_animation.py --lsoa-code E01033543 \
    --geojson data/epc_abm_newcastle.geojson \
    --climate data/ncc_2t_timeseries_2010_2039.parquet \
    --output results/abm_demo.mp4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
import matplotlib

# Headless rendering (FuncAnimation will still use the configured writer)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from household_energy.model import EnergyModel
from household_energy.config import load_config
try:
    import contextily as cx  # optional basemap
except Exception:  # pragma: no cover - optional dep
    cx = None


# ──────────────────────────────── CLI ─────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ABM and build a dual-panel animation.")
    parser.add_argument(
        "--geojson",
        default="data/epc_abm_newcastle.geojson",
        help="GeoJSON with household polygons (default: data/ncc_neighborhood_full.geojson)",
    )
    parser.add_argument(
        "--climate",
        default="data/ncc_2t_timeseries_2010_2039.parquet",
        help="Hourly climate parquet prepared by 01-climate-prep.ipynb",
    )
    parser.add_argument(
        "--lsoa-code",
        default="E01008452",
        help="LSOA code to filter the GeoJSON (default: E01008452). Pass '' to skip filtering.",
    )
    parser.add_argument(
        "--area-column",
        default=None,
        help="Optional column name to filter on (e.g., ward_code, msoa_code). "
             "If set, this takes precedence over the LSOA auto-detect.",
    )
    parser.add_argument(
        "--area-code",
        default=None,
        help="Area code to match in --area-column. If omitted, falls back to --lsoa-code.",
    )
    parser.add_argument(
        "--basemap",
        choices=("osm", "none"),
        default="osm",
        help="Background map: 'osm' (Carto/OSM via contextily) or 'none' (blank). Default: osm.",
    )
    parser.add_argument(
        "--start-utc",
        default=None,
        help="Optional UTC timestamp for the run start (e.g. 2022-01-10T00:00:00Z). "
             "Defaults to the first timestamp in the climate parquet.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Simulation duration in days (default: 7).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of highest-consuming households to animate (default: 40). Set to 0 to use all households.",
    )
    parser.add_argument(
        "--all-households",
        action="store_true",
        help="Color all households (ignore --top-n filtering).",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=None,
        help="Optional fixed max for colorbar (kWh). If omitted, uses 99th percentile; when --all-households, also caps at 12 kWh unless overridden.",
    )
    parser.add_argument(
        "--black-threshold",
        type=float,
        default=0.2,
        help="Household hourly kWh at/under which the dot is rendered black (default: 0.2 kWh).",
    )
    parser.add_argument(
        "--output",
        default="results/abm_animation.mp4",
        help="Path for the rendered animation (default: results/abm_animation.mp4).",
    )
    parser.add_argument(
        "--writer",
        choices=("ffmpeg", "pillow"),
        default="ffmpeg",
        help="Matplotlib animation writer (default: ffmpeg → MP4).",
    )
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for the animation (default: 6; slower to see changes).")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving (default: 150).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--figure-size",
        type=float,
        nargs=2,
        default=(12.0, 6.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 12 6.5).",
    )
    parser.add_argument(
        "--hidp-csv",
        default=None,
        help="Optional household enrichment CSV (HIDP, hh_n_people, tenure, income band, dwelling bucket, schedule_type).",
    )
    return parser.parse_args()


# ───────────────────────────── helpers ───────────────────────────────
def pick_lsoa_column(columns: Iterable[str]) -> str | None:
    """Pick the first column that looks like an LSOA code."""
    for col in columns:
        lc = col.lower()
        if "lsoa" in lc and "code" in lc:
            return col
    return None


def flatten_agent_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the Mesa DataCollector agent dataframe into tabular form with
    `step` and `agent_id` columns.
    """
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "AgentID" in df.columns:
        df = df.rename(columns={"AgentID": "agent_id"})
    elif "AgentID_1" in df.columns:
        df = df.rename(columns={"AgentID_1": "agent_id"})
    elif "agent_id" not in df.columns:
        raise ValueError("Could not find agent identifier column in DataCollector output.")

    if "Step" in df.columns:
        df = df.rename(columns={"Step": "step"})
    elif "step" not in df.columns:
        df = df.reset_index().rename(columns={"index": "step"})
    return df


def compute_hourly_index(model: EnergyModel, steps: int) -> pd.DatetimeIndex:
    """Return timezone-aware timestamps for each simulated hour."""
    if getattr(model, "climate", None) is None:
        # Fallback: synthetic hourly index
        return pd.date_range("2000-01-01", periods=steps, freq="H", tz=getattr(model, "_local_tz", "UTC"))

    times = getattr(model.climate, "times", None)
    if times is None or len(times) == 0:
        # Fallback: synthetic hourly index
        return pd.date_range("2000-01-01", periods=steps, freq="H", tz=getattr(model, "_local_tz", "UTC"))

    start_idx = int(getattr(model, "_t0", 0))
    tz_name = getattr(model, "_local_tz", "UTC")
    end_idx = min(len(times), start_idx + steps)
    if end_idx - start_idx < steps:
        raise ValueError("Climate parquet does not cover the requested simulation window.")
    ts = pd.to_datetime(times[start_idx:end_idx]).tz_localize("UTC")
    try:
        ts = ts.tz_convert(tz_name)
    except Exception:
        pass
    return ts


def extract_household_positions(model: EnergyModel) -> pd.DataFrame:
    """Return lon/lat positions and metadata for every household agent."""
    records: List[dict] = []
    for house in model.household_agents:
        geom = getattr(house, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        pt = geom
        if pt.geom_type != "Point":
            pt = geom.centroid
        records.append(
            {
                "agent_id": str(house.unique_id),
                "lon": pt.x,
                "lat": pt.y,
                "property_type": getattr(house, "property_type", ""),
                "sap_rating": getattr(house, "sap_rating", np.nan),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No household geometries available for mapping.")
    return df.drop_duplicates("agent_id").reset_index(drop=True)


def derive_top_households(agent_df: pd.DataFrame, top_n: int) -> pd.Series:
    """Return a Series of total kWh per agent sorted descending."""
    totals = (
        agent_df.groupby("agent_id", as_index=True)["energy_consumption"]
        .sum()
        .sort_values(ascending=False)
    )
    if totals.empty:
        raise ValueError("Agent-level DataCollector did not capture household energy.")
    return totals.head(top_n)


def fixed_sizes(n: int, size: float = 80.0) -> np.ndarray:
    """Return a fixed-size array for scatter markers."""
    return np.full(n, size, dtype=float)


# ───────────────────────────── pipeline ─────────────────────────────
def run_weekly_model(args: argparse.Namespace, gdf: gpd.GeoDataFrame) -> EnergyModel:
    """Instantiate and run the model for the configured number of days."""
    hours = int(args.days) * 24
    if hours <= 0:
        raise ValueError("--days must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    model = EnergyModel(
        gdf=gdf,
        climate_parquet=args.climate,
        climate_start=args.start_utc,
        collect_agent_level=True,
        agent_collect_every=1,
    )
    print(f"🏠 Households: {len(model.household_agents):,} | Persons: {len(model.person_agents):,}")
    print(f"▶️  Running {hours:,} hourly steps …")
    for h in range(hours):
        model.step()
        if (h + 1) % 24 == 0:
            print(f"  progressed {h+1:,}/{hours:,} hours")
    return model


def prepare_animation_payload(
    model: EnergyModel,
    gdf: gpd.GeoDataFrame,
    top_n: int,
    use_all: bool = False,
) -> dict:
    """Assemble all arrays/dataframes needed for the animation."""
    model_df = model.model_dc.get_model_vars_dataframe().copy()
    if model_df.empty:
        raise ValueError("Model DataCollector returned no rows.")
    model_df = model_df.iloc[1:].reset_index().rename(columns={"index": "step"})
    model_df["hour_index"] = model_df["step"] - 1

    agent_df = model.agent_dc.get_agent_vars_dataframe()
    agent_df = flatten_agent_dataframe(agent_df)
    agent_df = agent_df[agent_df["agent_type"] == "household"].copy()
    agent_df = agent_df[agent_df["step"] > 0]
    agent_df["hour_index"] = agent_df["step"] - 1

    if use_all or top_n <= 0:
        top_ids = agent_df["agent_id"].unique().tolist()
        totals = agent_df.groupby("agent_id", as_index=True)["energy_consumption"].sum().sort_values(ascending=False)
    else:
        totals = derive_top_households(agent_df, top_n=top_n)
        top_ids = list(totals.index)

    hh_filtered = agent_df[agent_df["agent_id"].isin(top_ids)].copy()
    hourly_matrix = (
        hh_filtered.pivot(index="hour_index", columns="agent_id", values="energy_consumption")
        .sort_index()
    )
    occ_matrix = (
        hh_filtered.pivot(index="hour_index", columns="agent_id", values="occupancy_count")
        .sort_index()
    )

    total_hours = model_df.shape[0]
    hourly_matrix = hourly_matrix.reindex(range(total_hours), fill_value=0.0)

    timestamps = compute_hourly_index(model, total_hours)
    if len(timestamps) != total_hours:
        raise ValueError("Timestamp index length mismatch.")

    positions_all = extract_household_positions(model)
    top_positions = (
        positions_all.set_index("agent_id")
        .reindex(top_ids)
        .reset_index()
    )

    return {
        "model_df": model_df,
        "hourly_matrix": hourly_matrix,
        "top_totals": totals,
        "timestamps": timestamps,
        "top_positions": top_positions,
        "all_positions": positions_all,
        "occupancy_matrix": occ_matrix.reindex(range(total_hours), fill_value=np.nan),
    }


def make_animation(
    payload: dict,
    args: argparse.Namespace,
    *,
    lsoa_name: str | None = None,
) -> None:
    """Create the animation using Matplotlib's FuncAnimation."""
    hourly = payload["hourly_matrix"]
    occ = payload.get("occupancy_matrix")
    model_df = payload["model_df"]
    timestamps = payload["timestamps"]
    totals = payload["top_totals"]
    top_positions = payload["top_positions"]
    all_positions = payload["all_positions"]

    total_energy = model_df["total_energy"].to_numpy()
    ambient_temp = model_df["ambient_mean_tempC"].to_numpy()
    if np.isnan(ambient_temp).all():
        ambient_temp = np.zeros_like(total_energy)

    energy_values = hourly.to_numpy()
    if args.color_max is not None:
        energy_vmax = float(args.color_max)
    else:
        energy_vmax = np.nanpercentile(energy_values, 99.0)
        if not np.isfinite(energy_vmax) or energy_vmax <= 0:
            energy_vmax = np.nanmax(energy_values)
        if not np.isfinite(energy_vmax) or energy_vmax <= 0:
            energy_vmax = 1.0
        if args.all_households or args.top_n <= 0:
            energy_vmax = min(energy_vmax, 12.0)

    # Plot scaffolding
    fig = plt.figure(figsize=tuple(args.figure_size))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.28)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    xmin = all_positions["lon"].min()
    xmax = all_positions["lon"].max()
    ymin = all_positions["lat"].min()
    ymax = all_positions["lat"].max()
    pad_x = (xmax - xmin) * 0.05 or 0.001
    pad_y = (ymax - ymin) * 0.05 or 0.001
    extent = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    # Optional basemap / coordinate prep
    use_basemap = args.basemap == "osm" and cx is not None
    if use_basemap:
        def _merc(lon, lat):
            k = 6378137.0
            x = np.radians(lon) * k
            y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * k
            return x, y
        all_x, all_y = _merc(all_positions["lon"].to_numpy(), all_positions["lat"].to_numpy())
        top_x, top_y = _merc(top_positions["lon"].to_numpy(), top_positions["lat"].to_numpy())
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
        pad_x = (xmax - xmin) * 0.05 or 100
        pad_y = (ymax - ymin) * 0.05 or 100
        extent = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
        scatter_coords = (top_x, top_y)
        back_coords = (all_x, all_y)
    else:
        scatter_coords = (top_positions["lon"], top_positions["lat"])
        back_coords = (all_positions["lon"], all_positions["lat"])

    # Optionally show grey context only if top-N filtering is used
    if not (args.all_households or args.top_n <= 0):
        ax_map.scatter(
            back_coords[0],
            back_coords[1],
            s=20,
            color="#b0b0b0",
            alpha=0.35,
            linewidths=0,
            label="All households",
            zorder=1,
        )

    initial_energy = hourly.iloc[0].to_numpy()
    scatter = ax_map.scatter(
        scatter_coords[0],
        scatter_coords[1],
        s=fixed_sizes(len(top_positions), size=30.0),
        c=initial_energy,
        cmap="inferno",
        norm=Normalize(vmin=0, vmax=energy_vmax),
        edgecolors="#1a1a1a",
        linewidths=0.35,
        label="Households",
        zorder=3,
    )

    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])

    # Optional basemap (OSM via contextily)
    if use_basemap:
        try:
            cx.add_basemap(
                ax_map,
                crs="EPSG:3857",
                source=cx.providers.CartoDB.Positron,
                alpha=1.0,
                attribution_size=6,
            )
        except Exception as e:
            print(f"⚠️  Basemap fetch failed ({e}); continuing without background.")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    map_title = "Weekly household energy hot-spots"
    if lsoa_name:
        map_title += f"\n{lsoa_name}"
    ax_map.set_title(map_title, fontsize=12, weight="bold")
    map_time_text = ax_map.text(
        0.02,
        0.96,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffffcc", ec="none"),
        zorder=5,
    )
    map_temp_text = ax_map.text(
        0.02,
        0.06,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#333333",
        zorder=5,
    )
    cbar = fig.colorbar(
        scatter,
        ax=ax_map,
        fraction=0.046,
        pad=0.02,
        label="Hourly household kWh",
    )
    cbar.ax.tick_params(labelsize=8)

    # Right panel: energy vs temp
    ax_plot.scatter(
        ambient_temp,
        total_energy,
        s=10,
        color="#c0c0c0",
        alpha=0.5,
        label="Hourly history",
    )
    hist_line, = ax_plot.plot([], [], color="#1f77b4", linewidth=2.0, label="Up to current hour")
    curr_marker, = ax_plot.plot([], [], marker="o", color="#d62728", markersize=8, label="Current hour")
    ax_plot.set_xlabel("Ambient temperature (°C)")
    ax_plot.set_ylabel("Total model energy (kWh)")
    ax_plot.grid(True, alpha=0.2, linestyle="--", linewidth=0.6)
    ax_plot.set_title("Energy response to temperature", fontsize=12, weight="bold")
    ax_plot.legend(loc="upper right", fontsize=8, frameon=False)
    plot_text = ax_plot.text(
        0.02,
        0.95,
        "",
        transform=ax_plot.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffffcc", ec="none"),
    )

    fig.suptitle("Household Energy ABM – climate × schedules × building fabric", fontsize=14, weight="bold")

    # Subtle temperature haze so brightness tracks ambient temperature
    temp_min = np.nanmin(ambient_temp)
    temp_max = np.nanmax(ambient_temp)
    if not np.isfinite(temp_min):
        temp_min = -5.0
    if not np.isfinite(temp_max):
        temp_max = 25.0
    temp_grid = np.full((2, 2), ambient_temp[0])
    temp_img = ax_map.imshow(
        temp_grid,
        cmap="Greys",
        alpha=0.12,  # subtle overlay
        extent=extent,
        origin="lower",
        vmin=temp_min,
        vmax=temp_max,
        zorder=0,
    )

    energy_cmap = scatter.cmap
    energy_norm = scatter.norm

    def update(frame: int):
        energy_slice = hourly.iloc[frame].to_numpy()
        scatter.set_sizes(fixed_sizes(len(energy_slice), size=30.0))
        # start from colormap for all points
        base_colors = energy_cmap(energy_norm(energy_slice))
        # determine away by occupancy (preferred) or energy threshold
        away_energy = energy_slice <= args.black_threshold
        if occ is not None:
            occ_slice = occ.iloc[frame].to_numpy()
            away_occ = np.nan_to_num(occ_slice, nan=0.0) <= 0
            away = away_energy | away_occ
        else:
            away = away_energy
        present = ~away
        colors = base_colors
        colors[away] = np.array([0.0, 0.0, 0.0, 1.0])
        colors[present, 3] = 1.0  # ensure alpha=1 for present
        scatter.set_array(energy_slice)  # keep colorbar in sync
        scatter.set_facecolors(colors)
        scatter.set_edgecolors(colors)
        scatter._facecolors = colors
        scatter._edgecolors = colors

        temp_val = float(ambient_temp[frame])
        if temp_img is not None:
            temp_img.set_array(np.full((2, 2), temp_val))
        timestamp = timestamps[frame]
        ts_str = timestamp.strftime("%a %d %b %Y %H:%M")
        map_time_text.set_text(ts_str)
        map_temp_text.set_text(f"Ambient: {temp_val:5.1f} °C")

        hist_line.set_data(ambient_temp[: frame + 1], total_energy[: frame + 1])
        curr_marker.set_data([ambient_temp[frame]], [total_energy[frame]])
        plot_text.set_text(f"Total demand: {total_energy[frame]:,.0f} kWh")
        return scatter, temp_img, hist_line, curr_marker, map_time_text, map_temp_text, plot_text

    frames = hourly.shape[0]
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / max(args.fps, 1),
        blit=False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.writer == "ffmpeg":
        writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    else:
        writer = PillowWriter(fps=args.fps)
    print(f"💾 Writing animation → {output_path}")
    anim.save(str(output_path), writer=writer, dpi=args.dpi)
    plt.close(fig)


# ───────────────────────────── entry point ──────────────────────────
def main() -> None:
    args = parse_args()
    geo_path = Path(args.geojson)
    if not geo_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geo_path}")
    climate_path = Path(args.climate)
    if not climate_path.exists():
        raise FileNotFoundError(f"Climate parquet not found: {climate_path}")

    print(f"📍 Loading households from {geo_path}")
    gdf = gpd.read_file(geo_path)
    if gdf.empty:
        raise ValueError("GeoJSON contains no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Optional household enrichment (HIDP + socio-demographics)
    cfg_defaults = load_config(None)
    hidp_path = args.hidp_csv or cfg_defaults.households.get("hidp_csv")
    if hidp_path:
        hidp_csv = Path(hidp_path)
        if not hidp_csv.exists():
            raise FileNotFoundError(f"HIDP CSV not found: {hidp_csv}")
        hidp_df = pd.read_csv(hidp_csv, low_memory=False)
        hidp_df.columns = [c.strip() for c in hidp_df.columns]
        geo_uprn_field = cfg_defaults.households.get("geojson_uprn_field", "UPRN")
        if geo_uprn_field not in gdf.columns:
            for alt in ["UPRN", "uprn", "fid"]:
                if alt in gdf.columns:
                    geo_uprn_field = alt
                    break
        if geo_uprn_field not in gdf.columns:
            raise KeyError("No UPRN-like field found in GeoJSON.")
        merge_on_csv = cfg_defaults.households.get("merge_on", "uprn_chr")
        gdf[geo_uprn_field] = gdf[geo_uprn_field].astype(str).str.strip()
        hidp_df[merge_on_csv] = hidp_df[merge_on_csv].astype(str).str.strip()
        before = len(gdf)
        gdf = gdf.merge(hidp_df, how="left", left_on=geo_uprn_field, right_on=merge_on_csv)
        unmatched = gdf[merge_on_csv].isna().sum() if merge_on_csv in gdf.columns else 0
        if unmatched:
            print(f"⚠️  {unmatched:,} households missing HIDP match (left join).")
        print(f"✅ Enriched households: {before:,} → {len(gdf):,} rows (merge on {geo_uprn_field} ↔ {merge_on_csv})")

    lsoa_label = None
    code = (args.area_code or args.lsoa_code or "").strip()
    area_col = args.area_column

    if code:
        if area_col:
            if area_col not in gdf.columns:
                raise ValueError(f"GeoJSON is missing column {area_col!r} for filtering.")
            col = area_col
        else:
            # prefer explicit LSOA, then ward/local authority fallbacks
            for candidate in ["lsoa_code", "ward_code", "local_authority"]:
                if candidate in gdf.columns:
                    col = candidate
                    break
            else:
                col = pick_lsoa_column(gdf.columns)
            if col is None:
                raise ValueError("GeoJSON is missing an area code column (looked for lsoa_code/ward_code/local_authority).")

        mask = gdf[col].astype(str).str.upper() == code.upper()
        if not mask.any():
            raise ValueError(f"No households found for {col} == {code!r}.")
        gdf = gdf.loc[mask].copy()
        name_col = next((c for c in gdf.columns if col.split('_')[0] in c.lower() and "name" in c.lower()), None)
        if name_col and pd.notna(gdf[name_col].iloc[0]):
            lsoa_label = f"{gdf[name_col].iloc[0]} ({code})"
        else:
            lsoa_label = f"{col}: {code}"
        print(f"   → Filtered to {len(gdf):,} dwellings in {lsoa_label}")
    else:
        print(f"   → Using all {len(gdf):,} dwellings (no area filter)")

    model = run_weekly_model(args, gdf)
    payload = prepare_animation_payload(model, gdf, top_n=args.top_n, use_all=args.all_households)
    make_animation(payload, args, lsoa_name=lsoa_label)
    print("✅ Animation complete.")


if __name__ == "__main__":
    main()
