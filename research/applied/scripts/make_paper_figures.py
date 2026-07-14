#!/usr/bin/env python3
"""
Paper 1 §Results figures, all read from committed transfer_confidence CSVs (every
point traceable; nothing recomputed by hand).

Outputs (into the overleaf project):
  figure_coverage_scatter.png — five-city per-LSOA validation, coloured by city:
                                (A) modelled vs DESNZ-metered electricity with the
                                1:1 line; (B) ratio vs coverage with the pooled fit.
  figure_confidence_map.png   — Newcastle LSOA-polygon choropleth by reliability
                                tier on an OpenStreetMap-derived basemap.

Figure styling is deliberately spare: panel letters, axis labels, one legend.
All interpretation lives in the LaTeX captions, not on the canvas.

Usage:
  .venv/bin/python research/applied/scripts/make_paper_figures.py
Needs network the first time for the basemap tiles (contextily).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
CONF = REPO / "research/applied"                                   # v1 (v5-calibration) CSVs
CONF_V2 = REPO / "research/applied/results/transfer_v2"            # v2 (SERL-ledger) CSVs
OUT = REPO / "docs/research/paper draft/too_hot_overleaf_v2"
BOUNDARIES = REPO / "data/boundaries/newcastle_lsoa_2021.geojson"

# version switch: v1 reads transfer_confidence_<slug>_2023.csv from research/applied;
# v2 reads transfer_confidence_<slug>_2023_v2.csv from results/transfer_v2 and writes
# figures with a _v2 suffix so the v5 figures are kept for side-by-side review.
V2 = False


def _conf_csv(slug: str) -> Path:
    if V2:
        return CONF_V2 / f"transfer_confidence_{slug}_2023_v2.csv"
    return CONF / f"transfer_confidence_{slug}_2023.csv"


def _out(name: str) -> Path:
    return OUT / (name.replace(".png", "_v2.png") if V2 else name)

CITIES = [
    ("newcastle", "Newcastle"),
    ("sunderland", "Sunderland"),
    ("waltham_forest", "Waltham Forest"),
    ("manchester", "Manchester"),
    ("brighton", "Brighton & Hove"),
]
# Okabe-Ito qualitative palette (colour-blind safe), one hue per city.
CITY_COLOUR = {
    "Newcastle": "#0072B2", "Sunderland": "#E69F00", "Waltham Forest": "#009E73",
    "Manchester": "#CC79A7", "Brighton & Hove": "#D55E00",
}
# Traffic-light reliability tiers (shared with the choropleth).
TIER_COLOUR = {"High": "#2E8B57", "Medium": "#E1A100", "Low": "#C44E52"}
TIER_ORDER = ["High", "Medium", "Low"]


def _load_all_cities() -> pd.DataFrame:
    frames = []
    for slug, name in CITIES:
        d = pd.read_csv(_conf_csv(slug))
        d["city"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _ols(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2


# ── Figure: five-city validation scatter, coloured by city ──────────────────

def coverage_scatter() -> None:
    d = _load_all_cities()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)

    # A — modelled vs DESNZ-metered electricity, per LSOA (GWh, log-log so the
    # bulk near the 1:1 line is legible despite a few large-metered outliers).
    lo, hi = np.inf, 0.0
    for _, name in CITIES:
        s = d[d["city"] == name]
        x = s["total_kwh_elec"].to_numpy() / 1e6
        y = s["abm_elec_kwh"].to_numpy() / 1e6
        ok = (x > 0) & (y > 0)
        axA.scatter(x[ok], y[ok], s=11, alpha=0.6, color=CITY_COLOUR[name], edgecolor="none")
        lo = min(lo, np.nanmin(x[ok]), np.nanmin(y[ok]))
        hi = max(hi, np.nanmax(x[ok]), np.nanmax(y[ok]))
    lo, hi = lo * 0.8, hi * 1.25
    axA.plot([lo, hi], [lo, hi], color="grey", lw=1.0, ls="--")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(lo, hi)
    axA.set_ylim(lo, hi)
    axA.set_xlabel("DESNZ metered electricity (GWh per LSOA)", fontsize=9)
    axA.set_ylabel("Modelled electricity (GWh per LSOA)", fontsize=9)
    axA.set_title("A", fontsize=11, loc="left", fontweight="bold")
    axA.tick_params(labelsize=8)

    # B — per-LSOA ratio vs coverage, pooled fit
    v = d[["coverage", "tot_ratio", "city"]].replace([np.inf, -np.inf], np.nan).dropna()
    for _, name in CITIES:
        s = v[v["city"] == name]
        axB.scatter(s["coverage"], s["tot_ratio"], s=11, alpha=0.6,
                    color=CITY_COLOUR[name], edgecolor="none")
    b, r2 = _ols(v["tot_ratio"].to_numpy(), v["coverage"].to_numpy())
    xs = np.linspace(v["coverage"].min(), v["coverage"].max(), 50)
    axB.plot(xs, b[0] + b[1] * xs, color="black", lw=1.2)
    axB.axhline(1.0, color="grey", lw=0.8, ls="--")
    axB.text(0.04, 0.95, f"pooled R² = {r2:.2f}", transform=axB.transAxes,
             fontsize=8.5, va="top")
    axB.set_xlabel("Coverage (modelled dwellings / DESNZ meters)", fontsize=9)
    axB.set_ylabel("Ratio (modelled / metered electricity)", fontsize=9)
    axB.set_title("B", fontsize=11, loc="left", fontweight="bold")
    axB.tick_params(labelsize=8)

    handles = [Line2D([0], [0], marker="o", ls="", color=CITY_COLOUR[n], markersize=6,
                      label=n) for _, n in CITIES]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(_out("figure_coverage_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {_out('figure_coverage_scatter.png')}")


# ── Figure: Newcastle reliability choropleth ────────────────────────────────

def confidence_map() -> None:
    import contextily as cx

    bound = gpd.read_file(BOUNDARIES)
    conf = pd.read_csv(_conf_csv("newcastle"))[["lsoa_code", "confidence"]]
    g = bound.merge(conf, left_on="LSOA21CD", right_on="lsoa_code", how="inner")
    g = g.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8, 8.4), constrained_layout=True)
    for tier in TIER_ORDER:
        sub = g[g["confidence"] == tier]
        if not sub.empty:
            sub.plot(ax=ax, color=TIER_COLOUR[tier], edgecolor="white",
                     linewidth=0.4, alpha=0.72)
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution_size=5)
    except Exception as e:  # offline fallback: keep the choropleth, drop tiles
        print(f"  basemap tiles unavailable ({e}); rendering without basemap")
    ax.set_axis_off()
    counts = g["confidence"].value_counts()
    handles = [plt.Rectangle((0, 0), 1, 1, fc=TIER_COLOUR[t], ec="white",
                             alpha=0.72, label=f"{t}  (n={int(counts.get(t, 0))})")
               for t in TIER_ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=True,
              title="Reliability tier", title_fontsize=10)
    fig.savefig(_out("figure_confidence_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {_out('figure_confidence_map.png')} ({len(g)} LSOAs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", action="store_true",
                    help="read the v2 (SERL-ledger) transfer CSVs and write _v2 figures")
    if ap.parse_args().v2:
        V2 = True
        print("v2 mode: reading results/transfer_v2 CSVs, writing *_v2.png")
    coverage_scatter()
    confidence_map()
