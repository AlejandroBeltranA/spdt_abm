#!/usr/bin/env python3
"""
Paper 1 §3.5 validation figure (v5), regenerated from committed v5 outputs only.

Four panels, all traceable to files on disk (nothing recomputed by hand,
nothing invented):

  A  Per-LSOA modelled vs DESNZ-metered electricity, Newcastle 2023, with the
     1:1 line and the Pearson correlation. Points coloured by reliability tier.
       source: research/applied/transfer_confidence_newcastle_2023.csv
  B  Inter-annual electricity, modelled vs metered, 2021-2023, citywide totals.
     Shows the model reproducing the warm-year (2022) dip.
       source: transfer_confidence_newcastle_{2021,2022,2023}.csv
  C  Intraday electricity: modelled winter and summer diurnal shape against the
     SERL half-hourly empirical envelope (10-90 band + mean).
  D  Intraday gas: same construction.
       sources: results_lsoa/transfer_newcastle/*/run_newcastle_2023/
                  model_timeseries_*_newcastle_2023.parquet  (v5 hourly)
                data/serl_profiles/serl_profiles_num_occupants.csv

Writes figure_2_validation.png into the overleaf project (replacing the stale
pre-v5 figure). Palette matches make_paper_figures.py.

Usage: .venv/bin/python research/applied/scripts/make_validation_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
CONF = REPO / "research/applied"
TS_DIR = REPO / "results_lsoa/transfer_newcastle"
SERL_CSV = REPO / "data/serl_profiles/serl_profiles_num_occupants.csv"
OUT = REPO / "docs/research/paper draft/too_hot_overleaf_v2/figure_2_validation.png"

# Shared with the §Results figures: traffic-light reliability tiers + accents.
TIER_COLOUR = {"High": "#2E8B57", "Medium": "#E1A100", "Low": "#C44E52"}
TIER_ORDER = ["High", "Medium", "Low"]
MODEL_C = "#2E8B57"   # modelled
METER_C = "#4C72B0"   # metered / DESNZ
WINTER_C = "#2E8B57"
SUMMER_C = "#E1A100"
YEARS = [2021, 2022, 2023]


# ── Panels A-B: annual electricity, modelled vs metered ─────────────────────

def panel_a_scatter(ax) -> None:
    d = pd.read_csv(CONF / "transfer_confidence_newcastle_2023.csv")
    d = d[["abm_elec_kwh", "total_kwh_elec", "confidence"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    x = d["total_kwh_elec"].to_numpy() / 1e6   # GWh/LSOA, DESNZ
    y = d["abm_elec_kwh"].to_numpy() / 1e6     # GWh/LSOA, modelled
    for tier in TIER_ORDER:
        s = d["confidence"] == tier
        ax.scatter(x[s], y[s], s=16, alpha=0.75, color=TIER_COLOUR[tier],
                   edgecolor="none")
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1.0, ls="--")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("DESNZ metered electricity (GWh per LSOA)", fontsize=9)
    ax.set_ylabel("Modelled electricity (GWh per LSOA)", fontsize=9)
    ax.set_title("A", fontsize=11, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)
    handles = [Line2D([0], [0], marker="o", ls="", color=TIER_COLOUR[t],
                      markersize=6, label=t) for t in TIER_ORDER]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False)


def panel_b_interannual(ax) -> None:
    model_twh, meter_twh = [], []
    for yr in YEARS:
        d = pd.read_csv(CONF / f"transfer_confidence_newcastle_{yr}.csv")
        model_twh.append(d["abm_elec_kwh"].sum() / 1e9)
        meter_twh.append(d["total_kwh_elec"].sum() / 1e9)
    xs = np.arange(len(YEARS))
    w = 0.38
    ax.bar(xs - w / 2, model_twh, w, color=MODEL_C, label="Modelled")
    ax.bar(xs + w / 2, meter_twh, w, color=METER_C, label="DESNZ metered")
    ax.set_xticks(xs)
    ax.set_xticklabels(YEARS)
    ax.set_ylabel("Citywide electricity (TWh)", fontsize=9)
    ax.set_ylim(0, max(max(model_twh), max(meter_twh)) * 1.22)
    ax.set_title("B", fontsize=11, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")


# ── Panels C-D: intraday shape, modelled vs SERL envelope ───────────────────

WINTER = [12, 1, 2]
SUMMER = [6, 7, 8]


def _modelled_diurnal(fuel_col: str) -> dict:
    """Citywide modelled diurnal shape (mean=1) for winter and summer, 2023.

    Sums hourly kWh across all Newcastle LSOAs, converts to local time, then
    takes the by-hour mean within each season and normalises to mean 1.0.
    """
    files = sorted(TS_DIR.glob(f"*/run_newcastle_2023/model_timeseries_*_newcastle_2023.parquet"))
    total = None
    for f in files:
        s = pd.read_parquet(f, columns=[fuel_col])[fuel_col]
        total = s if total is None else total.add(s, fill_value=0.0)
    total.index = total.index.tz_convert("Europe/London")
    df = pd.DataFrame({"kwh": total.values, "hour": total.index.hour,
                       "month": total.index.month})
    out = {}
    for name, months in (("winter", WINTER), ("summer", SUMMER)):
        prof = df[df["month"].isin(months)].groupby("hour")["kwh"].mean()
        out[name] = prof / prof.mean()
    return out, len(files)


def _serl_envelope(fuel: str) -> pd.DataFrame:
    """SERL empirical diurnal envelope across occupancy segments (mean=1)."""
    s = pd.read_csv(SERL_CSV)
    s = s[(s["kind"] == "hourly") & (s["fuel"] == fuel)]
    g = s.groupby("idx")["mult"]
    return pd.DataFrame({"lo": g.quantile(0.10), "hi": g.quantile(0.90),
                         "mean": g.mean()}).sort_index()


def panel_intraday(ax, fuel_col: str, serl_fuel: str, label: str, n_lsoa: int) -> None:
    env = _serl_envelope(serl_fuel)
    hours = env.index.to_numpy()
    ax.fill_between(hours, env["lo"], env["hi"], color="grey", alpha=0.22,
                    label="SERL 10-90%")
    ax.plot(hours, env["mean"], color="grey", lw=1.4, ls="--", label="SERL mean")
    prof, _ = _modelled_diurnal_cache[fuel_col]
    ax.plot(prof["winter"].index, prof["winter"].values, color=WINTER_C, lw=2.0,
            label="Modelled winter")
    ax.plot(prof["summer"].index, prof["summer"].values, color=SUMMER_C, lw=2.0,
            label="Modelled summer")
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("hour of day (local)", fontsize=9)
    ax.set_ylabel("relative demand (mean = 1)", fontsize=9)
    ax.set_title(label, fontsize=11, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")


_modelled_diurnal_cache: dict = {}


def main() -> None:
    print("aggregating v5 hourly parquets for the intraday panels...")
    elec_prof, n_lsoa = _modelled_diurnal("total_electric_kwh")
    gas_prof, _ = _modelled_diurnal("total_gas_kwh")
    _modelled_diurnal_cache["total_electric_kwh"] = (elec_prof, n_lsoa)
    _modelled_diurnal_cache["total_gas_kwh"] = (gas_prof, n_lsoa)
    print(f"  aggregated {n_lsoa} Newcastle LSOA timeseries (2023)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)
    panel_a_scatter(axes[0, 0])
    panel_b_interannual(axes[0, 1])
    panel_intraday(axes[1, 0], "total_electric_kwh", "electric", "C", n_lsoa)
    panel_intraday(axes[1, 1], "total_gas_kwh", "gas", "D", n_lsoa)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
