#!/usr/bin/env python3
"""
Five-city comparison of the Morris top-3 influence ranking (Phase-D robustness).

Reads the per-city sa_morris_<city>.csv + _meta.json written by sa_morris.py and
produces a single μ* comparison table and grouped-bar figure. Newcastle's μ*
comes from the full 10-knob screen (interactions are negligible — see the σ≈0
diagnostic — so its restricted-vs-full μ* are equivalent); the other four are
the 3-knob runs.

Output (results/sensitivity_analysis/):
  sa_morris_5city.csv  — μ* (% base energy), σ, R²(cov) μ*, per city × knob
  sa_morris_5city.png  — grouped bars: μ* by knob across cities
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SA = REPO / "results/sensitivity_analysis"

CITIES = ["newcastle", "sunderland", "waltham_forest", "manchester", "brighton"]
TOP3 = ["heating_setpoint_C", "building_age_mult_heating_gas", "sap_band_mult_heating_gas"]
LABELS = {
    "heating_setpoint_C": "setpoint",
    "building_age_mult_heating_gas": "building-age mult",
    "sap_band_mult_heating_gas": "SAP-band mult",
}


def main() -> int:
    rows = []
    for c in CITIES:
        df = pd.read_csv(SA / f"sa_morris_{c}.csv").set_index("knob")
        be = json.load(open(SA / f"sa_morris_{c}_meta.json"))["base_energy_kwh"]
        for k in TOP3:
            rows.append({
                "city": c, "knob": k,
                "mu_star_pct": df.loc[k, "mu_star_energy"] / be * 100,
                "sigma_pct":   df.loc[k, "sigma_energy"] / be * 100,
                "mu_star_r2cov": df.loc[k, "mu_star_r2cov"],
            })
    t = pd.DataFrame(rows)
    t.to_csv(SA / "sa_morris_5city.csv", index=False)

    piv = t.pivot(index="knob", columns="city", values="mu_star_pct").loc[TOP3, CITIES]

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    x = np.arange(len(TOP3))
    w = 0.16
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    for i, c in enumerate(CITIES):
        ax.bar(x + (i - 2) * w, piv[c].to_numpy(), w,
               label=c.replace("_", " "), color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in TOP3])
    ax.set_ylabel("μ*  (mean |EE| on citywide energy, % of base over ±2·SE)")
    ax.set_title("Morris top-3 influence ranking — five-city robustness")
    ax.legend(title="city", fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(SA / "sa_morris_5city.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("μ* (% base energy), knob × city:")
    print(piv.round(3).to_string())
    print(f"\nσ max across all cities/knobs: {t['sigma_pct'].max():.3f}% of base")
    print(f"R²(coverage) μ* max:           {t['mu_star_r2cov'].max():.4f}")
    print(f"\nwrote sa_morris_5city.{{csv,png}} to {SA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
