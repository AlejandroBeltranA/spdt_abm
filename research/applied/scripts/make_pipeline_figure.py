#!/usr/bin/env python3
"""
Figure 1 — the v5 model-and-data pipeline schematic (Paper 1, §3.3/§3.4).

Redraws the data → calibration → model → validation flow to match the v5
architecture: SERL drives a reproducible CLI fit pipeline (six fits → a
calibrated config), the MABM runs on the EPC/Census/ERA5 stack, and validation
is the DESNZ coverage-aware reliability layer plus the five-city transfer, with
a Morris sensitivity screen. No data run — pure schematic.

Output: docs/research/paper draft/too_hot_overleaf_v2/figure_1_pipeline_v5.png
(written to a v5-suffixed name so the existing figure_1_pipeline.png is not
overwritten until reviewed).

Usage: .venv/bin/python research/applied/scripts/make_pipeline_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[3] / "docs/research/paper draft/too_hot_overleaf_v2"

FILL = {"data": "#dce8f5", "calib": "#e7e0f3", "model": "#daf0da",
        "validate": "#fde8d0", "output": "#f5e6f7"}
EDGE = {"data": "#4c78a8", "calib": "#7d5bbe", "model": "#54a24b",
        "validate": "#f58518", "output": "#9467bd"}
LINE_H = 0.030


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, title, subtitle="", role="data"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.012,rounding_size=0.012",
                     fc=FILL[role], ec=EDGE[role], lw=2.0, zorder=3))
        cx, cy = x + w / 2, y + h / 2
        if subtitle:
            n = subtitle.count("\n") + 1
            ax.text(cx, cy + n * LINE_H * 0.55, title, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="#1a1a1a", zorder=4)
            ax.text(cx, cy - LINE_H * 0.85, subtitle, ha="center", va="center",
                    fontsize=8.6, color="#444", zorder=4, linespacing=1.5)
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="#1a1a1a", zorder=4)

    def arr(x0, y0, x1, y1, rad=0.0, color="#555"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                    mutation_scale=14,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=2)

    # ── Column 1: data inputs ────────────────────────────────────────────────
    box(0.02, 0.72, 0.20, 0.15, "EPC Register",
        "97,714 dwellings\nUPRN · SAP · fuel · floor area", role="data")
    box(0.02, 0.52, 0.20, 0.15, "Census 2021 +\nUnderstanding Society",
        "Income · tenure\noccupancy schedules", role="data")
    box(0.02, 0.32, 0.20, 0.15, "ERA5 Climate",
        "Hourly outdoor temperature\n2010–2026", role="data")
    box(0.02, 0.10, 0.20, 0.15, "SERL Panel",
        "~13k smart-meter dwellings\ndaily + diurnal aggregates", role="data")
    ax.text(0.12, 0.93, "Data inputs", ha="center", fontsize=11, fontweight="bold", color=EDGE["data"])

    # ── Column 2: calibration + model ────────────────────────────────────────
    box(0.27, 0.06, 0.22, 0.20, "SERL Calibration",
        "Six reproducible CLI fits →\nsetpoint · HDD slopes · area/SAP/age\nmultipliers · elec baseline · per-person\n→ calibrated config", role="calib")
    box(0.27, 0.64, 0.22, 0.15, "Synthetic Population",
        "UPRN matching (99.5%)\nHIDP household attributes", role="model")
    box(0.27, 0.34, 0.22, 0.22, "Household Energy MABM",
        "HouseholdAgents + PersonAgents\nhourly demand · 97,714 dwellings\ngas · electric · other", role="model")
    ax.text(0.38, 0.93, "Calibration & model", ha="center", fontsize=11, fontweight="bold", color=EDGE["model"])

    # ── Column 3: validation, robustness, outputs ────────────────────────────
    box(0.55, 0.70, 0.22, 0.16, "DESNZ Reliability Layer",
        "per-LSOA coverage tiers\nHigh tier within ±15%", role="validate")
    box(0.55, 0.48, 0.22, 0.16, "Five-City Transfer",
        "no per-city refit\nR²(ratio~coverage) 0.71–0.80", role="validate")
    box(0.55, 0.27, 0.22, 0.15, "Sensitivity (Morris)",
        "SERL-bootstrap parameter bands\nnear-linear · ≈±4% envelope", role="validate")
    box(0.55, 0.05, 0.22, 0.16, "Outputs",
        "LSOA-scale annual demand\n+ per-LSOA reliability layer", role="output")
    ax.text(0.66, 0.93, "Validation & outputs", ha="center", fontsize=11, fontweight="bold", color=EDGE["validate"])

    ax.axvline(0.245, 0.04, 0.91, color="#ccc", lw=0.8, ls="--", zorder=1)
    ax.axvline(0.515, 0.04, 0.91, color="#ccc", lw=0.8, ls="--", zorder=1)

    # ── Arrows ───────────────────────────────────────────────────────────────
    arr(0.22, 0.79, 0.27, 0.74, color=EDGE["data"])      # EPC → synth pop
    arr(0.22, 0.59, 0.27, 0.71, color=EDGE["data"])      # Census → synth pop
    arr(0.22, 0.39, 0.27, 0.45, color=EDGE["data"])      # ERA5 → MABM
    arr(0.22, 0.16, 0.27, 0.16, color=EDGE["data"])      # SERL → calibration
    arr(0.38, 0.64, 0.38, 0.56, color=EDGE["model"])     # synth pop → MABM
    arr(0.38, 0.26, 0.38, 0.34, color=EDGE["calib"])     # calibration → MABM
    arr(0.49, 0.50, 0.55, 0.74, color=EDGE["model"], rad=-0.15)   # MABM → reliability
    arr(0.49, 0.46, 0.55, 0.55, color=EDGE["model"])             # MABM → transfer
    arr(0.49, 0.42, 0.55, 0.34, color=EDGE["model"], rad=0.12)   # MABM → SA
    arr(0.49, 0.38, 0.55, 0.13, color=EDGE["model"], rad=0.18)   # MABM → outputs

    legend_items = [
        mpatches.Patch(fc=FILL["data"], ec=EDGE["data"], label="Empirical data inputs"),
        mpatches.Patch(fc=FILL["calib"], ec=EDGE["calib"], label="SERL calibration pipeline"),
        mpatches.Patch(fc=FILL["model"], ec=EDGE["model"], label="Simulation model"),
        mpatches.Patch(fc=FILL["validate"], ec=EDGE["validate"], label="Validation & robustness"),
        mpatches.Patch(fc=FILL["output"], ec=EDGE["output"], label="Outputs"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=9,
              framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.02))

    fig.savefig(OUT / "figure_1_pipeline_v5.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'figure_1_pipeline_v5.png'}")


if __name__ == "__main__":
    main()
