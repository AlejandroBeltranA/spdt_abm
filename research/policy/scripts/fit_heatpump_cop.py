#!/usr/bin/env python3
"""
Anchor the heat-pump COP parameters to OpenEnergyMonitor field data.

Background
----------
The model's heat-pump efficiency was an unsourced assumption: a flat
``heatpump_cop_ref = 2.80`` plus an unused temperature curve
``systems.heat_pump.cop_curve`` (config_defaults.yaml). 2.80 happens to
match the DESNZ *Electrification of Heat* trial median ASHP SPFH4
(representative new-install sample), but it carried no citation and the
temperature curve was hand-drawn.

This script anchors both to **heatpumpmonitor.org** — OpenEnergyMonitor's
open, continuously-updated registry of field-monitored UK heat pumps.
Every system publishes measured electricity and heat (class-2 heat
meters) via emoncms; the site exposes a no-auth public JSON API:

    https://heatpumpmonitor.org/system/list/public.json   (per-system metadata)
    https://heatpumpmonitor.org/system/stats/last365      (per-system measured COP)

Sample-bias caveat (carried into the YAML)
------------------------------------------
heatpumpmonitor is a **self-selected, well-commissioned** sample —
owner-enthusiasts who instrument their systems. It is the *good-install*
end of the distribution and reads high relative to the representative
EoH trial. We therefore emit BOTH:

  - ``cop_field_median`` (well-installed anchor, from this data), and
  - ``cop_representative`` (DESNZ EoH median, conservative default),

so the model can run either as a scenario lever rather than baking in
the optimistic number. See the four-paper roadmap / DESNZ confidence
layer notes.

Which COP maps to the model?
----------------------------
``hp_effect_mult = boiler_efficiency / heatpump_cop_ref`` multiplies the
**space-heating** slope (agent.py), so the field metric that maps to it
is **space-heating COP** (SPFH4 space), not the combined space+DHW SPF.
Both are reported; the point anchor uses space COP.

Temperature curve
-----------------
Field mean-outdoor-temperatures only span ~3-14 C, so a raw linear fit
cannot be trusted out to the -5 C the curve needs. We instead fit a
constant **Carnot efficiency** (eta = COP * (T_flow - T_out)/T_flow,
medianed across systems) and evaluate COP(T_out) = eta * T_flow /
(T_flow - T_out) at a documented representative flow temperature. This
gives a physically-grounded curve that extrapolates sanely. The raw
binned medians are emitted alongside as a sanity check.

Output
------
``results/field_fits/heatpump_cop.yaml`` with point anchors, the derived
cop_curve, by-type breakdown, and full diagnostics + provenance.

CLI
---
    python research/applied/scripts/fit_heatpump_cop.py            # fetch live
    python research/applied/scripts/fit_heatpump_cop.py --systems-json /tmp/list.json \
        --stats-json /tmp/stats.json   # offline from cached pulls
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

LIST_URL = "https://heatpumpmonitor.org/system/list/public.json"
STATS_URL = "https://heatpumpmonitor.org/system/stats/last365"

# DESNZ Electrification of Heat trial — representative new-install median
# SPFH4. Conservative default anchor; see module docstring.
EOH_REPRESENTATIVE_COP = 2.80
EOH_CITATION = (
    "Energy Systems Catapult (2023), Electrification of Heat Demonstration "
    "Project: Heat Pump Performance Data Analysis. Median ASHP SPFH4 ~2.80."
)

# Curve anchor temperatures the model's cop_curve is sampled at.
CURVE_TEMPS_C = [15.0, 7.0, 0.0, -5.0]


# ---------------------------------------------------------------------------
# data acquisition
# ---------------------------------------------------------------------------
def _fetch_json(url: str, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": "spdt_abm-fit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_data(
    systems_json: Optional[Path],
    stats_json: Optional[Path],
) -> tuple[list[dict], dict, dict]:
    """Return (joined_rows, raw_systems_by_id, raw_stats)."""
    systems = (
        json.loads(systems_json.read_text()) if systems_json else _fetch_json(LIST_URL)
    )
    stats = (
        json.loads(stats_json.read_text()) if stats_json else _fetch_json(STATS_URL)
    )
    by_id = {int(s["id"]): s for s in systems}

    rows = []
    for sid, s in stats.items():
        sid = int(sid)
        meta = by_id.get(sid, {})
        rows.append(
            {
                "id": sid,
                "hp_type": meta.get("hp_type"),
                "combined_cop": s.get("combined_cop"),
                "space_cop": s.get("space_cop"),
                "water_cop": s.get("water_cop"),
                "outsideT": s.get("combined_outsideT_mean"),
                "flowT": s.get("combined_flowT_mean"),
                "data_seconds": s.get("combined_data_length") or 0,
            }
        )
    return rows, by_id, stats


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _summ(vals: list[float], weights: Optional[list[float]] = None) -> dict:
    a = np.asarray(vals, dtype=float)
    out = {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
    }
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        out["weighted_mean"] = float(np.average(a, weights=w))
    return out


def _valid(rows, key, min_seconds):
    """Rows with a sane COP in (0, 8) and enough coverage."""
    out = []
    for r in rows:
        v = r.get(key)
        if v is None or not (0 < v < 8):
            continue
        if r["data_seconds"] < min_seconds:
            continue
        out.append(r)
    return out


def carnot_curve(rows, min_seconds) -> dict:
    """Constant-Carnot-efficiency fit -> COP at the model's curve temps."""
    eta, flowKs, used = [], [], 0
    for r in rows:
        cop, t_out, t_flow = r["combined_cop"], r["outsideT"], r["flowT"]
        if cop is None or t_out is None or t_flow is None:
            continue
        if not (0 < cop < 8) or r["data_seconds"] < min_seconds:
            continue
        dt = t_flow - t_out
        if dt <= 1.0:  # guard tiny/negative lifts
            continue
        t_flow_k = t_flow + 273.15
        eta.append(cop * dt / t_flow_k)
        flowKs.append(t_flow_k)
        used += 1

    eta_med = float(np.median(eta))
    flow_med_c = float(np.median(flowKs)) - 273.15
    flow_k = flow_med_c + 273.15
    curve = []
    for t in CURVE_TEMPS_C:
        dt = max(flow_k - (t + 273.15), 1.0)
        curve.append([t, round(eta_med * flow_k / dt, 2)])
    return {
        "cop_curve": curve,
        "carnot_efficiency_median": round(eta_med, 4),
        "assumed_flow_temp_C": round(flow_med_c, 1),
        "n_systems": used,
    }


def binned_curve(rows, min_seconds) -> list:
    """Raw median combined COP per 1 C outdoor-temp bin (sanity check)."""
    buckets: dict[int, list[float]] = {}
    for r in _valid(rows, "combined_cop", min_seconds):
        if r["outsideT"] is None:
            continue
        b = int(round(r["outsideT"]))
        buckets.setdefault(b, []).append(r["combined_cop"])
    return [
        [b, round(float(np.median(v)), 2), len(v)]
        for b, v in sorted(buckets.items())
        if len(v) >= 3
    ]


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def fit(rows, min_days: int) -> dict:
    min_seconds = min_days * 86400

    space = _valid(rows, "space_cop", min_seconds)
    comb = _valid(rows, "combined_cop", min_seconds)
    water = _valid(rows, "water_cop", min_seconds)

    space_summ = _summ(
        [r["space_cop"] for r in space],
        weights=[r["data_seconds"] for r in space],
    )
    comb_summ = _summ(
        [r["combined_cop"] for r in comb],
        weights=[r["data_seconds"] for r in comb],
    )
    water_summ = _summ([r["water_cop"] for r in water])

    by_type = {}
    for t in sorted({r["hp_type"] for r in space if r["hp_type"]}):
        vals = [r["space_cop"] for r in space if r["hp_type"] == t]
        if len(vals) >= 3:
            by_type[t] = _summ(vals)

    cc = carnot_curve(rows, min_seconds)

    return {
        # ---- model-bound anchors ----------------------------------------
        "heatpump_cop_ref": round(space_summ["median"], 2),  # well-installed
        "cop_field_median_space": round(space_summ["median"], 2),
        "cop_field_median_combined": round(comb_summ["median"], 2),
        "cop_representative": EOH_REPRESENTATIVE_COP,  # conservative default
        "cop_curve": cc["cop_curve"],
        # ---- provenance / caveats ---------------------------------------
        "provenance": {
            "source": "heatpumpmonitor.org (OpenEnergyMonitor) public API",
            "endpoints": {"systems": LIST_URL, "stats": STATS_URL},
            "window": "last 365 days",
            "min_coverage_days": min_days,
            "sample_bias": (
                "Self-selected, well-commissioned owner-enthusiast systems; "
                "biased ABOVE the representative installed base. Use "
                "cop_representative for a conservative default and "
                "cop_field_median_* for a well-installed scenario."
            ),
            "maps_to_model": (
                "hp_effect_mult = boiler_efficiency / heatpump_cop_ref "
                "scales the SPACE-heating slope, so the anchor uses space COP."
            ),
            "representative_citation": EOH_CITATION,
        },
        # ---- diagnostics -------------------------------------------------
        "diagnostics": {
            "space_cop": space_summ,
            "combined_cop": comb_summ,
            "water_cop": water_summ,
            "by_hp_type_space_cop": by_type,
            "carnot_curve": cc,
            "binned_combined_cop_vs_outdoorT": binned_curve(rows, min_seconds),
            "n_systems_in_stats": len(rows),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--systems-json", type=Path, default=None,
                   help="Cached public.json (offline). Default: fetch live.")
    p.add_argument("--stats-json", type=Path, default=None,
                   help="Cached stats/last365 (offline). Default: fetch live.")
    p.add_argument("--min-days", type=int, default=90,
                   help="Minimum monitoring coverage per system (default 90).")
    p.add_argument("--output", type=Path,
                   default=Path("results/field_fits/heatpump_cop.yaml"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        rows, _, _ = load_data(args.systems_json, args.stats_json)
    except Exception as e:  # network or parse failure
        print(f"ERROR: could not load heatpumpmonitor data: {e}", file=sys.stderr)
        return 2

    result = fit(rows, min_days=args.min_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    if not args.quiet:
        d = result["diagnostics"]
        print(f"heatpumpmonitor.org — {d['n_systems_in_stats']} systems in stats, "
              f">={args.min_days}d coverage:\n")
        for label, key in [("space", "space_cop"), ("combined", "combined_cop"),
                           ("water (DHW)", "water_cop")]:
            s = d[key]
            print(f"  {label:13s} n={s['n']:3d}  median={s['median']:.2f}  "
                  f"mean={s['mean']:.2f}  p10-p90={s['p10']:.2f}-{s['p90']:.2f}")
        print("\n  by type (space COP):")
        for t, s in d["by_hp_type_space_cop"].items():
            print(f"    {t:14s} n={s['n']:3d}  median={s['median']:.2f}")
        cc = d["carnot_curve"]
        print(f"\n  Carnot curve (eta={cc['carnot_efficiency_median']}, "
              f"flow={cc['assumed_flow_temp_C']}C): {cc['cop_curve']}")
        print(f"\n  ANCHORS: field space-COP median={result['heatpump_cop_ref']} "
              f"(well-installed) | representative={result['cop_representative']} (EoH)")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
