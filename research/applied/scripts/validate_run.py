#!/usr/bin/env python3
"""
Per-(city, year) validation run.

Loads city stock + applies calibrated parameters + runs EnergyModel for a full
year → aggregates to LSOA → compares against DESNZ totals → writes metrics
and per-LSOA residuals to disk.

Used by tasks #3 sanity check, #7 (Sunderland), #8 (Waltham Forest),
#9 (Newcastle 2024 temporal), #15 (Manchester, Cornwall once data lands).

Usage:
  python research/applied/scripts/validate_run.py \
      --city newcastle \
      --year 2023 \
      --params research/applied/results/calibration/2020_2022/params.yaml \
      [--kind transfer|temporal]      # output subdir; default 'transfer'
      [--window-hours 8760]
      [--collect-agent-level]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[3]))

from utils import (  # noqa: E402
    cache_dir, climate_path, load_city_stock, load_desnz,
    load_params, run_model, aggregate_to_lsoa,
    compute_validation_metrics, save_validation_result,
    CITY_CONVENTIONS,
)
from progress import ProgressTracker  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", required=True,
                   choices=sorted(CITY_CONVENTIONS.keys()),
                   help="City to validate against")
    p.add_argument("--year", type=int, required=True,
                   help="Target year (climate forcing + DESNZ ground truth)")
    p.add_argument("--params", type=Path, required=True,
                   help="Path to calibrated params.yaml (from calibrate_serl.py)")
    p.add_argument("--kind", default="transfer",
                   choices=["transfer", "temporal", "sanity"],
                   help="Output category subdir (default: transfer)")
    p.add_argument("--window-hours", type=int, default=8760,
                   help="Model run length in hours (default 8760 = full year)")
    p.add_argument("--collect-agent-level", action="store_true",
                   help="Enable Mesa agent-level datacollector (default off)")
    p.add_argument("--checkpoint-every-hours", type=int, default=720,
                   help="Pickle the model every N hours so a crash can resume "
                        "(default 720h ≈ 30 days simulated; set 0 to disable)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start fresh")
    args = p.parse_args()

    out_dir = cache_dir(args.kind, f"{args.city}_{args.year}")
    tracker = ProgressTracker(out_dir,
                                job_name=f"validate_{args.city}_{args.year}_{args.kind}")
    tracker.start(f"city={args.city} year={args.year} window={args.window_hours}h")
    tracker.milestone(f"params_file={args.params}")
    tracker.milestone(f"output_dir={out_dir}")

    # ── Load inputs ─────────────────────────────────────────────────────────
    with tracker.section("load city stock"):
        stock = load_city_stock(args.city)
    tracker.milestone("stock loaded",
                       n_dwellings=len(stock),
                       n_lsoas=stock["lsoa_code"].nunique() if "lsoa_code" in stock.columns else 0)

    with tracker.section("load DESNZ"):
        desnz = load_desnz(args.city, args.year)
    tracker.milestone("DESNZ loaded", n_lsoas=len(desnz))

    with tracker.section("load params"):
        params = load_params(args.params)

    # ── Run the model ────────────────────────────────────────────────────────
    with tracker.section(f"run model ({args.window_hours}h)"):
        abm = run_model(
            stock,
            params,
            year=args.year,
            window_hours=args.window_hours,
            climate_path_override=climate_path(args.city),
            collect_agent_level=args.collect_agent_level,
            tracker=tracker,
            checkpoint_dir=(out_dir if args.checkpoint_every_hours > 0 else None),
            checkpoint_every_hours=max(1, args.checkpoint_every_hours),
            resume=not args.no_resume,
        )
    tracker.milestone("model finished",
                       n_dwellings_out=len(abm),
                       total_elec_GWh=f"{abm['elec_kwh'].sum()/1e6:.2f}",
                       total_gas_GWh=f"{abm['gas_kwh'].sum()/1e6:.2f}")

    # ── Aggregate + compare ──────────────────────────────────────────────────
    with tracker.section("aggregate to LSOA + metrics"):
        abm_lsoa = aggregate_to_lsoa(abm)
        metrics = compute_validation_metrics(abm_lsoa, desnz)

    metrics["city"] = args.city
    metrics["year"] = args.year
    metrics["window_hours"] = args.window_hours

    tracker.milestone(
        "validation",
        mape_elec=f"{metrics['mape_elec']:.2f}%",
        mape_gas=f"{metrics['mape_gas']:.2f}%",
        mape_total=f"{metrics['mape_total']:.2f}%",
        bias_total=f"{metrics['bias_total']:+.2f}%",
        n_lsoas=metrics["n_lsoas"],
    )

    save_validation_result(metrics, out_dir)
    tracker.finish(f"wrote metrics.json + residuals.parquet to {out_dir}")


if __name__ == "__main__":
    main()
