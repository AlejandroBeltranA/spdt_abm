#!/usr/bin/env python3
"""
Bootstrap calibration → N parameter draws.

Method: row-level bootstrap on the SERL aggregates table.

For each draw i in 1..N:
  1. Resample (with replacement) all rows of `daily_targets` filtered to the
     calibration years.
  2. Re-run the calibration on the resampled set.
  3. Record (anchors, slopes, R², n_points) as one row.

The output `draws.parquet` is the parameter distribution that feeds Monte
Carlo propagation (task #11): the MC sampler draws one row per replication
and uses it as the calibration input.

Method note (for the methods section):
  Row-level bootstrap on the aggregated SERL targets table captures the
  sampling variance of the calibration set. With three years pooled, this
  also propagates year-to-year variance (resamples will sometimes
  over/under-represent each year). It does *not* capture within-cell
  SERL respondent-level variance — that would require the SERL microdata,
  which is not in scope for this paper.

Usage:
  python research/applied/scripts/bootstrap_serl.py \
      --years 2020 2021 2022 \
      --n-draws 200 \
      --seed 7 \
      --label 2020_2022
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[3]))

from utils import repo_root, cache_dir  # noqa: E402
from progress import ProgressTracker  # noqa: E402
from calibrate_serl import calibrate, DEFAULT_MIN_N  # noqa: E402
from household_energy.serl_calibration_v2 import load_serl_targets  # noqa: E402


def _flatten_draw(cal: dict, draw_id: int) -> dict:
    """Extract a one-row summary of a calibration result for the draws table."""
    diag = cal["fit_diagnostics"]
    return {
        "draw":           int(draw_id),
        "gas_anchor":     cal["anchors"]["gas"],
        "elec_anchor":    cal["anchors"]["electric"],
        "gas_slope":      cal["hdd_slopes"]["gas"],
        "elec_slope":     cal["hdd_slopes"]["electric"],
        "gas_r2":         diag["gas"]["r_squared"],
        "elec_r2":        diag["electric"]["r_squared"],
        "gas_n_points":   diag["gas"]["n_points"],
        "elec_n_points":  diag["electric"]["n_points"],
        "n_btype_mults_gas":  len(cal["building_type_multipliers"]["gas"]),
        "n_btype_mults_elec": len(cal["building_type_multipliers"]["electric"]),
    }


def bootstrap(
    daily_targets: pd.DataFrame,
    *,
    years: list[int],
    n_draws: int,
    seed: int = 7,
    min_n: int = DEFAULT_MIN_N,
    tracker: ProgressTracker | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run a row-level bootstrap of the multi-year calibration.

    Returns:
      draws_df: one row per successful draw (flattened summary)
      btype_records: list of dicts {draw, fuel, building_type, multiplier}
                     (kept separately so the draws table stays flat)
    """
    rng = np.random.default_rng(seed)
    base = daily_targets[
        daily_targets["year"].astype(int).isin([int(y) for y in years])
    ].copy().reset_index(drop=True)
    n_base = len(base)
    if n_base == 0:
        raise ValueError(f"No rows for years={years} in daily_targets.")

    if tracker is not None:
        tracker.milestone("bootstrap setup",
                           base_rows=n_base, n_draws=n_draws, seed=seed)

    rows: list[dict] = []
    btype_records: list[dict] = []
    n_skipped = 0
    iterator = range(n_draws)
    if tracker is not None:
        iterator = tracker.iter(iterator, desc="bootstrap")

    for i in iterator:
        idx = rng.integers(0, n_base, n_base)
        resampled = base.iloc[idx].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal = calibrate(resampled, years=years, min_n=min_n)
        except Exception as exc:
            n_skipped += 1
            if tracker is not None:
                tracker.warn(f"draw {i} failed: {type(exc).__name__}: {exc}")
            continue

        rows.append(_flatten_draw(cal, draw_id=i))
        for fuel, mults in cal["building_type_multipliers"].items():
            for btype, m in mults.items():
                btype_records.append({"draw": i, "fuel": fuel,
                                      "building_type": btype, "multiplier": m})

    if tracker is not None and n_skipped > 0:
        tracker.warn(f"{n_skipped}/{n_draws} draws failed and were skipped")

    return pd.DataFrame(rows), btype_records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022])
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    p.add_argument("--label", type=str, default=None,
                   help="Output subdir label (default: '<min>_<max>' from --years)")
    args = p.parse_args()

    years = sorted(set(args.years))
    label = args.label or f"{min(years)}_{max(years)}"
    out_dir = cache_dir("bootstrap", label)

    tracker = ProgressTracker(out_dir,
                                job_name=f"bootstrap_serl_{label}",
                                total=args.n_draws,
                                heartbeat_every=20)
    tracker.start(f"years={years} n_draws={args.n_draws} seed={args.seed}")

    with tracker.section("load SERL targets"):
        daily_targets, _ = load_serl_targets(repo_root())

    with tracker.section("bootstrap calibration"):
        draws_df, btype_records = bootstrap(
            daily_targets,
            years=years,
            n_draws=args.n_draws,
            seed=args.seed,
            min_n=args.min_n,
            tracker=tracker,
        )

    # ── persist ──
    draws_df.to_parquet(out_dir / "draws.parquet", index=False)
    pd.DataFrame(btype_records).to_parquet(out_dir / "building_type_multipliers.parquet",
                                            index=False)

    # Summary stats
    def _qsum(s: pd.Series) -> dict:
        s = s.dropna()
        return {
            "mean": float(s.mean()),
            "std":  float(s.std(ddof=1)),
            "p2.5":  float(s.quantile(0.025)),
            "p50":   float(s.quantile(0.50)),
            "p97.5": float(s.quantile(0.975)),
            "n":    int(len(s)),
        }

    summary = {
        col: _qsum(draws_df[col])
        for col in ["gas_anchor", "elec_anchor", "gas_slope", "elec_slope",
                    "gas_r2", "elec_r2"]
    }
    summary["meta"] = {
        "years":        years,
        "n_draws_requested": args.n_draws,
        "n_draws_succeeded": int(len(draws_df)),
        "seed":         args.seed,
        "min_n":        args.min_n,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    tracker.milestone(
        "summary",
        gas_anchor_ci=f"[{summary['gas_anchor']['p2.5']:.4f}, {summary['gas_anchor']['p97.5']:.4f}]",
        elec_anchor_ci=f"[{summary['elec_anchor']['p2.5']:.4f}, {summary['elec_anchor']['p97.5']:.4f}]",
        gas_slope_ci=f"[{summary['gas_slope']['p2.5']:.4f}, {summary['gas_slope']['p97.5']:.4f}]",
        elec_slope_ci=f"[{summary['elec_slope']['p2.5']:.4f}, {summary['elec_slope']['p97.5']:.4f}]",
    )
    tracker.finish(
        f"wrote draws.parquet ({len(draws_df)} rows), "
        f"building_type_multipliers.parquet, summary.json to {out_dir}"
    )


if __name__ == "__main__":
    main()
