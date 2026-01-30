"""
Quick enrichment/schedule sanity check.

Usage:
    python notebooks/enrichment_check.py --results results_smoke_epc

Expects `agent_timeseries.parquet` in the results folder (run without
`--no-agent-level`). Falls back gracefully if file is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Check enrichment coverage and schedules.")
    ap.add_argument("--results", default="results", help="Path to run output directory.")
    args = ap.parse_args()

    res_dir = Path(args.results)
    agent_path = res_dir / "agent_timeseries.parquet"
    if not agent_path.exists():
        print(f"⚠️  {agent_path} not found. Rerun without --no-agent-level to produce it.")
        return

    df = pd.read_parquet(agent_path)

    # household-only view
    hh = df[df["agent_type"] == "household"]
    if hh.empty:
        print("⚠️  No household rows found in agent_timeseries.parquet.")
        return

    total = len(hh)
    def pct(col):
        return 100 * hh[col].notna().mean()

    print(f"Households in agent file: {total:,}")
    print(f"hidp coverage        : {pct('hidp'):.1f}%")
    print(f"hh_n_people coverage : {pct('hh_n_people'):.1f}%")
    print(f"schedule_type cov    : {pct('schedule_type'):.1f}%")
    print(f"dwelling_bucket cov  : {pct('dwelling_bucket'):.1f}%")
    print(f"tenure coverage      : {pct('tenure'):.1f}%")
    print(f"income_band coverage : {pct('hh_income_band'):.1f}%")
    print(f"education coverage   : {pct('hh_edu_detail'):.1f}%")

    print("\nTop schedule_type:")
    print(hh["schedule_type"].value_counts(dropna=False).head(10))

    print("\nTop schedule_profile (post-assignment):")
    print(hh.get("schedule_profile", pd.Series(dtype=object)).value_counts(dropna=False).head(10))

    # People-level schedules
    ppl = df[df["agent_type"] == "person"]
    if not ppl.empty:
        print("\nPerson schedule_profile counts:")
        print(ppl["schedule_profile"].value_counts(dropna=False).head(12))


if __name__ == "__main__":
    main()
