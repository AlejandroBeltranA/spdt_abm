"""Run the live decomposition on a deterministic representative LSOA sample.

Selects a fixed (seed=42) sample of Newcastle LSOAs whose pooled electric-heated
share tracks the city (~11.5%), runs the full-year live decomposition, and saves
a per-dwelling CSV for the transparency notebook (Goal 1).

    .venv/bin/python research/applied/scripts/run_decomp_sample.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from decompose_demand import run_decomposition, summarize

YEAR = 2023
N_LSOA = 30
SEED = 42
OUT = "results_lsoa/decomp_sample_newcastle_2023.csv"
ABM = "results_lsoa/transfer_newcastle/abm_year_all_newcastle_2023.csv"


def pick_sample() -> list[str]:
    abm = pd.read_csv(ABM)
    abm["lsoa_code"] = abm["lsoa_code"].astype(str)
    codes = abm["lsoa_code"].tolist()
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(codes, size=N_LSOA, replace=False).tolist())
    sub = abm[abm["lsoa_code"].isin(sample)]
    share = sub["run_electric_heated_dwellings"].sum() / sub["run_dwellings"].sum()
    print(f"Sample: {N_LSOA} LSOAs, {sub['run_dwellings'].sum()} dwellings, "
          f"electric-heated share={share:.4f} (city=0.1151)")
    return sample


def main() -> None:
    sample = pick_sample()
    t0 = time.time()
    df = run_decomposition(lsoa_codes=sample, year=YEAR, progress_every=2000)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} dwellings -> {OUT}  ({time.time()-t0:.0f}s)")
    print("\nPer-dwelling means by heating bucket (kWh/yr):")
    print(summarize(df).round(0).to_string())


if __name__ == "__main__":
    main()
