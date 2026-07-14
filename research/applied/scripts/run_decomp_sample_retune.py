"""Live confirmation of the joint electricity retune on the 30-LSOA sample:
anchor recentring corrected + electric heating slope calibrated to SERL cohorts.
"""
from __future__ import annotations
import time
from decompose_demand import run_decomposition, summarize
from run_decomp_sample import pick_sample, YEAR

OUT = "results_lsoa/decomp_sample_newcastle_2023_retune.csv"
CFG = "results_lsoa/_tmp_config_retune.yaml"


def main() -> None:
    sample = pick_sample()
    t0 = time.time()
    df = run_decomposition(lsoa_codes=sample, year=YEAR, config_path=CFG, progress_every=2000)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} dwellings -> {OUT}  ({time.time()-t0:.0f}s)")
    print("\nRETUNE per-dwelling means by heating bucket (kWh/yr):")
    print(summarize(df).round(0).to_string())


if __name__ == "__main__":
    main()
