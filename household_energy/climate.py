# climate.py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Optional, Literal

class ClimateField:
    """
    Holds hourly 2m temps as a 2D array [T, P], timestamps [T],
    and maps households to nearest climate point.
    """

    # ─────────────────────────────────────────────────────────────
    # Public validator: quick T×P sanity check (no array building)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def validate_parquet(
        parquet_path: str,
        *,
        verbose: bool = True,
        allow_duplicates: bool = True,
        return_summary: bool = False,
    ):
        """
        Inspect a climate parquet and report whether it is a full T×P grid.

        Parameters
        ----------
        parquet_path : str
            Parquet file to check. Must contain columns:
            ['timestamp','latitude','longitude','temp_C'].
        verbose : bool
            Print a short human-readable report (default: True).
        allow_duplicates : bool
            If True, drop exact duplicate (timestamp, lat, lon) rows before checking.
        return_summary : bool
            If True, return a dict with detailed stats.

        Returns
        -------
        is_rectangular : bool  (if return_summary=False)
        OR
        summary : dict         (if return_summary=True) with keys:
            rows, T, P, TP, dupes, missing_timestamps, min_per_ts, med_per_ts, max_per_ts
        """
        cols = ["timestamp", "latitude", "longitude", "temp_C"]
        df = pd.read_parquet(parquet_path, engine="pyarrow")[cols].copy()

        # basic hygiene
        if allow_duplicates:
            dupes = int(df.duplicated(["timestamp", "latitude", "longitude"]).sum())
            if dupes:
                df = df.drop_duplicates(["timestamp", "latitude", "longitude"], keep="first")
        else:
            dupes = int(df.duplicated(["timestamp", "latitude", "longitude"]).sum())

        # compute T, P and per-timestamp counts
        df = df.sort_values(["timestamp", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)
        T  = df["timestamp"].nunique()
        P  = df[["latitude","longitude"]].drop_duplicates().shape[0]
        per_ts = df.groupby("timestamp").size()
        min_per_ts = int(per_ts.min())
        med_per_ts = int(per_ts.median())
        max_per_ts = int(per_ts.max())
        gaps = int((per_ts != P).sum())
        is_rect = (len(df) == T * P) and (gaps == 0)

        if verbose:
            print(f"[Climate parquet check]")
            print(f"  File: {parquet_path}")
            print(f"  Rows: {len(df):,} | T: {T:,} | P: {P:,} | T×P: {T*P:,}")
            print(f"  Duplicates dropped: {dupes}")
            print(f"  Rows per timestamp → min/median/max: {min_per_ts}/{med_per_ts}/{max_per_ts}")
            if is_rect:
                print("  ✅ Rectangular grid (full T×P).")
            else:
                print(f"  ❌ Not rectangular. Timestamps with missing points: {gaps:,}")
                # small hint: show a few problematic timestamps
                bad = per_ts[per_ts != P]
                example = bad.head(5)
                if not example.empty:
                    print("  Examples (timestamp → row-count):")
                    for ts, n in example.items():
                        print(f"    • {ts} → {int(n)}")

        if return_summary:
            return dict(
                rows=len(df), T=int(T), P=int(P), TP=int(T*P),
                dupes=dupes, missing_timestamps=gaps,
                min_per_ts=min_per_ts, med_per_ts=med_per_ts, max_per_ts=max_per_ts,
                rectangular=is_rect,
            )
        return is_rect
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path, engine="pyarrow")[
            ["timestamp", "latitude", "longitude", "temp_C"]
        ]
        # ✅ normalise timestamps to UTC first
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(["timestamp", "latitude", "longitude"], kind="mergesort")

        # Drop exact dupes (safe) to prevent false negatives
        if df.duplicated(["timestamp","latitude","longitude"]).any():
            df = df.drop_duplicates(["timestamp","latitude","longitude"], keep="first")

        points = df[["latitude","longitude"]].drop_duplicates().to_numpy(dtype=np.float32)

        # ✅ store as tz-naive numpy datetime64[ns] representing UTC instants
        times  = df["timestamp"].drop_duplicates().to_numpy(dtype="datetime64[ns]")

        T, P   = len(times), len(points)
        assert len(df) == T * P, (
            "Climate file is not a full T×P grid.\n"
            "Hint: run ClimateField.validate_parquet('your_file.parquet') to see gaps."
        )

        temps = df["temp_C"].to_numpy(dtype=np.float32).reshape(T, P)
        self.points = points
        self.times  = times              # dtype datetime64[ns], UTC-normalised
        self.temps  = temps
        self._tree  = cKDTree(points[:, [0,1]])

    def map_households(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Return nearest climate point index for each (lat, lon)."""
        _, idx = self._tree.query(np.c_[lats.astype(np.float32), lons.astype(np.float32)], k=1)
        return idx.astype(np.int32)

    def time_index_for(self, start_ts) -> int:
        """Index in self.times for a given timestamp (>=), robust to tz-aware inputs."""
        ts = pd.to_datetime(start_ts, utc=True).to_datetime64()  # -> datetime64[ns] UTC
        return int(np.searchsorted(self.times, ts, side="left"))

    def temps_at_index(self, t: int) -> np.ndarray:
        """Return temps for all P points at time index t  (shape [P])."""
        return self.temps[t, :]


# ─────────────────────────────────────────────────────────────────────
# Optional CLI: python -m climate /path/to/file.parquet
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Validate climate parquet T×P rectangularity.")
    p.add_argument("parquet", help="Path to climate parquet")
    p.add_argument("--quiet", action="store_true", help="Do not print report; exit code only")
    args = p.parse_args()

    ok = ClimateField.validate_parquet(args.parquet, verbose=not args.quiet)
    raise SystemExit(0 if ok else 1)
