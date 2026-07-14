"""
Shared utilities for the Paper 1 (Applied Energy) methodology pipeline.

Every CLI script in research/applied/scripts/ imports from here.
Notebooks in research/applied/notebooks/ load cached results — they do not
re-run the model.

Conventions (see research/applied/PROGRESS.md for design rationale):
- EPC stock files:  data/epc_abm_{epc_slug}.geojson
- HIDP synthpop:    data/{epc_slug}_hidp_uprn_matches_tiered.csv if it exists,
                    else data/hidp_uprn_matches_tiered.csv (national)
- Climate forcing:  data/{climate_prefix}_2t_timeseries_2010_2026.parquet
- DESNZ totals:     data/LSOA_domestic_{fuel}_2010-2024.xlsx (preferred)
                    or  data/LSOA_domestic_{fuel}_2010-2023.xlsx (fallback)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Path discovery
# ─────────────────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    """Locate the spdt_abm repository root by walking up from this file."""
    p = Path(__file__).resolve()
    for ancestor in [p] + list(p.parents):
        if (ancestor / "household_energy").is_dir() and (ancestor / "data").is_dir():
            return ancestor
    raise FileNotFoundError("Could not locate spdt_abm root from " + str(p))


def results_root() -> Path:
    return repo_root() / "research" / "applied" / "results"


def epc_stock_path(city: str) -> Path:
    """Resolve a city's EPC stock file, the single contract every entry point
    shares. Prefer ``.geojson`` (legacy main-repo convention); fall back to
    ``.gpkg`` (epc-to-abm sister-repo native output). ``gpd.read_file`` reads
    both transparently. Raises if neither exists."""
    conv = city_convention(city)
    data = repo_root() / "data"
    geojson = data / f"epc_abm_{conv.epc_slug}.geojson"
    gpkg = data / f"epc_abm_{conv.epc_slug}.gpkg"
    if geojson.exists():
        return geojson
    if gpkg.exists():
        return gpkg
    raise FileNotFoundError(
        f"EPC stock not found for '{city}': tried {geojson} and {gpkg}"
    )


def hidp_path_for(city: str) -> Optional[Path]:
    """Resolve a city's HIDP synthpop file: city-specific if present, else the
    national file, else None (callers run EPC-only)."""
    conv = city_convention(city)
    data = repo_root() / "data"
    candidates = [
        data / f"{conv.epc_slug}_hidp_uprn_matches_tiered.csv",
        data / "wf_hidp_uprn_matches_tiered.csv" if conv.epc_slug == "waltham_forest" else None,
        data / "hidp_uprn_matches_tiered.csv",
    ]
    return next((c for c in candidates if c is not None and c.exists()), None)


# ─────────────────────────────────────────────────────────────────────────────
# City conventions
#
# To add a new city, append one entry. Everything downstream picks it up
# automatically.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CityConvention:
    epc_slug: str          # filename slug used in data/epc_abm_{slug}.geojson
    climate_prefix: str    # filename prefix used in data/{prefix}_2t_timeseries_…
    desnz_la: str          # local-authority string as it appears in DESNZ xlsx


CITY_CONVENTIONS: dict[str, CityConvention] = {
    "newcastle":      CityConvention("newcastle",      "ncc",            "Newcastle upon Tyne"),
    "sunderland":     CityConvention("sunderland",     "ncc",            "Sunderland"),
    "waltham_forest": CityConvention("waltham_forest", "waltham_forest", "Waltham Forest"),
    # Scaffolded — entries live so `transfer.py --city manchester` resolves
    # cleanly once these three files land in data/:
    #   data/epc_abm_manchester.geojson         (from epc-to-abm pipeline)
    #   data/manchester_2t_timeseries_2010_2026.parquet
    #   data/manchester_hidp_uprn_matches_tiered.csv  (or fall back to national)
    "manchester":     CityConvention("manchester",        "manchester",     "Manchester"),
    "brighton":       CityConvention("brighton_and_hove", "brighton",       "Brighton and Hove"),
    # Cornwall scaffolded for whenever climate-to-abm regenerates a valid
    # parquet (the current sister-repo cornwall_era5_land_2t_2010_2026.parquet
    # is corrupt — Parquet magic bytes not found in footer).
    "cornwall":       CityConvention("cornwall",          "cornwall",       "Cornwall"),
}


def city_convention(city: str) -> CityConvention:
    if city not in CITY_CONVENTIONS:
        raise KeyError(
            f"Unknown city '{city}'. Add an entry to CITY_CONVENTIONS in "
            "research/applied/scripts/utils.py."
        )
    return CITY_CONVENTIONS[city]


# ─────────────────────────────────────────────────────────────────────────────
# City stock loader (EPC + HIDP merge)
# ─────────────────────────────────────────────────────────────────────────────

def load_city_stock(city: str, hidp_path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load EPC stock joined with HIDP synthpop attributes for `city`.

    Default HIDP source: city-specific file if it exists
    (e.g. `data/wf_hidp_uprn_matches_tiered.csv` for Waltham Forest),
    otherwise the national file `data/hidp_uprn_matches_tiered.csv`.
    """
    geo_path = epc_stock_path(city)
    if hidp_path is None:
        hidp_path = hidp_path_for(city)

    g = gpd.read_file(geo_path)
    g["UPRN"] = g["UPRN"].astype(str).str.strip()

    if hidp_path and hidp_path.exists():
        hidp = pd.read_csv(hidp_path, low_memory=False)
        hidp.columns = [c.strip() for c in hidp.columns]
        hidp["uprn_chr"] = hidp["uprn_chr"].astype(str).str.strip()
        # Drop HIDP columns that already exist in EPC so the merge doesn't
        # clobber EPC values into <name>_x / <name>_y suffixes. EPC is the
        # authoritative source for dwelling-attached fields like lsoa_code,
        # local_authority, ward_code, property_type — HIDP carries household
        # demographics keyed by UPRN.
        epc_cols = set(g.columns)
        drop_from_hidp = [c for c in hidp.columns
                          if c in epc_cols and c not in ("UPRN", "uprn_chr")]
        if drop_from_hidp:
            hidp = hidp.drop(columns=drop_from_hidp)
        g = g.merge(hidp, left_on="UPRN", right_on="uprn_chr", how="left")

    return g


def climate_path(city: str) -> Path:
    """Return the climate timeseries parquet for a city."""
    conv = city_convention(city)
    return repo_root() / "data" / f"{conv.climate_prefix}_2t_timeseries_2010_2026.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# DESNZ LSOA totals
# ─────────────────────────────────────────────────────────────────────────────

_DESNZ_COL_MAP = {
    "Local authority code":           "la_code",
    "Local authority":                "local_authority",
    "MSOA code":                      "msoa_code",
    "Middle layer super output area": "msoa_name",
    "LSOA code":                      "lsoa_code",
    "Lower layer super output area":  "lsoa_name",
    "Number of meters":               "meters",
    "Total consumption (kWh)":        "total_kwh",
}


@lru_cache(maxsize=8)
def _read_desnz_book(path_str: str, mtime_ns: int, fuel: str) -> pd.DataFrame:
    """Parse + tidy *all* year sheets of a DESNZ LSOA workbook (header on row 5,
    one sheet per year). Cached by (path, mtime) so repeated per-year/per-city
    validation reuses one parse instead of re-reading the whole workbook each
    call. ``mtime_ns`` is a cache key only — it invalidates the entry if the
    file changes on disk."""
    book = pd.read_excel(Path(path_str), sheet_name=None, header=4)
    out_frames = []
    for sheet_name, df in book.items():
        m = re.search(r"(\d{4})", str(sheet_name))
        if m is None:
            continue
        yr = int(m.group(1))
        df = df.copy()
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
        df = df.rename(columns=_DESNZ_COL_MAP)
        keep = [c for c in ["la_code", "local_authority", "lsoa_code",
                            "meters", "total_kwh"] if c in df.columns]
        df = df[keep].dropna(subset=["lsoa_code"])
        df["year"] = yr
        df["fuel"] = fuel
        out_frames.append(df)
    if not out_frames:
        return pd.DataFrame(columns=["year", "fuel", "lsoa_code", "local_authority",
                                     "meters", "total_kwh"])
    return pd.concat(out_frames, ignore_index=True)


def _tidy_lsoa_desnz(xlsx_path: Path, fuel: str, years: list[int]) -> pd.DataFrame:
    """Years-filtered view over the cached full-workbook parse."""
    full = _read_desnz_book(str(xlsx_path), xlsx_path.stat().st_mtime_ns, fuel)
    if not years:
        return full.copy()
    return full[full["year"].isin(years)].reset_index(drop=True)


def desnz_xlsx_paths() -> tuple[Path, Path]:
    """Return (elec_xlsx, gas_xlsx), preferring the 2010-2024 versions."""
    root = repo_root()
    elec_2024 = root / "data" / "LSOA_domestic_elec_2010-2024.xlsx"
    gas_2024  = root / "data" / "LSOA_domestic_gas_2010-2024.xlsx"
    elec_2023 = root / "data" / "LSOA_domestic_elec_2010-2023.xlsx"
    gas_2023  = root / "data" / "LSOA_domestic_gas_2010-2023.xlsx"
    elec = elec_2024 if elec_2024.exists() else elec_2023
    gas  = gas_2024  if gas_2024.exists()  else gas_2023
    return elec, gas


def load_desnz(city: str, year: int) -> pd.DataFrame:
    """Load DESNZ LSOA-level kWh totals for `city` and `year`.

    Returns a DataFrame with one row per LSOA and columns:
      lsoa_code, local_authority,
      meters_elec, total_kwh_elec,
      meters_gas,  total_kwh_gas
    """
    conv = city_convention(city)
    elec_xlsx, gas_xlsx = desnz_xlsx_paths()
    elec = _tidy_lsoa_desnz(elec_xlsx, "elec", [year])
    gas  = _tidy_lsoa_desnz(gas_xlsx,  "gas",  [year])

    la_lower = conv.desnz_la.lower()
    elec = elec[elec["local_authority"].astype(str).str.lower() == la_lower]
    gas  = gas [gas ["local_authority"].astype(str).str.lower() == la_lower]

    out = elec[["lsoa_code", "local_authority", "meters", "total_kwh"]].rename(
        columns={"meters": "meters_elec", "total_kwh": "total_kwh_elec"}
    ).merge(
        gas[["lsoa_code", "meters", "total_kwh"]].rename(
            columns={"meters": "meters_gas", "total_kwh": "total_kwh_gas"}
        ),
        on="lsoa_code", how="outer",
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Calibration parameter I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_params(path: str | Path) -> dict:
    """Load a parameter file (.yaml/.yml/.json)."""
    path = Path(path)
    with open(path) as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if path.suffix == ".json":
            return json.load(f)
    raise ValueError(f"Unknown parameter file format: {path.suffix}")


def save_params(params: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if path.suffix in {".yaml", ".yml"}:
            yaml.safe_dump(params, f, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(params, f, indent=2, default=float)
        else:
            raise ValueError(f"Unknown parameter file format: {path.suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation metrics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_lsoa(abm_per_dwelling: pd.DataFrame) -> pd.DataFrame:
    """Sum per-dwelling ABM output to LSOA-level totals.

    Expects columns: lsoa_code, elec_kwh, gas_kwh, UPRN.
    """
    return (
        abm_per_dwelling
        .groupby("lsoa_code", as_index=False)
        .agg(
            abm_elec_kwh=("elec_kwh", "sum"),
            abm_gas_kwh =("gas_kwh",  "sum"),
            n_dwellings =("UPRN",     "count"),
        )
    )


def _safe_mape(a: np.ndarray, d: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(d) & (d > 0)
    return float(np.mean(np.abs((a[ok] - d[ok]) / d[ok])) * 100) if ok.any() else float("nan")


def _safe_bias(a: np.ndarray, d: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(d) & (d > 0)
    return float(np.mean((a[ok] - d[ok]) / d[ok]) * 100) if ok.any() else float("nan")


def compute_validation_metrics(
    abm_lsoa: pd.DataFrame,
    desnz_lsoa: pd.DataFrame,
) -> dict:
    """Compute LSOA-level MAPE, bias, residual distribution.

    Returns dict with scalar metrics, plus a `residuals_df` (one row per LSOA)
    for downstream plotting / stratification.
    """
    merged = abm_lsoa.merge(desnz_lsoa, on="lsoa_code", how="inner")
    merged["abm_total_kwh"]   = merged["abm_elec_kwh"]   + merged["abm_gas_kwh"]
    merged["desnz_total_kwh"] = merged["total_kwh_elec"] + merged["total_kwh_gas"]
    merged["residual_total"]  = merged["abm_total_kwh"]  - merged["desnz_total_kwh"]
    merged["residual_pct"]    = 100 * merged["residual_total"] / merged["desnz_total_kwh"]

    return {
        "n_lsoas":     int(len(merged)),
        "mape_elec":   _safe_mape(merged["abm_elec_kwh"].values,  merged["total_kwh_elec"].values),
        "mape_gas":    _safe_mape(merged["abm_gas_kwh"].values,   merged["total_kwh_gas"].values),
        "mape_total":  _safe_mape(merged["abm_total_kwh"].values, merged["desnz_total_kwh"].values),
        "bias_elec":   _safe_bias(merged["abm_elec_kwh"].values,  merged["total_kwh_elec"].values),
        "bias_gas":    _safe_bias(merged["abm_gas_kwh"].values,   merged["total_kwh_gas"].values),
        "bias_total":  _safe_bias(merged["abm_total_kwh"].values, merged["desnz_total_kwh"].values),
        "residuals_df": merged[[
            "lsoa_code", "abm_elec_kwh", "abm_gas_kwh", "abm_total_kwh",
            "total_kwh_elec", "total_kwh_gas", "desnz_total_kwh",
            "residual_total", "residual_pct",
        ]].copy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Result cache layout
# ─────────────────────────────────────────────────────────────────────────────

def cache_dir(kind: str, *parts: str) -> Path:
    """Standardised output location.

    Examples:
      cache_dir("calibration", "2020_2022")           → results/calibration/2020_2022/
      cache_dir("transfer", "sunderland_2023")        → results/transfer/sunderland_2023/
      cache_dir("mc", "newcastle_2023")               → results/mc/newcastle_2023/
    """
    d = results_root() / kind
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_validation_result(metrics: dict, out_dir: Path) -> None:
    """Persist a validate() result: residuals.parquet + metrics.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    residuals = metrics.pop("residuals_df")
    residuals.to_parquet(out_dir / "residuals.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)},
                  f, indent=2, default=float)


# ─────────────────────────────────────────────────────────────────────────────
# Model runner + checkpointing
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import pickle
import tempfile


def _params_signature(params: dict, year: int, window_hours: int,
                       climate_path: Path, n_dwellings: int) -> str:
    """Stable hash of the inputs that define a model run.

    Used to validate that an on-disk checkpoint matches the current invocation
    before resuming. If any input changes, the checkpoint is invalidated and
    we start fresh.
    """
    payload = json.dumps({
        "params": params,
        "year": int(year),
        "window_hours": int(window_hours),
        "climate_path": str(climate_path),
        "n_dwellings": int(n_dwellings),
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _atomic_pickle_dump(obj, path: Path) -> None:
    """Pickle to `path` via a tmp + rename so a mid-write crash doesn't
    corrupt the checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _atomic_json_dump(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins on conflicts."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_model(
    stock: gpd.GeoDataFrame,
    params: dict,
    year: int,
    *,
    window_hours: int = 8760,
    climate_path_override: Optional[Path] = None,
    collect_agent_level: bool = False,
    tracker=None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every_hours: int = 720,
    resume: bool = True,
) -> pd.DataFrame:
    """Run EnergyModel on `stock` with `params` over `year`.

    `params` is a model-config-shaped dict (with a top-level "model" key);
    it is written to a temp YAML and passed to EnergyModel via `config_path=…`.

    Returns: per-dwelling DataFrame with columns
      [UPRN, lsoa_code, elec_kwh, gas_kwh, total_kwh]
    where the kWh values are summed across all hours of the run (i.e. annual
    if window_hours=8760).

    Notes:
      - Climate slice is read from `climate_path_override` if given, else
        the city default would be supplied by the caller (validate() does this).
      - `electric_kwh` / `gas_kwh` annual totals come from the per-fuel
        accumulators added to EnergyModel._accumulate_annual_kwh in 2026-05;
        if those attrs are missing the run will return NaN for the per-fuel
        columns (caller should treat that as a build-error signal).

    Checkpoint / resume:
      If `checkpoint_dir` is provided, the EnergyModel object is pickled to
      `<checkpoint_dir>/_model_checkpoint.pkl` every `checkpoint_every_hours`
      hours. On a fresh invocation with `resume=True`, an existing checkpoint
      is loaded if and only if its signature (hash of params + year +
      window_hours + climate_path + n_dwellings) matches the current call —
      so changing any input invalidates the checkpoint automatically.
      On clean exit the checkpoint file is deleted. Pickling failures are
      logged and the run continues without checkpointing (so a Mesa version
      that resists pickle won't break the run, you just lose the safety net).
    """
    # Late import — keeps `from utils import …` cheap for code paths that
    # don't actually run the model (e.g. analysis notebooks).
    from household_energy.model import EnergyModel  # noqa: E402

    if climate_path_override is None:
        raise ValueError(
            "run_model(): pass climate_path_override (resolve via climate_path(city) "
            "before calling)."
        )

    start_utc = pd.Timestamp(f"{int(year)}-01-01", tz="UTC")

    # UPRN → lsoa_code lookup for downstream LSOA aggregation
    if "lsoa_code" not in stock.columns:
        raise KeyError(
            "stock GeoDataFrame missing 'lsoa_code' column — required for "
            "LSOA-level validation."
        )
    uprn_to_lsoa: dict[str, str] = (
        stock[["UPRN", "lsoa_code"]]
        .assign(UPRN=lambda d: d["UPRN"].astype(str).str.strip())
        .set_index("UPRN")["lsoa_code"]
        .astype(str)
        .to_dict()
    )

    # Persist params to a temp YAML for EnergyModel(config_path=…)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(params, tmp, sort_keys=False)
        cfg_path = tmp.name

    # Checkpoint locations + signature
    signature = _params_signature(
        params, year, window_hours, climate_path_override, len(stock),
    )
    ckpt_pkl: Optional[Path] = None
    ckpt_meta: Optional[Path] = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_pkl  = checkpoint_dir / "_model_checkpoint.pkl"
        ckpt_meta = checkpoint_dir / "_model_checkpoint.json"

    def _log(msg: str, **extras) -> None:
        if tracker is not None:
            tracker.milestone(msg, **extras)

    def _warn(msg: str) -> None:
        if tracker is not None:
            tracker.warn(msg)

    try:
        m = None
        start_hour = 0

        # Try to resume from a compatible checkpoint
        if resume and ckpt_pkl is not None and ckpt_pkl.exists() and ckpt_meta.exists():
            try:
                with open(ckpt_meta) as f:
                    meta = json.load(f)
                if meta.get("signature") == signature:
                    with open(ckpt_pkl, "rb") as f:
                        m = pickle.load(f)
                    start_hour = int(meta.get("hour", 0))
                    _log("resumed from checkpoint",
                         hour=start_hour, total=window_hours,
                         saved_at=meta.get("saved_at"))
                else:
                    _warn(f"checkpoint signature mismatch — discarding "
                          f"(saved={meta.get('signature')!s} now={signature})")
            except Exception as exc:
                _warn(f"checkpoint load failed: {type(exc).__name__}: {exc}")
                m = None
                start_hour = 0

        # Build a fresh model if we didn't resume
        if m is None:
            m = EnergyModel(
                gdf=stock,
                climate_parquet=str(climate_path_override),
                climate_start=start_utc,
                collect_agent_level=collect_agent_level,
                agent_collect_every=168,
                config_path=cfg_path,
            )

        # Throttled checkpoint writer — logs but doesn't crash on pickle failure.
        ckpt_disabled_after_failure = False
        def _save_ckpt(model_obj, hour_done: int) -> None:
            nonlocal ckpt_disabled_after_failure
            if ckpt_pkl is None or ckpt_disabled_after_failure:
                return
            try:
                _atomic_pickle_dump(model_obj, ckpt_pkl)
                _atomic_json_dump({
                    "hour": int(hour_done),
                    "total": int(window_hours),
                    "signature": signature,
                    "saved_at": pd.Timestamp.utcnow().isoformat(),
                }, ckpt_meta)
            except Exception as exc:
                _warn(f"checkpoint write failed: {type(exc).__name__}: {exc}; "
                       "continuing without resume safety net")
                ckpt_disabled_after_failure = True

        # Main loop
        if tracker is not None:
            _prev_total = tracker.total
            _prev_count = tracker._count
            tracker.total = int(window_hours)
            tracker._count = int(start_hour)
            try:
                for i in range(start_hour, int(window_hours)):
                    m.step()
                    tracker.tick("h")
                    if ckpt_pkl is not None and (i + 1) % checkpoint_every_hours == 0:
                        _save_ckpt(m, i + 1)
            finally:
                tracker.total = _prev_total
                tracker._count = _prev_count
        else:
            for i in range(start_hour, int(window_hours)):
                m.step()
                if ckpt_pkl is not None and (i + 1) % checkpoint_every_hours == 0:
                    _save_ckpt(m, i + 1)

        # Walk agents → per-dwelling annual totals
        rows = []
        for h in m.household_agents:
            uprn = str(getattr(h, "unique_id", ""))
            elec_dict = getattr(h, "annual_electric_kwh_by_year", None) or {}
            gas_dict  = getattr(h, "annual_gas_kwh_by_year", None) or {}
            total_dict = getattr(h, "annual_kwh_by_year", None) or {}
            rows.append({
                "UPRN":      uprn,
                "lsoa_code": uprn_to_lsoa.get(uprn),
                "elec_kwh":  float(sum(elec_dict.values()))  if elec_dict  else float("nan"),
                "gas_kwh":   float(sum(gas_dict.values()))   if gas_dict   else float("nan"),
                "total_kwh": float(sum(total_dict.values())) if total_dict else float("nan"),
            })

        # Clean exit — delete the checkpoint files so the next run starts fresh
        if ckpt_pkl is not None and ckpt_pkl.exists():
            try:
                ckpt_pkl.unlink()
                if ckpt_meta is not None and ckpt_meta.exists():
                    ckpt_meta.unlink()
            except Exception:
                pass

        return pd.DataFrame(rows)
    finally:
        try:
            Path(cfg_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    city: str,
    year: int,
    params: dict,
    stock: Optional[gpd.GeoDataFrame] = None,
    window_hours: int = 8760,
) -> dict:
    """Load stock → run model → compare to DESNZ → return metrics.

    The single function that every downstream script calls. Result is
    cacheable as JSON (scalar metrics) + parquet (per-LSOA residuals)
    via save_validation_result().
    """
    if stock is None:
        stock = load_city_stock(city)

    abm = run_model(stock, params, year,
                    climate_path_override=climate_path(city),
                    window_hours=window_hours)
    abm_lsoa = aggregate_to_lsoa(abm)
    desnz_lsoa = load_desnz(city, year)
    metrics = compute_validation_metrics(abm_lsoa, desnz_lsoa)
    metrics["city"] = city
    metrics["year"] = year
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Coverage-aware confidence tiers
#
# Decision (2026-06-04): DESNZ LSOA electricity is an independent benchmark
# with known coverage contamination (empties, second meters, non-EPC stock),
# not a calibration target. Quantify *where* the model is trustworthy by
# tiering each LSOA on three computable covariates. Same rule for every city
# so cross-city results stay comparable. Symmetric coverage band (2026-06-04):
# best when |coverage−1| is small — punishes overcounts as well as undercounts.
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence_tiers(
    abm_rollup: pd.DataFrame,
    city: str,
    year: int = 2023,
) -> pd.DataFrame:
    """Merge DESNZ electricity + compute coverage / tot_ratio / confidence tiers.

    Inputs
    ------
    abm_rollup : DataFrame with columns
        year, lsoa_code, run_dwellings, run_electric_heated_dwellings, abm_elec_kwh
    city : one of CITY_CONVENTIONS
    year : DESNZ year to merge against (default 2023)

    Returns
    -------
    DataFrame with the merge plus:
        coverage, elec_heat_share, tot_ratio,
        confidence_score (0–8), confidence ('High'|'Medium'|'Low').

    Tier rule (score 0–8) = 2*cov_pts + elec_pts + size_pts
        cov_pts  : |coverage−1| ≤0.20 → 2 ; ≤0.40 → 1 ; else 0
        elec_pts : elec_heat_share <0.20 → 2 ; <0.40 → 1 ; else 0
        size_pts : meters_elec ≥500 → 2 ; ≥300 → 1 ; else 0
    Bucket : score ≥6 High ; ≥4 Medium ; else Low.
    """
    des = load_desnz(city, year)[["lsoa_code", "meters_elec", "total_kwh_elec"]]
    c = abm_rollup[abm_rollup["year"] == year].merge(des, on="lsoa_code", how="inner").copy()
    c["coverage"] = c["run_dwellings"] / c["meters_elec"]
    c["elec_heat_share"] = c["run_electric_heated_dwellings"] / c["run_dwellings"]
    c["tot_ratio"] = c["abm_elec_kwh"] / c["total_kwh_elec"]

    cov_dev = (c["coverage"] - 1.0).abs()
    cov_pts = np.select([cov_dev <= 0.20, cov_dev <= 0.40], [2, 1], 0)
    elec_pts = np.select([c["elec_heat_share"] < 0.20, c["elec_heat_share"] < 0.40], [2, 1], 0)
    size_pts = np.select([c["meters_elec"] >= 500, c["meters_elec"] >= 300], [2, 1], 0)
    c["confidence_score"] = 2 * cov_pts + elec_pts + size_pts
    c["confidence"] = np.select(
        [c["confidence_score"] >= 6, c["confidence_score"] >= 4], ["High", "Medium"], "Low")
    return c


def summarize_confidence_tiers(c: pd.DataFrame, city: str) -> None:
    """Print the coverage-fit headline + per-tier breakdown to stdout."""
    def _r2(y, *xs):
        X = np.column_stack([np.ones(len(y))] + list(xs))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return 1.0 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()

    v = c[["tot_ratio", "coverage", "elec_heat_share"]].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n[{city}] n={len(c)} LSOAs")
    print(f"  R2(tot_ratio ~ coverage)              = {_r2(v['tot_ratio'].values, v['coverage'].values):.3f}")
    print(f"  R2(tot_ratio ~ coverage + elec share) = {_r2(v['tot_ratio'].values, v['coverage'].values, v['elec_heat_share'].values):.3f}")
    print(f"  corr(model_total, DESNZ_total)        = {c['abm_elec_kwh'].corr(c['total_kwh_elec']):.3f}")
    print(f"  city model/DESNZ elec total           = {c['abm_elec_kwh'].sum() / c['total_kwh_elec'].sum():.3f}")
    print(f"  dwelling-wt coverage                  = {c['run_dwellings'].sum() / c['meters_elec'].sum():.3f}")

    ts = c.groupby("confidence").agg(n=("tot_ratio", "size"), mean_ratio=("tot_ratio", "mean")).reindex(["High", "Medium", "Low"])
    ts["corr_total"] = [c.loc[c["confidence"] == t, "abm_elec_kwh"].corr(c.loc[c["confidence"] == t, "total_kwh_elec"]) for t in ts.index]
    print(ts.round(3).to_string())


CONFIDENCE_OUT_COLS = [
    "lsoa_code", "meters_elec", "run_dwellings", "coverage", "elec_heat_share",
    "abm_elec_kwh", "total_kwh_elec", "tot_ratio", "confidence_score", "confidence",
]


__all__ = [
    "repo_root", "results_root",
    "CityConvention", "CITY_CONVENTIONS", "city_convention",
    "load_city_stock", "climate_path",
    "load_desnz", "desnz_xlsx_paths",
    "load_params", "save_params",
    "aggregate_to_lsoa", "compute_validation_metrics",
    "cache_dir", "save_validation_result",
    "run_model", "validate",
    "compute_confidence_tiers", "summarize_confidence_tiers", "CONFIDENCE_OUT_COLS",
]
