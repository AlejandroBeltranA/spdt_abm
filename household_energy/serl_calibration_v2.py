from __future__ import annotations

import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from .climate import ClimateField
from .model import EnergyModel


@dataclass(frozen=True)
class Window:
    name: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp


@dataclass(frozen=True)
class CitySource:
    name: str
    geojson: str
    hidp_csv: str
    climate_parquet: str
    serl_region: str


CITY_SOURCES: dict[str, CitySource] = {
    "newcastle": CitySource(
        name="newcastle",
        geojson="data/epc_abm_newcastle.geojson",
        hidp_csv="data/hidp_uprn_matches_tiered.csv",
        climate_parquet="data/ncc_2t_timeseries_2010_2026.parquet",
        serl_region="North East",
    ),
    "sunderland": CitySource(
        name="sunderland",
        geojson="data/epc_abm_sunderland.geojson",
        hidp_csv="data/hidp_uprn_matches_tiered.csv",
        climate_parquet="data/ncc_2t_timeseries_2010_2026.parquet",
        serl_region="North East",
    ),
    "waltham_forest": CitySource(
        name="waltham_forest",
        geojson="data/epc_abm_waltham_forest.geojson",
        hidp_csv="data/wf_hidp_uprn_matches_tiered.csv",
        climate_parquet="data/waltham_forest_2t_timeseries_2010_2026.parquet",
        serl_region="Greater London",
    ),
}


STATIC_SEG3_MAP = {
    "building_age": "serl_building_age",
    "building_type": "serl_building_type",
    "central_heating_type": "serl_central_heating_type",
    "currentEnergyRating": "serl_currentEnergyRating",
    "floor_area_m2": "serl_floor_area_m2",
    "num_bedrooms": "serl_num_bedrooms",
    "num_occupants": "serl_num_occupants",
    "region": "serl_region",
    "temperature_band": "temperature_band",
    "tenure": "serl_tenure",
}


def repo_root_from(module_file: str | Path = __file__) -> Path:
    return Path(module_file).resolve().parents[1]


def load_enriched_gdf(repo_root: Path, city: str) -> gpd.GeoDataFrame:
    src = CITY_SOURCES[city]
    g = gpd.read_file(repo_root / src.geojson)
    g["UPRN"] = g["UPRN"].astype(str).str.strip()

    hidp = pd.read_csv(repo_root / src.hidp_csv, low_memory=False)
    hidp.columns = [c.strip() for c in hidp.columns]
    hidp["uprn_chr"] = hidp["uprn_chr"].astype(str).str.strip()
    hidp = hidp.drop_duplicates(subset=["uprn_chr"])
    g = g.merge(hidp, how="left", left_on="UPRN", right_on="uprn_chr", suffixes=("_geo", "_hidp"))

    for base in ["lsoa_code", "ward_code", "local_authority"]:
        geo_col, hidp_col = f"{base}_geo", f"{base}_hidp"
        if base not in g.columns and (geo_col in g.columns or hidp_col in g.columns):
            if geo_col in g.columns and hidp_col in g.columns:
                g[base] = g[geo_col].combine_first(g[hidp_col])
            elif geo_col in g.columns:
                g[base] = g[geo_col]
            else:
                g[base] = g[hidp_col]

    g["city"] = city
    g["AgentID"] = g["UPRN"].astype(str)
    return g


def _band_floor_area(s: pd.Series) -> pd.Series:
    a = pd.to_numeric(s, errors="coerce")
    out = pd.Series("No data", index=s.index, dtype="object")
    out[a <= 50] = "50 or less"
    out[(a > 50) & (a <= 100)] = "51 to 100"
    out[(a > 100) & (a <= 150)] = "101 to 150"
    out[(a > 150) & (a <= 200)] = "151 to 200"
    out[a > 200] = "Over 200"
    return out


def _band_occupants(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series("No data", index=s.index, dtype="object")
    for val in [1, 2, 3, 4, 5]:
        out[n == val] = str(val)
    out[n >= 6] = ">=6"
    return out


def _band_bedrooms(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series("No data", index=s.index, dtype="object")
    for val in [1, 2, 3]:
        out[n == val] = str(val)
    out[n >= 4] = "4+"
    return out


def _band_property_type(s: pd.Series) -> pd.Series:
    t = s.astype("string").fillna("").str.strip().str.lower()
    out = pd.Series("Commercial building or no answer", index=s.index, dtype="object")
    out[t.str.contains("detached", na=False) & ~t.str.contains("semi", na=False)] = "Detached"
    out[t.str.contains("semi-detached", na=False) | t.str.contains("semi detached", na=False)] = "Semi-detached"
    out[t.str.contains("terraced", na=False)] = "Terraced"
    out[t.isin(["large block of flats", "block of flats"])] = "Purpose-built flat"
    out[t.str.contains("dwelling converted in to flats", na=False)] = "Converted flat or shared house"
    return out


def _band_property_age(s: pd.Series) -> pd.Series:
    raw = s.astype("string").fillna("").str.strip().str.lower()
    mapping = {
        "pre-1900": "Before 1900",
        "1900-1929": "1900 - 1929",
        "1930-1949": "1930 - 1949",
        "1950-1966": "1950 - 1975",
        "1967-1982": "1976 - 1990",
        "1983-1995": "1991 - 2002",
        "post-1996": "2003 onwards",
    }
    return raw.map(mapping).fillna("No data")


def _band_tenure(s: pd.Series) -> pd.Series:
    raw = s.astype("string").fillna("").str.strip().str.lower()
    mapping = {
        "owner_occupied": "Own outright or mortgage",
        "private_rent": "Private rent",
        "social_rent": "Social rent",
    }
    return raw.map(mapping).fillna("No answer")


def _band_heating_type(main_heating_system: pd.Series, main_fuel_type: pd.Series) -> pd.Series:
    heat = main_heating_system.astype("string").fillna("").str.strip().str.lower()
    fuel = main_fuel_type.astype("string").fillna("").str.strip().str.lower()
    out = pd.Series("none", index=heat.index, dtype="object")

    gas_mask = fuel.eq("mains gas") & heat.eq("boiler")
    out[gas_mask] = "Gas boiler"
    out[heat.eq("communal")] = "District or community"
    out[heat.eq("storage heaters")] = "Electric storage radiators"

    other_electric = heat.isin(["room heaters", "heat pump"]) & fuel.isin(["electricity", "no fuel", ""])
    out[other_electric] = "Other electric"

    fossil_other = fuel.isin(["oil", "solid", "biomass", "lpg"])
    out[fossil_other] = "Oil, solid fuel or biomass"

    other_mix = out.eq("none") & heat.ne("")
    out[other_mix] = "Other or other mix"
    return out


def _band_epc(current_energy_rating: Optional[pd.Series], sap_band_ord: Optional[pd.Series]) -> pd.Series:
    if current_energy_rating is not None:
        raw = current_energy_rating.astype("string").fillna("").str.strip().str.upper()
        out = pd.Series("No data", index=raw.index, dtype="object")
        out[raw.isin(["A", "B"])] = "A and B"
        out[raw.eq("C")] = "C"
        out[raw.eq("D")] = "D"
        out[raw.eq("E")] = "E"
        out[raw.isin(["F", "G"])] = "F and G"
        return out

    band = pd.to_numeric(sap_band_ord, errors="coerce") if sap_band_ord is not None else pd.Series(dtype=float)
    out = pd.Series("No data", index=band.index, dtype="object")
    out[band >= 6] = "A and B"
    out[band == 5] = "C"
    out[band == 4] = "D"
    out[band == 3] = "E"
    out[band <= 2] = "F and G"
    return out


def add_serl_crosswalk_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["serl_region"] = out["city"].map({k: v.serl_region for k, v in CITY_SOURCES.items()}).fillna("No data")
    out["serl_num_occupants"] = _band_occupants(out.get("hh_n_people", pd.Series(index=out.index, dtype=float)))
    out["serl_num_bedrooms"] = _band_bedrooms(out.get("size_band", pd.Series(index=out.index, dtype=float)))
    out["serl_floor_area_m2"] = _band_floor_area(out.get("floor_area_m2", pd.Series(index=out.index, dtype=float)))
    out["serl_building_type"] = _band_property_type(out.get("property_type", pd.Series(index=out.index, dtype="object")))
    out["serl_building_age"] = _band_property_age(out.get("property_age", pd.Series(index=out.index, dtype="object")))
    out["serl_tenure"] = _band_tenure(out.get("tenure", pd.Series(index=out.index, dtype="object")))
    out["serl_central_heating_type"] = _band_heating_type(
        out.get("main_heating_system", pd.Series(index=out.index, dtype="object")),
        out.get("main_fuel_type", pd.Series(index=out.index, dtype="object")),
    )
    epc = out["currentEnergyRating"] if "currentEnergyRating" in out.columns else None
    sap_band = out["sap_band_ord"] if "sap_band_ord" in out.columns else None
    out["serl_currentEnergyRating"] = _band_epc(epc, sap_band)
    return out


def load_serl_targets(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_dir = repo_root / "data" / "serl_8963_targets"
    daily_targets = pd.read_csv(target_dir / "daily_targets.csv")
    hourly_targets = pd.read_csv(target_dir / "diurnal_targets_hourly_mean.csv")
    for df in [daily_targets, hourly_targets]:
        is_bed = df["seg3_var"].astype(str) == "num_bedrooms"
        bed = df["seg3_value"].astype(str)
        df.loc[is_bed & bed.isin(["4", ">=5", "4+"]), "seg3_value"] = "4+"
    return daily_targets, hourly_targets


def supported_seg3_vars(*, include_region: bool = True) -> list[str]:
    vals = list(STATIC_SEG3_MAP)
    if not include_region:
        vals = [v for v in vals if v != "region"]
    return vals


def calibration_segmentations(seg3_vars: list[str]) -> list[dict]:
    return [
        {"name": seg3, "attr": STATIC_SEG3_MAP[seg3], "fallback_value": "No data"}
        for seg3 in seg3_vars
        if seg3 != "temperature_band"
    ]


def make_override_yaml(*, name: str, params: dict) -> dict:
    payload = {
        "meta": {"name": name, "date": pd.Timestamp.utcnow().date().isoformat(), "notes": "SERL calibration v2"},
        "model": {},
        "schedules": {},
        "calibration": {},
        "serl_profiles": {},
        "systems": {},
    }
    for key, value in params.items():
        if "." not in key:
            continue
        top, rest = key.split(".", 1)
        if top not in payload:
            continue
        cur = payload[top]
        parts = rest.split(".")
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
    return payload


def deep_merge_dict(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _write_cfg(payload: dict) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(payload, tmp, sort_keys=False)
    return tmp.name


def _run_parallel_starmap(func, tasks, n_procs: int):
    if n_procs <= 1:
        return [func(*t) if isinstance(t, tuple) else func(t) for t in tasks]
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_procs) as pool:
            if tasks and isinstance(tasks[0], tuple):
                return pool.starmap(func, tasks)
            return pool.map(func, tasks)
    except Exception:
        if tasks and isinstance(tasks[0], tuple):
            return [func(*t) for t in tasks]
        return [func(t) for t in tasks]


def build_full_year_window(target_year: int) -> list[Window]:
    return [Window("full_year", pd.Timestamp(f"{target_year}-01-01T00:00:00Z"), pd.Timestamp(f"{target_year+1}-01-01T00:00:00Z"))]


def _run_city_shard(
    shard_path: str,
    city: str,
    climate_parquet: str,
    cfg_path: str,
    target_year: int,
    local_tz: str,
) -> pd.DataFrame:
    gdf = gpd.read_parquet(shard_path)
    cf = ClimateField(climate_parquet)
    frames: list[pd.DataFrame] = []
    for window in build_full_year_window(target_year):
        i0 = cf.time_index_for(window.start_utc)
        i1 = cf.time_index_for(window.end_utc)
        start_aligned = pd.to_datetime(cf.times[i0], utc=True)
        model = EnergyModel(
            gdf=gdf,
            climate_parquet=climate_parquet,
            climate_start=start_aligned,
            local_tz=local_tz,
            collect_agent_level=False,
            config_path=cfg_path,
        )
        for _ in range(int(i1 - i0)):
            model.step()
        seg = model.get_segmentation_timeseries()
        seg["city"] = city
        frames.append(seg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_city_segmentations(
    *,
    repo_root: Path,
    city: str,
    target_year: int,
    seg3_vars: list[str],
    max_homes: Optional[int] = None,
    seed: int = 7,
    model_overrides: Optional[dict] = None,
    extra_config: Optional[dict] = None,
    local_tz: str = "Europe/London",
    process_column_preferred: str = "lsoa_code",
    n_procs: int = 1,
) -> pd.DataFrame:
    src = CITY_SOURCES[city]
    gdf = add_serl_crosswalk_columns(load_enriched_gdf(repo_root, city))
    if max_homes is not None and len(gdf) > int(max_homes):
        rng = np.random.default_rng(int(seed))
        gdf = gdf.iloc[rng.choice(gdf.index.to_numpy(), size=int(max_homes), replace=False)].copy()

    override_params = {
        "calibration.segmentations": calibration_segmentations(seg3_vars),
        "model.random_seed": int(seed),
    }
    if model_overrides:
        override_params.update(model_overrides)

    payload = make_override_yaml(name=f"serl_v2_{city}", params=override_params)
    if extra_config:
        payload = deep_merge_dict(payload, extra_config)
    cfg_path = _write_cfg(payload)
    process_column = process_column_preferred if process_column_preferred in gdf.columns else "city"
    units = gdf[process_column].astype(str).replace("", np.nan).dropna().unique().tolist()
    if not units:
        units = [city]
        gdf[process_column] = city

    with tempfile.TemporaryDirectory(prefix=f"serl_v2_{city}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        shard_tasks = []
        for unit in units:
            g_unit = gdf[gdf[process_column].astype(str) == str(unit)].copy()
            if g_unit.empty:
                continue
            shard_path = tmpdir_path / f"{process_column}={unit}.parquet"
            g_unit.to_parquet(shard_path, index=False)
            shard_tasks.append(
                (
                    str(shard_path),
                    city,
                    str(repo_root / src.climate_parquet),
                    cfg_path,
                    int(target_year),
                    str(local_tz),
                )
            )

        frames = _run_parallel_starmap(
            _run_city_shard,
            shard_tasks,
            n_procs=max(1, int(n_procs)),
        )
    return pd.concat([f for f in frames if isinstance(f, pd.DataFrame) and not f.empty], ignore_index=True) if frames else pd.DataFrame()


def run_pooled_segmentations(
    *,
    repo_root: Path,
    target_year: int,
    seg3_vars: list[str],
    cities: list[str],
    max_homes_per_city: Optional[int] = None,
    seed: int = 7,
    model_overrides: Optional[dict] = None,
    extra_config: Optional[dict] = None,
    local_tz: str = "Europe/London",
    process_column_preferred: str = "lsoa_code",
    n_procs: int = 1,
) -> pd.DataFrame:
    frames = []
    for i, city in enumerate(cities):
        frames.append(
            run_city_segmentations(
                repo_root=repo_root,
                city=city,
                target_year=target_year,
                seg3_vars=seg3_vars,
                max_homes=max_homes_per_city,
                seed=seed + i,
                model_overrides=model_overrides,
                extra_config=extra_config,
                local_tz=local_tz,
                process_column_preferred=process_column_preferred,
                n_procs=n_procs,
            )
        )
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()


def abm_profiles_from_seg_ts(seg_ts: pd.DataFrame, *, local_tz: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = seg_ts.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    loc = df["timestamp_utc"].dt.tz_convert(local_tz)
    df["local_hour"] = loc.dt.hour
    df["local_date"] = loc.dt.floor("D")
    df["local_month"] = loc.dt.month

    df["elec_kwh_per_home"] = df["electric_kwh"] / df["n_homes"].replace({0: np.nan})
    df["gas_kwh_per_home"] = df["gas_kwh"] / df["n_homes"].replace({0: np.nan})

    hourly = (
        df.groupby(["segmentation", "value", "local_hour"], as_index=False)
        .agg(elec_kwh=("elec_kwh_per_home", "mean"), gas_kwh=("gas_kwh_per_home", "mean"))
        .sort_values(["segmentation", "value", "local_hour"])
    )

    daily = (
        df.groupby(["segmentation", "value", "local_date", "local_month"], as_index=False)
        .agg(electric_kwh_day=("electric_kwh", "sum"), gas_kwh_day=("gas_kwh", "sum"), n=("n_homes", "min"))
    )
    daily["elec_kwh_per_home_day"] = daily["electric_kwh_day"] / daily["n"].replace({0: np.nan})
    daily["gas_kwh_per_home_day"] = daily["gas_kwh_day"] / daily["n"].replace({0: np.nan})
    monthly = (
        daily.groupby(["segmentation", "value", "local_month"], as_index=False)
        .agg(elec_kwh_per_home_day=("elec_kwh_per_home_day", "mean"), gas_kwh_per_home_day=("gas_kwh_per_home_day", "mean"))
        .rename(columns={"local_month": "month"})
        .sort_values(["segmentation", "value", "month"])
    )
    return hourly, monthly


def temperature_band_from_temp(temp_c: float) -> str:
    if not np.isfinite(temp_c):
        return "No data"
    if temp_c < 0:
        return "-5_to_0"
    if temp_c < 5:
        return "0_to_5"
    if temp_c < 10:
        return "5_to_10"
    if temp_c < 15:
        return "10_to_15"
    if temp_c < 20:
        return "15_to_20"
    return "20_to_25"


def build_city_temperature_lookup(repo_root: Path, city: str, target_year: int) -> pd.DataFrame:
    src = CITY_SOURCES[city]
    cf = ClimateField(str(repo_root / src.climate_parquet))
    start = pd.Timestamp(f"{target_year}-01-01T00:00:00Z")
    end = pd.Timestamp(f"{target_year+1}-01-01T00:00:00Z")
    i0 = cf.time_index_for(start)
    i1 = cf.time_index_for(end)
    rows = []
    for i in range(int(i0), int(i1)):
        ts = pd.to_datetime(cf.times[i], utc=True)
        vec = cf.temps_at_index(i)
        mean_temp = float(np.nanmean(vec)) if vec is not None and len(vec) else float("nan")
        rows.append(
            {
                "city": city,
                "timestamp_utc": ts,
                "mean_temp_c": mean_temp,
                "temperature_band": temperature_band_from_temp(mean_temp),
            }
        )
    return pd.DataFrame(rows)


def abm_temperature_band_profiles(
    *,
    seg_ts: pd.DataFrame,
    repo_root: Path,
    target_year: int,
    local_tz: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region = seg_ts[seg_ts["segmentation"].astype(str) == "region"].copy()
    if region.empty:
        return pd.DataFrame(), pd.DataFrame()

    lookups = [build_city_temperature_lookup(repo_root, city, target_year) for city in sorted(region["city"].astype(str).unique())]
    temp_df = pd.concat(lookups, ignore_index=True) if lookups else pd.DataFrame(columns=["city", "timestamp_utc", "temperature_band"])

    region["timestamp_utc"] = pd.to_datetime(region["timestamp_utc"], utc=True)
    region = region.merge(temp_df[["city", "timestamp_utc", "temperature_band"]], on=["city", "timestamp_utc"], how="left")
    region["temperature_band"] = region["temperature_band"].fillna("No data")
    loc = region["timestamp_utc"].dt.tz_convert(local_tz)
    region["local_hour"] = loc.dt.hour
    region["local_date"] = loc.dt.floor("D")
    region["local_month"] = loc.dt.month

    region["elec_kwh_per_home"] = region["electric_kwh"] / region["n_homes"].replace({0: np.nan})
    region["gas_kwh_per_home"] = region["gas_kwh"] / region["n_homes"].replace({0: np.nan})

    hourly = (
        region.groupby(["temperature_band", "local_hour"], as_index=False)
        .agg(elec_kwh=("elec_kwh_per_home", "mean"), gas_kwh=("gas_kwh_per_home", "mean"))
        .rename(columns={"temperature_band": "value"})
        .assign(segmentation="temperature_band")
        .sort_values(["value", "local_hour"])
    )

    daily = (
        region.groupby(["temperature_band", "local_date", "local_month"], as_index=False)
        .agg(electric_kwh_day=("electric_kwh", "sum"), gas_kwh_day=("gas_kwh", "sum"), n=("n_homes", "sum"))
    )
    daily["elec_kwh_per_home_day"] = daily["electric_kwh_day"] / daily["n"].replace({0: np.nan})
    daily["gas_kwh_per_home_day"] = daily["gas_kwh_day"] / daily["n"].replace({0: np.nan})
    monthly = (
        daily.groupby(["temperature_band", "local_month"], as_index=False)
        .agg(elec_kwh_per_home_day=("elec_kwh_per_home_day", "mean"), gas_kwh_per_home_day=("gas_kwh_per_home_day", "mean"))
        .rename(columns={"temperature_band": "value", "local_month": "month"})
        .assign(segmentation="temperature_band")
        .sort_values(["value", "month"])
    )
    return hourly, monthly


def _weighted_serl_hourly_targets(hourly_targets: pd.DataFrame, *, target_year: int, seg3_var: str, min_n: int) -> pd.DataFrame:
    sub = hourly_targets[
        (hourly_targets["year"].astype(int) == int(target_year))
        & (hourly_targets["weekday_weekend"].astype(str) == "both")
        & (hourly_targets["heating_fuel"].astype(str) == "All")
        & (hourly_targets["has_pv"].astype(str).isin(["No", "All"]))
        & (hourly_targets["seg3_var"].astype(str) == str(seg3_var))
    ].copy()
    sub["n_rounded"] = pd.to_numeric(sub["n_rounded"], errors="coerce").fillna(0.0)
    sub = sub[sub["n_rounded"] >= float(min_n)].copy()
    if sub.empty:
        return sub
    sub["wx"] = sub["mean_kwh"] * sub["n_rounded"]
    out = (
        sub.groupby(["quantity", "seg3_value", "hour"], as_index=False)
        .agg(wx=("wx", "sum"), n_rounded=("n_rounded", "sum"))
    )
    out["target_kwh"] = out["wx"] / out["n_rounded"].replace({0: np.nan})
    return out


def _weighted_serl_monthly_targets(daily_targets: pd.DataFrame, *, target_year: int, seg3_var: str, min_n: int) -> pd.DataFrame:
    sub = daily_targets[
        (daily_targets["period_type"].astype(str) == "monthly")
        & (daily_targets["year"].astype(int) == int(target_year))
        & (daily_targets["weekday_weekend"].astype(str) == "both")
        & (daily_targets["heating_fuel"].astype(str) == "All")
        & (daily_targets["has_pv"].astype(str).isin(["No", "All"]))
        & (daily_targets["seg3_var"].astype(str) == str(seg3_var))
    ].copy()
    sub["n_rounded"] = pd.to_numeric(sub["n_rounded"], errors="coerce").fillna(0.0)
    sub = sub[sub["n_rounded"] >= float(min_n)].copy()
    if sub.empty:
        return sub
    sub["wx"] = sub["mean"] * sub["n_rounded"]
    out = (
        sub.groupby(["quantity", "seg3_value", "month"], as_index=False)
        .agg(wx=("wx", "sum"), n_rounded=("n_rounded", "sum"))
    )
    out["target_kwh_day"] = out["wx"] / out["n_rounded"].replace({0: np.nan})
    return out


def derive_profile_multipliers(
    *,
    abm_hourly: pd.DataFrame,
    abm_monthly: pd.DataFrame,
    hourly_targets: pd.DataFrame,
    daily_targets: pd.DataFrame,
    target_year: int,
    seg3_vars: list[str],
    min_n: int = 80,
    shrinkage_n: float = 1000.0,
    clip_min: float = 0.6,
    clip_max: float = 1.8,
) -> pd.DataFrame:
    rows: list[dict] = []
    eps = 1e-9
    fuel_map = [("Electricity imports", "electric", "elec_kwh", "elec_kwh_per_home_day"), ("Gas", "gas", "gas_kwh", "gas_kwh_per_home_day")]

    for seg3 in seg3_vars:
        sh = _weighted_serl_hourly_targets(hourly_targets, target_year=target_year, seg3_var=seg3, min_n=min_n)
        sm = _weighted_serl_monthly_targets(daily_targets, target_year=target_year, seg3_var=seg3, min_n=min_n)
        ah = abm_hourly[abm_hourly["segmentation"].astype(str) == str(seg3)].copy()
        am = abm_monthly[abm_monthly["segmentation"].astype(str) == str(seg3)].copy()

        for quantity, fuel, abm_hour_col, abm_month_col in fuel_map:
            if not sh.empty:
                z = ah.merge(
                    sh[sh["quantity"].astype(str) == quantity][["seg3_value", "hour", "target_kwh", "n_rounded"]].rename(
                        columns={"seg3_value": "value", "hour": "local_hour"}
                    ),
                    on=["value", "local_hour"],
                    how="inner",
                )
                for _, r in z.iterrows():
                    abm_val = float(r[abm_hour_col])
                    tgt = float(r["target_kwh"])
                    if not np.isfinite(abm_val) or abm_val <= 0 or not np.isfinite(tgt) or tgt <= 0:
                        continue
                    raw_mult = tgt / max(abm_val, eps)
                    w = float(r["n_rounded"]) / (float(r["n_rounded"]) + float(shrinkage_n))
                    mult = 1.0 + w * (raw_mult - 1.0)
                    mult = float(np.clip(mult, clip_min, clip_max))
                    rows.append(
                        {
                            "seg3_var": seg3,
                            "kind": "hourly",
                            "fuel": fuel,
                            "seg3_value": str(r["value"]),
                            "idx": int(r["local_hour"]),
                            "mult": mult,
                            "raw_mult": raw_mult,
                            "n_rounded": float(r["n_rounded"]),
                        }
                    )

            if not sm.empty:
                z = am.merge(
                    sm[sm["quantity"].astype(str) == quantity][["seg3_value", "month", "target_kwh_day", "n_rounded"]].rename(columns={"seg3_value": "value"}),
                    on=["value", "month"],
                    how="inner",
                )
                for _, r in z.iterrows():
                    abm_val = float(r[abm_month_col])
                    tgt = float(r["target_kwh_day"])
                    if not np.isfinite(abm_val) or abm_val <= 0 or not np.isfinite(tgt) or tgt <= 0:
                        continue
                    raw_mult = tgt / max(abm_val, eps)
                    w = float(r["n_rounded"]) / (float(r["n_rounded"]) + float(shrinkage_n))
                    mult = 1.0 + w * (raw_mult - 1.0)
                    mult = float(np.clip(mult, clip_min, clip_max))
                    rows.append(
                        {
                            "seg3_var": seg3,
                            "kind": "monthly",
                            "fuel": fuel,
                            "seg3_value": str(r["value"]),
                            "idx": int(r["month"]),
                            "mult": mult,
                            "raw_mult": raw_mult,
                            "n_rounded": float(r["n_rounded"]),
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["seg3_var", "kind", "fuel", "seg3_value", "idx"]).reset_index(drop=True)


def evaluate_against_serl(
    *,
    abm_hourly: pd.DataFrame,
    abm_monthly: pd.DataFrame,
    hourly_targets: pd.DataFrame,
    daily_targets: pd.DataFrame,
    target_year: int,
    seg3_vars: list[str],
    min_n: int = 80,
) -> pd.DataFrame:
    rows: list[dict] = []
    eps = 1e-9
    fuel_map = [("Electricity imports", "electric", "elec_kwh", "elec_kwh_per_home_day"), ("Gas", "gas", "gas_kwh", "gas_kwh_per_home_day")]

    for seg3 in seg3_vars:
        sh = _weighted_serl_hourly_targets(hourly_targets, target_year=target_year, seg3_var=seg3, min_n=min_n)
        sm = _weighted_serl_monthly_targets(daily_targets, target_year=target_year, seg3_var=seg3, min_n=min_n)
        ah = abm_hourly[abm_hourly["segmentation"].astype(str) == str(seg3)].copy()
        am = abm_monthly[abm_monthly["segmentation"].astype(str) == str(seg3)].copy()

        for quantity, fuel, abm_hour_col, abm_month_col in fuel_map:
            shf = sh[sh["quantity"].astype(str) == quantity].copy()
            smf = sm[sm["quantity"].astype(str) == quantity].copy()

            if not shf.empty:
                z = ah.merge(
                    shf[["seg3_value", "hour", "target_kwh", "n_rounded"]].rename(columns={"seg3_value": "value", "hour": "local_hour"}),
                    on=["value", "local_hour"],
                    how="inner",
                )
                if not z.empty:
                    z["sq_err"] = (pd.to_numeric(z[abm_hour_col], errors="coerce") - pd.to_numeric(z["target_kwh"], errors="coerce")) ** 2
                    z["abs_log_ratio"] = np.abs(np.log((pd.to_numeric(z[abm_hour_col], errors="coerce") + eps) / (pd.to_numeric(z["target_kwh"], errors="coerce") + eps)))
                    z = z[np.isfinite(z["sq_err"]) & np.isfinite(z["abs_log_ratio"])].copy()
                    if not z.empty:
                        w = pd.to_numeric(z["n_rounded"], errors="coerce").fillna(0.0)
                        rows.append(
                            {
                                "seg3_var": seg3,
                                "fuel": fuel,
                                "metric": "hourly_rmse",
                                "value": float(np.sqrt(np.average(z["sq_err"], weights=w))),
                                "rows": int(len(z)),
                            }
                        )
                        rows.append(
                            {
                                "seg3_var": seg3,
                                "fuel": fuel,
                                "metric": "hourly_abs_log_ratio",
                                "value": float(np.average(z["abs_log_ratio"], weights=w)),
                                "rows": int(len(z)),
                            }
                        )

            if not smf.empty:
                z = am.merge(
                    smf[["seg3_value", "month", "target_kwh_day", "n_rounded"]].rename(columns={"seg3_value": "value"}),
                    on=["value", "month"],
                    how="inner",
                )
                if not z.empty:
                    z["sq_err"] = (pd.to_numeric(z[abm_month_col], errors="coerce") - pd.to_numeric(z["target_kwh_day"], errors="coerce")) ** 2
                    z["abs_log_ratio"] = np.abs(np.log((pd.to_numeric(z[abm_month_col], errors="coerce") + eps) / (pd.to_numeric(z["target_kwh_day"], errors="coerce") + eps)))
                    z = z[np.isfinite(z["sq_err"]) & np.isfinite(z["abs_log_ratio"])].copy()
                    if not z.empty:
                        w = pd.to_numeric(z["n_rounded"], errors="coerce").fillna(0.0)
                        rows.append(
                            {
                                "seg3_var": seg3,
                                "fuel": fuel,
                                "metric": "monthly_rmse",
                                "value": float(np.sqrt(np.average(z["sq_err"], weights=w))),
                                "rows": int(len(z)),
                            }
                        )
                        rows.append(
                            {
                                "seg3_var": seg3,
                                "fuel": fuel,
                                "metric": "monthly_abs_log_ratio",
                                "value": float(np.average(z["abs_log_ratio"], weights=w)),
                                "rows": int(len(z)),
                            }
                        )

    return pd.DataFrame(rows)


def compare_runs(baseline_scores: pd.DataFrame, calibrated_scores: pd.DataFrame) -> pd.DataFrame:
    if baseline_scores.empty or calibrated_scores.empty:
        return pd.DataFrame()
    out = baseline_scores.merge(
        calibrated_scores,
        on=["seg3_var", "fuel", "metric"],
        how="inner",
        suffixes=("_baseline", "_calibrated"),
    )
    out["abs_improvement"] = out["value_baseline"] - out["value_calibrated"]
    out["pct_improvement"] = np.where(
        pd.to_numeric(out["value_baseline"], errors="coerce") > 0,
        100.0 * out["abs_improvement"] / out["value_baseline"],
        np.nan,
    )
    return out.sort_values(["metric", "fuel", "seg3_var"]).reset_index(drop=True)


def export_profile_layers(multipliers: pd.DataFrame, out_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    layers: list[dict] = []
    audit_rows: list[dict] = []
    for seg3, g in multipliers.groupby("seg3_var", observed=True):
        csv_path = out_dir / f"serl_profiles_{seg3}.csv"
        g[["kind", "fuel", "seg3_value", "idx", "mult"]].to_csv(csv_path, index=False)
        layers.append(
            {
                "name": str(seg3),
                "seg3_var": str(seg3),
                "seg3_column": STATIC_SEG3_MAP[str(seg3)],
                "fallback_value": "No data",
                "profiles_csv": str(csv_path),
                "alpha": 1.0,
                "use_hourly": True,
                "use_monthly": True,
                "source_type": "temperature_band" if str(seg3) == "temperature_band" else "household_attribute",
            }
        )
        audit_rows.append(
            {
                "seg3_var": str(seg3),
                "profiles_csv": str(csv_path),
                "rows": int(len(g)),
                "values": int(g["seg3_value"].astype(str).nunique()),
            }
        )
    return layers, pd.DataFrame(audit_rows)


def build_serl_profile_config(*, name: str, layers: list[dict]) -> dict:
    return {
        "meta": {
            "name": name,
            "date": pd.Timestamp.utcnow().date().isoformat(),
            "notes": "SERL calibration v2 layered profile export",
        },
        "serl_profiles": {
            "layers": [
                {
                    "name": layer["name"],
                    "seg3_column": layer["seg3_column"],
                    "fallback_value": layer["fallback_value"],
                    "profiles_csv": layer["profiles_csv"],
                    "alpha": layer.get("alpha", 1.0),
                    "use_hourly": layer.get("use_hourly", True),
                    "use_monthly": layer.get("use_monthly", True),
                    "source_type": layer.get("source_type", "household_attribute"),
                }
                for layer in layers
            ]
        },
    }


def export_serl_profile_config(*, name: str, layers: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_serl_profile_config(name=name, layers=layers)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return out_path


def build_pooled_crosswalk_audit(repo_root: Path, cities: list[str]) -> pd.DataFrame:
    frames = []
    for city in cities:
        gdf = add_serl_crosswalk_columns(load_enriched_gdf(repo_root, city))
        for seg3_var, col in STATIC_SEG3_MAP.items():
            if col not in gdf.columns:
                continue
            vc = gdf[col].astype(str).value_counts(dropna=False).rename_axis("seg3_value").reset_index(name="rows")
            vc["city"] = city
            vc["seg3_var"] = seg3_var
            frames.append(vc[["city", "seg3_var", "seg3_value", "rows"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["city", "seg3_var", "seg3_value", "rows"])
