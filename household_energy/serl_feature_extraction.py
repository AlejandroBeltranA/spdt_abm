#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .serl_calibration_v2 import load_serl_targets, repo_root_from


FUEL_MAP = {
    "Electricity imports": "electric",
    "Gas": "gas",
}

DEFAULT_AM_HOURS = [5, 6, 7, 8, 9, 10, 11]
DEFAULT_PM_HOURS = [16, 17, 18, 19, 20, 21, 22]


@dataclass(frozen=True)
class WlsResult:
    coef: pd.Series
    stderr: pd.Series
    n_obs: int


def _base_filter(df: pd.DataFrame, *, target_year: int) -> pd.DataFrame:
    return df[
        (df["year"].astype(int) == int(target_year))
        & (df["weekday_weekend"].astype(str) == "both")
        & (df["has_pv"].astype(str).isin(["No", "All"]))
    ].copy()


def _weighted_group_mean(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    weight_col: str,
    out_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["_w"] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)
    out["_x"] = pd.to_numeric(out[value_col], errors="coerce")
    out["_wx"] = out["_x"] * out["_w"]
    grouped = out.groupby(group_cols, as_index=False).agg(_wx=("_wx", "sum"), _w=("_w", "sum"))
    grouped[out_col] = grouped["_wx"] / grouped["_w"].replace({0: np.nan})
    return grouped.drop(columns=["_wx"])


def _period_from_hour(hour: int, am_hours: set[int], pm_hours: set[int]) -> str:
    h = int(hour)
    if h in am_hours:
        return "am"
    if h in pm_hours:
        return "pm"
    return "other"


def _wls(design: pd.DataFrame, y: pd.Series, weights: pd.Series) -> WlsResult:
    x = design.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    m = np.isfinite(yv) & np.isfinite(w) & (w > 0)
    m &= np.all(np.isfinite(x), axis=1)
    x = x[m]
    yv = yv[m]
    w = w[m]
    if len(yv) == 0:
        raise ValueError("No valid rows for weighted regression")

    sw = np.sqrt(w)
    xw = x * sw[:, None]
    yw = yv * sw
    beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("Regression design matrix is rank deficient")

    resid = yv - (x @ beta)
    dof = max(1, len(yv) - x.shape[1])
    sigma2 = float(np.sum(w * resid * resid) / dof)
    xtwx = x.T @ (w[:, None] * x)
    cov = sigma2 * np.linalg.pinv(xtwx)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return WlsResult(
        coef=pd.Series(beta, index=design.columns),
        stderr=pd.Series(se, index=design.columns),
        n_obs=int(len(yv)),
    )


def _run_am_pm_regression(df: pd.DataFrame) -> WlsResult:
    z = df[df["period"].isin(["am", "pm"])].copy()
    z["is_pm"] = (z["period"] == "pm").astype(int)
    z["y"] = np.log(pd.to_numeric(z["mean_kwh"], errors="coerce").clip(lower=1e-9))

    dummies = pd.get_dummies(z["seg3_value"].astype(str), prefix="seg3", drop_first=True, dtype=float)
    design = pd.concat([pd.Series(1.0, index=z.index, name="intercept"), z["is_pm"].astype(float), dummies], axis=1)
    return _wls(design, z["y"], z["n_rounded"])


def _run_temperature_band_regression(df: pd.DataFrame, *, reference_band: str) -> WlsResult:
    z = df[df["period"].isin(["am", "pm"])].copy()
    z["is_pm"] = (z["period"] == "pm").astype(int)
    z["y"] = np.log(pd.to_numeric(z["mean_kwh"], errors="coerce").clip(lower=1e-9))
    z["band"] = z["seg3_value"].astype(str)

    bands = sorted(b for b in z["band"].unique() if b != reference_band)
    design_parts = [pd.Series(1.0, index=z.index, name="intercept"), z["is_pm"].astype(float)]
    for band in bands:
        bcol = (z["band"] == band).astype(float)
        design_parts.append(pd.Series(bcol, index=z.index, name=f"band[{band}]"))
        design_parts.append(pd.Series(bcol * z["is_pm"], index=z.index, name=f"is_pm:band[{band}]"))
    design = pd.concat(design_parts, axis=1)
    return _wls(design, z["y"], z["n_rounded"])


def extract_summer_baseline_by_property_type(
    daily_targets: pd.DataFrame,
    *,
    target_year: int,
    summer_months: list[int],
    min_n: int,
) -> pd.DataFrame:
    d = _base_filter(daily_targets, target_year=target_year)
    d = d[
        (d["period_type"].astype(str) == "monthly")
        & (d["heating_fuel"].astype(str) == "All")
        & (d["seg3_var"].astype(str) == "building_type")
        & (d["quantity"].astype(str).isin(FUEL_MAP))
    ].copy()
    d["month"] = pd.to_numeric(d["month"], errors="coerce").astype("Int64")
    d = d[d["month"].isin([int(m) for m in summer_months])].copy()
    d["n_rounded"] = pd.to_numeric(d["n_rounded"], errors="coerce").fillna(0.0)
    d = d[d["n_rounded"] >= float(min_n)].copy()
    d["fuel"] = d["quantity"].astype(str).map(FUEL_MAP)
    d["seg3_value"] = d["seg3_value"].astype(str)

    baseline = _weighted_group_mean(
        d,
        group_cols=["fuel", "seg3_value"],
        value_col="mean",
        weight_col="n_rounded",
        out_col="summer_kwh_per_home_day",
    ).drop(columns=["_w"], errors="ignore")
    months = (
        d.groupby(["fuel", "seg3_value"], as_index=False)
        .agg(months_observed=("month", lambda s: int(pd.Series(s).nunique())), n_total=("n_rounded", "sum"))
    )
    out = baseline.merge(months, on=["fuel", "seg3_value"], how="left")
    return out.sort_values(["fuel", "summer_kwh_per_home_day"], ascending=[True, False]).reset_index(drop=True)


def build_hourly_profiles(
    hourly_targets: pd.DataFrame,
    *,
    target_year: int,
    seg3_var: str,
    min_n: int,
    am_hours: list[int],
    pm_hours: list[int],
) -> pd.DataFrame:
    h = _base_filter(hourly_targets, target_year=target_year)
    h = h[
        (h["heating_fuel"].astype(str) == "All")
        & (h["seg3_var"].astype(str) == str(seg3_var))
        & (h["quantity"].astype(str).isin(FUEL_MAP))
    ].copy()
    h["n_rounded"] = pd.to_numeric(h["n_rounded"], errors="coerce").fillna(0.0)
    h = h[h["n_rounded"] >= float(min_n)].copy()
    h["hour"] = pd.to_numeric(h["hour"], errors="coerce").astype(int)
    h["fuel"] = h["quantity"].astype(str).map(FUEL_MAP)
    h["seg3_value"] = h["seg3_value"].astype(str)

    grouped = _weighted_group_mean(
        h,
        group_cols=["fuel", "seg3_value", "hour"],
        value_col="mean_kwh",
        weight_col="n_rounded",
        out_col="mean_kwh",
    ).rename(columns={"_w": "n_rounded"})

    grouped["period"] = grouped["hour"].apply(lambda x: _period_from_hour(x, set(am_hours), set(pm_hours)))
    totals = grouped.groupby(["fuel", "seg3_value"], as_index=False).agg(total_kwh=("mean_kwh", "sum"))
    grouped = grouped.merge(totals, on=["fuel", "seg3_value"], how="left")
    grouped["hourly_share"] = grouped["mean_kwh"] / grouped["total_kwh"].replace({0: np.nan})
    return grouped.sort_values(["fuel", "seg3_value", "hour"]).reset_index(drop=True)


def summarize_am_pm_shares(hourly_profiles: pd.DataFrame) -> pd.DataFrame:
    z = hourly_profiles[hourly_profiles["period"].isin(["am", "pm", "other"])].copy()
    out = (
        z.groupby(["fuel", "seg3_value", "period"], as_index=False)
        .agg(period_kwh=("mean_kwh", "sum"), period_share=("hourly_share", "sum"))
    )
    piv = out.pivot_table(index=["fuel", "seg3_value"], columns="period", values="period_share", aggfunc="first").reset_index()
    for col in ["am", "pm", "other"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv["pm_to_am_share_ratio"] = piv["pm"] / piv["am"].replace({0: np.nan})
    return piv.sort_values(["fuel", "seg3_value"]).reset_index(drop=True)


def floor_residual_decomposition(
    hourly_profiles: pd.DataFrame,
    *,
    floor_quantile: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = hourly_profiles.copy()
    if z.empty:
        return z, pd.DataFrame(columns=["fuel", "seg3_value", "floor_kwh"])

    floors = (
        z.groupby(["fuel", "seg3_value"], as_index=False)
        .agg(floor_kwh=("mean_kwh", lambda s: float(pd.to_numeric(pd.Series(s), errors="coerce").quantile(float(floor_quantile)))))
    )
    out = z.merge(floors, on=["fuel", "seg3_value"], how="left")
    out["floor_kwh"] = pd.to_numeric(out["floor_kwh"], errors="coerce").fillna(0.0)
    out["residual_kwh"] = (pd.to_numeric(out["mean_kwh"], errors="coerce") - out["floor_kwh"]).clip(lower=0.0)
    return out, floors


def build_residual_profile_24h(
    hourly_with_residual: pd.DataFrame,
    *,
    fuel: str,
) -> list[float]:
    z = hourly_with_residual[hourly_with_residual["fuel"].astype(str) == str(fuel)].copy()
    if z.empty:
        return [1.0] * 24
    agg = (
        z.groupby("hour", as_index=False)
        .agg(
            wr=("residual_kwh", lambda s: float((pd.to_numeric(s, errors="coerce") * z.loc[s.index, "n_rounded"]).sum())),
            w=("n_rounded", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        )
        .sort_values("hour")
    )
    agg["residual_mean"] = agg["wr"] / agg["w"].replace({0: np.nan})
    full = pd.DataFrame({"hour": np.arange(24, dtype=int)}).merge(agg[["hour", "residual_mean"]], on="hour", how="left")
    y = pd.to_numeric(full["residual_mean"], errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    mean_y = float(y.mean())
    if mean_y <= 0:
        return [1.0] * 24
    prof = (y / mean_y).clip(lower=0.2, upper=3.0)
    prof_mean = float(np.nanmean(prof.to_numpy(dtype=float)))
    if prof_mean > 0:
        prof = prof / prof_mean
    return [float(v) for v in prof.to_numpy()]


def window_peak_multipliers_from_profile(
    profile_24h: list[float],
    *,
    morning_hours: list[int],
    evening_hours: list[int],
    shoulder_hours: list[int] | None = None,
    clip: tuple[float, float] = (0.7, 1.6),
) -> tuple[float, float]:
    p = np.asarray(profile_24h, dtype=float)
    if p.size != 24 or not np.isfinite(p).any():
        return 1.0, 1.0
    if shoulder_hours is None:
        shoulder_hours = [10, 11, 12, 13, 14, 15]
    sh = float(np.nanmean(p[np.asarray(shoulder_hours, dtype=int)]))
    sh = max(sh, 1e-9)
    am = float(np.nanmean(p[np.asarray(morning_hours, dtype=int)]))
    pm = float(np.nanmean(p[np.asarray(evening_hours, dtype=int)]))
    lo, hi = float(clip[0]), float(clip[1])
    return max(lo, min(hi, am / sh)), max(lo, min(hi, pm / sh))


def estimate_am_pm_peak_coefficients(hourly_profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fuel, g in hourly_profiles.groupby("fuel", observed=True):
        fit = _run_am_pm_regression(g)
        beta_pm = float(fit.coef.get("is_pm", np.nan))
        se_pm = float(fit.stderr.get("is_pm", np.nan))
        pm_over_am = float(np.exp(beta_pm))
        morning_mult = float(1.0 / np.sqrt(pm_over_am))
        evening_mult = float(np.sqrt(pm_over_am))
        rows.append(
            {
                "fuel": str(fuel),
                "n_obs": fit.n_obs,
                "beta_pm_log": beta_pm,
                "beta_pm_log_stderr": se_pm,
                "pm_over_am_multiplier": pm_over_am,
                "pm_over_am_pct": 100.0 * (pm_over_am - 1.0),
                "suggested_morning_mult": morning_mult,
                "suggested_evening_mult": evening_mult,
            }
        )
    return pd.DataFrame(rows).sort_values("fuel").reset_index(drop=True)


def estimate_temperature_band_coefficients(
    hourly_targets: pd.DataFrame,
    *,
    target_year: int,
    min_n: int,
    am_hours: list[int],
    pm_hours: list[int],
    reference_band: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = build_hourly_profiles(
        hourly_targets,
        target_year=target_year,
        seg3_var="temperature_band",
        min_n=min_n,
        am_hours=am_hours,
        pm_hours=pm_hours,
    )
    rows = []
    model_rows = []
    for fuel, g in hourly.groupby("fuel", observed=True):
        fit = _run_temperature_band_regression(g, reference_band=reference_band)
        beta_pm_base = float(fit.coef.get("is_pm", 0.0))
        all_bands = sorted(g["seg3_value"].astype(str).unique())
        for band in all_bands:
            if band == reference_band:
                beta_band = 0.0
                beta_int = 0.0
            else:
                beta_band = float(fit.coef.get(f"band[{band}]", np.nan))
                beta_int = float(fit.coef.get(f"is_pm:band[{band}]", np.nan))

            am_mult = float(np.exp(beta_band))
            pm_mult = float(np.exp(beta_band + beta_pm_base + beta_int))
            pm_over_am = float(np.exp(beta_pm_base + beta_int))
            rows.append(
                {
                    "fuel": str(fuel),
                    "temperature_band": str(band),
                    "reference_band": str(reference_band),
                    "n_obs": fit.n_obs,
                    "am_multiplier_vs_ref_am": am_mult,
                    "pm_multiplier_vs_ref_am": pm_mult,
                    "pm_over_am_multiplier": pm_over_am,
                    "band_log_coef": beta_band,
                    "pm_interaction_log_coef": beta_int,
                }
            )
            model_rows.append(
                {
                    "fuel": str(fuel),
                    "temperature_band": str(band),
                    "suggested_am_mult": am_mult,
                    "suggested_pm_mult": pm_mult,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["fuel", "temperature_band"]).reset_index(drop=True),
        pd.DataFrame(model_rows).sort_values(["fuel", "temperature_band"]).reset_index(drop=True),
    )


def build_suggested_override_yaml(
    *,
    target_year: int,
    summer_months: list[int],
    baseline_property: pd.DataFrame,
    am_pm_coeffs: pd.DataFrame,
    temp_coeffs: pd.DataFrame,
    heating_profile_24h: list[float] | None = None,
    dhw_profile_24h: list[float] | None = None,
    awake_home_spike_mult: float = 1.0,
    sleep_home_spike_mult: float = 0.3,
) -> dict:
    gas_row = am_pm_coeffs[am_pm_coeffs["fuel"] == "gas"]
    elec_row = am_pm_coeffs[am_pm_coeffs["fuel"] == "electric"]
    gas_morning = float(gas_row["suggested_morning_mult"].iloc[0]) if not gas_row.empty else 1.0
    gas_evening = float(gas_row["suggested_evening_mult"].iloc[0]) if not gas_row.empty else 1.0
    elec_morning = float(elec_row["suggested_morning_mult"].iloc[0]) if not elec_row.empty else 1.0
    elec_evening = float(elec_row["suggested_evening_mult"].iloc[0]) if not elec_row.empty else 1.0

    base_rows = []
    for _, r in baseline_property.iterrows():
        base_rows.append(
            {
                "fuel": str(r["fuel"]),
                "building_type": str(r["seg3_value"]),
                "summer_kwh_per_home_day": float(r["summer_kwh_per_home_day"]),
            }
        )

    temp_rows = []
    for _, r in temp_coeffs.iterrows():
        temp_rows.append(
            {
                "fuel": str(r["fuel"]),
                "temperature_band": str(r["temperature_band"]),
                "am_mult": float(r["suggested_am_mult"]),
                "pm_mult": float(r["suggested_pm_mult"]),
            }
        )

    return {
        "meta": {
            "name": f"serl_feature_extraction_{target_year}",
            "date": pd.Timestamp.utcnow().date().isoformat(),
            "notes": "Direct SERL feature extraction for baseline + AM/PM peaks + temperature-band effects",
        },
        "inputs": {
            "target_year": int(target_year),
            "summer_months": [int(m) for m in summer_months],
        },
        "model": {
            # Gas-led mapping for heating and electric-led mapping for DHW peaks.
            "heating_peak_morning_mult": gas_morning,
            "heating_peak_evening_mult": gas_evening,
            "dhw_peak_morning_mult": elec_morning,
            "dhw_peak_evening_mult": elec_evening,
            "heating_profile_24h": heating_profile_24h if heating_profile_24h is not None else [1.0] * 24,
            "dhw_profile_24h": dhw_profile_24h if dhw_profile_24h is not None else [1.0] * 24,
            "awake_home_spike_mult": float(awake_home_spike_mult),
            "sleep_home_spike_mult": float(sleep_home_spike_mult),
        },
        "derived_tables": {
            "summer_baseline_by_building_type": base_rows,
            "temperature_band_am_pm_coefficients": temp_rows,
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simplified SERL feature extraction pipeline")
    ap.add_argument("--target-year", type=int, default=2023)
    ap.add_argument("--summer-months", nargs="+", type=int, default=[6, 7, 8])
    ap.add_argument("--min-n", type=int, default=80)
    ap.add_argument("--am-hours", nargs="+", type=int, default=DEFAULT_AM_HOURS)
    ap.add_argument("--pm-hours", nargs="+", type=int, default=DEFAULT_PM_HOURS)
    ap.add_argument("--reference-band", type=str, default="15_to_20")
    ap.add_argument("--out-dir", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from()
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (repo_root / "results" / f"serl_feature_extract_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_targets, hourly_targets = load_serl_targets(repo_root)

    baseline = extract_summer_baseline_by_property_type(
        daily_targets,
        target_year=args.target_year,
        summer_months=[int(m) for m in args.summer_months],
        min_n=args.min_n,
    )
    baseline.to_csv(out_dir / "summer_baseline_by_building_type.csv", index=False)

    hourly_building = build_hourly_profiles(
        hourly_targets,
        target_year=args.target_year,
        seg3_var="building_type",
        min_n=args.min_n,
        am_hours=[int(h) for h in args.am_hours],
        pm_hours=[int(h) for h in args.pm_hours],
    )
    hourly_building.to_csv(out_dir / "hourly_profiles_building_type.csv", index=False)

    am_pm_shares = summarize_am_pm_shares(hourly_building)
    am_pm_shares.to_csv(out_dir / "am_pm_shares_building_type.csv", index=False)

    am_pm_coeffs = estimate_am_pm_peak_coefficients(hourly_building)
    am_pm_coeffs.to_csv(out_dir / "am_pm_peak_coefficients.csv", index=False)

    temp_effects, temp_model_coeffs = estimate_temperature_band_coefficients(
        hourly_targets,
        target_year=args.target_year,
        min_n=args.min_n,
        am_hours=[int(h) for h in args.am_hours],
        pm_hours=[int(h) for h in args.pm_hours],
        reference_band=str(args.reference_band),
    )
    temp_effects.to_csv(out_dir / "temperature_band_am_pm_effects.csv", index=False)
    temp_model_coeffs.to_csv(out_dir / "temperature_band_am_pm_model_coefficients.csv", index=False)

    payload = build_suggested_override_yaml(
        target_year=args.target_year,
        summer_months=[int(m) for m in args.summer_months],
        baseline_property=baseline,
        am_pm_coeffs=am_pm_coeffs,
        temp_coeffs=temp_model_coeffs,
    )
    with (out_dir / "serl_feature_extraction_override.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)

    print("Output dir:", out_dir)
    print("Wrote:", out_dir / "summer_baseline_by_building_type.csv")
    print("Wrote:", out_dir / "hourly_profiles_building_type.csv")
    print("Wrote:", out_dir / "am_pm_shares_building_type.csv")
    print("Wrote:", out_dir / "am_pm_peak_coefficients.csv")
    print("Wrote:", out_dir / "temperature_band_am_pm_effects.csv")
    print("Wrote:", out_dir / "temperature_band_am_pm_model_coefficients.csv")
    print("Wrote:", out_dir / "serl_feature_extraction_override.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
