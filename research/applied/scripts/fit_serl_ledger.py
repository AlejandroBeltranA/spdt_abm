"""SERL -> parameter ledger -> ABM config. The single source of truth.

This script replaces the old multi-script + model-in-the-loop calibration
(fit_cohort_params.py, calibrate_serl.py, fit_*.py x13 and the v5/v6/v7 configs).
Every ABM parameter is read DIRECTLY off SERL aggregate cells -- there is NO
model run anywhere in this file. You can trace each number to a SERL filter and
a one-line formula in the emitted PARAMETER_LEDGER.md.

The model the ABM computes, per dwelling, per hour:

    E_elec(i,h,mo) = Base_elec(i)*shape_elec(h,mo)              # house standing load
                   + People(i)                                  # per occupant
                   + Heat_elec(i)*HDD(i,t)*shape_heat(h,mo)     # if electric-heated
    E_gas(i,h,mo)  = Base_gas(i)*shape_gas(h,mo)                # cooking / hot water
                   + Heat_gas(i)*HDD(i,t)*shape_heat(h,mo)      # if gas-heated

where each per-dwelling factor is an anchor times SERL marginal multipliers:

    Base_gas(i)  = A_gas  * type_b(i) * imd(i)
    Heat_gas(i)  = S_gas  * area(i) * sap(i) * age(i)
    HDD(i,t)     = max(0, setpoint - T_outdoor(i,t))

ONE-EFFECT-ONE-DIMENSION RULE. SERL only gives one-way marginals and the
dimensions are correlated (a detached house is also large-area). Multiplying all
13 marginals would double-count. So each physical effect is assigned to exactly
one dimension and read only from that marginal:
    size        -> floor_area_m2        (heating slope)
    efficiency  -> currentEnergyRating + building_age   (heating slope)
    fuel/system -> heating_fuel cohort  (separate anchors + slopes)
    standing    -> building_type        (summer baseline level)
    deprivation -> IMD_quintile         (summer baseline level)
This assignment is the only modelling judgement in the file; it is printed in
the ledger so it is auditable.

    .venv/bin/python research/applied/scripts/fit_serl_ledger.py
Writes the config straight to household_energy/calibrated_config.yaml (the tracked
path the engine reads -- single source of truth, no promote step; git diff is the
audit) and the audit docs to results/serl_ledger/{PARAMETER_LEDGER.md, ASSUMPTIONS.md}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[3]
DAILY = REPO / "data/serl_8963_targets/daily_targets.csv"
DIURNAL = REPO / "data/serl_8963_targets/diurnal_targets_hourly_mean.csv"
OUTDIR = REPO / "results/serl_ledger"           # audit docs + caches (gitignored)
# The config is written straight to the tracked path the engine reads. There is
# no separate candidate + promote step: running this file (or notebook 1) IS the
# calibration, and `git diff household_energy/calibrated_config.yaml` is the audit.
CONFIG_OUT = REPO / "household_energy" / "calibrated_config.yaml"

# ---- fixed analysis choices (documented in the ledger) ----------------------
YEAR = 2023            # latest full SERL year (held out from any prior fitting)
WKND = "both"          # weekday+weekend combined
PV = "No"              # exclude PV homes: their imports understate baseline;
                       # PV generation is a separate term, not part of demand.
SUMMER = [6, 7, 8]     # baseline window: negligible space heating

# reference categories (the denominator of every marginal multiplier)
REF = {
    "building_type": "Terraced",
    "currentEnergyRating": "D",
    "building_age": "1950 - 1975",
    "floor_area_m2": "51 to 100",
    "IMD_quintile": "3",
}

# SERL building_type label -> ABM property-type key(s). SERL does not split
# terraced into mid/end or flats into block sizes, so one SERL class maps to
# several ABM keys with the same multiplier (noted in the ledger).
TYPE_MAP = {
    "Detached": ["detached house"],
    "Semi-detached": ["semi-detached house"],
    "Terraced": ["mid-terraced house", "end-terraced house"],
    "Purpose-built flat": ["block of flats", "large block of flats"],
    "Converted flat or shared house": [
        "small block of flats/dwelling converted in to flats",
        "flat in mixed use building",
    ],
}

LEDGER: list[dict] = []   # one row per emitted parameter, full provenance


def rec(param, value, source, formula, n=None, ci=None, note=""):
    """Append a provenance row and return the value (so callers stay terse)."""
    LEDGER.append(dict(param=param, value=value, source=source, formula=formula,
                       n=n, ci=ci, note=note))
    return value


# =============================================================================
# SERL accessors
# =============================================================================
_D = pd.read_csv(DAILY)


def cells(quantity, hf, seg3_var="none", seg3_value="none", period_type="monthly"):
    """Return the SERL daily-target rows matching one fully-specified cell."""
    s = _D[(_D.quantity == quantity) & (_D.heating_fuel == hf) &
           (_D.seg3_var == seg3_var) & (_D.seg3_value.astype(str) == str(seg3_value)) &
           (_D.period_type == period_type) & (_D.year == YEAR) &
           (_D.weekday_weekend == WKND) & (_D.has_pv == PV)]
    return s


def summer_baseline(quantity, hf, seg3_var="none", seg3_value="none"):
    """Mean daily kWh over Jun-Aug for a cell (the standing/baseline load)."""
    s = cells(quantity, hf, seg3_var, seg3_value).set_index("month")["mean"]
    s = pd.to_numeric(s, errors="coerce").reindex(SUMMER)
    return float(s.mean())


def hdd_slope(quantity, hf, seg3_var="none", seg3_value="none"):
    """Weighted OLS slope of monthly daily-kWh on SERL mean_hdd (kWh/HDD/day).

    Returns (slope, intercept, r2, n). The intercept is an independent estimate
    of the baseline; we cross-check it against summer_baseline in the ledger.
    """
    s = cells(quantity, hf, seg3_var, seg3_value).dropna(subset=["mean", "mean_hdd"])
    if len(s) < 3:
        return None
    x = s["mean_hdd"].to_numpy(float)
    y = s["mean"].to_numpy(float)
    w = np.sqrt(pd.to_numeric(s["n_rounded"], errors="coerce").fillna(1).to_numpy(float))
    W = np.diag(w)
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(W @ X, w * y, rcond=None)[0]
    a, b = float(beta[0]), float(beta[1])
    pred = a + b * x
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - np.average(y, weights=w)) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b, a, r2, len(s)


def joint_setpoint_slope(quantity, hf):
    """Fit baseline + slope*max(0, setpoint - T) across the 6 temperature bands.

    This pins the heating hinge (setpoint) and the per-degree slope on the SAME
    temperature convention, so there is no HDD-base mismatch to fudge later.
    Returns dict(baseline, setpoint, slope_per_day, n).
    """
    s = _D[(_D.quantity == quantity) & (_D.heating_fuel == hf) &
           (_D.seg3_var == "temperature_band") & (_D.period_type == "annual") &
           (_D.year == YEAR) & (_D.weekday_weekend == WKND) & (_D.has_pv == PV)]
    s = s.dropna(subset=["mean", "mean_temp"]).sort_values("mean_temp")
    T = s["mean_temp"].to_numpy(float)
    y = s["mean"].to_numpy(float)
    w = np.sqrt(pd.to_numeric(s["n_rounded"], errors="coerce").fillna(1).to_numpy(float))
    fit = least_squares(
        lambda p: w * (p[0] + p[1] * np.maximum(0.0, p[2] - T) - y),
        x0=[y.min(), 1.0, 15.0], bounds=([0, 0, 8], [50, 30, 22]))
    base, slope, setp = (float(v) for v in fit.x)
    return dict(baseline=base, setpoint=setp, slope_per_day=slope, n=len(s))


def marginal(quantity, hf, seg3_var, *, kind):
    """Per-category multipliers vs the reference category.

    kind='slope'    -> ratio of HDD slopes  (heating sensitivity by category)
    kind='baseline' -> ratio of summer baselines (standing load by category)
    Returns {seg3_value: (mult, n)}; reference category is exactly 1.0.
    """
    vals = sorted(_D[_D.seg3_var == seg3_var].seg3_value.dropna().unique().astype(str))
    ref = REF[seg3_var]
    if kind == "slope":
        ref_fit = hdd_slope(quantity, hf, seg3_var, ref)
        if ref_fit is None:
            raise ValueError(f"no slope for ref {seg3_var}={ref}")
        ref_v = ref_fit[0]
        get = lambda v: (hdd_slope(quantity, hf, seg3_var, v) or (None,))[0]
    else:
        ref_v = summer_baseline(quantity, hf, seg3_var, ref)
        get = lambda v: summer_baseline(quantity, hf, seg3_var, v)
    out = {}
    for v in vals:
        gv = get(v)
        if gv is None or not np.isfinite(gv) or ref_v in (None, 0):
            continue
        n = int(pd.to_numeric(
            cells(quantity, hf, seg3_var, v)["n_rounded"], errors="coerce").max() or 0)
        out[v] = (round(gv / ref_v, 4), n)
    return out


# =============================================================================
# SHAPES (mean-1.0 hourly and seasonal profiles)
# =============================================================================
_H = pd.read_csv(DIURNAL)


def diurnal_shape(quantity, hf):
    """24-element mean-1.0 hourly profile for a fuel cohort (seg3=none)."""
    s = _H[(_H.quantity == quantity) & (_H.heating_fuel == hf) &
           (_H.seg3_var == "none") & (_H.year == YEAR) &
           (_H.weekday_weekend == WKND) & (_H.has_pv == PV)]
    p = pd.to_numeric(s.set_index("hour")["mean_kwh"], errors="coerce").reindex(range(24))
    return (p / p.mean()).round(6).tolist()


def heating_diurnal_storage():
    """24-element mean-1.0 charge profile for electric STORAGE heaters only.

    SERL's ``central_heating_type`` splits the electric cohort: ``Electric storage
    radiators`` peak overnight (hour 2, night/day ~2.9 -- Economy-7 off-peak
    charge), while ``Electric radiators`` (direct) peak at 18:00 like gas. So only
    STORAGE homes need a night profile; direct electric follows demand and keeps
    the gas temperature-conditioned profile. Fit the charge shape from the storage
    curve, subtracting the appliance floor (gas-boiler homes' electricity, the
    pure no-electric-heat shape) scaled so the midday residual is ~0 -- storage
    draws almost nothing 10:00-16:00 (it releases stored heat passively). Applied
    mean-preservingly, so only the timing of storage heating moves.

    Caveat: SERL hourly is annual-average (no seasonal split); the appliance floor
    is a cross-cohort proxy. Documented in ASSUMPTIONS.
    """
    def _hourly(val):
        s = _H[(_H.quantity == "Electricity imports") & (_H.seg3_var == "central_heating_type") &
               (_H.seg3_value.astype(str) == val) & (_H.year == YEAR) &
               (_H.weekday_weekend == WKND) & (_H.has_pv == PV)]
        return s.groupby("hour")["mean_kwh"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()).reindex(range(24)).to_numpy(float)
    S = _hourly("Electric storage radiators")
    G = _hourly("Gas boiler")                            # pure-appliance shape (no electric heat)
    Sn, Gn = S / np.nansum(S), G / np.nansum(G)
    # appliance floor: storage draws ~nothing midday (10-16), so midday elec is appliances
    k = float(np.nanmean(Sn[10:16]) / np.nanmean(Gn[10:16]))
    heating = np.clip(Sn - k * Gn, 0.0, None)
    prof = heating / heating.mean()
    return [round(float(x), 6) for x in prof]


def seasonal_shape(quantity, hf, *, hdd_weighted):
    """12-element monthly profile. Baseline shapes use a plain mean-1.0; heating
    shapes are HDD-weighted-mean-1.0 so the annual heating energy is preserved
    when the smooth profile replaces the old hard on/off heating-months gate."""
    s = cells(quantity, hf).copy()
    s = s.dropna(subset=["mean"])
    m = pd.to_numeric(s.set_index("month")["mean"], errors="coerce").reindex(range(1, 13))
    if hdd_weighted:
        hdd = pd.to_numeric(s.set_index("month")["mean_hdd"], errors="coerce").reindex(range(1, 13))
        # isolate the heating component (subtract summer floor), weight by HDD share
        heat = (m - m.loc[SUMMER].mean()).clip(lower=0)
        wmean = float((heat * hdd).sum() / hdd.sum()) if hdd.sum() else float(heat.mean())
        prof = (heat / wmean) if wmean else heat
    else:
        prof = m / m.mean()
    return prof.round(6).tolist()


# ---- diurnal-shape calibration (schedule-retired architecture) --------------
COOKING_BAND = "15_to_20"   # above the ~16.5C balance point -> ~no space heat
HEATING_BANDS = {"-5_to_0": -2.5, "0_to_5": 2.5, "5_to_10": 7.5, "10_to_15": 12.5}


def diurnal_band(quantity, hf, band):
    """24-vector of hourly mean kWh for one outdoor-temperature band."""
    s = _H[(_H.quantity == quantity) & (_H.heating_fuel == hf) &
           (_H.seg3_var == "temperature_band") & (_H.seg3_value.astype(str) == band) &
           (_H.year == YEAR) & (_H.weekday_weekend == WKND) & (_H.has_pv == PV)]
    return pd.to_numeric(s.set_index("hour")["mean_kwh"],
                         errors="coerce").reindex(range(24)).to_numpy(float)


def gas_cooking_profile():
    """Cooking/DHW gas day-shape (mean-1.0): the warm-band gas diurnal, where
    space heating is ~zero, so the twin meal peaks are the pure non-heating
    signal. Fixes the overnight gas floor the schedule path never had."""
    v = diurnal_band("Gas", "Gas", COOKING_BAND)
    return (v / v.mean()).round(6).tolist()


def heating_temp_profiles():
    """Space-heating day-shape per outdoor-temperature band (each mean-1.0).

    Subtract the cooking day-shape in ABSOLUTE kWh (cooking is ~temperature-
    invariant), clip at zero, normalise. Cold bands come out flat (heating runs
    all day), mild bands peaky (morning/evening bursts) -- the runtime
    interpolates between bands by ambient temperature, which reproduces the
    HDD-by-hour interaction by construction."""
    cook = diurnal_band("Gas", "Gas", COOKING_BAND)
    out = {}
    for band, tmid in HEATING_BANDS.items():
        heat = np.maximum(diurnal_band("Gas", "Gas", band) - cook, 0.0)
        if heat.sum() <= 0:
            continue
        p = heat / heat.mean()
        out[band] = {"temp_mid_C": tmid, "profile": [round(float(x), 6) for x in p]}
    return out


def serl_shares(seg3_var, quantity, hf):
    """SERL population share per category (annual n_rounded) for one seg3_var."""
    s = _D[(_D.quantity == quantity) & (_D.heating_fuel == hf) &
           (_D.seg3_var == seg3_var) & (_D.period_type == "annual") &
           (_D.year == YEAR) & (_D.weekday_weekend == WKND) & (_D.has_pv == PV)]
    return s.groupby(s.seg3_value.astype(str))["n_rounded"].max()


def recenter(mults, seg3_var, quantity, hf):
    """Renormalise a reference-normalised multiplier dict to SERL-population
    mean 1.0, so that anchor(=cohort mean) x multipliers reproduces the cohort
    mean. Without this the engine multiplies a cohort-mean anchor by multipliers
    that average >1 over the stock, inflating the total. Returns (dict, popmean)."""
    sh = serl_shares(seg3_var, quantity, hf)
    num = den = 0.0
    for k, m in mults.items():
        w = float(sh.get(str(k), 0.0))
        num += w * m
        den += w
    mean = num / den if den else 1.0
    return {k: round(m / mean, 4) for k, m in mults.items()}, round(mean, 4)


ABM2SERL = {abm: serl for serl, abms in TYPE_MAP.items() for abm in abms}


def _area_band(a):
    try:
        a = float(a)
        if not np.isfinite(a):
            a = 90.0
    except (TypeError, ValueError):
        a = 90.0
    return ("50 or less" if a <= 50 else "51 to 100" if a <= 100 else
            "101 to 150" if a <= 150 else "151 to 200" if a <= 200 else "Over 200")


def type_area_joint():
    """P(floor-area band | SERL building type), pooled over the five-city EPC
    stocks (~580k dwellings). Cached at OUTDIR/type_area_joint.csv.

    SERL publishes only ONE-WAY marginals, so the observed type ratio mixes the
    type's own effect with its size mix (detached homes are also large). This
    joint -- a stock DISTRIBUTION, not an energy fit -- is the deconvolution
    key: expected-from-area(type) = sum_b P(band|type) * area_mult(band), and
    the residual type effect is observed / expected. Validation (2026-07)
    showed neutralising the type entirely under-disperses demand exactly where
    detached/semi share is high; this restores the beyond-size part only."""
    cache = OUTDIR / "type_area_joint.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    import geopandas as gpd
    frames = []
    for slug in ["newcastle", "sunderland", "waltham_forest", "manchester", "brighton_and_hove"]:
        p = REPO / f"data/epc_abm_{slug}.geojson"
        if not p.exists():
            p = REPO / f"data/epc_abm_{slug}.gpkg"
        frames.append(gpd.read_file(p, ignore_geometry=True,
                                    columns=["property_type", "floor_area_m2"]))
    g = pd.concat(frames, ignore_index=True)
    g["serl_type"] = g.property_type.map(ABM2SERL)
    g["band"] = [_area_band(a) for a in g.floor_area_m2]
    j = g.dropna(subset=["serl_type"]).groupby(["serl_type", "band"]).size().unstack(fill_value=0)
    j = j.div(j.sum(axis=1), axis=0).round(6)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    j.to_csv(cache)
    return j


def area_occupancy_mean():
    """E[household occupancy | floor-area band], pooled over the five-city
    HIDP-matched synthpop. Cached at OUTDIR/area_occupancy_mean.csv.

    SERL publishes floor_area and num_occupants only as separate one-way
    marginals, so its area-baseline gradient silently absorbs the fact that
    larger homes hold more people. The engine then re-adds a per-person
    deviation load ``(n_occ - panel_mean) * slope`` on top, double-counting
    that occupancy (measured 2026-07: the realised electricity-area gradient
    ran ~2x DESNZ, ~60% from an over-steep base, ~20% from this double-count).
    This map -- occupancy by area on the population the model actually runs --
    is the deconvolution key that lets the area bands carry the standing-load
    gradient NET of occupancy, exactly mirroring ``type_area_joint()``."""
    cache = OUTDIR / "area_occupancy_mean.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0)["occ"]
        return {str(k): float(v) for k, v in s.items()}
    import geopandas as gpd
    frames = []
    for slug in ["newcastle", "sunderland", "waltham_forest", "manchester", "brighton_and_hove"]:
        epc = REPO / f"data/epc_abm_{slug}.geojson"
        if not epc.exists():
            epc = REPO / f"data/epc_abm_{slug}.gpkg"
        g = gpd.read_file(epc, ignore_geometry=True, columns=["UPRN", "floor_area_m2"])
        g["UPRN"] = g["UPRN"].astype(str).str.strip()
        hh = pd.read_csv(REPO / f"data/{slug}_hidp_uprn_matches_tiered.csv",
                         usecols=["uprn_chr", "hh_n_people"])
        hh["uprn_chr"] = hh["uprn_chr"].astype(str).str.strip()
        m = g.merge(hh, left_on="UPRN", right_on="uprn_chr", how="inner")
        frames.append(m[["floor_area_m2", "hh_n_people"]])
    d = pd.concat(frames, ignore_index=True)
    d["hh_n_people"] = pd.to_numeric(d["hh_n_people"], errors="coerce")
    d = d.dropna(subset=["hh_n_people"])
    d["band"] = [_area_band(a) for a in d.floor_area_m2]
    occ = d.groupby("band")["hh_n_people"].mean()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    occ.rename("occ").to_csv(cache)
    return {str(k): float(v) for k, v in occ.items()}


def deconfounded_area_baseline():
    """Summer-baseline electricity area ratios, NET of per-person occupancy load.

    Raw SERL summer baseline by area band ``S(a)`` mixes area-correlated standing
    load with the fact that bigger homes hold more people. The engine re-adds
    ``(n_occ - panel_mean) * sum(slope_24h)`` per home
    (``model.py:_apply_serl_person_loads``), so if the area bands carried the raw
    ``S(a)`` the occupancy gradient would be counted twice. Subtract the expected
    per-person load per band so that ``base(area) + people(occ)`` reproduces
    ``S(a)`` on the modelled population::

        S_base(a)    = S(a) - (occ_bar(a) - panel_mean) * sum(pps)
        area_mult(a) = S_base(a) / S_base(ref)

    ``occ_bar(a)`` is the synthpop ``area_occupancy_mean``; ``panel_mean`` and the
    daily per-person load ``sum(pps)`` are the SAME SERL values the engine uses,
    so the subtraction cancels the engine's addition exactly. Returns
    ``{band: (ratio_vs_ref, n)}`` like ``marginal(kind='baseline')``.
    """
    ref = REF["floor_area_m2"]
    occ = area_occupancy_mean()
    pm = panel_mean_occupants()
    pp_daily = float(sum(hourly_per_person_slope("Electricity imports")))
    vals = sorted(_D[_D.seg3_var == "floor_area_m2"].seg3_value.dropna().unique().astype(str))
    absval, ncell = {}, {}
    for v in vals:
        S = summer_baseline("Electricity imports", "All", "floor_area_m2", v)
        ob = occ.get(str(v))
        if S is None or not np.isfinite(S) or ob is None:
            continue
        absval[v] = S - (ob - pm) * pp_daily
        ncell[v] = int(pd.to_numeric(
            cells("Electricity imports", "All", "floor_area_m2", v)["n_rounded"],
            errors="coerce").max() or 0)
    ref_v = absval.get(ref)
    if not ref_v or ref_v <= 0:
        raise ValueError(f"deconfounded reference band {ref} non-positive: {ref_v}")
    return {v: (round(sb / ref_v, 4), ncell[v]) for v, sb in absval.items()}


_CORR: dict = {}   # cohort anchor/slope correction factors, for the self-check


def cohort_mult_mean(mult_map, seg3_var, hf, keymap=None):
    """Subpopulation-weighted mean of a multiplier map over heating_fuel=hf.

    The multipliers are recentred to mean 1.0 over the WHOLE stock, but each fuel
    cohort has its own anchor/slope (its measured mean). A skewed cohort (e.g.
    electric-heated = mostly small flats) has a sub-population multiplier mean
    well below 1, so anchor x multipliers lands below the cohort mean. Dividing
    the anchor by this returns the correct cohort level. Uses SERL shares WITHIN
    the cohort, so it self-cancels (=1.0) for the cohort that matches the stock.
    """
    sh = serl_shares(seg3_var, "Electricity imports", hf)
    num = den = 0.0
    for val, w in sh.items():
        key = keymap.get(val) if keymap else val
        if key in mult_map:
            num += float(w) * mult_map[key]
            den += float(w)
    return num / den if den else 1.0


def heat_mult_stock():
    """Pooled five-city EPC stock, the columns needed to evaluate the engine's
    heating-slope multiplier stack. Cached at OUTDIR/heat_mult_stock.csv.

    A stock DISTRIBUTION, not an energy fit -- the same device as
    ``type_area_joint()``. SERL publishes only one-way marginals, so the joint
    distribution of (area x SAP x property type x system) that the engine's
    multipliers actually sample can only come from the stock."""
    cache = OUTDIR / "heat_mult_stock.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import geopandas as gpd
    cols = ["property_type", "floor_area_m2", "sap_rating", "retrofit_envelope_score",
            "main_heating_system", "is_electric_heating", "is_gas"]
    frames = []
    for slug in ["newcastle", "sunderland", "waltham_forest", "manchester", "brighton_and_hove"]:
        p = REPO / f"data/epc_abm_{slug}.geojson"
        if not p.exists():
            p = REPO / f"data/epc_abm_{slug}.gpkg"
        frames.append(gpd.read_file(p, ignore_geometry=True, columns=cols))
    g = pd.concat(frames, ignore_index=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    g.to_csv(cache, index=False)
    return g


def engine_heat_mult_mean(cfg, cohort):
    """Cohort mean of the FULL multiplier product the ENGINE applies to the
    heating slope, evaluated on the pooled stock.

    ``cohort_mult_mean`` above corrects for cohort skew across the SERL
    marginals it can see -- area and SAP. But ``agent.py`` multiplies the base
    slope by EIGHT factors, not two: a hard-coded SAP nudge, ``property_type_
    mult_heat``, ``_sap_multiplier(kind="slope")``, ``heat_slope_area_bands``,
    the retrofit-envelope factor, ``systems.*.heating_slope_mult``,
    ``hp_effect_mult``, and ``sap_band_mult_heating_*``. Their product averages
    ~0.50 over electric-heated homes (small, well-rated flats) against ~0.78
    over gas-heated ones, so a slope fitted on the electric-heated cohort is
    discounted a second time for structure that is already inside it -- the
    same double-count ``deconfounded_area_baseline()`` removes from the
    baseline. Dividing the fitted slope by this mean makes the multipliers pure
    within-cohort redistribution, which is what the ledger claims they are.

    Imports the engine's own helpers so the two cannot drift apart."""
    from household_energy.agent import (
        PROPERTY_TYPE_MULT_HEAT, _area_band_multiplier, _serl_sap_band)

    defaults = yaml.safe_load((REPO / "household_energy/config_defaults.yaml").read_text())
    sysc = defaults.get("systems", {})
    sapp = {"sap_lo": 40.0, "sap_hi": 90.0, "slope_mult_hi": 1.30, "slope_mult_lo": 0.70}
    sapp.update(defaults.get("model", {}).get("sap_scaling", {}))
    hp_eff = (float(defaults["model"].get("boiler_efficiency", 0.90))
              / float(defaults["model"].get("heatpump_cop_ref", 2.8)))

    g = heat_mult_stock()
    electric = (cohort == "Electric")
    sel = g.is_electric_heating == 1 if electric else ((g.is_gas == 1) & (g.is_electric_heating != 1))
    g = g[sel]
    sap = g.sap_rating.fillna(70.0).to_numpy(float)
    fa = g.floor_area_m2.fillna(90.0).to_numpy(float)
    retro = g.retrofit_envelope_score.fillna(0.5).to_numpy(float)
    hs = g.main_heating_system.astype(str).str.lower()

    def _arche(t):
        t = (t or "").strip().lower()
        if "detached" in t and "semi" not in t: return "detached"
        if "semi-detached" in t or "semi detached" in t: return "semi-detached"
        if "terraced" in t: return "terraced"
        if "flat" in t: return "flat"
        return "default"

    def _pt(t):
        m = PROPERTY_TYPE_MULT_HEAT
        v = m.get(t)
        if v is None: v = m.get(_arche(t))
        if v is None: v = m.get("default", 1.0)
        return float(v)

    idx = np.clip((np.clip(sap, sapp["sap_lo"], sapp["sap_hi"]) - sapp["sap_lo"])
                  / (sapp["sap_hi"] - sapp["sap_lo"]), 0, 1)
    sap_lin = sapp["slope_mult_hi"] + (sapp["slope_mult_lo"] - sapp["slope_mult_hi"]) * idx
    sap_heat = cfg.get("sap_band_mult_heating_electric" if electric
                       else "sap_band_mult_heating_gas", {}) or {}
    band_mult = np.array([float(sap_heat.get(_serl_sap_band(s), 1.0)) for s in sap])

    p = np.where(sap < 50, 1.10, np.where(sap > 80, 0.90, 1.0))          # SAP nudge
    p = p * g.property_type.map(_pt).to_numpy(float)                     # property type
    p = p * sap_lin                                                      # linear SAP
    p = p * np.array([_area_band_multiplier(a, cfg["heat_slope_area_bands"]) for a in fa])
    p = p * (1.0 - 0.20 * np.clip(retro, 0.0, 1.0))                      # envelope
    p = p * np.where(hs.str.contains("communal"),
                     float(sysc.get("communal", {}).get("heating_slope_mult", 0.85)),
                     np.where(hs.str.contains("heat pump"),
                              float(sysc.get("heat_pump", {}).get("heating_slope_mult", 0.70)), 1.0))
    p = p * np.where(hs.str.contains("heat pump"), hp_eff, 1.0)          # hp effectiveness
    p = p * band_mult                                                    # SERL SAP-heat band
    return float(p.mean()), int(len(g))


def hourly_per_person_slope(quantity):
    """24-element per-occupant load (kWh/h/person), by hour, from SERL.

    For each hour, regress the hourly mean kWh on num_occupants across the
    occupancy bands; the slope is the marginal per-occupant load at that hour.
    This IS the daily schedule -- evening peak, overnight trough -- read from
    data, replacing the leave/wake/sleep archetypes.
    """
    out = []
    for h in range(24):
        s = _H[(_H.quantity == quantity) & (_H.heating_fuel == "All") &
               (_H.seg3_var == "num_occupants") & (_H.year == YEAR) &
               (_H.weekday_weekend == WKND) & (_H.has_pv == PV) & (_H.hour == h)].copy()
        s = s[s.seg3_value.astype(str).isin(["1", "2", "3", "4", "5", ">=6"])]
        s["n"] = s.seg3_value.replace({">=6": "6"}).astype(int)
        x = s["n"].to_numpy(float)
        y = pd.to_numeric(s["mean_kwh"], errors="coerce").to_numpy(float)
        w = np.sqrt(pd.to_numeric(s["n_rounded"], errors="coerce").fillna(1).to_numpy(float))
        X = np.vstack([np.ones_like(x), x]).T
        b = np.linalg.lstsq((w[:, None] * X), w * y, rcond=None)[0][1]
        out.append(round(float(b), 5))
    return out


def panel_mean_occupants():
    """SERL n-weighted mean household occupancy (the per-person deviation centre)."""
    s = _D[(_D.quantity == "Electricity imports") & (_D.heating_fuel == "All") &
           (_D.seg3_var == "num_occupants") & (_D.period_type == "annual") &
           (_D.year == YEAR) & (_D.weekday_weekend == WKND) & (_D.has_pv == PV)].copy()
    s = s[s.seg3_value.astype(str).isin(["1", "2", "3", "4", "5", ">=6"])]
    s["n"] = s.seg3_value.replace({">=6": "6"}).astype(int)
    return round(float(np.average(
        s["n"], weights=pd.to_numeric(s["n_rounded"], errors="coerce"))), 3)


def light_profile():
    """Non-heating electricity seasonality, summer-normalised (=1.0 in summer).

    Read from the GAS-heated cohort whose electricity has no space-heat
    contamination, so it is a clean lighting/behaviour seasonal shape.
    """
    s = cells("Electricity imports", "Gas")
    m = pd.to_numeric(s.set_index("month")["mean"], errors="coerce").reindex(range(1, 13))
    return (m / m.loc[SUMMER].mean()).tolist()


def elec_heating_slope_on_residual(summer_floor_day, light):
    """Electric heating slope (kWh/HDD/day) from the residual electricity after
    removing the lighting-profiled baseline, so lighting is not double-counted."""
    s = cells("Electricity imports", "Electric").copy().dropna(subset=["mean", "mean_hdd"])
    m = pd.to_numeric(s.set_index("month")["mean"], errors="coerce")
    hdd = pd.to_numeric(s.set_index("month")["mean_hdd"], errors="coerce")
    resid = {mo: m[mo] - summer_floor_day * light[int(mo) - 1] for mo in m.index}
    x = np.array([hdd[mo] for mo in m.index], float)
    y = np.array([resid[mo] for mo in m.index], float)
    # slope through origin (residual baseline is ~0 by construction)
    return float((x * y).sum() / (x * x).sum())


# =============================================================================
# BUILD
# =============================================================================
def build():
    cfg: dict = {}

    # ---- House standing load: anchors (summer baseline, direct cell read) ----
    a_gas_e = summer_baseline("Electricity imports", "Gas")
    a_gas_g = summer_baseline("Gas", "Gas")
    a_ele_e = summer_baseline("Electricity imports", "Electric")
    cfg["baseline_anchor_elec_kwh_per_hour"] = rec(
        "baseline_anchor_elec_kwh_per_hour", round(a_gas_e / 24, 4),
        f"daily_targets: Electricity imports, heating_fuel=Gas, summer {SUMMER}",
        "mean(Jun,Jul,Aug daily kWh) / 24",
        note="gas-heated home's non-heating electricity")
    cfg["baseline_anchor_gas_kwh_per_hour"] = rec(
        "baseline_anchor_gas_kwh_per_hour", round(a_gas_g / 24, 4),
        f"daily_targets: Gas, heating_fuel=Gas, summer {SUMMER}",
        "mean(Jun,Jul,Aug daily kWh) / 24",
        note="gas cooking + hot water (no space heat in summer)")
    cfg["baseline_anchor_elec_kwh_per_hour_electric"] = rec(
        "baseline_anchor_elec_kwh_per_hour_electric", round(a_ele_e / 24, 4),
        f"daily_targets: Electricity imports, heating_fuel=Electric, summer {SUMMER}",
        "mean(Jun,Jul,Aug daily kWh) / 24",
        note="electric-heated home's non-heating electricity (elec hot water/cooking)")
    cfg["use_separate_fuel_baseline_anchors"] = True

    # ---- Heating: setpoint + slope (joint temperature-band fit) --------------
    gj = joint_setpoint_slope("Gas", "Gas")
    ej = joint_setpoint_slope("Electricity imports", "Electric")
    setpoint = round((gj["setpoint"] + ej["setpoint"]) / 2, 3)
    cfg["heating_trigger_temp_C"] = rec(
        "heating_trigger_temp_C", setpoint,
        "daily_targets: temperature_band, annual, gas & elec cohorts",
        "argmin baseline+slope*max(0,setpoint-T) over 6 temp bands; avg of fuels",
        n=gj["n"] + ej["n"],
        note=f"gas hinge={gj['setpoint']:.2f}C elec hinge={ej['setpoint']:.2f}C")
    cfg["heating_slope_kWh_per_deg"] = rec(
        "heating_slope_kWh_per_deg", round(gj["slope_per_day"] / 24, 4),
        "daily_targets: Gas, temperature_band, annual", "slope / 24", n=gj["n"],
        note="gas space-heating sensitivity, kWh per degree-hour")
    # Non-heating electricity has a real winter rise (lighting/behaviour in dark
    # months), NOT space heat. Read its seasonal shape from the GAS-heated cohort
    # -- their electricity is pure non-heating, so it is an uncontaminated lighting
    # profile. Summer-normalised (=1.0 in summer) so it ADDS the winter rise on top
    # of the summer anchor for BOTH cohorts.
    light = light_profile()                       # 12-element, 1.0 in summer
    cfg["base_profile_12_electric"] = rec(
        "base_profile_12_electric", [round(x, 6) for x in light],
        "daily_targets: Electricity imports, heating_fuel=Gas, monthly",
        "gas-heated monthly elec / summer mean (=1.0 in summer, lighting rise in winter)",
        note="non-heating electricity seasonality; applied to the elec baseline anchor")

    # Electric heating slope: fit on the RESIDUAL after removing the profiled
    # baseline, so the lighting rise is not double-counted by the slope.
    es = elec_heating_slope_on_residual(a_ele_e, light)
    cfg["heating_slope_kWh_per_deg_electric"] = rec(
        "heating_slope_kWh_per_deg_electric", round(es / 24, 5),
        "daily_targets: Electricity imports, heating_fuel=Electric, monthly residual",
        "OLS of (monthly elec - summer_floor*lighting_profile) on monthly HDD, /24",
        n=12, note="electric space-heating sensitivity, lighting removed first")
    cfg["heat_slope_max"] = 5.0  # uncap: structural package default 0.10 would clip SERL slopes

    # cross-check: two independent baseline reads should agree. The temperature-
    # band joint fit estimates the gas baseline at its own hinge; the summer-cell
    # read estimates it directly. Agreement validates both. (A naive 12-month
    # OLS intercept does NOT belong here -- the gas~HDD curve bends at the summer
    # floor, so its intercept is not a baseline.)
    rec("_check_gas_baseline_jointfit_vs_summer",
        f"{gj['baseline']:.2f} vs {a_gas_g:.2f} kWh/day "
        f"({100*abs(gj['baseline']-a_gas_g)/a_gas_g:.0f}% apart)",
        "daily_targets: temperature_band joint fit  vs  summer cell read",
        "two independent SERL baseline estimates", n=gj["n"],
        note="cross-validation, not an emitted parameter")

    # All multiplier sets below are recentred to SERL-population mean 1.0 (see
    # recenter()): the anchors/slopes are cohort MEANS, so the multipliers must
    # average 1.0 over the stock or the engine inflates the total.

    # ---- Standing-load level by floor area (summer baseline ratio) -----------
    # Derived BEFORE the property-type block: the area multipliers are the
    # deconvolution key for the residual type effect below.
    # Occupancy-deconfounded: raw S(a) double-counts the size/occupancy overlap
    # with the engine's per-person deviation term. deconfounded_area_baseline()
    # subtracts exactly what the engine re-adds (see its docstring).
    base_area = {k: v[0] for k, v in deconfounded_area_baseline().items()}
    base_area, mba = recenter(base_area, "floor_area_m2", "Electricity imports", "All")
    cfg["baseline_elec_area_bands"] = base_area
    for k, mult in base_area.items():
        rec(f"baseline_elec_area_bands[{k}]", mult,
            "daily_targets: Electricity imports, heating_fuel=All, summer, floor_area_m2",
            f"summer baseline NET of (occ_bar-panel_mean)*per-person, ratio vs {REF['floor_area_m2']}, recentred /{mba}",
            note="occupancy-deconfounded via 5-city synthpop E[occ|area] "
                 "(area_occupancy_mean); removes the double-count with the per-person deviation term")

    # ---- Standing-load level by property type: RESIDUAL (beyond-size) effect --
    # SERL's raw type ratio mixes the type's own effect with its size mix
    # (detached homes are also large), so applying it alongside the area bands
    # would double-count size -- the reason an earlier version neutralised the
    # type entirely. But validation vs DESNZ (2026-07) showed full neutralisation
    # UNDER-disperses: the model reads systematically low exactly where
    # detached/semi share is high (pooled std beta -0.50). Detached-ness carries
    # real standing load beyond floor area. Fix: divide the observed type ratio
    # by the part the type's own size mix already explains via the area bands
    # (type x area joint from the pooled five-city EPC stock -- a stock
    # distribution, not an energy fit), keep the residual, recentre to stock
    # mean 1.0. Gas stays neutralised: its DESNZ miss is a level, not a
    # composition, story.
    tbe_serl = {k: v[0] for k, v in marginal("Electricity imports", "All", "building_type", kind="baseline").items()}
    tbe_serl, mpe = recenter(tbe_serl, "building_type", "Electricity imports", "All")
    joint = type_area_joint()
    exp_area = {t: float(sum(joint.loc[t].get(b, 0.0) * base_area.get(b, 1.0)
                             for b in joint.columns))
                for t in tbe_serl if t in joint.index}
    resid_raw = {t: round(tbe_serl[t] / exp_area[t], 4) for t in exp_area}
    resid, mres = recenter(resid_raw, "building_type", "Electricity imports", "All")
    pte = {abm: resid[t] for t, abms in TYPE_MAP.items() if t in resid for abm in abms}
    cfg["property_type_mult_base_electric"] = pte
    for t in resid:
        rec(f"property_type_mult_base_electric[{t}] (residual, beyond size)", resid[t],
            "daily_targets: Electricity imports, heating_fuel=All, summer, building_type"
            " + five-city EPC type x area joint",
            f"SERL type ratio {tbe_serl[t]} / area-expected {exp_area[t]:.4f}, recentred /{mres}",
            note=f"-> ABM keys {TYPE_MAP[t]}")
    ptg = {abm: 1.0 for abms in TYPE_MAP.values() for abm in abms}
    cfg["property_type_mult_base_gas"] = ptg
    rec("property_type_mult_base_gas (neutralised)", 1.0,
        "one-effect-one-dimension", "gas baseline size -> (none); gas DESNZ miss is level, not composition",
        note="no residual applied on gas")

    # ---- Standing-load level by EFFICIENCY band (summer baseline ratio) ------
    # Read from the GAS-heated cohort's electricity, which contains no space
    # heat, so the gradient is pure non-heating load: inefficient homes run
    # MORE baseline electricity (older appliances, immersion / secondary
    # electric heat). Validation vs DESNZ (2026-07) found the model most-too-
    # low exactly in low-SAP LSOAs (pooled std beta +0.46 on mean SAP); this
    # is the SERL-supported part of that gradient. Recentred over the whole
    # stock since it multiplies every home's electricity baseline.
    sap_base = {k: v[0] for k, v in marginal("Electricity imports", "Gas", "currentEnergyRating", kind="baseline").items()}
    sap_base, msb = recenter(sap_base, "currentEnergyRating", "Electricity imports", "All")
    cfg["sap_band_mult_base_electric"] = sap_base
    for k, mult in sap_base.items():
        rec(f"sap_band_mult_base_electric[{k}]", mult,
            "daily_targets: Electricity imports, heating_fuel=Gas, summer, currentEnergyRating",
            f"summer baseline ratio vs {REF['currentEnergyRating']}, recentred /{msb}",
            note="EFFICIENCY effect on the standing load (gas cohort = heating-free signal)")

    # ---- Heating sensitivity by size / efficiency (slope ratios) -------------
    area = {k: v[0] for k, v in marginal("Gas", "Gas", "floor_area_m2", kind="slope").items()}
    area, ma = recenter(area, "floor_area_m2", "Gas", "Gas")
    cfg["heat_slope_area_bands"] = area
    for k, mult in area.items():
        rec(f"heat_slope_area_bands[{k}]", mult,
            "daily_targets: Gas, heating_fuel=Gas, floor_area_m2",
            f"gas slope ratio vs {REF['floor_area_m2']}, recentred /{ma}",
            note="SIZE effect (one-effect-one-dimension)")

    sap_g = {k: v[0] for k, v in marginal("Gas", "Gas", "currentEnergyRating", kind="slope").items()}
    sap_g, msg = recenter(sap_g, "currentEnergyRating", "Gas", "Gas")
    cfg["sap_band_mult_heating_gas"] = sap_g
    for k, mult in sap_g.items():
        rec(f"sap_band_mult_heating_gas[{k}]", mult,
            "daily_targets: Gas, heating_fuel=Gas, currentEnergyRating",
            f"gas slope ratio vs {REF['currentEnergyRating']}, recentred /{msg}",
            note="EFFICIENCY effect")

    age_raw = {k: v[0] for k, v in marginal("Gas", "Gas", "building_age", kind="slope").items()}
    age_raw.setdefault("No data", 1.0)
    age_g, mag = recenter(age_raw, "building_age", "Gas", "Gas")
    cfg["building_age_mult_heating_gas"] = age_g
    for k, mult in age_g.items():
        rec(f"building_age_mult_heating_gas[{k}]", mult,
            "daily_targets: Gas, heating_fuel=Gas, building_age",
            f"gas slope ratio vs {REF['building_age']}, recentred /{mag}",
            note="EFFICIENCY (era) effect; 'No data'->1.0")

    # electric-heated efficiency multipliers (noisier; still SERL-direct)
    sap_e = {k: v[0] for k, v in marginal("Electricity imports", "Electric", "currentEnergyRating", kind="slope").items()}
    if sap_e:
        sap_e, mse = recenter(sap_e, "currentEnergyRating", "Electricity imports", "Electric")
        cfg["sap_band_mult_heating_electric"] = sap_e
        for k, mult in sap_e.items():
            rec(f"sap_band_mult_heating_electric[{k}]", mult,
                "daily_targets: Electricity imports, heating_fuel=Electric, currentEnergyRating",
                f"elec slope ratio vs {REF['currentEnergyRating']}, recentred /{mse}")

    # ---- Deprivation gradient (replaces the neutralised wealth map) ----------
    # SERL IMD-quintile baseline ratios, coerced onto the HIDP economic quintile
    # hh_income_band (q1_lowest=most deprived=IMD1 .. q5_highest=least=IMD5).
    # Applied as a BASELINE multiplier (a standing-level effect), keyed by
    # hh_income_band so it survives the SERL-profile path. Income quintile is a
    # stand-in for area IMD until a real IMD field is joined. Recentred to mean 1.0
    # over the SERL IMD distribution (≈uniform quintiles) so it shifts shape, not level.
    imd = marginal("Electricity imports", "All", "IMD_quintile", kind="baseline")
    if imd:
        q2band = {"1": "q1_lowest", "2": "q2_low", "3": "q3_mid", "4": "q4_high", "5": "q5_highest"}
        raw_q = {str(q): m for q, (m, _) in imd.items() if str(q) in q2band}
        cen_q, mdep = recenter(raw_q, "IMD_quintile", "Electricity imports", "All")
        dep = {q2band[q]: v for q, v in cen_q.items()}
        cfg["baseline_deprivation_mult"] = dep
        for q, (mult, n) in imd.items():
            band = q2band.get(str(q))
            rec(f"baseline_deprivation_mult[{band}<-IMD{q}]", dep.get(band),
                "daily_targets: Electricity imports, heating_fuel=All, summer, IMD_quintile",
                f"summer baseline ratio vs IMD={REF['IMD_quintile']}, recentred /{mdep}",
                n=n, note="HIDP hh_income_band as IMD stand-in; swap for real IMD later")

    # ---- Occupancy: per-person load + its SERL intraday SHAPE -----------------
    # The hourly per-occupant slope IS the daily schedule, read from SERL. The
    # engine's SERL-profile path applies (n_occ - panel_mean) * slope(hour),
    # retiring the leave/wake/sleep archetypes entirely.
    cfg["use_serl_profiles"] = True
    # NOTE the key: the engine reads config.model["panel_mean_occupants"]
    # (model.py:264). An earlier draft emitted "serl_panel_mean_occupants",
    # which the engine silently ignored in favour of its 2.4 default.
    cfg["panel_mean_occupants"] = rec(
        "panel_mean_occupants", panel_mean_occupants(),
        "daily_targets: Electricity imports, heating_fuel=All, num_occupants",
        "n_rounded-weighted mean of num_occupants",
        note="centre of the per-person deviation; baseline anchor holds the avg-occupancy level")
    pps = hourly_per_person_slope("Electricity imports")
    cfg["per_person_slope_24h_electric"] = rec(
        "per_person_slope_24h_electric", pps,
        "diurnal_targets_hourly_mean: Electricity imports, heating_fuel=All, num_occupants",
        "per hour: slope of hourly kWh ~ num_occupants",
        n=24, note=f"daily sum {sum(pps):.2f} kWh/person; evening-peak shape replaces the schedule")
    # legacy flat per-person value kept for provenance; inert under use_serl_profiles.
    rec("energy_per_person_home (legacy, inert)", round(sum(pps) / 24, 5),
        "= mean of per_person_slope_24h_electric",
        "daily per-person / 24", note="superseded by the 24h slope above")

    # ---- Diurnal (24h) shape; seasonal baseline shape set above (lighting) ----
    cfg["base_profile_24h_electric"] = rec(
        "base_profile_24h_electric", diurnal_shape("Electricity imports", "All"),
        "diurnal_targets_hourly_mean: Electricity imports, heating_fuel=All, seg3=none",
        "hourly mean / 24h mean (mean-1.0)", note="house+occupancy electricity day-shape")
    cfg["heating_month_profile_12"] = rec(
        "heating_month_profile_12", seasonal_shape("Gas", "Gas", hdd_weighted=True),
        "daily_targets: Gas, heating_fuel=Gas, monthly",
        "(monthly - summer floor) / HDD-weighted mean (HDD-weighted-mean-1.0)",
        note="smooth replacement for the hard heating-months gate; preserves annual heat")

    # ---- Intraday gas + heating shapes (the diurnal-shape calibration) -------
    # These two parameters ARE the schedule-retired hourly architecture for gas:
    # without them the engine's gas day-shape is flat. Derived here under the
    # ledger's own slice conventions (has_pv=No, heating_fuel=Gas), superseding
    # the standalone profile pack (fit_serl_profiles.py, which mixed PV=All).
    cfg["base_profile_24h_gas"] = rec(
        "base_profile_24h_gas", gas_cooking_profile(),
        f"diurnal_targets_hourly_mean: Gas, heating_fuel=Gas, temperature_band={COOKING_BAND}",
        "warm-band hourly gas / its mean (mean-1.0)",
        note="cooking/DHW twin-peak day-shape; warm band = above balance point, ~no space heat")
    htp = heating_temp_profiles()
    cfg["heating_temp_profile"] = htp
    for b, d in htp.items():
        rec(f"heating_temp_profile[{b}]", d["profile"],
            f"diurnal_targets_hourly_mean: Gas, heating_fuel=Gas, temperature_band={b}",
            f"max(band gas - cooking band, 0) / mean (mean-1.0); runtime interpolates at T={d['temp_mid_C']}C",
            note="space-heating day-shape: flat when cold, peaky when mild")

    # Off-peak electric heating charges overnight (Economy-7): storage heaters
    # today, and smart/off-peak HEAT PUMPS in electrification scenarios. DIRECT
    # electric (Electric radiators) peaks in the evening like gas and keeps the gas
    # temperature-conditioned profile. The overnight shape is fit from SERL's
    # storage-radiator curve -- the cleanest empirical off-peak signal (SERL has too
    # few heat pumps to fit their own); the engine's heatpump_offpeak_charging flag
    # decides whether heat pumps use it. SERL's central_heating_type separates
    # storage from direct, so both diurnal regimes are calibrated.
    cfg["heating_profile_24h_offpeak"] = rec(
        "heating_profile_24h_offpeak", heating_diurnal_storage(),
        "diurnal_targets_hourly_mean: Electricity imports, central_heating_type=Electric storage radiators",
        "storage hourly - appliance floor (gas-boiler elec, midday-matched), clipped, mean-1.0",
        note="Economy-7 overnight charge shape; applied mean-preservingly to storage + off-peak heat pumps")

    # ---- Cohort-aware anchor/slope correction --------------------------------
    # The multipliers are mean-1.0 over the WHOLE stock, but each fuel cohort has
    # its own anchor (its measured mean). For a skewed cohort (electric-heated =
    # mostly small flats) the sub-population multiplier mean is <1, so anchor x
    # multipliers under-shoots. Divide each cohort's anchor/slope by ITS sub-pop
    # multiplier mean. Self-cancels (=1.0) for the cohort that matches the stock,
    # so gas-heated is unchanged; electric-heated is re-inflated to its true level.
    _CORR.clear()
    pt_keymap = {serl: keys[0] for serl, keys in TYPE_MAP.items()}
    pte, bea = cfg["property_type_mult_base_electric"], cfg["baseline_elec_area_bands"]
    for cohort, akey in [("Gas", "baseline_anchor_elec_kwh_per_hour"),
                         ("Electric", "baseline_anchor_elec_kwh_per_hour_electric")]:
        prod = (cohort_mult_mean(pte, "building_type", cohort, pt_keymap)
                * cohort_mult_mean(bea, "floor_area_m2", cohort)
                * cohort_mult_mean(cfg["sap_band_mult_base_electric"], "currentEnergyRating", cohort))
        _CORR[akey] = prod
        old = cfg[akey]
        cfg[akey] = round(old / prod, 4)
        rec(f"{akey} (cohort-corrected)", cfg[akey],
            f"baseline anchor / {cohort}-heated sub-pop mean(property_type x area x sap_base)",
            f"{old} / {prod:.3f}", note=f"x{1/prod:.2f}; corrects whole-stock recentring for this cohort")
    # Electric heating slope: divide by the mean of the FULL multiplier product
    # the engine applies to it, not just the two marginals SERL exposes.
    #
    # The earlier pairing (area bands x elec SAP, weighted by SERL marginals)
    # inverted a 1.23x discount while agent.py was applying a 1.98x one, so
    # electric-heated homes received ~49% of the space heating SERL measures for
    # them (811 vs 1641 kWh/yr; measured 2026-07-10 via decompose_demand on four
    # Manchester LSOAs). Gas-heated homes are the modal stock, so their product
    # (~0.78) sits near the whole-stock mean and their heating already lands at
    # 0.977 of SERL -- they are deliberately left uncorrected here.
    prod_h, n_h = engine_heat_mult_mean(cfg, "Electric")
    _CORR["heating_slope_kWh_per_deg_electric"] = prod_h
    old = cfg["heating_slope_kWh_per_deg_electric"]
    cfg["heating_slope_kWh_per_deg_electric"] = round(old / prod_h, 5)
    rec("heating_slope_kWh_per_deg_electric (cohort-corrected)",
        cfg["heating_slope_kWh_per_deg_electric"],
        "elec heating slope / electric-heated stock mean(full engine multiplier product)",
        f"{old} / {prod_h:.4f}", n=n_h,
        note=f"x{1/prod_h:.2f}; undoes the engine's structural double-discount on a "
             "cohort whose structure is already inside the fitted slope")

    # ---- Heating-fuel misclassification lever (DESNZ-fitted, NOT a SERL read)
    # REMOVED 2026-07-07: elec_heat_share_by_sap was a hard-coded {E:0.08,F:0.16,G:0.16}
    # guess (never fitted, despite the old comment describing a Newcastle beta-zeroing).
    # Neutralise-and-check (30 NCL LSOAs, on vs off) showed it was not load-bearing and
    # worsened the electricity fit (+9.5% -> +7.2% bias); gas ~unchanged. Demand path is
    # now 100% SERL-read; emit empty so the config carries no hand-set demand parameter.
    cfg["elec_heat_share_by_sap"] = rec(
        "elec_heat_share_by_sap", {}, "(removed; not fitted, not load-bearing)",
        "neutralised 2026-07-07", note="see ASSUMPTIONS.md / PAPER1_REWRITE_TRACKER.md")

    return cfg


def self_check(cfg):
    """Analytic reconstruction of each cohort's annual energy from the emitted
    parameters, vs the SERL annual cell. No model run -- this checks the formula,
    not the engine. Printed so the fit can never silently drift."""
    DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    def monthly(q, hf):
        return cells(q, hf).set_index("month")[["mean", "mean_hdd"]].apply(
            pd.to_numeric, errors="coerce").reindex(range(1, 13))

    def annual(q, hf):
        s = _D[(_D.quantity == q) & (_D.heating_fuel == hf) & (_D.seg3_var == "none") &
               (_D.period_type == "annual") & (_D.year == YEAR) &
               (_D.weekday_weekend == WKND) & (_D.has_pv == PV)]
        return float(pd.to_numeric(s["mean"], errors="coerce").iloc[0]) * 365

    light = cfg["base_profile_12_electric"]
    # The cohort correction divides anchors/slope by each cohort's sub-pop
    # multiplier mean; this analytic check is at the cohort MEAN, so multiply it
    # back to recover the level the engine produces on a SERL-representative stock.
    c = lambda k: cfg[k] * _CORR.get(k, 1.0)
    out = []
    # gas-heated gas = flat gas anchor + gas heating slope * HDD
    mg = monthly("Gas", "Gas")
    pred = sum((c("baseline_anchor_gas_kwh_per_hour") * 24
                + c("heating_slope_kWh_per_deg") * 24 * mg["mean_hdd"][mo]) * DAYS[mo]
               for mo in range(1, 13))
    out.append(("gas-heated gas", annual("Gas", "Gas"), pred))
    # gas-heated elec = lighting-profiled elec anchor (no electric heat)
    pred = sum(c("baseline_anchor_elec_kwh_per_hour") * 24 * light[mo - 1] * DAYS[mo]
               for mo in range(1, 13))
    out.append(("gas-heated elec", annual("Electricity imports", "Gas"), pred))
    # elec-heated elec = lighting-profiled elec anchor + electric heating slope * HDD
    me = monthly("Electricity imports", "Electric")
    pred = sum((c("baseline_anchor_elec_kwh_per_hour_electric") * 24 * light[mo - 1]
                + c("heating_slope_kWh_per_deg_electric") * 24 * me["mean_hdd"][mo]) * DAYS[mo]
               for mo in range(1, 13))
    out.append(("elec-heated elec", annual("Electricity imports", "Electric"), pred))
    print("\nSELF-CHECK (analytic, formula vs SERL annual):")
    worst = 0.0
    for lbl, act, pred in out:
        err = 100 * (pred - act) / act
        worst = max(worst, abs(err))
        print(f"  {lbl:18s} SERL={act:7.0f}  formula={pred:7.0f}  err={err:+5.1f}%")
    print(f"  worst |err| = {worst:.1f}%  ({'PASS' if worst < 5 else 'REVIEW'})")
    return worst


def _native(o):
    """Recursively coerce numpy scalars/strings to plain Python types so
    yaml.safe_dump works under any numpy (numpy>=2 makes np.str_ a distinct
    type SafeDumper refuses; area-band keys come out of .astype(str) as np.str_)."""
    if isinstance(o, dict):
        return {_native(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(x) for x in o]
    if isinstance(o, np.generic):   # np.str_, np.float64, np.int64, ...
        return o.item()
    return o


def main():
    cfg = build()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    self_check(cfg)

    # config (model block only, the shape the engine consumes) -> straight to the
    # tracked engine path. This is the single source of truth; no promote step.
    yaml.safe_dump({"model": _native(cfg)}, open(CONFIG_OUT, "w"),
                   sort_keys=False, default_flow_style=False)

    # human-readable ledger
    lines = ["# SERL Parameter Ledger",
             "",
             f"Every value below is read directly from SERL aggregate cells "
             f"(year {YEAR}, weekday_weekend={WKND}, has_pv={PV}). No model run is "
             f"involved. Reference categories: "
             + ", ".join(f"{k}={v}" for k, v in REF.items()) + ".",
             "",
             "| Parameter | Value | SERL source | Formula | n | Note |",
             "|---|---|---|---|---|---|"]
    for r in LEDGER:
        v = r["value"]
        vs = (f"[{len(v)}]" if isinstance(v, list) else
              (str(v) if not isinstance(v, float) else f"{v:.5g}"))
        lines.append("| {param} | {v} | {source} | {formula} | {n} | {note} |".format(
            v=vs, n=r["n"] if r["n"] is not None else "", **{
                k: str(r[k]).replace("|", "\\|") for k in ("param", "source", "formula", "note")}))
    (OUTDIR / "PARAMETER_LEDGER.md").write_text("\n".join(lines) + "\n")

    # assumptions register: everything the model needs that SERL CANNOT supply.
    # SERL describes the current stock's observed behaviour; it cannot give the
    # causal levers for interventions that have not happened, nor phenomena it
    # never observes (UK summers have negligible cooling load).
    ASSUMPTIONS = [
        ("cooling_slope_kWh_per_deg / cooling_threshold_C", "0.03 / 24.0",
         "Not in SERL: UK AC penetration ~5-15%, summer cooling signal is noise.",
         "structural placeholder; revisit if/when a cooling-equipped panel exists"),
        ("heatpump_cop_ref", "2.8 (3.0-4.2 well-installed)",
         "Counterfactual lever: SERL has few heat pumps and reports demand, not COP.",
         "field data: heatpumpmonitor.org / DESNZ Electrification of Heat trial medians"),
        ("boiler_efficiency", "0.90",
         "Efficiency, not demand: SERL meters delivered energy, not boiler losses.",
         "literature / SAP appliance assumptions"),
        ("envelope retrofit multipliers (cavity_wall, etc.)", "lever set",
         "Counterfactual: SERL cannot observe a retrofit that has not occurred.",
         "intervention deltas; SERL SAP-band slopes bound the plausible range"),
        ("energy_per_person_away", "0.01 kWh/h",
         "Vacant-home standby; SERL aggregates cannot isolate per-absent-person load.",
         "literature placeholder; small and second-order"),
        ("elec_heat_share_by_sap", "{} (REMOVED 2026-07-07)",
         "Was a hand-set lever, never a SERL read (the old 'fitted on Newcastle' note "
         "described a beta-zeroing that had no code).",
         "REMOVED: neutralise-and-check showed it was not load-bearing and worsened the "
         "electricity fit; demand path is now 100% SERL-read"),
    ]
    alines = ["# Assumptions Register (non-SERL parameters)", "",
              "These are the ONLY parameters not read from SERL in PARAMETER_LEDGER.md. "
              "Each is here because SERL structurally cannot supply it, with the reason "
              "and the source we use instead.", "",
              "| Parameter | Value | Why not SERL | Source / status |",
              "|---|---|---|---|"]
    for p, v, why, src in ASSUMPTIONS:
        alines.append(f"| {p} | {v} | {why} | {src} |")
    (OUTDIR / "ASSUMPTIONS.md").write_text("\n".join(alines) + "\n")

    print(f"wrote {CONFIG_OUT}  (the tracked engine config; git diff = the audit)")
    print(f"wrote {OUTDIR/'PARAMETER_LEDGER.md'}  ({len(LEDGER)} parameter rows)")
    print(f"wrote {OUTDIR/'ASSUMPTIONS.md'}  ({len(ASSUMPTIONS)} non-SERL params)")
    # echo the load-bearing scalars
    for k in ("baseline_anchor_elec_kwh_per_hour", "baseline_anchor_gas_kwh_per_hour",
              "baseline_anchor_elec_kwh_per_hour_electric", "heating_trigger_temp_C",
              "heating_slope_kWh_per_deg", "heating_slope_kWh_per_deg_electric",
              "panel_mean_occupants"):
        print(f"  {k} = {cfg.get(k)}")


if __name__ == "__main__":
    main()
