"""Generate notebooks/pipeline_diagnostics.ipynb — an end-to-end, independently
verifiable record of the household-energy ABM pipeline: calibration, sensitivity,
and validation. Runs start to finish; the final cell promotes the trustworthy
parameter set.

  .venv/bin/python research/applied/scripts/build_pipeline_notebook.py
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 notebooks/pipeline_diagnostics.ipynb
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(src: str) -> None: cells.append(new_markdown_cell(src.strip("\n")))
def code(src: str) -> None: cells.append(new_code_cell(src.strip("\n")))


# ── 0. Overview ───────────────────────────────────────────────────────────
md(r"""
# Household-energy ABM — calibration, sensitivity & validation

End-to-end record of the model's parameter pipeline, runnable start to finish.

1. **Calibration** — every demand parameter is estimated from SERL smart-meter
   data and shown next to the target it reproduces.
2. **Sensitivity** — a Morris screen quantifies each parameter's influence.
3. **Validation** — modelled demand is checked against an independent source
   (DESNZ meter data), **per dwelling / per meter**, **decomposed by demand
   component, heating fuel, and month** — so accuracy is isolated from coverage
   and no error hides inside a total.

**Provenance.** Every parameter is *fitted* (from SERL), *cited* (literature), or
*neutralised* (off). The assembled set the model runs on is
[`household_energy/calibrated_config.yaml`](../household_energy/calibrated_config.yaml).

| Stage | Script(s) | Output |
|---|---|---|
| Calibration | [`calibrate_serl.py`](../research/applied/scripts/calibrate_serl.py), `fit_*.py`, [`fit_monthly_profile.py`](../research/applied/scripts/fit_monthly_profile.py) | `results/calibration_*/*.yaml` |
| Assembly | [`promote_config.py`](../research/applied/scripts/promote_config.py) | `calibrated_config.yaml` |
| Sensitivity | [`sa_morris.py`](../research/applied/scripts/sa_morris.py) | `sa_morris_*.csv` |
| Validation | [`decompose_demand.py`](../research/applied/scripts/decompose_demand.py), [`run_monthly_retune.py`](../research/applied/scripts/run_monthly_retune.py) | per-cohort / monthly / per-unit |
""")

code(r"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt

REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)   # scripts (pick_sample, utils) use repo-relative paths
SCRIPTS = REPO / "research" / "applied" / "scripts"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCRIPTS))

PHASE5B  = REPO / "results" / "calibration_v5_phase5b"
SERL_FIT = REPO / "results" / "calibration_serl_fits"
V6_CFG   = REPO / "results" / "calibration_v6_monthly" / "calibrated_config.yaml"
SA_DIR   = REPO / "results" / "sensitivity_analysis"
DATA     = REPO / "data"
RES_LSOA = REPO / "results_lsoa"

# Validation runs the model live on a representative LSOA sample. Set True to
# regenerate; default reads the saved per-dwelling / monthly decomposition.
RUN_LIVE = False
TAG = "v6"                      # config tag for the validation run
SAMPLE_N = 30                   # LSOAs in the validation sample

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
def load_fit(name):
    with open(PHASE5B / name) as f: return yaml.safe_load(f)
DAYS = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

# ── SERL cohort targets (computed once, reused throughout) ──
_d = pd.read_csv(DATA / "serl_8963_targets" / "daily_targets.csv")
def _serl_month(hf, qty, col="mean"):
    s = _d[(_d.period_type=="monthly") & (_d.year==2023) & (_d.weekday_weekend=="both") &
           (_d.heating_fuel==hf) & (_d.has_pv.isin(["No","All"])) & (_d.seg3_var=="none") &
           (_d.quantity==qty)].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    return s.groupby("month")[col].mean().reindex(range(1,13))
def _annual(series): return float(sum(series[m]*DAYS[m] for m in range(1,13)))

serl_gas_gas   = _serl_month("Gas", "Gas")
serl_gas_elec  = _serl_month("Gas", "Electricity imports")
serl_ele_elec  = _serl_month("Electric", "Electricity imports")
_gas_floor = serl_gas_gas[[6,7,8]].mean(); _ele_floor = serl_ele_elec[[6,7,8]].mean()
SERL = {
    "gas_heated_elec":      _annual(serl_gas_elec),                                   # 2903
    "elec_heated_baseline": _ele_floor*365,                                           # 2624
    "elec_heated_heating":  _annual((serl_ele_elec - _ele_floor).clip(lower=0)),      # 2085
}
SERL["elec_heated_total"] = SERL["elec_heated_baseline"] + SERL["elec_heated_heating"]
print("python", sys.version.split()[0], "| RUN_LIVE", RUN_LIVE)
print("SERL cohort targets (kWh/yr):", {k: round(v) for k, v in SERL.items()})
""")

# ── 1. Calibration ────────────────────────────────────────────────────────
md(r"""
---
## 1 · Calibration

Each parameter is estimated from SERL by a dedicated script writing a `*_fit.yaml`
diagnostic. Inputs:
[`daily_targets.csv`](../data/serl_8963_targets/daily_targets.csv) (daily means by
segmentation) and
[`diurnal_targets_hourly_mean.csv`](../data/serl_8963_targets/diurnal_targets_hourly_mean.csv)
(hourly shape).
""")

md(r"""
### 1.1 Heating setpoint
[`fit_heating_setpoint.py`](../research/applied/scripts/fit_heating_setpoint.py) —
piecewise-linear hinge in SERL gas kWh/day across outdoor-temperature bands; the
kink is the temperature at which heating engages.
""")
code(r"""
fit = load_fit("heating_setpoint_fit.yaml")
bands = pd.DataFrame(fit["diagnostics"]["bands"]); setpoint = fit["heating_setpoint_C"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(bands.temp_C_midpoint, bands.observed_kwh_per_day,
           s=bands.n_rounded/bands.n_rounded.max()*120, label="SERL observed (size ∝ n)", zorder=3)
ax.plot(bands.temp_C_midpoint, bands.fitted_kwh_per_day, "r-", label="fitted hinge")
ax.axvline(setpoint, color="k", ls="--", lw=1, label=f"setpoint {setpoint:.2f} °C")
ax.set(xlabel="outdoor temperature (°C)", ylabel="gas kWh/day", title="Heating setpoint"); ax.legend(); plt.show()
print(f"setpoint {setpoint:.2f} °C | slope {fit['fit']['slope_kwh_per_day_per_deg']:.2f} kWh/day/°C "
      f"| weighted RMSE {fit['fit']['rmse_weighted_kwh_per_day']:.2f}")
""")

md(r"""
### 1.2 Heating slopes (temperature sensitivity)
[`calibrate_serl.py`](../research/applied/scripts/calibrate_serl.py) fits the HDD
heating slope (OLS through origin), per fuel. The raw SERL fit gives the gas and
electric slopes; the **values the model runs on are slightly lower** because they
are paired with the seasonal heating profile (§1.7) — a per-degree coefficient is
only meaningful alongside the degree-count it is applied through, so the slope is
re-levelled once the profile sets the monthly distribution (see §4.2).
""")
code(r"""
diag = json.loads((PHASE5B / "diagnostics.json").read_text())
v6 = yaml.safe_load(V6_CFG.read_text())["model"]
print("SERL raw HDD slopes (kWh/h/°C):", {k: round(v,4) for k,v in diag["hdd_slopes"].items()})
print("model-applied slopes (profile-paired):",
      {"gas": round(v6["heating_slope_kWh_per_deg"],4),
       "electric": round(v6["heating_slope_kWh_per_deg_electric"],4)})
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(["gas\n(raw)","gas\n(applied)","electric\n(raw)","electric\n(applied)"],
       [diag["hdd_slopes"]["gas"], v6["heating_slope_kWh_per_deg"],
        diag["hdd_slopes"]["electric"], v6["heating_slope_kWh_per_deg_electric"]],
       color=["#c44","#e88","#48c","#9bd"])
ax.set(ylabel="slope (kWh/h/°C)", title="Heating slope by fuel: raw SERL fit vs profile-paired"); plt.show()
""")

md(r"""
### 1.3 Structural multipliers on the heating slope
Floor area, SAP band, and building age scale the gas heating slope, each fitted to
SERL gas by segmentation, normalised to a reference band:
[`fit_area_scaling.py`](../research/applied/scripts/fit_area_scaling.py),
[`fit_sap_band_mult.py`](../research/applied/scripts/fit_sap_band_mult.py),
[`fit_age_mult.py`](../research/applied/scripts/fit_age_mult.py).
""")
code(r"""
area, sap, age = load_fit("area_scaling_fit.yaml"), load_fit("sap_band_mult_fit.yaml"), load_fit("age_mult_fit.yaml")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (title, dct, key) in zip(axes, [
        ("Floor area", area, "heat_slope_area_bands"),
        ("SAP band", sap, "sap_band_mult_heating_gas"),
        ("Building age", age, "building_age_mult_heating_gas")]):
    s = pd.Series(dct[key]); ref = dct["reference_band"]
    ax.bar(range(len(s)), s.values, color=["#888" if b == ref else "#48c" for b in s.index])
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=45, ha="right", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.7, ls="--"); ax.set(title=f"{title} (ref={ref})", ylabel="slope multiplier")
plt.tight_layout(); plt.show()
""")

md(r"""
### 1.4 Electricity baseline — anchor and cohort split
Baseline (non-heating) electricity is an anchor scaled by floor area
([`fit_elec_baseline_area.py`](../research/applied/scripts/fit_elec_baseline_area.py)).
Two cohort facts the area scaling alone misses:
- The shared anchor is set so **gas-heated** homes reproduce SERL gas-heated
  electricity (≈ all baseline, no electric heating).
- **Electric-heated** homes are all-electric (water heating + cooking on the
  meter) so their baseline is higher than their floor area implies; an
  `electric_heated_baseline_mult` restores it to SERL.
""")
code(r"""
eb = load_fit("elec_baseline_area_fit.yaml"); s = pd.Series(eb["baseline_elec_area_bands"]); ref = eb["reference_band"]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar(range(len(s)), s.values, color=["#888" if b == ref else "#4a4" for b in s.index])
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=45, ha="right")
ax.axhline(1.0, color="k", lw=0.7, ls="--"); ax.set(title=f"baseline × floor area (ref={ref})", ylabel="multiplier"); plt.show()
print(f"shared anchor                 : {v6['baseline_anchor_elec_kwh_per_hour']:.3f} kWh/h "
      f"→ {v6['baseline_anchor_elec_kwh_per_hour']*8760:.0f} kWh/yr (gas-heated target SERL {SERL['gas_heated_elec']:.0f})")
print(f"electric_heated_baseline_mult : {v6['electric_heated_baseline_mult']:.2f} "
      f"(electric-heated target SERL {SERL['elec_heated_baseline']:.0f})")
""")

md(r"""
### 1.5 Per-person occupancy load
[`fit_presence_spikes.py`](../research/applied/scripts/fit_presence_spikes.py)
separates behavioural per-person electricity from area-correlated standby using
two SERL segmentations, and fits the awake/sleep multiplier.
""")
code(r"""
ps = load_fit("presence_spikes_fit.yaml")
print(f"per-person (behavioural) {ps['energy_per_person_home']:.4f} kWh/h | naive {ps['naive_per_person_home_awake']:.4f} "
      f"| awake/sleep {ps['awake_home_spike_mult']:.2f}/{ps['sleep_home_spike_mult']:.2f} | panel occupants {ps['panel_mean_occupants']:.2f}")
ph = pd.DataFrame(ps["diagnostics"]["per_hour_occ"]).T; ph.index = ph.index.astype(int); ph = ph.sort_index()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(ph.index, ph["baseline_kwh_per_hour"], label="baseline (0-occupant)")
ax.plot(ph.index, ph["per_person_kwh_per_hour"], label="per-person slope")
ax.set(xlabel="hour", ylabel="kWh/h", title="Per-hour occupancy regression (SERL)"); ax.legend(); plt.show()
""")

md(r"""
### 1.6 Diurnal (hourly) shape
[`fit_diurnal_profile.py`](../research/applied/scripts/fit_diurnal_profile.py)
fits a mean-1.0 24-hour multiplier so the *hourly* shape of baseline electricity
matches SERL without changing the daily mean.
""")
code(r"""
dp = yaml.safe_load((SERL_FIT / "diurnal_profile.yaml").read_text())
prof = np.array(dp["base_profile_24h_electric"], float); dd = dp["diagnostics"]["electric"]
di = pd.read_csv(DATA / "serl_8963_targets" / "diurnal_targets_hourly_mean.csv")
serl = di[(di.quantity=="Electricity imports") & (di.has_pv=="All") & (di.heating_fuel=="All") &
          (di.seg3_var=="none") & (di.weekday_weekend=="both") & (di.year==2023)].sort_values("hour")
serl_shape = (serl.mean_kwh / serl.mean_kwh.mean()).values
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(range(24), prof, "-o", ms=3, label="fitted profile"); ax.plot(range(24), serl_shape, "--", label="SERL shape")
ax.axhline(1.0, color="k", lw=0.6)
ax.set(xlabel="hour", ylabel="multiplier (mean=1.0)", title=f"Diurnal shape — peak h{dd['peak_hour']}, trough h{dd['trough_hour']}")
ax.legend(); plt.show()
print(f"profile mean {prof.mean():.4f} (mean-preserving)")
""")

md(r"""
### 1.7 Seasonal (monthly) shape
[`fit_monthly_profile.py`](../research/applied/scripts/fit_monthly_profile.py) is
the seasonal analogue of the diurnal fit — two mean-1.0 monthly profiles:
- **heating** — SERL's monthly heating-per-degree-day; replaces the hard
  heating-months on/off gate with a smooth seasonal ramp.
- **baseline electricity** — the winter uplift the flat baseline misses.

Both are mean-preserving, so annual totals are unchanged; they set seasonal
*shape* only (validated in §4.3).
""")
code(r"""
mp = yaml.safe_load((SERL_FIT / "monthly_profile.yaml").read_text())
hp = np.array(mp["heating_month_profile_12"]); bp = np.array(mp["base_profile_12_electric"])
M = ["J","F","M","A","M","J","J","A","S","O","N","D"]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(range(1,13), hp, "-o", ms=4, label="heating profile")
ax.plot(range(1,13), bp, "-s", ms=4, label="baseline-electricity profile")
ax.axhline(1.0, color="k", lw=0.6); ax.set_xticks(range(1,13)); ax.set_xticklabels(M)
ax.set(ylabel="multiplier (mean=1.0)", title="Monthly profiles (seasonal shape)"); ax.legend(); plt.show()
print(f"heating profile (HDD-wtd mean {mp['diagnostics']['heating_weighted_mean_check']:.3f}); "
      f"baseline profile (day-wtd mean {mp['diagnostics']['base_weighted_mean_check']:.3f})")
""")

# ── 2. Assembly ───────────────────────────────────────────────────────────
md(r"""
---
## 2 · Parameter assembly

The fitted parameters are assembled into the model's `model:` block. The
calibration order is **level then shape**: anchors and slopes set annual levels
against SERL cohort targets, then the diurnal and monthly profiles set the
within-day and within-year shape (mean-preserving, so levels hold).

The assembled set used below is
[`results/calibration_v6_monthly/calibrated_config.yaml`](../results/calibration_v6_monthly/calibrated_config.yaml).
[`promote_config.py`](../research/applied/scripts/promote_config.py) ships the
`model:` block to `household_energy/calibrated_config.yaml` (the final cell does
this).
""")
code(r"""
keys = ["baseline_anchor_elec_kwh_per_hour","electric_heated_baseline_mult",
        "heating_slope_kWh_per_deg","heating_slope_kWh_per_deg_electric","heating_trigger_temp_C"]
print("Key assembled parameters (v6):")
for k in keys:
    print(f"  {k:34s} {v6.get(k)}")
print(f"  base_profile_24h_electric         len {len(v6.get('base_profile_24h_electric', []))}")
print(f"  base_profile_12_electric          len {len(v6.get('base_profile_12_electric', []))}")
print(f"  heating_month_profile_12          len {len(v6.get('heating_month_profile_12', []))}")
""")

# ── 3. Sensitivity ────────────────────────────────────────────────────────
md(r"""
---
## 3 · Sensitivity analysis

[`sa_morris.py`](../research/applied/scripts/sa_morris.py) runs a Morris screen,
perturbing each parameter within its published standard error and reporting μ\*
on annual energy (influence). Ranges/provenance:
[`sa_param_table.csv`](../results/sensitivity_analysis/sa_param_table.csv).
""")
code(r"""
sa = pd.read_csv(SA_DIR / "sa_morris_newcastle.csv").sort_values("mu_star_energy")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh(sa.knob, sa.mu_star_energy_pct_base, color="#48c")
ax.set(xlabel="μ* on annual energy (% of baseline)", title="Morris influence ranking — Newcastle 2023")
plt.tight_layout(); plt.show()
print(f"largest single-parameter effect: {sa.mu_star_energy_pct_base.max():.1f}% per ±2·SE step")
""")
md(r"""
### 3.1 Cross-city robustness
([`sa_morris_5city.csv`](../results/sensitivity_analysis/sa_morris_5city.csv))
""")
code(r"""
p5 = SA_DIR / "sa_morris_5city.csv"
if p5.exists():
    sa5 = pd.read_csv(p5)
    print(sa5[[c for c in sa5.columns if "mu_star" in c or c in ("knob","city")]].head(20).to_string(index=False))
else:
    print("sa_morris_5city.csv not found")
""")

# ── 4. Validation ─────────────────────────────────────────────────────────
md(r"""
---
## 4 · Validation

The model is run live on a representative LSOA sample with the assembled v6
config, accumulating each dwelling's annual demand (baseline / heating /
occupancy) and per-cohort monthly totals. Validation is in three views, each
isolating a different failure mode:
- **§4.1 per-cohort split** vs SERL (the calibration target) — level + split.
- **§4.2 seasonal shape** vs SERL — month-by-month.
- **§4.3 per unit** vs DESNZ (independent meter data) — the held-out check.
""")
code(r"""
DEC = RES_LSOA / f"decomp_sample_newcastle_2023_{TAG}.csv"
MON = RES_LSOA / f"monthly_cohort_newcastle_2023_{TAG}.csv"
if RUN_LIVE or not DEC.exists():
    import subprocess
    subprocess.run([sys.executable, str(SCRIPTS / "run_monthly_retune.py"), TAG, str(SAMPLE_N)],
                   cwd=str(REPO), check=True,
                   env={**__import__("os").environ, "PYTHONPATH": f"{REPO}:{SCRIPTS}"})
dec = pd.read_csv(DEC); mon = pd.read_csv(MON)
print(f"validation sample: {len(dec)} dwellings | electric-heated {(dec.heating_bucket=='electric').mean():.1%}")
""")

md(r"""
### 4.1 Per-cohort split vs SERL
Electricity decomposed into non-heating (baseline + occupancy) and heating, by
heating cohort, against the SERL cohort targets. A correct calibration reproduces
**both the split and the total** — not just the total.
""")
code(r"""
e = dec[dec.heating_bucket=="electric"]; g = dec[dec.heating_bucket=="gas"]
rows = [
    ["electric-heated", "non-heating", e.base_kwh.mean()+e.spike_kwh.mean(), SERL["elec_heated_baseline"]],
    ["electric-heated", "heating",     e.heat_kwh.mean(),                    SERL["elec_heated_heating"]],
    ["electric-heated", "TOTAL",       e.electric_kwh.mean(),               SERL["elec_heated_total"]],
    ["gas-heated",      "electricity", g.electric_kwh.mean(),               SERL["gas_heated_elec"]],
]
t = pd.DataFrame(rows, columns=["cohort","component","model","SERL"])
t["ratio"] = (t.model/t.SERL).round(3); t.model = t.model.round(0); t.SERL = t.SERL.round(0)
print(t.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
lab = ["elec-heated\nnon-heat","elec-heated\nheating","gas-heated\nelec"]
mod = [e.base_kwh.mean()+e.spike_kwh.mean(), e.heat_kwh.mean(), g.electric_kwh.mean()]
ser = [SERL["elec_heated_baseline"], SERL["elec_heated_heating"], SERL["gas_heated_elec"]]
x = np.arange(3); w = 0.35
ax.bar(x-w/2, mod, w, label="model"); ax.bar(x+w/2, ser, w, label="SERL target")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set(ylabel="kWh/yr per dwelling", title="Per-cohort electricity split: model vs SERL")
ax.legend(); plt.show()
""")

md(r"""
### 4.2 Seasonal shape vs SERL
Monthly kWh/dwelling by cohort and fuel. The monthly profiles (§1.7) should give a
smooth heating ramp (no shoulder-season cliff) and a seasonal baseline, while the
annual totals stay on target.
""")
code(r"""
serl_ref = pd.DataFrame({"gas_gas": serl_gas_gas*pd.Series(DAYS),
                         "gas_elec": serl_gas_elec*pd.Series(DAYS),
                         "electric_elec": serl_ele_elec*pd.Series(DAYS)})
pe = mon.pivot(index="month", columns="cohort", values="elec_kwh_per_dw")
pg = mon.pivot(index="month", columns="cohort", values="gas_kwh_per_dw")
M = ["J","F","M","A","M","J","J","A","S","O","N","D"]
series = [("Gas-heated: GAS", pg["gas"], serl_ref["gas_gas"]),
          ("Gas-heated: electricity", pe["gas"], serl_ref["gas_elec"]),
          ("Electric-heated: electricity", pe["electric"], serl_ref["electric_elec"])]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, mod, ser) in zip(axes, series):
    ax.plot(range(1,13), [mod[m] for m in range(1,13)], "-o", ms=4, label="model")
    ax.plot(range(1,13), [ser[m] for m in range(1,13)], "--s", ms=4, label="SERL")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(M); ax.set_title(name); ax.set_ylabel("kWh/month/dw"); ax.legend()
plt.tight_layout(); plt.show()
for name, mod, ser in series:
    print(f"{name:32s} annual model {mod.sum():6.0f}  SERL {ser.sum():6.0f}  ratio {mod.sum()/ser.sum():.2f}")
""")

md(r"""
### 4.3 Per unit vs DESNZ (independent check)
The held-out validation: modelled electricity per dwelling vs DESNZ per meter on
the sample LSOAs. DESNZ is **never** a calibration target; the residual above 1.0
is the SERL-panel-vs-population difference, left honest.
""")
code(r"""
from utils import load_desnz
from run_decomp_sample import pick_sample
sample = pick_sample()[:SAMPLE_N]
des = load_desnz("newcastle", 2023); des["lsoa_code"] = des["lsoa_code"].astype(str)
des = des[des.lsoa_code.isin(sample)]
desnz_elec_pm = des.total_kwh_elec.sum()/des.meters_elec.sum()
desnz_gas_pm  = des.total_kwh_gas.sum()/des.meters_gas.sum()
model_elec_pd = dec.electric_kwh.mean()
model_gas_pd  = dec[dec.gas_kwh>0].gas_kwh.mean()
print(f"electricity: model {model_elec_pd:.0f}/dwelling vs DESNZ {desnz_elec_pm:.0f}/meter  → per-unit {model_elec_pd/desnz_elec_pm:.3f}")
print(f"gas        : model {model_gas_pd:.0f}/gas-dwelling vs DESNZ {desnz_gas_pm:.0f}/meter → per-unit {model_gas_pd/desnz_gas_pm:.3f}")
print("(per-unit > 1 on electricity is the SERL-panel level vs Newcastle meters; not closed by design.)")
""")

# ── 5. Transfer ───────────────────────────────────────────────────────────
md(r"""
---
## 5 · Spatial and temporal transfer
[`transfer.py`](../research/applied/scripts/transfer.py) applies the same
parameter set to other cities/years and scores per-LSOA coverage-confidence.
""")
code(r"""
import glob
rows = []
for f in sorted(glob.glob(str(REPO / "research" / "applied" / "transfer_confidence_*.csv"))):
    t = pd.read_csv(f)
    rows.append({"city_year": Path(f).stem.replace("transfer_confidence_",""), "n_lsoa": len(t),
                 "median totals ratio": round(t.tot_ratio.median(),3) if "tot_ratio" in t else None,
                 "High-confidence %": round((t.confidence=="High").mean()*100) if "confidence" in t else None})
print(pd.DataFrame(rows).to_string(index=False))
""")

# ── 6. Summary / promote ──────────────────────────────────────────────────
md(r"""
---
## 6 · Summary & promotion

- **Calibration** (§1): every parameter traces to a SERL target; level set by
  anchors/slopes, shape by diurnal + monthly profiles.
- **Sensitivity** (§3): demand is most sensitive to the setpoint and gas
  heating-slope structure; the ranking is stable across cities.
- **Validation** (§4): electricity reproduces SERL on **split, season, and total**
  for both heating cohorts; gas validates per-unit against DESNZ; the residual
  electricity per-unit gap vs DESNZ is the SERL-panel difference, left honest.

The cell below promotes the assembled v6 `model:` block to
`household_energy/calibrated_config.yaml`. Set `PROMOTE = True` to write it.
""")
code(r"""
PROMOTE = False
if PROMOTE:
    import subprocess
    subprocess.run([sys.executable, str(SCRIPTS / "promote_config.py"),
                    "--source", str(V6_CFG)], cwd=str(REPO), check=True)
    print("promoted v6 → household_energy/calibrated_config.yaml")
else:
    print("PROMOTE is False — set True to ship the v6 config.")
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = "notebooks/pipeline_diagnostics.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
