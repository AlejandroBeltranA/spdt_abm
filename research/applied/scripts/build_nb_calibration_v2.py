"""Build research/applied/notebooks/1_calibration_v2.ipynb.

The SERL-direct calibration notebook, written on a "best of both worlds"
contract:

  - every derivation is written OUT IN FULL in the notebook cell (visible
    pandas/numpy, in the style of the original serl_calibration_clean.ipynb),
    so a reader can follow the math without opening another file;
  - fit_serl_ledger.py remains the single source of truth: each cell ends by
    asserting its inline result equals what the script computes, so the
    notebook and the pipeline can never silently drift apart.

Walks every parameter's derivation in the order the model uses them: the SERL
data itself -> house standing load -> people -> heating -> shapes ->
deprivation -> cohort correction (live, not inside L.main()) -> assemble ->
validate against the running model.

    .venv/bin/python research/applied/scripts/build_nb_calibration_v2.py
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "research/applied/notebooks/1_calibration_v2.ipynb"

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# 1 · Calibration v2: building the model's numbers straight from smart-meter data

## What this notebook is

This model estimates **how much electricity and gas every home in an area uses,
hour by hour**. To be trusted for policy it can't run on guesses; every number
it uses has to come from real measured data. This notebook is where those numbers
are produced, and it shows the origin of **every single one**.

The data source is **SERL** (the Smart Energy Research Lab): anonymised smart-meter
readings from ~13,000 GB homes, published as **aggregate tables**, averages for
groups of homes (e.g. "average gas use of gas-heated homes in February"). We never
see an individual home, only group averages. Those averages are what we read here.

**The golden rule of v2:** *nothing in this notebook runs the model to decide a
parameter.* Every number is read directly from a SERL table with a one-line
formula. (The older v1 pipeline ran the full simulation 8760 times to back the
numbers out, which made it impossible to tell what came from data and what came
from the model. v2 removes that circularity.) At the very end (§G) we *do* run the
model once, but only to **check** the finished numbers reproduce SERL, never to
set them.

**How to read this notebook (the notebook ↔ script contract).** The pipeline's
authoritative implementation is one script, `fit_serl_ledger.py` (imported as `L`).
You should never need to open it: **every derivation below is written out in
full in the cell**, the actual filters, the actual regression, the actual
division. Each cell then ends with an `assert` that the inline result equals the
script's result, so what you read here *is* what the pipeline computes, provably.
If someone edits the script, the corresponding assert here fails on the next run.

## The one equation the whole model computes

For a single dwelling *d*, in hour *h* of month *m*:

$$
\underbrace{E^{\text{elec}}_{d,h,m}}_{\text{electricity used}}
=
\underbrace{A^{e}_{c(d)}\;\mu^{\text{area}}_{d}\;\mu^{\text{imd}}_{d}\;s^{\text{light}}_{m}}_{\text{1. house standing load}}
\;+\;
\underbrace{(n_d-\bar n)\,\sigma_h}_{\text{2. people}}
\;+\;
\underbrace{S^{e}\;\mu^{\text{area}}_{d}\,\mu^{\text{sap}}_{d}\;\max(0,\tau-T_{d,h})\;s^{\text{heat}}_{m}}_{\text{3. electric heating (if electrically heated)}}
$$

$$
\underbrace{E^{\text{gas}}_{d,h,m}}_{\text{gas used}}
=
\underbrace{A^{g}}_{\text{cooking / hot water}}
\;+\;
\underbrace{S^{g}\;\mu^{\text{area}}_{d}\,\mu^{\text{sap}}_{d}\,\mu^{\text{age}}_{d}\;\max(0,\tau-T_{d,h})\;s^{\text{heat}}_{m}}_{\text{gas heating (if gas heated)}}
$$

In words: **a house has a standing load; the people in it add more; when it's cold
they heat; bigger/leakier houses lose heat faster; and use rises and falls by hour
and month.** Every symbol is defined below, and every symbol is set by one section
of this notebook.

| symbol | plain meaning | set in |
|---|---|---|
| $A^{e},A^{g}$ | **anchor**: the average kWh/hour of a home when it isn't heating (the summer floor), for electricity / gas | §A.1 |
| $c(d)$ | the **cohort** of dwelling *d*: is it gas-heated or electrically heated? | (data) |
| $\mu^{\text{area}}$ (baseline) | **size multiplier on the standing load**: bigger homes have more of everything on | §A.2 |
| $s^{\text{light}}_{m}$ | **lighting-season shape**: non-heating electricity is higher in dark winter months | §A.3 |
| $n_d,\ \bar n$ | number of people in the home; $\bar n$ = the SERL average (2.29) | §B |
| $\sigma_h$ | **per-person load at hour** *h*: how much one extra occupant adds, hour by hour | §B |
| $\tau$ | **heating setpoint**: the outdoor temperature below which homes start heating | §C.1 |
| $T_{d,h}$ | the outdoor temperature at the dwelling that hour | (climate data) |
| $\max(0,\tau-T)$ | **how cold it is**: degrees below the setpoint (zero in summer) | §C.1 |
| $S^{e},S^{g}$ | **heating slope**: extra kWh per degree of cold, for electric / gas heating | §C.1 |
| $\mu^{\text{area}},\mu^{\text{sap}},\mu^{\text{age}}$ (heating) | heating multipliers for floor area, energy-efficiency rating, building age | §C.2 |
| $s^{\text{heat}}_{m}$ | **heating-season shape**: heating is concentrated in winter months | §C.3 |
| $\mu^{\text{imd}}$ | **deprivation multiplier**: richer homes use a bit more electricity | §D |

## A few words you'll see repeatedly

- **Cohort**: a group of homes that share a heating fuel. The two that matter are
  *gas-heated* and *electrically-heated*. They behave differently, so each gets its
  own anchor and heating slope.
- **Anchor vs multiplier**: the *anchor* is the cohort's **average** (one number).
  The *multipliers* spread that average across individual homes by their
  attributes. The anchor sets the level; the multipliers set the differences.
- **Marginal**: SERL publishes averages **one attribute at a time** ("by property
  type", "by floor area"), not for every combination at once. So we read each
  attribute's effect separately and multiply them together.
- **Mean-1.0 / recentring**: every multiplier is scaled so the *average* home gets
  ×1.0. That way the multipliers only **redistribute** energy between homes; they
  don't secretly raise or lower the total. (§A.2 shows this happening, weights and
  all.)
""")

code(r"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Find the repo root (the folder containing household_energy/) and make imports work.
REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)
SCRIPTS = REPO / "research" / "applied" / "scripts"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCRIPTS))

# fit_serl_ledger is the pipeline's authoritative implementation. The notebook
# re-derives everything inline and ASSERTS agreement with it, cell by cell.
import importlib, fit_serl_ledger as L
importlib.reload(L)

SHIPPED = REPO / "household_energy" / "calibrated_config.yaml"   # the currently-shipped config (v7)
print("SERL input tables :", L.DAILY.name, "+", L.DIURNAL.name)
print(f"fixed slice we always read: year={L.YEAR}, weekday+weekend={L.WKND!r}, homes without solar PV={L.PV!r}")
print("reference categories (the '1.0' point for each multiplier):", L.REF)
""")

# ----------------------------------------------------------------- 0. the data
md(r"""
## 0 · Meet the data: what a "SERL cell" actually is

Everything below reads from **two CSV files**, so before deriving anything, look at
them. `daily_targets.csv` holds daily-average kWh for groups of homes; each row is
one group in one month (or one year). The columns that matter:

| column | meaning |
|---|---|
| `quantity` | what the meter measured: `"Electricity imports"` or `"Gas"` |
| `heating_fuel` | which cohort: `"Gas"`, `"Electric"`, or `"All"` homes |
| `seg3_var` / `seg3_value` | an optional **third split**: `"none"` = no further split; or e.g. `seg3_var="floor_area_m2"`, `seg3_value="51 to 100"` = only homes of that size |
| `period_type` | `"monthly"` (12 rows per group) or `"annual"` (1 row) |
| `month`, `year` | which period |
| `weekday_weekend` | `"weekday"`, `"weekend"`, or `"both"` |
| `has_pv` | homes with solar panels (`"Yes"`/`"No"`); we use `"No"` (their imports understate what they actually use) |
| `mean` | **the number: average daily kWh per home in that group** |
| `mean_hdd`, `mean_temp` | how cold that period was (heating-degree-days / °C), which lets us relate use to weather |
| `n_rounded` | how many homes are in the group (bigger = more trustworthy) |

A **"cell"** = one fully-specified group: pick a quantity, a cohort, a third split,
a period. That returns 12 monthly rows (or 1 annual row). Every parameter in this
notebook is a one-line formula over one or two cells.

The second file, `diurnal_targets_hourly_mean.csv`, is the same idea but with 24
`hour` rows instead of months; it gives the *within-day* shape (§B).
""")
code(r"""
daily = pd.read_csv(L.DAILY)         # the exact file the pipeline reads
print(f"daily_targets.csv: {len(daily):,} rows")
for c in ["quantity", "heating_fuel", "seg3_var", "period_type", "year", "weekday_weekend", "has_pv"]:
    vals = sorted(daily[c].dropna().astype(str).unique())
    print(f"  {c:16s}: {vals if len(vals) <= 8 else vals[:8] + ['...']}")

# One inline helper used for EVERY read below; these eight filters ARE the whole
# access pattern. (Identical to L.cells; asserted right here.)
def cell(quantity, hf, seg3_var="none", seg3_value="none", period_type="monthly"):
    return daily[(daily.quantity == quantity) & (daily.heating_fuel == hf) &
                 (daily.seg3_var == seg3_var) & (daily.seg3_value.astype(str) == str(seg3_value)) &
                 (daily.period_type == period_type) & (daily.year == L.YEAR) &
                 (daily.weekday_weekend == L.WKND) & (daily.has_pv == L.PV)]

ex = cell("Gas", "Gas")
assert ex.reset_index(drop=True).equals(L.cells("Gas", "Gas").reset_index(drop=True)), "notebook filter drifted from fit_serl_ledger.cells()"
print("\nExample cell, 'gas use of gas-heated homes, monthly, 2023' (12 rows, one per month):")
ex[["month", "mean", "mean_hdd", "n_rounded"]].set_index("month").round(2)
""")

# ----------------------------------------------------------------- A. House
md(r"""
## A · The house's standing load (the part that's always on)

### A.1 Anchors: read the summer floor straight off SERL

**Plain English.** Every home draws a baseline of power that has nothing to do with
heating: fridge, lights, cooking, hot water, standby. The cleanest place to
measure it is **summer**, when almost no one runs space heating, so the meter shows
only that baseline. We take the average June–August daily total and divide by 24 to
get kWh per hour.

**Formula.** For each cohort *c* and fuel:
$$A \;=\; \frac{\text{mean daily kWh over June, July, August}}{24}\quad(\text{kWh/hour})$$

**Where it comes from.** Three SERL cells, one number each:
- electricity of **gas-heated** homes  → `baseline_anchor_elec_kwh_per_hour`
- gas of **gas-heated** homes (cooking + hot water) → `baseline_anchor_gas_kwh_per_hour`
- electricity of **electrically-heated** homes → `baseline_anchor_elec_kwh_per_hour_electric`

> **Heads-up: these raw means get one correction later (§E).** The multipliers below
> are scaled so the *average home in the whole country* is ×1.0. Yet electrically-
> heated homes are mostly small flats, so *within that cohort* the multipliers
> average below 1.0. Left alone, that would push electric homes' energy down twice
> (once because their anchor is already low, once via the multipliers). §E computes
> that correction **live, in front of you**: it bumps the electric anchor from
> ~0.30 to ~0.41, which happens to land on the old v7 value (0.402), reached here
> transparently instead of by running the model.
""")
code(r"""
def summer_floor(quantity, hf, seg3_var="none", seg3_value="none"):
    s = cell(quantity, hf, seg3_var, seg3_value).set_index("month")["mean"]
    s = pd.to_numeric(s, errors="coerce").reindex([6, 7, 8])       # June, July, August
    return float(s.mean())

ge = summer_floor("Electricity imports", "Gas")       # gas-heated home's electricity
gg = summer_floor("Gas", "Gas")                       # gas-heated home's gas (cooking/hot water)
ee = summer_floor("Electricity imports", "Electric")  # electric-heated home's electricity

# contract check: the inline read equals the pipeline's
for q, hf, v in [("Electricity imports", "Gas", ge), ("Gas", "Gas", gg), ("Electricity imports", "Electric", ee)]:
    assert np.isclose(v, L.summer_baseline(q, hf)), f"drifted from script: {q}/{hf}"
print("inline == fit_serl_ledger  ✓")

pd.DataFrame({
    "anchor (kWh/hour)": [round(ge/24, 4), round(gg/24, 4), round(ee/24, 4)],
    "= SERL summer (kWh/day)": [round(ge, 2), round(gg, 2), round(ee, 2)],
}, index=["baseline_anchor_elec_kwh_per_hour   (gas-heated home, electricity)",
          "baseline_anchor_gas_kwh_per_hour    (gas cooking + hot water)",
          "baseline_anchor_elec_kwh_per_hour_electric (electric-heated home)"])
""")

md(r"""
### A.2 How the standing load varies with the home, and the recentring step

**Plain English.** A bigger home has more of everything (rooms, appliances, lights),
so its standing load is higher. We measure *how much* higher by comparing each
group's summer baseline to a reference group's; that's a **marginal multiplier**.
Then we **recentre**: rescale the whole set so the *stock-average home* gets exactly
×1.0, using SERL's own population counts as weights. Recentring is what makes
multipliers redistribute energy between homes without changing the total.

**Worked example first** (one division, nothing hidden), then the full tables.

**One number, one job, but done right.** Property type and floor area overlap: a
detached house is also a large house. If both raw ratios multiplied the baseline, a
small flat would be double-shrunk (type ~0.79 × area ~0.72 = 0.57 instead of the
true ~0.75). An earlier version solved this by neutralising the type multiplier
entirely (size → floor area alone). Validation against real meters then showed that
**over-corrects**: the model read systematically low exactly where detached/semi
share is high. Detached-ness carries real standing load *beyond* its floor area
(outbuildings, more appliances, gardens). So we keep the **residual**:

$$\mu^{\text{type,resid}}_{t}
=\frac{\text{observed SERL type ratio}_t}
      {\underbrace{\sum_b P(\text{area band }b\mid t)\;\mu^{\text{area}}_{b}}_{\text{what the type's own size mix already explains}}}$$

$P(b\mid t)$, which sizes each type comes in, is a *stock distribution* (pooled
five-city EPC, ~580k dwellings), not an energy fit. Houses come out slightly above
1.0 (more standing load than their size implies), flats below. Gas keeps the full
neutralisation: its validation miss is a level story, not a composition one.

**Formula for every multiplier set.** For each category *k* of a split:
$$\mu_{k}=\frac{\text{summer baseline of group }k}{\text{summer baseline of the reference group}}
\qquad\text{then}\qquad
\mu_{k}\leftarrow\frac{\mu_{k}}{\sum_k w_k\mu_k / \sum_k w_k}$$
where $w_k$ = SERL's home count for category *k* (the recentring weights).
""")
code(r"""
# --- worked example: the Detached multiplier, by hand -------------------------
det = summer_floor("Electricity imports", "All", "building_type", "Detached")
ter = summer_floor("Electricity imports", "All", "building_type", "Terraced")   # the reference
print(f"detached homes' summer electricity {det:.2f} kWh/day  ÷  terraced {ter:.2f} kWh/day  =  ×{det/ter:.3f}")

# --- the machinery, in full ---------------------------------------------------
def shares(seg3_var, quantity, hf):
    '''SERL home-count per category (the recentring weights), from the annual rows.'''
    s = daily[(daily.quantity == quantity) & (daily.heating_fuel == hf) &
              (daily.seg3_var == seg3_var) & (daily.period_type == "annual") &
              (daily.year == L.YEAR) & (daily.weekday_weekend == L.WKND) & (daily.has_pv == L.PV)]
    return s.groupby(s.seg3_value.astype(str))["n_rounded"].max()

def baseline_marginal(quantity, hf, seg3_var):
    '''Each category's summer baseline / the reference category's. (== L.marginal kind="baseline")'''
    ref_v = summer_floor(quantity, hf, seg3_var, L.REF[seg3_var])
    out = {}
    for v in sorted(daily[daily.seg3_var == seg3_var].seg3_value.dropna().unique().astype(str)):
        gv = summer_floor(quantity, hf, seg3_var, v)
        if np.isfinite(gv):
            out[v] = round(gv / ref_v, 4)
    return out

def recentre(mults, seg3_var, quantity, hf):
    '''Divide by the stock-weighted mean so the average home is exactly x1.0. (== L.recenter)'''
    sh = shares(seg3_var, quantity, hf)
    num = den = 0.0
    for k, m_ in mults.items():
        w = float(sh.get(str(k), 0.0)); num += w * m_; den += w
    mean = num / den if den else 1.0
    return {k: round(m_ / mean, 4) for k, m_ in mults.items()}, round(mean, 4)

# --- floor area: the multiplier the baseline ACTUALLY uses --------------------
raw_area  = baseline_marginal("Electricity imports", "All", "floor_area_m2")
base_area, area_mean = recentre(raw_area, "floor_area_m2", "Electricity imports", "All")
w_area = shares("floor_area_m2", "Electricity imports", "All")
assert base_area == L.recenter({k: v[0] for k, v in L.marginal("Electricity imports", "All", "floor_area_m2", kind="baseline").items()},
                               "floor_area_m2", "Electricity imports", "All")[0], "drifted from script"
print(f"\nbaseline_elec_area_bands: raw stock-weighted mean {area_mean} -> divided out so avg home = 1.0   (inline == script ✓)")
display(pd.DataFrame({"homes in SERL (weight)": w_area.reindex(raw_area.keys()).astype("Int64"),
                      "ref-normalised (vs 51-100 m²)": raw_area,
                      "recentred (avg home = 1.0)": base_area}))

# --- property type: keep only the RESIDUAL effect (beyond what size explains) ---
raw_type = baseline_marginal("Electricity imports", "All", "building_type")
cen_type, type_mean = recentre(raw_type, "building_type", "Electricity imports", "All")

joint = L.type_area_joint()          # P(area band | type): pooled five-city EPC stock (cached CSV)
print("which sizes each property type comes in (P(area band | type), from ~580k EPC dwellings):")
display(joint.round(3))

exp_area = {t: float(sum(joint.loc[t].get(b, 0.0) * base_area.get(b, 1.0) for b in joint.columns))
            for t in cen_type if t in joint.index}
resid_raw = {t: round(cen_type[t] / exp_area[t], 4) for t in exp_area}
resid, resid_mean = recentre(resid_raw, "building_type", "Electricity imports", "All")
pte = {abm: resid[t] for t, abms in L.TYPE_MAP.items() if t in resid for abm in abms}

d_ = "Detached"
print(f"worked example, {d_}: observed ×{cen_type[d_]:.3f}; its size mix alone predicts ×{exp_area[d_]:.3f}; "
      f"residual = {cen_type[d_]:.3f}/{exp_area[d_]:.3f} = ×{cen_type[d_]/exp_area[d_]:.3f} beyond size")
pd.DataFrame({"observed SERL type ratio (recentred)": cen_type,
              "expected from its size mix alone": exp_area,
              "residual (applied in config)": resid})
""")

md(r"""
### A.2c Inefficient homes run more non-heating electricity (efficiency gradient)

**Plain English.** An EPC band-F home doesn't just leak heat; it tends to run more
*non-heating* electricity too: older appliances, immersion heaters, supplementary
plug-in heat. Validation against real meters found the model most-too-low exactly
in low-SAP neighbourhoods, so this gradient was being left on the table.

**The clean read.** Heating would contaminate this signal, so we read it from the
**gas-heated cohort's electricity** (their space heat is on the gas meter), summer
baselines by EPC band, ratio vs band D, recentred over the whole stock (it
multiplies every home's electricity baseline). → `sap_band_mult_base_electric`.
""")
code(r"""
raw_sap = baseline_marginal("Electricity imports", "Gas", "currentEnergyRating")
sap_base, sap_base_mean = recentre(raw_sap, "currentEnergyRating", "Electricity imports", "All")
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.bar(range(len(sap_base)), list(sap_base.values()), color="#5b6676")
ax.set_xticks(range(len(sap_base))); ax.set_xticklabels(sap_base.keys())
ax.axhline(1.0, color="grey", ls="--", lw=1)
ax.set_ylabel("× stock average"); ax.set_xlabel("EPC band")
ax.set_title("sap_band_mult_base_electric: standing load rises as efficiency falls")
plt.show()
print(f"band C homes ×{sap_base['C']}, band F-G ×{sap_base['F and G']}: a "
      f"{100*(sap_base['F and G']/sap_base['C']-1):.0f}% swing in NON-heating electricity")
""")

md(r"""
### A.3 Non-heating electricity is higher in winter (the "lighting" shape)

**Plain English.** Even setting heating aside, homes use ~14% more electricity in
dark winter months than in summer: lights on longer, more time indoors. We read
this seasonal shape from **gas-heated homes' electricity**, because those homes do
their heating with gas, so their electricity is a clean non-heating signal.

**Formula.** For each month *m*:
$$s^{\text{light}}_{m}=\frac{\text{gas-heated homes' electricity in month }m}{\text{their summer average}}$$
So it equals 1.0 in summer and rises above 1.0 in winter. The chart should be a
smile: high in Jan/Dec, low in Jun/Jul. → `base_profile_12_electric`.
""")
code(r"""
m12 = pd.to_numeric(cell("Electricity imports", "Gas").set_index("month")["mean"],
                    errors="coerce").reindex(range(1, 13))
light = (m12 / m12.loc[[6, 7, 8]].mean()).tolist()          # divide by the summer mean
assert np.allclose(light, L.light_profile()), "drifted from script"

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(1, 13), light, marker="o", color="#f2a900")
ax.axhline(1.0, color="grey", ls="--", lw=1)
ax.set_xticks(range(1, 13)); ax.set_xlabel("month (1=Jan ... 12=Dec)")
ax.set_ylabel("× summer level"); ax.set_title("base_profile_12_electric: non-heating electricity by month")
plt.show()
print(f"January is ×{light[0]:.3f} the summer level; July ×{light[6]:.3f}.   (inline == script ✓)")
""")

# ----------------------------------------------------------------- B. People
md(r"""
## B · People: how much each occupant adds, hour by hour

**Plain English.** More people in a home means more energy. And *when* they use it
follows the day: low overnight, a bump in the morning, a big peak in the evening.
Instead of inventing daily schedules for imaginary people (the old v1 approach used
three hand-built "archetypes"), we read the real pattern from SERL: for each hour,
compare homes with different numbers of occupants and see how much each extra
person adds.

**Formula.** For each hour *h*, fit a straight line across occupancy levels
(weighted by group size):
$$\text{electricity}(h)=\text{constant} + \sigma_h\cdot(\text{number of occupants})$$
The slope $\sigma_h$ is the **per-person load at hour** *h*. The model then adds
$(n_d-\bar n)\,\sigma_h$ to each home: homes with more people than the average
$\bar n$ get a bump, fewer get a reduction. → `per_person_slope_24h_electric`.

**Worked example first:** the 6 pm regression, points and line, so you can see with
your eyes what the slope *is*. Then the same fit for all 24 hours.
""")
code(r"""
diurnal = pd.read_csv(L.DIURNAL)

def occupancy_rows(hour):
    '''SERL hourly electricity by number of occupants, for one hour of the day.'''
    s = diurnal[(diurnal.quantity == "Electricity imports") & (diurnal.heating_fuel == "All") &
                (diurnal.seg3_var == "num_occupants") & (diurnal.year == L.YEAR) &
                (diurnal.weekday_weekend == L.WKND) & (diurnal.has_pv == L.PV) &
                (diurnal.hour == hour)].copy()
    s = s[s.seg3_value.astype(str).isin(["1", "2", "3", "4", "5", ">=6"])]
    s["n_occ"] = s.seg3_value.replace({">=6": "6"}).astype(int)
    return s

def per_person_slope(hour):
    '''Weighted straight-line fit: hourly kWh ~ occupants. The slope = one extra person's load.'''
    s = occupancy_rows(hour)
    x = s["n_occ"].to_numpy(float)
    y = pd.to_numeric(s["mean_kwh"], errors="coerce").to_numpy(float)
    w = np.sqrt(pd.to_numeric(s["n_rounded"], errors="coerce").fillna(1).to_numpy(float))
    X = np.vstack([np.ones_like(x), x]).T                 # [constant, occupants]
    const, slope = np.linalg.lstsq(w[:, None] * X, w * y, rcond=None)[0]
    return float(const), float(slope)

# --- worked example: 6 pm ------------------------------------------------------
s18 = occupancy_rows(18)
c18, b18 = per_person_slope(18)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
axes[0].scatter(s18["n_occ"], pd.to_numeric(s18["mean_kwh"]), color="#c0392b", zorder=3,
                s=pd.to_numeric(s18["n_rounded"]) / 30, label="SERL groups (size = # homes)")
xx = np.array([1, 6]); axes[0].plot(xx, c18 + b18 * xx, color="#1d8a55", label=f"fit: slope {b18:.3f} kWh/h/person")
axes[0].set_xlabel("occupants"); axes[0].set_ylabel("kWh in the 6pm hour"); axes[0].legend(fontsize=8)
axes[0].set_title("6 pm: each extra person adds a fixed amount")

# --- all 24 hours ---------------------------------------------------------------
pps = [round(per_person_slope(h)[1], 5) for h in range(24)]
assert pps == L.hourly_per_person_slope("Electricity imports"), "drifted from script"
axes[1].bar(range(24), pps, color="#f2a900")
axes[1].set_xlabel("hour of day (0-23)"); axes[1].set_ylabel("kWh/hour per extra person")
axes[1].set_title(f"per_person_slope_24h_electric: one person adds {sum(pps):.2f} kWh over a day")
plt.tight_layout(); plt.show()

# the deviation centre: SERL's average household size, weighted by group counts
occ = daily[(daily.quantity == "Electricity imports") & (daily.heating_fuel == "All") &
            (daily.seg3_var == "num_occupants") & (daily.period_type == "annual") &
            (daily.year == L.YEAR) & (daily.weekday_weekend == L.WKND) & (daily.has_pv == L.PV)].copy()
occ = occ[occ.seg3_value.astype(str).isin(["1", "2", "3", "4", "5", ">=6"])]
occ["n_occ"] = occ.seg3_value.replace({">=6": "6"}).astype(int)
panel_mean = round(float(np.average(occ["n_occ"], weights=pd.to_numeric(occ["n_rounded"]))), 3)
assert panel_mean == L.panel_mean_occupants(), "drifted from script"
print(f"panel_mean_occupants = {panel_mean}  (homes above this get a bump, below it a cut)   (inline == script ✓)")
""")

# ----------------------------------------------------------------- C. Heating
md(r"""
## C · Heating

### C.1 When does heating switch on, and how hard? (setpoint + slope)

**Plain English.** Homes heat when it gets cold. Two numbers describe this: the
**setpoint** (the outdoor temperature at which heating kicks in) and the **slope**
(how many extra kWh per degree colder it gets). SERL groups homes into six
outdoor-temperature bands; plotting each band's daily energy against its temperature
gives a hockey-stick: flat when warm, rising steeply once it's cold. We fit both
numbers from that one curve at once.

**Formula.** Fit to the six temperature-band points (weighted by group size):
$$\text{daily kWh}=\text{baseline}+S\cdot\max(0,\ \tau-T)$$
- $\tau$ = **setpoint** (the "hinge" where the line bends up) → `heating_trigger_temp_C`
- $S$ = **slope** (steepness of the cold arm), ÷24 for per-hour → `heating_slope_kWh_per_deg` (gas)
- $\max(0,\tau-T)$ = "how many degrees below the hinge" = how cold it is

Fitting both together means the slope and the setpoint use the *same* temperature
definition, so there's no mismatch to fudge afterwards. The fit below is four
visible lines of `scipy.optimize.least_squares`.
""")
code(r"""
def temp_band_rows(quantity, hf):
    '''The six SERL temperature-band cells (annual) for one cohort.'''
    s = daily[(daily.quantity == quantity) & (daily.heating_fuel == hf) &
              (daily.seg3_var == "temperature_band") & (daily.period_type == "annual") &
              (daily.year == L.YEAR) & (daily.weekday_weekend == L.WKND) & (daily.has_pv == L.PV)]
    return s.dropna(subset=["mean", "mean_temp"]).sort_values("mean_temp")

def hinge_fit(quantity, hf):
    '''Fit daily kWh = baseline + slope*max(0, setpoint - T) to the band points.'''
    s = temp_band_rows(quantity, hf)
    T = s["mean_temp"].to_numpy(float); y = s["mean"].to_numpy(float)
    w = np.sqrt(pd.to_numeric(s["n_rounded"], errors="coerce").fillna(1).to_numpy(float))
    fit = least_squares(lambda p: w * (p[0] + p[1] * np.maximum(0.0, p[2] - T) - y),
                        x0=[y.min(), 1.0, 15.0], bounds=([0, 0, 8], [50, 30, 22]))
    return dict(baseline=float(fit.x[0]), slope_per_day=float(fit.x[1]), setpoint=float(fit.x[2]))

gj = hinge_fit("Gas", "Gas")
ej = hinge_fit("Electricity imports", "Electric")
for q, hf, f in [("Gas", "Gas", gj), ("Electricity imports", "Electric", ej)]:
    ref = L.joint_setpoint_slope(q, hf)
    assert all(np.isclose(f[k], ref[k]) for k in ("baseline", "slope_per_day", "setpoint")), f"drifted: {hf}"
print(f"gas-heated : heating starts at {gj['setpoint']:.2f}°C,  slope {gj['slope_per_day']/24:.4f} kWh per degree per hour")
print(f"elec-heated: heating starts at {ej['setpoint']:.2f}°C,  slope {ej['slope_per_day']/24:.4f} kWh/deg/h (superseded in C.1b)")
setpoint = round((gj["setpoint"] + ej["setpoint"]) / 2, 3)
print(f"heating_trigger_temp_C = {setpoint}  (average of the two cohorts' hinges)   (inline == script ✓)")

s = temp_band_rows("Gas", "Gas")
T, y = s["mean_temp"].to_numpy(float), s["mean"].to_numpy(float)
xx = np.linspace(T.min(), T.max(), 60)
yy = gj["baseline"] + gj["slope_per_day"] * np.maximum(0.0, gj["setpoint"] - xx)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.scatter(T, y, color="#c0392b", zorder=3, label="SERL temperature bands (measured)")
ax.plot(xx, yy, color="#1d8a55", lw=2, label=f"fitted line, hinge at {gj['setpoint']:.1f}°C")
ax.axhline(gj["baseline"], color="grey", ls=":", lw=1, label=f"fitted baseline {gj['baseline']:.1f} ≈ §A.1's summer floor {gg:.1f}")
ax.set_xlabel("outdoor temperature (°C)"); ax.set_ylabel("gas used (kWh/day)")
ax.legend(fontsize=8); ax.set_title("Gas heating: colder outside → more gas. The bend is the setpoint.")
plt.show()
print(f"cross-check: two INDEPENDENT estimates of the gas baseline agree; hinge-fit "
      f"{gj['baseline']:.2f} vs summer cell {gg:.2f} kWh/day ({100*abs(gj['baseline']-gg)/gg:.0f}% apart)")
""")

md(r"""
### C.1b The electric heating slope: fit on the residual, or lighting gets counted twice

**The trap.** An electrically-heated home's winter electricity rises for TWO
reasons: heating, *and* the ordinary winter lighting rise every home has (§A.3). A
naive fit of monthly electricity on cold would blame *all* of the winter rise on
heating, double-counting the lighting.

**The fix, visible below:** first subtract the lighting-profiled baseline
(summer floor × the §A.3 shape) from each month, THEN fit the leftover (the
*residual*, which is pure heating) against how cold each month was. Since the
residual is ~0 by construction in summer, the line goes through the origin, and the
fit is one visible ratio of dot products (the same OLS-through-origin the original
`serl_calibration_clean.ipynb` used):
$$S^{e}=\frac{\sum_m \text{HDD}_m\cdot\text{resid}_m}{\sum_m \text{HDD}_m^2}$$
""")
code(r"""
s = cell("Electricity imports", "Electric").dropna(subset=["mean", "mean_hdd"])
m_  = pd.to_numeric(s.set_index("month")["mean"], errors="coerce")        # monthly elec, kWh/day
hdd = pd.to_numeric(s.set_index("month")["mean_hdd"], errors="coerce")    # how cold each month was

resid = pd.Series({mo: m_[mo] - ee * light[int(mo) - 1] for mo in m_.index})   # subtract lighting-profiled floor
x = np.array([hdd[mo] for mo in m_.index], float)
y = resid.to_numpy(float)
es = float((x * y).sum() / (x * x).sum())            # OLS through the origin: Σ(HDD·resid)/Σ(HDD²)
assert np.isclose(es, L.elec_heating_slope_on_residual(ee, light)), "drifted from script"

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].plot(m_.index, m_.values, "o-", color="#c0392b", label="measured monthly electricity")
axes[0].plot(m_.index, [ee * light[int(mo) - 1] for mo in m_.index], "s--", color="#f2a900",
             label="lighting-profiled baseline (subtracted)")
axes[0].set_xlabel("month"); axes[0].set_ylabel("kWh/day"); axes[0].legend(fontsize=8)
axes[0].set_title("electric-heated homes: baseline out first")
axes[1].scatter(x, y, color="#c0392b", zorder=3)
xx = np.linspace(0, x.max(), 20); axes[1].plot(xx, es * xx, color="#1d8a55", label=f"slope {es:.3f} kWh/day per HDD")
axes[1].set_xlabel("heating degree days"); axes[1].set_ylabel("residual kWh/day (pure heating)")
axes[1].legend(); axes[1].set_title("the leftover IS the heating signal")
plt.tight_layout(); plt.show()
print(f"heating_slope_kWh_per_deg_electric (pre-§E) = {round(es/24, 5)} kWh/deg/hour   (inline == script ✓)")
""")

md(r"""
### C.2 Heating varies by size and by how leaky the home is

**Plain English.** A big, old, poorly-rated home loses heat faster and needs more
energy to stay warm than a small, new, well-rated one. We capture this with three
multipliers on the heating slope: **floor area** (size), **SAP rating** (efficiency
band A–G), and **building age** (era).

**One number, one job (again).** Each effect is assigned to exactly one attribute:
**size → floor area**, **efficiency → SAP + age**. (Property type is derived only
for the ledger, §A.2.)

**How each multiplier is computed, visible below.** For each category, fit the
same weather regression the anchor sections used (daily kWh on heating-degree-days,
weighted by group size); its slope is that category's heating sensitivity. Each
category's slope ÷ the reference category's slope = the multiplier. Then recentre
to stock-average 1.0 with the §A.2 machinery. Worked example first.
""")
code(r"""
def hdd_ols(quantity, hf, seg3_var="none", seg3_value="none"):
    '''Weighted OLS of monthly daily-kWh on heating-degree-days -> (slope, intercept). (== L.hdd_slope)'''
    s = cell(quantity, hf, seg3_var, seg3_value).dropna(subset=["mean", "mean_hdd"])
    if len(s) < 3:
        return None
    x = s["mean_hdd"].to_numpy(float); y = s["mean"].to_numpy(float)
    w = np.sqrt(pd.to_numeric(s["n_rounded"], errors="coerce").fillna(1).to_numpy(float))
    X = np.vstack([np.ones_like(x), x]).T
    intercept, slope = np.linalg.lstsq(w[:, None] * X, w * y, rcond=None)[0]
    return float(slope), float(intercept)

# --- worked example: do big homes really heat harder? --------------------------
big  = hdd_ols("Gas", "Gas", "floor_area_m2", "Over 200")[0]
ref_ = hdd_ols("Gas", "Gas", "floor_area_m2", "51 to 100")[0]
print(f"'Over 200 m²' homes: {big:.2f} kWh/day per degree-day  ÷  '51-100 m²' {ref_:.2f}  =  ×{big/ref_:.2f} the heating sensitivity")

def slope_marginal(quantity, hf, seg3_var):
    '''Each category's HDD slope / the reference category's. (== L.marginal kind="slope")'''
    ref_v = hdd_ols(quantity, hf, seg3_var, L.REF[seg3_var])[0]
    out = {}
    for v in sorted(daily[daily.seg3_var == seg3_var].seg3_value.dropna().unique().astype(str)):
        f = hdd_ols(quantity, hf, seg3_var, v)
        if f is not None and np.isfinite(f[0]):
            out[v] = round(f[0] / ref_v, 4)
    return out

def slope_table(seg, label, quantity="Gas", hf="Gas", default_no_data=False):
    raw = slope_marginal(quantity, hf, seg)
    if default_no_data:
        raw.setdefault("No data", 1.0)
    cen, popmean = recentre(raw, seg, quantity, hf)
    # contract check against the script
    sref = {k: v[0] for k, v in L.marginal(quantity, hf, seg, kind="slope").items()}
    if default_no_data: sref.setdefault("No data", 1.0)
    assert cen == L.recenter(sref, seg, quantity, hf)[0], f"drifted from script: {seg}"
    print(f"{label}: raw stock-weighted mean = {popmean} -> divided out so the average home is exactly 1.0   (inline == script ✓)")
    return pd.DataFrame({"ref-normalised": raw, "recentred (avg home = 1.0)": cen}), cen

t, area_slope = slope_table("floor_area_m2", "SIZE -> heat_slope_area_bands"); display(t)
t, sap_g = slope_table("currentEnergyRating", "EFFICIENCY (SAP band) -> sap_band_mult_heating_gas"); display(t)
t, age_g = slope_table("building_age", "ERA (building age) -> building_age_mult_heating_gas", default_no_data=True); display(t)
# the electric cohort's own SAP gradient (noisier, but still SERL-direct); used by §E
t, sap_e = slope_table("currentEnergyRating", "EFFICIENCY, electric-heated -> sap_band_mult_heating_electric",
                       quantity="Electricity imports", hf="Electric"); t
""")

md(r"""
### C.3 Heating is concentrated in winter (the heating-season shape)

**Plain English.** Almost all heating happens Oct–Apr. Rather than a hard on/off
switch by month (old approach), we use a smooth monthly shape. The subtlety is the
**normalisation**: divide by the plain mean and the *annual total* of heating would
change when the shape is applied. Instead we isolate the heating part of each month
(subtract the summer floor) and divide by its **cold-weighted** mean, so that when
the engine multiplies `slope × HDD × shape`, the year's heating energy comes out
exactly as the slope dictates. Both steps are visible below.
""")
code(r"""
s = cell("Gas", "Gas").dropna(subset=["mean"])
m_  = pd.to_numeric(s.set_index("month")["mean"], errors="coerce").reindex(range(1, 13))
hdd = pd.to_numeric(s.set_index("month")["mean_hdd"], errors="coerce").reindex(range(1, 13))
heat_part = (m_ - m_.loc[[6, 7, 8]].mean()).clip(lower=0)          # take the summer floor out
wmean = float((heat_part * hdd).sum() / hdd.sum())                 # cold-weighted mean...
hmp = (heat_part / wmean).round(6).tolist()                        # ...divided out => HDD-weighted mean-1.0
assert hmp == L.seasonal_shape("Gas", "Gas", hdd_weighted=True), "drifted from script"

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(1, 13), hmp, color="#5b6676")
ax.axhline(1.0, color="grey", ls="--", lw=1)
ax.set_xticks(range(1, 13)); ax.set_xlabel("month (1=Jan ... 12=Dec)")
ax.set_title("heating_month_profile_12: heating weight by month (near zero in summer)")
plt.show()
print(f"HDD-weighted mean of the shape = {float((pd.Series(hmp, index=range(1,13)) * hdd).sum() / hdd.sum()):.4f} "
      f"(=1.0 by construction => annual heating preserved)   (inline == script ✓)")
""")

md(r"""
### C.4 The hourly shape of gas and heating: the diurnal-shape calibration

**Why this section exists.** Levels and seasons are set above; what's left is the
**time of day**. Electricity's day-shape is already covered (§B gave the per-person
hourly slope, and the whole-population hourly curve is read directly below). Gas
needs more care, because a gas meter mixes two very different rhythms:

- **Cooking / hot water**: twin peaks (breakfast and dinner), the same in any weather.
- **Space heating**: *changes shape with the weather*. On a mild day the boiler
  fires in short morning/evening bursts (peaky); on a freezing day it runs all day
  (flat). One fixed 24-hour heating profile can't capture that.

**The trick, visible below.** SERL publishes the gas day-curve separately for six
outdoor-temperature bands:

1. The **warm band (15–20 °C)** is above the ~16.5 °C heating balance point, so its
   gas curve contains *no space heat*: it IS the cooking/hot-water shape.
   Normalised to mean 1.0 → `base_profile_24h_gas`.
2. For each **colder band**, subtract that cooking curve in *absolute* kWh (cooking
   doesn't change with weather), clip at zero; the leftover is the pure heating
   day-shape for that outdoor temperature. Normalise each to mean 1.0
   → `heating_temp_profile`.
3. At runtime the engine **interpolates between the band shapes by the actual
   outdoor temperature** that hour, so the flat-when-cold / peaky-when-mild
   behaviour emerges from data, with no schedule machinery at all.

Watch the chart: the mild band (12.5 °C) should swing hard between night trough and
morning peak; the coldest band should be much flatter.
""")
code(r"""
diurnal_bands = {"-5_to_0": -2.5, "0_to_5": 2.5, "5_to_10": 7.5, "10_to_15": 12.5}
COOK_BAND = "15_to_20"                      # above the balance point -> no space heat

def diurnal_band(quantity, hf, band_):
    '''24-vector of hourly mean kWh for one outdoor-temperature band. (== L.diurnal_band)'''
    s = diurnal[(diurnal.quantity == quantity) & (diurnal.heating_fuel == hf) &
                (diurnal.seg3_var == "temperature_band") & (diurnal.seg3_value.astype(str) == band_) &
                (diurnal.year == L.YEAR) & (diurnal.weekday_weekend == L.WKND) & (diurnal.has_pv == L.PV)]
    return pd.to_numeric(s.set_index("hour")["mean_kwh"], errors="coerce").reindex(range(24)).to_numpy(float)

# 1. cooking/DHW shape = the warm band, normalised
cook_abs = diurnal_band("Gas", "Gas", COOK_BAND)
gas_day = (cook_abs / cook_abs.mean()).round(6).tolist()
assert gas_day == L.gas_cooking_profile(), "drifted from script"

# 2. heating shape per band = (band - cooking) in absolute kWh, clipped, normalised
htp = {}
for band_, tmid in diurnal_bands.items():
    heat_abs = np.maximum(diurnal_band("Gas", "Gas", band_) - cook_abs, 0.0)
    p = heat_abs / heat_abs.mean()
    htp[band_] = {"temp_mid_C": tmid, "profile": [round(float(x), 6) for x in p]}
assert htp == L.heating_temp_profiles(), "drifted from script"

# 3. the electricity day-shape (whole population), read the same way as §A's cells
p24 = pd.to_numeric(diurnal[(diurnal.quantity == "Electricity imports") & (diurnal.heating_fuel == "All") &
                            (diurnal.seg3_var == "none") & (diurnal.year == L.YEAR) &
                            (diurnal.weekday_weekend == L.WKND) & (diurnal.has_pv == L.PV)]
                    .set_index("hour")["mean_kwh"], errors="coerce").reindex(range(24))
elec_day = (p24 / p24.mean()).round(6).tolist()
assert elec_day == L.diurnal_shape("Electricity imports", "All"), "drifted from script"

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
axes[0].plot(range(24), elec_day, color="#f2a900", lw=2)
axes[0].set_title("base_profile_24h_electric: electricity day-shape")
axes[1].plot(range(24), gas_day, color="#5b6676", lw=2)
axes[1].set_title("base_profile_24h_gas: cooking/DHW twin peaks")
cmap = plt.cm.coolwarm_r
for k, (band_, d) in enumerate(htp.items()):
    axes[2].plot(range(24), d["profile"], lw=2, color=cmap(k / max(1, len(htp) - 1)),
                 label=f"{d['temp_mid_C']:+.1f}°C outside")
axes[2].legend(fontsize=7); axes[2].set_title("heating_temp_profile: flat when cold, peaky when mild")
for ax in axes:
    ax.axhline(1.0, color="grey", ls=":", lw=.8); ax.set_xlabel("hour of day"); ax.set_xticks(range(0, 24, 4))
axes[0].set_ylabel("relative level (mean = 1)")
plt.tight_layout(); plt.show()
for band_, d in htp.items():
    p = np.array(d["profile"]); print(f"  {band_:9s} ({d['temp_mid_C']:+5.1f}°C): peak/trough swing ×{p.max()/max(p.min(),1e-6):.1f}")
print("colder -> flatter, exactly the boiler behaviour the schedule machinery used to fake.   (inline == script ✓)")
""")

# ----------------------------------------------------------------- D. Deprivation
md(r"""
## D · Richer homes use a bit more electricity (deprivation gradient)

**Plain English.** Household income tracks electricity use: more appliances, more
gadgets. SERL publishes electricity by area-deprivation quintile (IMD 1 = most
deprived … 5 = least). We turn that into a small multiplier on the standing load,
using exactly the §A.2 machinery: summer-baseline ratios vs the middle quintile,
recentred to average 1.0 (so it tilts *between* homes without moving the total).

**How it's attached to homes.** Our dwellings don't carry an official IMD score yet,
but the household data (the "HIDP", joined by property reference number) carries an
**income quintile** `hh_income_band` (`q1_lowest` … `q5_highest`). We map the SERL
deprivation gradient onto those income bands as a stand-in until a real IMD field is
joined.
""")
code(r"""
raw_q = baseline_marginal("Electricity imports", "All", "IMD_quintile")
cen_q, dep_mean = recentre(raw_q, "IMD_quintile", "Electricity imports", "All")
q2band = {"1": "q1_lowest", "2": "q2_low", "3": "q3_mid", "4": "q4_high", "5": "q5_highest"}
dep = {q2band[q]: v for q, v in cen_q.items() if q in q2band}
print(f"summer-baseline ratios vs the middle quintile, recentred (raw stock mean {dep_mean}):")
pd.Series(dep).to_frame("standing-load multiplier (avg = 1.0)")
""")

# ----------------------------------------------------------------- E. correction + assemble
md(r"""
## E · The cohort correction, computed live, then assemble and self-check

This is the one step v1 buried and earlier drafts of v2 hid inside `L.main()`.
Watch it happen.

**The problem, concretely.** §A.2 recentred every multiplier so the average home
*in the whole national stock* is ×1.0. Yet the engine applies
`anchor × multipliers`, and each **cohort's** anchor is that cohort's own measured
mean. Electrically-heated homes are mostly small flats, so *within that cohort* the
size multipliers average well below 1.0; the engine would take an anchor that is
already "small flat sized" and shrink it again.

**The fix.** For each cohort, compute its **own** average multiplier (weighting by
SERL's count of that cohort's homes in each size band and property type) and divide
the anchor by it.
For the gas-heated cohort, which basically *is* the national stock, this comes out
≈1.0 and nothing changes. For the electric cohort it is ~0.75, and dividing restores
the anchor to the level the cohort actually metered. Same logic for the electric
heating slope (its multipliers are area × SAP).
""")
code(r"""
def cohort_wmean(mult_map, seg3_var, hf):
    '''A cohort's own average multiplier: weight each category by the cohort's home count. (== L.cohort_mult_mean)'''
    sh = shares(seg3_var, "Electricity imports", hf)   # counts WITHIN this cohort
    num = den = 0.0
    for val, w in sh.items():
        if val in mult_map:
            num += float(w) * mult_map[val]; den += float(w)
    return num / den if den else 1.0

print("Where each cohort's homes sit on the size bands (SERL counts, %):")
size_mix = pd.DataFrame({hf: (lambda s: s / s.sum() * 100)(shares("floor_area_m2", "Electricity imports", hf))
                         for hf in ["Gas", "Electric"]}).round(1).reindex(base_area.keys())
display(size_mix)

# baseline anchors: multiplier product = residual property-type x floor-area
pt_keymap = {serl: keys[0] for serl, keys in L.TYPE_MAP.items()}
def cohort_wmean_type(mult_map, hf):
    '''Cohort's average TYPE multiplier: SERL type shares within the cohort, mapped to ABM keys.'''
    sh = shares("building_type", "Electricity imports", hf)
    num = den = 0.0
    for val, w in sh.items():
        k = pt_keymap.get(val)
        if k in mult_map:
            num += float(w) * mult_map[k]; den += float(w)
    return num / den if den else 1.0

anchors = {}
for cohort, akey, raw_daily in [("Gas", "baseline_anchor_elec_kwh_per_hour", ge),
                                ("Electric", "baseline_anchor_elec_kwh_per_hour_electric", ee)]:
    m_type = cohort_wmean_type(pte, cohort)
    m_area = cohort_wmean(base_area, "floor_area_m2", cohort)
    m_sap  = cohort_wmean(sap_base, "currentEnergyRating", cohort)
    sub_mean = m_type * m_area * m_sap
    anchors[akey] = round(round(raw_daily / 24, 4) / sub_mean, 4)
    print(f"\n{cohort}-heated cohort: type {m_type:.3f} x size {m_area:.3f} x efficiency {m_sap:.3f} = {sub_mean:.3f}")
    print(f"  anchor {round(raw_daily/24, 4)} kWh/h ÷ {sub_mean:.3f} = {anchors[akey]}  -> {akey}")

# electric heating slope: multiplier product = floor-area (slope bands) x electric SAP
prod_h = cohort_wmean(area_slope, "floor_area_m2", "Electric") * cohort_wmean(sap_e, "currentEnergyRating", "Electric")
slope_e_corr = round(round(es / 24, 5) / prod_h, 5)
print(f"\nElectric cohort's average heating multiplier (area x SAP) = {prod_h:.3f}")
print(f"  slope {round(es/24, 5)} ÷ {prod_h:.3f} = {slope_e_corr}  -> heating_slope_kWh_per_deg_electric")
print("\nGas-heated ≈ the stock, so its correction ≈ 1.0 (self-cancelling by design).")
""")

md(r"""
### Assemble the config, and prove the notebook and the pipeline agree

`L.main()` re-runs every derivation you just watched and writes three files:
- `calibrated_config.yaml`: the numbers the model reads
- `PARAMETER_LEDGER.md`: every parameter with its SERL source and formula
- `ASSUMPTIONS.md`: the handful of numbers SERL can't provide

It also prints a **self-check**: rebuild each cohort's annual total from the
finished formula, on paper, and compare to SERL's own annual figure (arithmetic,
not a simulation; under ~5% is a pass).

The cell after it then loads the emitted YAML and **asserts, number by number,
that it equals what this notebook derived inline**: the contract, enforced.
""")
code(r"""
importlib.reload(L)            # clean slate, then the authoritative build
L.main()
""")
code(r"""
emitted = yaml.safe_load((L.OUTDIR / "calibrated_config.yaml").read_text())["model"]

derived = {   # everything this notebook computed with visible math
    "baseline_anchor_elec_kwh_per_hour":          anchors["baseline_anchor_elec_kwh_per_hour"],
    "baseline_anchor_gas_kwh_per_hour":           round(gg / 24, 4),
    "baseline_anchor_elec_kwh_per_hour_electric": anchors["baseline_anchor_elec_kwh_per_hour_electric"],
    "heating_trigger_temp_C":                     setpoint,
    "heating_slope_kWh_per_deg":                  round(gj["slope_per_day"] / 24, 4),
    "heating_slope_kWh_per_deg_electric":         slope_e_corr,
    "panel_mean_occupants":                       panel_mean,
    "per_person_slope_24h_electric":              pps,
    "base_profile_12_electric":                   [round(x, 6) for x in light],
    "heating_month_profile_12":                   hmp,
    "base_profile_24h_electric":                  elec_day,
    "base_profile_24h_gas":                       gas_day,
    "heating_temp_profile":                       htp,
    "baseline_elec_area_bands":                   base_area,
    "property_type_mult_base_electric":           pte,
    "property_type_mult_base_gas":                {abm: 1.0 for abms in L.TYPE_MAP.values() for abm in abms},
    "sap_band_mult_base_electric":                sap_base,
    "heat_slope_area_bands":                      area_slope,
    "sap_band_mult_heating_gas":                  sap_g,
    "building_age_mult_heating_gas":              age_g,
    "sap_band_mult_heating_electric":             sap_e,
    "baseline_deprivation_mult":                  dep,
    # REMOVED 2026-07-07: this was a hard-coded {E:0.08,F:0.16,G:0.16} guess, never
    # actually fitted despite the old comment; neutralise-and-check showed it was not
    # load-bearing and worsened the electricity fit. Demand path is now 100% SERL-read.
    "elec_heat_share_by_sap":                     {},
}
for k, v in derived.items():
    assert emitted.get(k) == v, f"MISMATCH on {k}: notebook {v!r} vs emitted {emitted.get(k)!r}"
print(f"all {len(derived)} parameters: notebook inline derivation == emitted calibrated_config.yaml  ✓")
pd.DataFrame({"derived inline above": {k: (f"[{len(v)} values]" if isinstance(v, (list, dict)) else v) for k, v in derived.items()},
              "emitted by fit_serl_ledger": {k: (f"[{len(emitted[k])} values]" if isinstance(emitted[k], (list, dict)) else emitted[k]) for k in derived}})
""")

md(r"""
### The full parameter ledger: "where did every number come from?"

One row per parameter: its value, the exact SERL cell it was read from, the formula,
and the sample size *n*. You've now *watched* every row of this table being
computed; the table is the portable audit trail.
""")
code(r"""
ledger = pd.DataFrame(L.LEDGER)[["param", "value", "source", "formula", "n"]]
pd.set_option("display.max_rows", 200, "display.max_colwidth", 90)
ledger
""")

md(r"""
### The assumptions register: the few numbers SERL *can't* give us

Everything not on the ledger above is here, and each row explains *why* SERL can't
supply it: e.g. cooling (UK homes barely have air-conditioning, so there's no
signal), or a heat-pump's efficiency for a home that doesn't have one yet (a
what-if, not something SERL ever measured).
""")
code(r"""
print((L.OUTDIR / "ASSUMPTIONS.md").read_text())
""")

# ----------------------------------------------------------------- compare + validate + promote
md(r"""
## F · Sanity-check against the previous approach (v7)

The old v7 config was produced by running the model in a loop (opaque, but a useful
cross-check). If v2's transparent reads land near v7's numbers, that's reassuring,
especially the electric anchor, where §E's cohort correction independently recovers
v7's value.
""")
code(r"""
old = yaml.safe_load(SHIPPED.read_text())["model"]
keys = ["baseline_anchor_elec_kwh_per_hour", "baseline_anchor_gas_kwh_per_hour",
        "baseline_anchor_elec_kwh_per_hour_electric", "heating_trigger_temp_C",
        "heating_slope_kWh_per_deg", "heating_slope_kWh_per_deg_electric"]
pd.DataFrame({"v7 (ran the model in a loop)": [old.get(k) for k in keys],
              "v2 (read straight from SERL)": [emitted.get(k) for k in keys]}, index=keys)
""")

md(r"""
## G · Now run the model and check it reproduces SERL

Everything so far was arithmetic. This is the real test: **load the numbers into the
actual simulation, run a full year on Newcastle, and compare its output to SERL's
monthly curves.** This is the step the original `serl_calibration_clean.ipynb` did
and v1 buried.

**This run is a check, not a fit**; nothing here changes any parameter. Read two
things separately:

- **Shape** (the month-by-month pattern) directly tests the calibration and doesn't
  care whether Newcastle differs from the national average. If the ABM curve tracks
  the SERL curve, the seasonal/heating logic is right.
- **Level** (absolute kWh) is run on the *Newcastle* housing stock, which is smaller
  and more flat-heavy than the *national* SERL panel. So a level below national is
  expected **stock transfer**, not an error; it's confirmed properly against
  Newcastle's own meter totals (DESNZ) in `3_validation_v2.ipynb`, not here.

**What to expect:** the two gas cohorts (most homes) land within a few percent. The
electric-heated cohort is a small, unusual Newcastle group (storage-heater flats);
after §E's cohort correction it matches SERL *national* to ~3% in the self-check,
and the remaining gap on the *Newcastle* run is genuine. Those homes really are
smaller than the national electric-heated average.

⏱️ This cell runs the simulation for a year and takes a couple of minutes.
""")
code(r"""
from household_energy.serl_calibration_v2 import run_pooled_segmentations
vcfg = yaml.safe_load((L.OUTDIR / "calibrated_config.yaml").read_text())["model"]
ovr = {f"model.{k}": v for k, v in vcfg.items()}
seg = run_pooled_segmentations(
    repo_root=REPO, target_year=L.YEAR, seg3_vars=["central_heating_type"],
    cities=["newcastle"], max_homes_per_city=1500, seed=7,
    model_overrides=ovr, process_column_preferred="lsoa_code", n_procs=4)
print(f"ran the ABM on Newcastle for {L.YEAR}: {len(seg):,} rows of hourly output")
""")
code(r"""
# Turn the raw hourly output into "average kWh per home per day, by month, by fuel
# cohort", so it lines up with the SERL monthly tables.
# NOTE the run is split across CPU processes, so a given (hour, cohort) appears in
# several chunks. We sum BOTH the energy and the home-count across chunks, THEN
# divide, otherwise per-home would be wrong.
df = seg[seg.segmentation == "central_heating_type"].copy()
df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
val = df.value.astype(str)
df["cohort"] = np.where(val.str.contains("gas", case=False), "gas",
               np.where(val.isin(["Electric storage radiators", "Other electric",
                                  "Electric radiators"]), "elec", None))
df = df.dropna(subset=["cohort"])
hh = df.groupby(["timestamp_utc", "cohort"], as_index=False).agg(
    e=("electric_kwh", "sum"), g=("gas_kwh", "sum"), n=("n_homes", "sum"))
loc = hh.timestamp_utc.dt.tz_convert("Europe/London")
hh["date"] = loc.dt.floor("D"); hh["month"] = loc.dt.month
hh["eph"] = hh.e / hh.n; hh["gph"] = hh.g / hh.n              # per-home per hour
daily_run = hh.groupby(["cohort", "date", "month"], as_index=False).agg(
    eday=("eph", "sum"), gday=("gph", "sum"))                 # per-home per day
mon = daily_run.groupby(["cohort", "month"], as_index=False).agg(
    elec=("eday", "mean"), gas=("gday", "mean"))              # avg per-home day, by month
gmo = mon[mon.cohort == "gas"].set_index("month").reindex(range(1, 13))
emo = mon[mon.cohort == "elec"].set_index("month").reindex(range(1, 13))

def serl_m(q, hf):   # the matching SERL monthly curve
    return pd.to_numeric(cell(q, hf).set_index("month")["mean"],
                         errors="coerce").reindex(range(1, 13)).values

series = [("gas-heated · gas", gmo["gas"].values, serl_m("Gas", "Gas")),
          ("gas-heated · elec", gmo["elec"].values, serl_m("Electricity imports", "Gas")),
          ("elec-heated · elec", emo["elec"].values, serl_m("Electricity imports", "Electric"))]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (lbl, abm, serl) in zip(axes, series):
    ax.plot(range(1, 13), serl, "o-", color="#c0392b", label="SERL (measured)")
    ax.plot(range(1, 13), abm, "s--", color="#1d8a55", label="ABM v2 (simulated)")
    ax.set_title(lbl); ax.set_xlabel("month"); ax.set_xticks(range(1, 13, 2))
axes[0].set_ylabel("kWh/home/day"); axes[0].legend()
plt.suptitle("Modelled vs measured (SERL) monthly demand per home, by fuel cohort", y=1.04)
plt.tight_layout()
# Save the paper figure (Paper 1 §3.4 calibration evidence). Shares the figures
# dir the validation notebook and the paper draft already point at.
PAPER_FIGDIR = REPO / "research/applied/results/transfer_v2/figures"
PAPER_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PAPER_FIGDIR / "figure_calibration_serl_match_v2.png", dpi=200, bbox_inches="tight")
print(f"wrote {PAPER_FIGDIR / 'figure_calibration_serl_match_v2.png'}")
plt.show()

DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
print(f"{'cohort':20s}{'ABM year':>10s}{'SERL year':>11s}{'level gap':>11s}   shape corr")
for lbl, abm, serl in series:
    A, S = float(np.nansum(abm * DAYS)), float(np.nansum(serl * DAYS))
    r = np.corrcoef(np.nan_to_num(abm), serl)[0, 1]
    print(f"{lbl:20s}{A:10.0f}{S:11.0f}{100*(A-S)/S:+10.1f}%   {r:.3f}")
print("\nshape corr near 1.0 = the seasonal pattern matches. Level gap on elec-heated")
print("is Newcastle's smaller electric stock (see §G notes); DESNZ settles the magnitude.")
""")

md(r"""
## H · Promote (optional)

When you're happy with the validation, set `PROMOTE = True` to make this v2 config
the one the model ships with. It writes a dated backup of the config it replaces
first (`calibrated_config.bak_YYYYMMDD_HHMMSS.yaml`), so nothing is lost. Left
`False` by default while the DESNZ check is pending.
""")
code(r"""
PROMOTE = False
src = L.OUTDIR / "calibrated_config.yaml"
dst = SHIPPED
if PROMOTE:
    import shutil
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = dst.with_name(f"{dst.stem}.bak_{stamp}{dst.suffix}")   # dated snapshot of the config being replaced
    shutil.copy(dst, backup)
    shutil.copy(src, dst)
    print(f"promoted -> {dst}  (previous config backed up to {backup.name})")
else:
    print("PROMOTE=False: config stays a candidate at", src, "; nothing shipped changed.")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
OUT.parent.mkdir(parents=True, exist_ok=True)

# RETIRED GENERATOR — DO NOT REGENERATE BLINDLY.
# 1_calibration_v2.ipynb is now HAND-AUTHORED: it carries edits that live only in
# the .ipynb and NOT in this script (e.g. sections C.4b/H, the single-source config
# wiring, the deconfounded-baseline reconciliation). Running this overwrites the
# notebook and silently drops those edits. Make figure/content changes in the
# notebook directly. To regenerate anyway (you will lose the hand edits), set
# ALLOW_NB_REGEN=1.
import os as _os
if _os.environ.get("ALLOW_NB_REGEN") != "1":
    raise SystemExit(
        "build_nb_calibration_v2.py is retired: 1_calibration_v2.ipynb is hand-authored.\n"
        "Regenerating would clobber notebook-only edits (sections C.4b/H, single-source\n"
        "config, deconfound reconciliation). Edit the .ipynb directly. To force a full\n"
        "regenerate and lose those edits, re-run with ALLOW_NB_REGEN=1.")
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
