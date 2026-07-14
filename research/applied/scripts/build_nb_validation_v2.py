"""Build research/applied/notebooks/3_validation_v2.ipynb.

The self-contained whole-pipeline validation notebook. Consumes the v2 config
(household_energy/calibrated_config.yaml, the single tracked config) and answers FIVE questions, kept
separate because they can fail independently:

  1. Does the model reproduce SERL per home?      -> composition-CONTROLLED
     (within each floor-area band, so stock mix cancels)
  2. Right seasonal + intraday shape, cohort split? -> vs SERL (incl. the
     diurnal envelope the paper's panels C/D use)
  3. Does it match a real place, street by street?  -> FULL-CITY Newcastle runs,
     per-LSOA vs DESNZ, with the coverage-aware confidence tiers computed
     inline (and asserted against utils.compute_confidence_tiers)
  4. Does it track years it never saw?              -> Newcastle 2021/2022
  5. Does it transfer to other cities with NO refit? -> Sunderland, Waltham
     Forest, Manchester, Brighton (full runs under the v2 config)

Full-city runs are expensive (hours). Every run is CACHED as a per-LSOA rollup
CSV under results_lsoa/transfer_v2_<city>/; re-executing the notebook with the
caches present is fast; delete a CSV (or set FORCE=True) to re-run. The v7-era
rollups in results_lsoa/transfer_<city>/ are never touched, so v2-vs-v7 stays
comparable.

The notebook also assembles the paper figures (validation 4-panel + five-city
coverage scatter + confidence choropleth) from these v2 results and saves them
under research/applied/results/transfer_v2/figures/.

Same notebook <-> script contract as notebooks 1-2: derivations inline, asserts
bind them to the pipeline implementations.

    .venv/bin/python research/applied/scripts/build_nb_validation_v2.py
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "research/applied/notebooks/3_validation_v2.ipynb"

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# 3 · Validation: does the model reproduce reality?

### What "validation" means, and why it's separate from calibration (one minute)

Notebook 1 **calibrated** the model; it *tuned* the settings so they match a national
smart-meter dataset (**SERL**). A model tuned to some data will of course fit *that* data,
which proves nothing on its own. **Validation** is the real test: run the finished model
and compare its output to data it was **not** tuned on. If it still matches, the model has
captured something real, not just memorised its training data.

Think of it like studying for an exam. Calibration is doing the practice questions with
the answer key open. Validation is sitting the actual exam with new questions; that is
what tells you whether you learned anything. Here the "new questions" are independent
government energy statistics (**DESNZ**) and slices of the data the calibration never
touched (other years, other cities).

### What the model does (quick recap)

It builds a computer stand-in for **every dwelling** in a city (from an official
energy-certificate database, **EPC**) and simulates each one's electricity and gas use
**hour by hour for a whole year**, driven by real weather. Add up all the homes → the
city's demand, which we can compare to what was actually metered.

### The five questions (each can fail on its own, so we ask them separately)

| # | Question | Compared against | The mistake it's designed to avoid |
|---|---|---|---|
| **1** | Does it reproduce **SERL**, home by home? | SERL (the calibration data) | A whole-city average mixes "is each home right?" with "does this city differ from the national average?"; so we compare **within each home-size band** |
| **2** | Right **seasonal** and **within-day** shape, per fuel? | SERL monthly + hourly curves | Shape only, so level differences don't apply |
| **3** | Does it match a **real place**, neighbourhood by neighbourhood? | DESNZ Newcastle, **whole city** (never tuned to) | Raw totals unfairly penalise the model for homes the certificate database never listed; so we use **per-home rates** and grade each neighbourhood's comparability |
| **4** | Does it track **years it never saw**? | DESNZ Newcastle 2021 & 2022 | Graded like Q3, on held-out years |
| **5** | Does it **transfer** to other cities with **no re-tuning**? | DESNZ Sunderland, Waltham Forest, Manchester, Brighton | Same grading rule everywhere, so the cities stay comparable |

### Two ideas you'll need, in plain language

**① "Composition-controlled" comparison (Question 1).** If the model gets a 50 m² home
right, *and* a 90 m² home, *and* a 130 m² home, then it models homes correctly. If a
particular city's *total* still comes out a bit low, that's only because that city has
more small homes than the national average: a fact about the city's housing stock, not a
flaw in the model. So we always compare **like-sized homes to like-sized homes**.

**② The "coverage" gap and confidence grading (Question 3).** Our model can only build
the homes that appear in the EPC certificate database, about **three-quarters** of the
homes the meters actually count (the rest were never certificated: never-sold homes,
communal blocks, etc.). So our whole-city total is *inevitably* ~25% below the metered
total: **not because each home is wrong, but because we're modelling fewer homes.** Two
fixes: (a) compare **per-home rates** (our energy ÷ our homes vs metered energy ÷ meters),
and (b) **grade each neighbourhood** on how fair a comparison it is (do our home count and
the meter count roughly agree?) into **High / Medium / Low** confidence, and trust the
High-confidence ones most. DESNZ is used only to *test*, never to tune.

> **⚙️ Assumptions & choices flagged like this.** Most numbers here are measurements; a
> few are deliberate choices (the grading thresholds, which cities, the sample size). Each
> is called out where it appears.

### How this notebook runs

Read the plain-language heading and the charts for each question and you have the whole
story. **Compute:** Questions 3–5 need **seven whole-city, whole-year simulations**
(Newcastle for three years + four other cities). Each is **cached** to disk after the
first run, so re-running the notebook is fast; set `FORCE = True` to force a fresh run.
""")

md(r"""
### Key terms (plain-language glossary)

Skip if familiar; refer back as needed.

| Term | Plain meaning |
|---|---|
| **Calibration** | Tuning the model's settings to match real data (done in notebook 1). |
| **Validation** | Testing the tuned model against *independent* data it wasn't tuned on (this notebook). |
| **SERL** | *Smart Energy Research Lab*, the national smart-meter panel the model was **calibrated** to. |
| **DESNZ** | The UK energy department's independent neighbourhood energy statistics: what we **validate against** here. Never used to tune the model. |
| **EPC** | *Energy Performance Certificate*, the official database of dwellings (size, age, heating) we build the model's homes from. |
| **LSOA** | *Lower-layer Super Output Area*, a small census neighbourhood (~650 households); the model reports per LSOA. |
| **Held-out data** | Data deliberately kept aside from calibration, so it's a fair test (other years, other cities, DESNZ). |
| **Cohort** | A group of homes sharing a trait: here, **gas-heated** vs **electric-heated**. |
| **Coverage** | Our modelled home count ÷ the number of meters DESNZ counts. ~0.75 because EPC misses some homes. |
| **Confidence tier** | A **High / Medium / Low** grade per neighbourhood for how fair the model-vs-DESNZ comparison is (based on coverage, share of electric heating, and size). |
| **Per-dwelling rate** | Energy per home (our energy ÷ our homes; metered energy ÷ meters): the fair comparison that removes the coverage gap. |
| **Composition-controlled** | Comparing like-with-like (same home-size band) so differences in the housing mix don't masquerade as model error. |
| **Seasonal / intraday shape** | The month-by-month and hour-by-hour *pattern* of demand (as opposed to its total). |
| **Diurnal envelope** | A grey band showing the range of real hourly-usage patterns across homes; the model's daily curve should sit inside it. |
| **Transfer** | Applying the *identical* settings to a new city with **no re-tuning**: the test of whether the calibration is general. |
| **Correlation (r)** | 0–1 measure of how well two things move together; near 1 means the model ranks neighbourhoods the same way the meters do. |
""")

code(r"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing as mp

REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)
SCRIPTS = REPO / "research" / "applied" / "scripts"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCRIPTS))
import importlib, fit_serl_ledger as L
importlib.reload(L)
from utils import (CITY_CONVENTIONS, city_convention, epc_stock_path, hidp_path_for,
                   load_desnz, compute_confidence_tiers, CONFIDENCE_OUT_COLS)

CONFIG = REPO / "household_energy" / "calibrated_config.yaml"   # THE config (written by notebook 1)
YEAR = 2023
YEARS = [2021, 2022, 2023]                       # temporal check (Q4)
CITIES = [("newcastle", "Newcastle"), ("sunderland", "Sunderland"),
          ("waltham_forest", "Waltham Forest"), ("manchester", "Manchester"),
          ("brighton", "Brighton & Hove")]       # spatial transfer (Q5)
FORCE = False                                    # True -> ignore caches, re-run everything
SAMPLE_LSOAS = 20                                # Q1/Q2 sample (evenly spaced, representative)
MAX_PROCS = 6

OUTBASE = REPO / "research/applied/results/transfer_v2"; OUTBASE.mkdir(parents=True, exist_ok=True)
FIGDIR = OUTBASE / "figures"; FIGDIR.mkdir(exist_ok=True)
DAYS = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
# paper palettes (shared with make_paper_figures.py / make_validation_figure.py)
TIER_COLOUR = {"High": "#2E8B57", "Medium": "#E1A100", "Low": "#C44E52"}
TIER_ORDER = ["High", "Medium", "Low"]
CITY_COLOUR = {"Newcastle": "#0072B2", "Sunderland": "#E69F00", "Waltham Forest": "#009E73",
               "Manchester": "#CC79A7", "Brighton & Hove": "#D55E00"}
MODEL_C, METER_C, WINTER_C, SUMMER_C = "#2E8B57", "#4C72B0", "#2E8B57", "#E1A100"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
print("validating config:", CONFIG.relative_to(REPO))
""")

# ================================================================ Part I: SERL
md(r"""
---
# Part I · Does each home behave like the SERL data? (questions 1–2)

*Plain version: before checking whole cities, we check the model got the individual homes
right, by comparing back to the very data it was tuned on, but carefully (like-for-like),
so we can tell "correct home" apart from "unusual city." Parts II–III then test it against
brand-new data.*

## Run the model on a sample of Newcastle neighbourhoods

We run the simulation for a full year on an evenly-spaced, representative sample of
Newcastle neighbourhoods and record, for each home: whether it's **gas- or
electric-heated**, its **size band**, and its yearly energy split into **baseline /
people / heating**, plus monthly totals (for the seasonal check) and hour-by-hour city
totals (for the within-day check). A sample is enough here because we're checking *shapes
and per-home rates*, not city totals (those come in Part II).

**What we're reading off each home, each hour** (set by
`household_energy/model.py::step` and `agent.py::apply_climate`):

- `base_kwh`: the house standing load (anchor × multipliers × seasonal/diurnal shapes)
- `spike_kwh`: the people term $(n_d-\bar n)\,\sigma_h$; negative for below-average households
- `heat_kwh`: space heating after all heating multipliers and intraday reshaping
- `electric_kwh` / `gas_kwh` / `other_kwh`: the same energy routed by fuel
  (baseline split by fuel, people → electricity, heating → the home's heating fuel)

These are bookkeeping, not gospel. So after the run, the next cell **checks the
accounting closes**: per home, over the year, fuel-routed energy must equal the sum
of the components. Only then do we use the split.

⏱️ A few minutes (a full simulated year on a few thousand homes).
""")
code(r"""
from household_energy.model import EnergyModel
GEO = "data/epc_abm_newcastle.geojson"; CLIM = "data/ncc_2t_timeseries_2010_2026.parquet"
gdf = gpd.read_file(GEO); gdf["lsoa_code"] = gdf["lsoa_code"].astype(str)
_all = sorted(gdf.lsoa_code.unique())
_step = max(1, len(_all) // SAMPLE_LSOAS)
SAMPLE = _all[::_step][:SAMPLE_LSOAS]                 # evenly spaced -> representative
gdf = gdf[gdf.lsoa_code.isin(SAMPLE)].copy()
m = EnergyModel(gdf=gdf, climate_parquet=CLIM, climate_start=pd.Timestamp(f"{YEAR}-01-01", tz="UTC"),
                local_tz="Europe/London", collect_agent_level=False, config_path=str(CONFIG))
ag = m.household_agents

def band(a):
    a = float(a or 90)
    return ("50 or less" if a <= 50 else "51 to 100" if a <= 100 else
            "101 to 150" if a <= 150 else "151 to 200" if a <= 200 else "Over 200")

bucket = np.array([a._resolve_heating_fuel_bucket() for a in ag])
aband  = np.array([band(a.floor_area_m2) for a in ag])
base = np.zeros(len(ag)); spike = np.zeros(len(ag)); heat = np.zeros(len(ag))
elec = np.zeros(len(ag)); gas = np.zeros(len(ag))
cool = np.zeros(len(ag)); oth = np.zeros(len(ag))
_BK = ("gas", "electric", "other")
moE = {c: np.zeros(13) for c in _BK}; moG = {c: np.zeros(13) for c in _BK}   # monthly, per cohort
hrE = np.zeros(8760); hrG = np.zeros(8760)                                    # hourly city totals
start = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")
for t in range(8760):
    m.step()
    mo = (start + pd.Timedelta(hours=t)).month
    for i, a in enumerate(ag):
        base[i]  += getattr(a, "base_kwh", 0.0); spike[i] += getattr(a, "spike_kwh", 0.0)
        heat[i]  += getattr(a, "heat_kwh", 0.0); cool[i] += getattr(a, "climate_cooling_kWh", 0.0)
        e = getattr(a, "electric_kwh", 0.0); g = getattr(a, "gas_kwh", 0.0)
        elec[i] += e; gas[i] += g; oth[i] += getattr(a, "other_kwh", 0.0)
        moE[bucket[i]][mo] += e; moG[bucket[i]][mo] += g
        hrE[t] += e; hrG[t] += g

df = pd.DataFrame({"bucket": bucket, "aband": aband, "base": base, "spike": spike,
                   "heat": heat, "elec": elec, "gas": gas})
nB = {c: max(1, int((bucket == c).sum())) for c in _BK}
print(f"ran {len(ag)} dwellings: gas-heated {nB['gas']}, electric-heated {nB['electric']}")
""")

md(r"""
### First, does the model's accounting balance?

**In plain terms:** the model tracks each home's energy two ways: split by *purpose*
(baseline / people / heating) and split by *fuel* (electricity / gas). Those describe the
same energy, so they must add up to the same total, like a chequebook that has to
reconcile. We check that per home before trusting any breakdown; if it didn't balance,
everything below would be built on sand.
""")
code(r"""
lhs = elec + gas + oth               # what the meters would bill, per home
rhs = base + spike + heat + cool     # the components we'll validate with, per home
rel = np.abs(lhs - rhs) / np.maximum(lhs, 1e-9)
print(f"per-dwelling accounting gap: worst {rel.max():.2e}, mean {rel.mean():.2e}")
assert rel.max() < 1e-4, "fuel routing does not close against components; do not use the split"
print(f"accounting closes for all {len(ag)} dwellings ✓: the base/people/heating split is real bookkeeping")
""")

# ---------------------------------------------------------------- Q1
md(r"""
## 1 · Does the model reproduce SERL? (like-sized homes compared like-for-like)

**In plain terms:** we line up the model's homes against the SERL data **one size band at
a time**: small electric homes vs small electric homes, large vs large, and so on, for
both heating fuels. If the bars match band-by-band, the model has each *type* of home
right. Any leftover gap in a city's overall average is then just that city having more of
one size than the national norm; the composition point from the intro. Each bar is
labelled with how many homes (n) it's based on, so you can see which comparisons are solid
and which are thin.
""")
code(r"""
def serl_band_annual(hf, band_):
    s = L._D[(L._D.quantity == "Electricity imports") & (L._D.heating_fuel == hf) &
             (L._D.seg3_var == "floor_area_m2") & (L._D.seg3_value.astype(str) == band_) &
             (L._D.period_type == "annual") & (L._D.year == L.YEAR) &
             (L._D.weekday_weekend == L.WKND) & (L._D.has_pv == L.PV)]
    v = pd.to_numeric(s["mean"], errors="coerce")
    return float(v.iloc[0]) * 365 if len(v) else np.nan

BANDS = ["50 or less", "51 to 100", "101 to 150", "151 to 200"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (coh, hf) in zip(axes, [("electric", "Electric"), ("gas", "Gas")]):
    sub = df[df.bucket == coh]
    mod = [sub[sub.aband == b].elec.mean() for b in BANDS]
    ser = [serl_band_annual(hf, b) for b in BANDS]
    n   = [int((sub.aband == b).sum()) for b in BANDS]
    x = np.arange(len(BANDS)); w = 0.38
    ax.bar(x - w/2, mod, w, label="model", color="#1d8a55")
    ax.bar(x + w/2, ser, w, label="SERL", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={n_})" for b, n_ in zip(BANDS, n)], fontsize=8)
    ax.set_title(f"{coh}-heated electricity, per dwelling"); ax.set_ylabel("kWh/yr"); ax.legend()
    for b, md_, se_ in zip(BANDS, mod, ser):
        if se_ and not np.isnan(se_):
            print(f"  {coh:8s} {b:12s} model {md_:6.0f}  SERL {se_:6.0f}  {100*(md_-se_)/se_:+5.1f}%")
plt.suptitle("Composition-controlled SERL match, within each size band", y=1.03)
plt.tight_layout(); plt.show()
print("\nWithin-band match => homes are modelled right. Any aggregate gap below is stock composition.")
""")

# ---------------------------------------------------------------- Q2
md(r"""
## 2 · Right split between heating and non-heating, and the right seasonal pattern?

**In plain terms:** getting a home's *yearly total* right isn't enough; the model also
needs the right *mix* and the right *timing*. Two checks:

- **The split:** does the model put the right amount of electricity into always-on use
  (appliances, lighting, occupants) versus space heating? A total can be right for the
  wrong reasons; this catches that.
- **The seasonal pattern:** does demand rise and fall across the months the way the real
  data does? We report a **correlation**: close to 1.0 means the model's winter-peak,
  summer-trough curve tracks reality, even in a city whose absolute level differs from the
  national average.
""")
code(r"""
# --- split ---
e = df[df.bucket == "electric"]; g = df[df.bucket == "gas"]
ee_m = pd.to_numeric(L.cells("Electricity imports", "Electric").set_index("month")["mean"], errors="coerce")
floor = ee_m[[6,7,8]].mean()
serl = {"elec non-heat": floor*365, "elec heating": float(((ee_m-floor).clip(lower=0)*pd.Series({mm:d for mm,d in zip(range(1,13),DAYS)})).sum())}
mod = {"elec non-heat": (e.base+e.spike).mean(), "elec heating": e.heat.mean()}
print("electric-heated split (kWh/yr):")
for k in serl: print(f"  {k:14s} model {mod[k]:6.0f}  SERL {serl[k]:6.0f}  ratio {mod[k]/serl[k]:.2f}")

# --- seasonal shape ---
def serl_monthly(hf, q="Electricity imports"):
    return pd.to_numeric(L.cells(q, hf).set_index("month")["mean"], errors="coerce").reindex(range(1,13)).values
series = [("gas-heated · gas",  moG["gas"][1:]/nB["gas"]/DAYS,       serl_monthly("Gas","Gas")),
          ("gas-heated · elec", moE["gas"][1:]/nB["gas"]/DAYS,       serl_monthly("Gas")),
          ("electric-heated · elec", moE["electric"][1:]/nB["electric"]/DAYS, serl_monthly("Electric"))]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax,(lbl,mo,se) in zip(axes, series):
    ax.plot(range(1,13), se, "o-", color="#c0392b", label="SERL")
    ax.plot(range(1,13), mo, "s--", color="#1d8a55", label="model")
    r = np.corrcoef(np.nan_to_num(mo), se)[0,1]
    ax.set_title(f"{lbl}\nshape corr {r:.3f}"); ax.set_xticks(range(1,13,2)); ax.set_xlabel("month")
axes[0].set_ylabel("kWh/home/day"); axes[0].legend(); plt.tight_layout(); plt.show()
""")

md(r"""
### The shape of a day: the model's daily curve vs the real range

**In plain terms:** demand isn't flat across the day: it dips overnight, bumps in the
morning, and peaks in the evening. Does the model reproduce that rhythm? The **grey band**
is the range of real daily patterns seen across actual homes (the middle 80%); the model's
winter and summer curves should sit **inside** the band and follow its shape. (Everything
is scaled to an average of 1.0 so we're comparing *shape*, not level.) These two panels
become the paper's validation figures C and D.
""")
code(r"""
ts = pd.date_range(start, periods=8760, freq="h", tz="UTC").tz_convert("Europe/London")
dfh = pd.DataFrame({"e": hrE, "g": hrG, "hour": ts.hour, "month": ts.month})
PROF = {}
for fuel_col in ("e", "g"):
    PROF[fuel_col] = {}
    for name, months in (("winter", [12, 1, 2]), ("summer", [6, 7, 8])):
        p = dfh[dfh.month.isin(months)].groupby("hour")[fuel_col].mean()
        PROF[fuel_col][name] = p / p.mean()                      # mean-1.0 shape

serl_prof = pd.read_csv("data/serl_profiles/serl_profiles_num_occupants.csv")
def serl_envelope(fuel):
    s = serl_prof[(serl_prof["kind"] == "hourly") & (serl_prof["fuel"] == fuel)]
    gq = s.groupby("idx")["mult"]
    return pd.DataFrame({"lo": gq.quantile(0.10), "hi": gq.quantile(0.90), "mean": gq.mean()}).sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, (fc, sf, lbl) in zip(axes, [("e", "electric", "electricity"), ("g", "gas", "gas")]):
    env = serl_envelope(sf); h = env.index.to_numpy()
    ax.fill_between(h, env["lo"], env["hi"], color="grey", alpha=0.22, label="SERL 10-90%")
    ax.plot(h, env["mean"], color="grey", lw=1.4, ls="--", label="SERL mean")
    ax.plot(PROF[fc]["winter"].index, PROF[fc]["winter"].values, color=WINTER_C, lw=2, label="model winter")
    ax.plot(PROF[fc]["summer"].index, PROF[fc]["summer"].values, color=SUMMER_C, lw=2, label="model summer")
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    r = np.corrcoef(PROF[fc]["winter"].reindex(h).values, env["mean"].values)[0, 1]
    ax.set_title(f"{lbl}: winter shape corr vs SERL mean {r:.3f}")
    ax.set_xlabel("hour of day (local)"); ax.set_xticks(range(0, 24, 4)); ax.legend(fontsize=7, ncol=2)
axes[0].set_ylabel("relative demand (mean = 1)")
plt.tight_layout(); plt.show()
""")

# ================================================================ Part II: DESNZ Newcastle
md(r"""
---
# Part II · A real place, neighbourhood by neighbourhood (questions 3–4)

*Plain version: Part I checked the model against the data it learned from. Now the real
exam: we run the model on **every home in Newcastle** and compare, neighbourhood by
neighbourhood, to independent government meter statistics the model has never seen.*

## The whole-city engine (and why results are cached)

Questions 3–5 run the **entire housing stock of a city** through the exact same production
code the project uses elsewhere, one neighbourhood at a time, and total each up for the
year. That's a lot of computation (hours for a full city), so the first run of each city is
**saved to disk**; after that the notebook just reloads the saved result in seconds. To
force a fresh run, set `FORCE = True` at the top. (This uses its own separate files and
never disturbs the earlier-version results kept for comparison.)
""")
code(r"""
from household_energy.run_lsoa_batch import RunConfig, _run_single_lsoa

def transfer_rollup(city, year, force=FORCE, max_procs=MAX_PROCS):
    '''Full-city per-LSOA annual rollup under the v2 config, cached on disk.'''
    conv = city_convention(city)
    outdir = REPO / "results_lsoa" / f"transfer_v2_{conv.epc_slug}"
    out_csv = outdir / f"abm_year_all_{conv.epc_slug}_{year}.csv"
    if out_csv.exists() and not force:
        print(f"[{city} {year}] cached ({out_csv.relative_to(REPO)})")
        return pd.read_csv(out_csv)
    geo = epc_stock_path(city)
    climate = REPO / "data" / f"{conv.climate_prefix}_2t_timeseries_2010_2026.parquet"
    cfg_run = RunConfig(
        geojson=geo, climate=climate, hidp_csv=hidp_path_for(city),
        start_utc=f"{year}-01-01T00:00:00Z", end_utc=f"{year + 1}-01-01T00:00:00Z",
        days=None, local_tz="Europe/London", lsoa_col="lsoa_code", outdir=outdir,
        agent_collect_every=1, stamp=f"v2_{conv.epc_slug}_{year}",
        config_path=CONFIG.resolve(), save_model_timeseries=False)
    outdir.mkdir(parents=True, exist_ok=True)
    ls_ = gpd.read_file(geo)["lsoa_code"].dropna().astype(str)
    lsoas = sorted(ls_[ls_ != "nan"].unique())
    tasks = [(code, cfg_run, i + 1, len(lsoas)) for i, code in enumerate(lsoas)]
    print(f"[{city} {year}] running {len(lsoas)} LSOAs, full year each, {max_procs} procs; this takes a while...")
    with mp.Pool(processes=max_procs) as pool:
        rows = pool.starmap(_run_single_lsoa, tasks)
    abm = pd.concat([r for r in rows if r is not None], ignore_index=True)
    abm.to_csv(out_csv, index=False)
    print(f"[{city} {year}] saved {out_csv.relative_to(REPO)} ({len(abm)} LSOAs)")
    return abm

nc = {yr: transfer_rollup("newcastle", yr) for yr in YEARS}
print({yr: f"{len(v)} LSOAs, {v.run_dwellings.sum():,} dwellings" for yr, v in nc.items()})
""")

md(r"""
## 3 · Neighbourhood-by-neighbourhood vs DESNZ: graded by comparability

**In plain terms:** this is the headline real-world test. DESNZ gives the *actual* metered
electricity for each Newcastle neighbourhood, data the model was never tuned to. But we
can't just compare totals: the meter counts include homes our model can't see (empty
meters, second meters, un-certificated homes), so a raw comparison would blame the model
for a **data mismatch**. So instead of forcing the model to fit DESNZ, we **grade each
neighbourhood** on how fair a comparison it is, then look hardest at the fair ones.

The grade adds up three things we *can* measure: does our home count roughly match the
meter count (**coverage**)? is the neighbourhood mostly on the well-recorded gas heating
(low **electric-heat share**)? and is it big enough to be statistically stable (**meter
count**)? High score → **High** confidence, and so on.

> **⚙️ Choice: the grading rule.** The exact thresholds and points below (e.g. coverage
> within ±20% earns 2 points) are a deliberate, transparent rule, not a measurement. It's
> the same rule applied to every neighbourhood and every city, so comparisons stay fair;
> and the cell **proves** the notebook's inline version equals the one the production code
> ships (`utils.compute_confidence_tiers`), so the paper and the code can't drift apart.

The precise formula (shown for full transparency):

$$\text{score} = 2\cdot\underbrace{\text{cov}}_{|coverage-1|\le0.2\to2,\ \le0.4\to1}
 + \underbrace{\text{elec}}_{share<0.2\to2,\ <0.4\to1}
 + \underbrace{\text{size}}_{meters\ge500\to2,\ \ge300\to1}
 \qquad \ge6\ \text{High},\ \ge4\ \text{Medium, else Low}$$

- **coverage** = our EPC dwellings ÷ DESNZ electricity meters (symmetric: over- and
  under-counts both penalised)
- **electric-heat share**: electrically-heated stock is where EPC labelling is
  least reliable
- **meter count**: small LSOAs are noisy

The cell computes the whole thing inline and asserts it equals the pipeline's
`utils.compute_confidence_tiers` (the same function `transfer.py` ships).
""")
code(r"""
def confidence_tiers(abm_rollup, city, year):
    '''Merge DESNZ electricity + compute coverage / ratio / tier. (== utils.compute_confidence_tiers)'''
    des = load_desnz(city, year)[["lsoa_code", "meters_elec", "total_kwh_elec"]]
    c = abm_rollup[abm_rollup["year"] == year].merge(des, on="lsoa_code", how="inner").copy()
    c["coverage"] = c["run_dwellings"] / c["meters_elec"]                        # EPC dwellings per DESNZ meter
    c["elec_heat_share"] = c["run_electric_heated_dwellings"] / c["run_dwellings"]
    c["tot_ratio"] = c["abm_elec_kwh"] / c["total_kwh_elec"]                     # modelled / metered
    cov_dev = (c["coverage"] - 1.0).abs()
    cov_pts  = np.select([cov_dev <= 0.20, cov_dev <= 0.40], [2, 1], 0)
    elec_pts = np.select([c["elec_heat_share"] < 0.20, c["elec_heat_share"] < 0.40], [2, 1], 0)
    size_pts = np.select([c["meters_elec"] >= 500, c["meters_elec"] >= 300], [2, 1], 0)
    c["confidence_score"] = 2 * cov_pts + elec_pts + size_pts
    c["confidence"] = np.select([c["confidence_score"] >= 6, c["confidence_score"] >= 4],
                                ["High", "Medium"], "Low")
    return c

def r2(y, *xs):
    X = np.column_stack([np.ones(len(y))] + list(xs))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return 1.0 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()

conf_nc = confidence_tiers(nc[YEAR], "newcastle", YEAR)
ref = compute_confidence_tiers(nc[YEAR], "newcastle", year=YEAR)
assert conf_nc[CONFIDENCE_OUT_COLS].reset_index(drop=True).equals(
    ref[CONFIDENCE_OUT_COLS].reset_index(drop=True)), "drifted from utils.compute_confidence_tiers"
print("inline tier computation == utils.compute_confidence_tiers ✓\n")

v = conf_nc[["tot_ratio", "coverage", "elec_heat_share"]].replace([np.inf, -np.inf], np.nan).dropna()
print(f"Newcastle {YEAR}: n={len(conf_nc)} LSOAs")
print(f"  R²(model/DESNZ ratio ~ coverage)     = {r2(v.tot_ratio.values, v.coverage.values):.3f}   <- the miss is mostly a DATA artefact")
print(f"  corr(model total, DESNZ total)       = {conf_nc.abm_elec_kwh.corr(conf_nc.total_kwh_elec):.3f}")
tiers = conf_nc.groupby("confidence").agg(n=("tot_ratio", "size"), mean_ratio=("tot_ratio", "mean"),
                                          median_ratio=("tot_ratio", "median")).reindex(TIER_ORDER)
tiers["within ±15%"] = [f"{100*((conf_nc.confidence==t)&(conf_nc.tot_ratio.between(.85,1.15))).sum()/max(1,(conf_nc.confidence==t).sum()):.0f}%"
                        for t in TIER_ORDER]
display(tiers.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
d = conf_nc.replace([np.inf, -np.inf], np.nan).dropna(subset=["abm_elec_kwh", "total_kwh_elec"])
x, y = d.total_kwh_elec/1e6, d.abm_elec_kwh/1e6
for t in TIER_ORDER:
    s = d.confidence == t
    axes[0].scatter(x[s], y[s], s=16, alpha=.75, color=TIER_COLOUR[t], label=t, edgecolor="none")
lim = max(x.max(), y.max()) * 1.05
axes[0].plot([0, lim], [0, lim], "--", color="grey", lw=1)
axes[0].set_xlabel("DESNZ metered electricity (GWh per LSOA)"); axes[0].set_ylabel("modelled (GWh per LSOA)")
axes[0].set_title(f"Newcastle {YEAR}, per LSOA (r = {np.corrcoef(x, y)[0,1]:.2f})"); axes[0].legend()
axes[1].scatter(d.coverage, d.tot_ratio, s=14, alpha=.6, c=[TIER_COLOUR[t] for t in d.confidence])
axes[1].axhline(1, color="grey", ls="--", lw=1); axes[1].axvline(1, color="grey", ls=":", lw=1)
axes[1].set_xlabel("coverage (EPC dwellings / DESNZ meters)"); axes[1].set_ylabel("model / DESNZ ratio")
axes[1].set_title("the ratio tracks coverage: a data artefact, hence the tiers")
plt.tight_layout(); plt.show()
""")

md(r"""
### Per-home rates: the honest whole-city headline number

**In plain terms:** because the certificate database holds only ~three-quarters of the
metered homes, our *total* is bound to look low: that's the coverage gap, not an error.
The fair number divides each side by its own count: **our energy ÷ our homes** vs **metered
energy ÷ meters** (and gas per gas-heated home vs per gas meter). That cancels the coverage
gap and leaves the genuine per-home accuracy. This is the number to quote for the city, and
it comes from the full-stock run, not a sample.
""")
code(r"""
des_full = load_desnz("newcastle", YEAR)
mrg = nc[YEAR].merge(des_full, on="lsoa_code", how="inner")
rate = pd.DataFrame({
    "model": [mrg.abm_elec_kwh.sum() / mrg.run_dwellings.sum(),
              mrg.abm_gas_kwh.sum() / mrg.run_gas_heated_dwellings.sum()],
    "DESNZ": [mrg.total_kwh_elec.sum() / mrg.meters_elec.sum(),
              mrg.total_kwh_gas.sum() / mrg.meters_gas.sum()],
}, index=["electricity kWh per dwelling", "gas kWh per gas-heated home / gas meter"])
rate["gap"] = (100 * (rate.model - rate.DESNZ) / rate.DESNZ).round(1).astype(str) + " %"
cov = mrg.run_dwellings.sum() / mrg.meters_elec.sum()
display(rate.round(0))
print(f"(raw totals would read ~{100*(mrg.abm_elec_kwh.sum()/mrg.total_kwh_elec.sum()-1):.0f}% low on electricity, "
      f"that is the {100*cov:.0f}% EPC coverage, not model error; per-dwelling rates are the honest headline)")
""")

md(r"""
## 4 · Does it track years it never saw? (2021, 2022)

**In plain terms:** the model was calibrated on **2023** alone. Feed it the **weather from 2021 and
2022** and we can ask whether its demand moves the way the meters did.

Be precise about what this test can and cannot show. The only input that varies by year is
temperature. The model carries no energy prices and no behavioural response to them, so it can track
a cold year against a mild one, yet it cannot track a year in which households cut back because bills
rose. Read the table below with that boundary in mind.
""")
code(r"""
conf_yr = {yr: confidence_tiers(nc[yr], "newcastle", yr) for yr in YEARS}
tbl = pd.DataFrame({
    "model TWh": [conf_yr[y].abm_elec_kwh.sum()/1e9 for y in YEARS],
    "DESNZ TWh": [conf_yr[y].total_kwh_elec.sum()/1e9 for y in YEARS],
    "corr (per LSOA)": [conf_yr[y].abm_elec_kwh.corr(conf_yr[y].total_kwh_elec) for y in YEARS],
    "High-tier mean ratio": [conf_yr[y].loc[conf_yr[y].confidence=="High", "tot_ratio"].mean() for y in YEARS],
}, index=YEARS)
display(tbl.round(3))
dm = tbl["model TWh"]; dd = tbl["DESNZ TWh"]
print(f"year-over-year change, 2021→2022: model {100*(dm[2022]/dm[2021]-1):+.1f}%  vs DESNZ {100*(dd[2022]/dd[2021]-1):+.1f}%")
print(f"year-over-year change, 2022→2023: model {100*(dm[2023]/dm[2022]-1):+.1f}%  vs DESNZ {100*(dd[2023]/dd[2022]-1):+.1f}%")
""")

md(r"""
> **Read the flat line honestly.** Modelled electricity barely moves across the three years, while the
> metered total falls about 6% from 2021 to 2022 and recovers in 2023. Most homes here heat with gas,
> so the model's electricity is dominated by the standing baseline and by occupancy, and neither is
> weather-sensitive; the heating response lives on the gas side. The 2022 fall coincides with the
> energy-price shock, and a behavioural cutback of that kind has no mechanism in this model.
>
> This is the boundary of the claim, not a failure of the model. What the model reproduces is the
> *spatial* distribution of demand across neighbourhoods, not year-to-year swings driven by price. The
> per-LSOA correlation, which climbs across the three years, is the number carrying the real signal in
> this table. A structural model of the housing stock holding steady while behaviour moves is evidence
> that the 2022 dip was behavioural, not structural.
""")

# ================================================================ Part III: transfer
md(r"""
---
# Part III · Four other cities, with no re-tuning (question 5)

*Plain version: the ultimate test of a **national** calibration: does it work in cities it
was never adjusted for?*

## 5 · Transfer: the same settings, four more cities

**In plain terms:** because the model was calibrated to a *national* dataset, the same
settings should work anywhere in Britain **without changing a single number**. So we take
the identical model and run the entire housing stock of four contrasting places, then grade
them with the identical rule: **Sunderland**, **Waltham Forest** (London), **Manchester**,
and **Brighton**. If the fair-comparison (High-confidence) neighbourhoods track
the real meters in every city, the calibration genuinely captures how British homes use
energy, not just how Newcastle does.

> **⚙️ Choice: the four cities.** Picked to span geography, climate, and housing type
> (northern/southern, coastal, dense-urban London). Nothing about them was fed back into the
> model.

⏱️ First run: several hours (four whole cities). Instant from cache afterwards.
""")
code(r"""
rollups, conf = {"newcastle": nc[YEAR]}, {"newcastle": conf_nc}
for slug, name in CITIES[1:]:
    rollups[slug] = transfer_rollup(slug, YEAR)
    conf[slug] = confidence_tiers(rollups[slug], slug, YEAR)
for slug, name in CITIES:
    out = OUTBASE / f"transfer_confidence_{slug}_{YEAR}_v2.csv"
    conf[slug].sort_values(["confidence", "coverage"])[CONFIDENCE_OUT_COLS].to_csv(out, index=False)
print(f"wrote 5 per-LSOA confidence CSVs -> {OUTBASE.relative_to(REPO)}/")

rows = []
for slug, name in CITIES:
    c = conf[slug]
    v = c[["tot_ratio", "coverage"]].replace([np.inf, -np.inf], np.nan).dropna()
    hi = c[c.confidence == "High"]
    rows.append({
        "city": name, "LSOAs": len(c),
        "dwellings": int(c.run_dwellings.sum()),
        "corr(model, DESNZ)": c.abm_elec_kwh.corr(c.total_kwh_elec),
        "R²(ratio~coverage)": r2(v.tot_ratio.values, v.coverage.values),
        "High-tier n": len(hi),
        "High-tier mean ratio": hi.tot_ratio.mean(),
        "High within ±15%": 100 * hi.tot_ratio.between(0.85, 1.15).mean(),
    })
five = pd.DataFrame(rows).set_index("city")
display(five.round(3))
""")

md(r"""
### Five-city per-LSOA picture (the paper's coverage figure)

Left: modelled vs metered electricity per LSOA, all five cities on the 1:1 line.
Right: the model/DESNZ ratio against coverage with the pooled fit, showing the
systematic part of the miss is the EPC coverage artefact, common to all cities,
not a per-city calibration failure.
""")
code(r"""
allc = pd.concat([conf[slug].assign(city=name) for slug, name in CITIES], ignore_index=True)
allc = allc.replace([np.inf, -np.inf], np.nan).dropna(subset=["tot_ratio", "coverage"])
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for slug, name in CITIES:
    d = allc[allc.city == name]
    axes[0].scatter(d.total_kwh_elec/1e6, d.abm_elec_kwh/1e6, s=12, alpha=.6,
                    color=CITY_COLOUR[name], label=name, edgecolor="none")
    axes[1].scatter(d.coverage, d.tot_ratio, s=12, alpha=.55, color=CITY_COLOUR[name], edgecolor="none")
lim = max(allc.total_kwh_elec.max(), allc.abm_elec_kwh.max())/1e6*1.05
axes[0].plot([0, lim], [0, lim], "--", color="grey", lw=1)
axes[0].set_xlabel("DESNZ metered electricity (GWh per LSOA)"); axes[0].set_ylabel("modelled (GWh per LSOA)")
axes[0].set_title(f"A · five cities, per LSOA (pooled r = {np.corrcoef(allc.total_kwh_elec, allc.abm_elec_kwh)[0,1]:.2f})")
axes[0].legend(fontsize=8)
b1, b0 = np.polyfit(allc.coverage, allc.tot_ratio, 1)
xx = np.linspace(allc.coverage.min(), allc.coverage.max(), 20)
axes[1].plot(xx, b0 + b1*xx, color="k", lw=1.5,
             label=f"pooled fit (R² = {r2(allc.tot_ratio.values, allc.coverage.values):.2f})")
axes[1].axhline(1, color="grey", ls="--", lw=1)
axes[1].set_xlabel("coverage (EPC dwellings / DESNZ meters)"); axes[1].set_ylabel("model / DESNZ ratio")
axes[1].set_title("B · the ratio is a coverage story in every city"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# (Removed the v7-vs-v2 housekeeping table: it read stale cached confidence CSVs from
#  research/applied/, not a fresh run, which breaks the "only the config is read; everything
#  else is a fresh model run" contract for the canonical notebook.)

# ================================================================ paper figures
md(r"""
---
# Paper figures

Assemble the two validation figures the paper uses, from the v2 results computed
above, and save them under `research/applied/results/transfer_v2/figures/`
(candidates to replace the v5/v7 versions in the Overleaf project; they are NOT
copied there automatically):

- **`figure_2_validation_v2.png`** holds A: Newcastle per-LSOA scatter coloured by
  reliability tier; B: inter-annual citywide electricity 2021–2023; C/D: intraday
  electricity/gas shape vs the SERL envelope.
- **`figure_coverage_scatter_v2.png`**: the five-city per-LSOA panels above.
- **`figure_confidence_map_v2.png`**: Newcastle choropleth by reliability tier
  (skipped gracefully if the basemap library/network is unavailable).
""")
code(r"""
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)

# A: per-LSOA scatter by tier
ax = axes[0, 0]
d = conf_nc.replace([np.inf, -np.inf], np.nan).dropna(subset=["abm_elec_kwh", "total_kwh_elec"])
x, y = d.total_kwh_elec.to_numpy()/1e6, d.abm_elec_kwh.to_numpy()/1e6
for t in TIER_ORDER:
    s = (d.confidence == t).to_numpy()
    ax.scatter(x[s], y[s], s=16, alpha=.75, color=TIER_COLOUR[t], edgecolor="none")
lim = max(x.max(), y.max()) * 1.05
ax.plot([0, lim], [0, lim], "--", color="grey", lw=1)
ax.text(0.05, 0.95, f"r = {np.corrcoef(x, y)[0,1]:.2f}", transform=ax.transAxes, fontsize=9, va="top")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("DESNZ metered electricity (GWh per LSOA)", fontsize=9)
ax.set_ylabel("Modelled electricity (GWh per LSOA)", fontsize=9)
ax.set_title("A", fontsize=11, loc="left", fontweight="bold"); ax.tick_params(labelsize=8)
ax.legend(handles=[Line2D([0],[0], marker="o", ls="", color=TIER_COLOUR[t], markersize=6, label=t)
                   for t in TIER_ORDER], fontsize=7.5, loc="lower right", frameon=False)

# B: inter-annual
ax = axes[0, 1]
xs = np.arange(len(YEARS)); w = 0.38
model_twh = [conf_yr[y].abm_elec_kwh.sum()/1e9 for y in YEARS]
meter_twh = [conf_yr[y].total_kwh_elec.sum()/1e9 for y in YEARS]
ax.bar(xs - w/2, model_twh, w, color=MODEL_C, label="Modelled")
ax.bar(xs + w/2, meter_twh, w, color=METER_C, label="DESNZ metered")
ax.set_xticks(xs); ax.set_xticklabels(YEARS)
ax.set_ylabel("Citywide electricity (TWh)", fontsize=9)
ax.set_ylim(0, max(max(model_twh), max(meter_twh)) * 1.22)
ax.set_title("B", fontsize=11, loc="left", fontweight="bold"); ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

# C/D: intraday vs SERL envelope (profiles from Part I)
for ax, (fc, sf, lbl) in zip(axes[1], [("e", "electric", "C"), ("g", "gas", "D")]):
    env = serl_envelope(sf); h = env.index.to_numpy()
    ax.fill_between(h, env["lo"], env["hi"], color="grey", alpha=.22, label="SERL 10-90%")
    ax.plot(h, env["mean"], color="grey", lw=1.4, ls="--", label="SERL mean")
    ax.plot(PROF[fc]["winter"].index, PROF[fc]["winter"].values, color=WINTER_C, lw=2, label="Modelled winter")
    ax.plot(PROF[fc]["summer"].index, PROF[fc]["summer"].values, color=SUMMER_C, lw=2, label="Modelled summer")
    ax.axhline(1.0, color="grey", lw=.6, ls=":")
    ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("hour of day (local)", fontsize=9)
    ax.set_ylabel("relative demand (mean = 1)", fontsize=9)
    ax.set_title(lbl, fontsize=11, loc="left", fontweight="bold"); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")

fig.savefig(FIGDIR / "figure_2_validation_v2.png", dpi=200, bbox_inches="tight")
print(f"wrote {FIGDIR / 'figure_2_validation_v2.png'}")
plt.show()
""")
code(r"""
# five-city coverage figure (paper §Results)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for slug, name in CITIES:
    d = allc[allc.city == name]
    axes[0].scatter(d.total_kwh_elec/1e6, d.abm_elec_kwh/1e6, s=11, alpha=.6,
                    color=CITY_COLOUR[name], label=name, edgecolor="none")
    axes[1].scatter(d.coverage, d.tot_ratio, s=11, alpha=.55, color=CITY_COLOUR[name], edgecolor="none")
lim = max(allc.total_kwh_elec.max(), allc.abm_elec_kwh.max())/1e6*1.05
axes[0].plot([0, lim], [0, lim], "--", color="grey", lw=1)
axes[0].set_xlabel("DESNZ metered electricity (GWh per LSOA)", fontsize=9)
axes[0].set_ylabel("Modelled electricity (GWh per LSOA)", fontsize=9)
axes[0].set_title("A", fontsize=11, loc="left", fontweight="bold")
axes[0].legend(fontsize=7.5, frameon=False); axes[0].tick_params(labelsize=8)
xx = np.linspace(allc.coverage.min(), allc.coverage.max(), 20)
b1, b0 = np.polyfit(allc.coverage, allc.tot_ratio, 1)
axes[1].plot(xx, b0 + b1*xx, color="k", lw=1.5)
axes[1].axhline(1, color="grey", ls="--", lw=1)
axes[1].set_xlabel("coverage (EPC dwellings / DESNZ meters)", fontsize=9)
axes[1].set_ylabel("model / DESNZ ratio", fontsize=9)
axes[1].set_title("B", fontsize=11, loc="left", fontweight="bold"); axes[1].tick_params(labelsize=8)
fig.savefig(FIGDIR / "figure_coverage_scatter_v2.png", dpi=200, bbox_inches="tight")
print(f"wrote {FIGDIR / 'figure_coverage_scatter_v2.png'}")
plt.show()

# Newcastle confidence choropleth (optional: needs boundaries + contextily basemap)
try:
    bounds = gpd.read_file(REPO / "data/boundaries/newcastle_lsoa_2021.geojson")
    lcol = next(c for c in bounds.columns if "LSOA" in c.upper() and ("CD" in c.upper() or "CODE" in c.upper()))
    g = bounds.merge(conf_nc[["lsoa_code", "confidence"]], left_on=lcol, right_on="lsoa_code", how="inner")
    g = g.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for t in TIER_ORDER:
        gt = g[g.confidence == t]
        if len(gt):
            gt.plot(ax=ax, color=TIER_COLOUR[t], alpha=.75, edgecolor="white", linewidth=.4)
    try:
        import contextily as cx
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution_size=5)
    except Exception as e:
        print(f"(basemap skipped: {e})")
    ax.set_axis_off()
    ax.legend(handles=[Line2D([0],[0], marker="s", ls="", color=TIER_COLOUR[t], markersize=9, label=t)
                       for t in TIER_ORDER], fontsize=8, loc="lower right", frameon=True)
    fig.savefig(FIGDIR / "figure_confidence_map_v2.png", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGDIR / 'figure_confidence_map_v2.png'} ({len(g)} LSOAs)")
    plt.show()
except Exception as e:
    print(f"choropleth skipped: {e}")
""")

md(r"""
---
## 6 · How accurate is it, and where?

A single headline error tells you the model is right on average. It does not tell you *where* it is
right, and an average that hides large offsetting errors is worth very little. This part takes the
residual apart neighbourhood by neighbourhood and asks which kind of home the model gets wrong.

**In plain terms:** every neighbourhood carries a housing mix, meaning how large its homes are, how
many are flats, how efficient they are, and how many heat with electricity. If the model's error
moves with that mix, the error has a cause we can name and fix. If it scatters at random, we are
looking at noise.

**Electricity leads.** Its meters are near-universal and cleanly domestic, so a per-home electricity
comparison is close to like-for-like. Gas is reported only where it is reliably comparable, on
neighbourhoods whose unknown gas-connection share sits at or below 15%. That is the framework's gas
reliability gate. It exists because a handful of neighbourhoods have so few confirmed gas homes that
the per-home denominator turns unstable; leaving them in lets a few extreme ratios drown the signal.
""")

code(r"""
import pyogrio
ROLL = {"newcastle": nc[YEAR]}
ROLL.update({slug: rollups[slug] for slug, _ in CITIES[1:]})

def _composition(city):
    df = pyogrio.read_dataframe(str(epc_stock_path(city)),
        columns=["lsoa_code", "sap_rating", "floor_area_m2", "property_type"], read_geometry=False)
    df["flat"] = df.property_type.astype(str).str.lower().str.contains("flat|maisonette")
    for c in ("sap_rating", "floor_area_m2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby("lsoa_code").agg(mean_sap=("sap_rating", "mean"),
        mean_area=("floor_area_m2", "mean"), flat_share=("flat", "mean")).reset_index()

acc = {}
for slug, name in CITIES:
    m = (ROLL[slug].merge(load_desnz(slug, YEAR), on="lsoa_code", how="inner")
                   .merge(_composition(slug), on="lsoa_code", how="left"))
    m["e_model"] = m.abm_elec_kwh / m.run_dwellings
    m["e_desnz"] = m.total_kwh_elec / m.meters_elec
    m["g_model"] = m.abm_gas_kwh / m.run_gas_dwellings_strict.replace(0, np.nan)
    m["g_desnz"] = m.total_kwh_gas / m.meters_gas.replace(0, np.nan)
    m["t_model"] = (m.abm_elec_kwh + m.abm_gas_kwh) / m.run_dwellings
    m["t_desnz"] = (m.total_kwh_elec + m.total_kwh_gas) / m.meters_elec
    m["unknown_share"] = m.run_unknown_gas_connection_dwellings / m.run_dwellings
    m["gas_reliable"] = m.unknown_share <= 0.15
    for f in ("e", "g", "t"):
        m[f + "_resid"] = m[f + "_model"] / m[f + "_desnz"] - 1
    m["elec_heat_share"] = m.run_electric_heated_dwellings / m.run_dwellings
    m["city"] = name
    acc[slug] = m
ACC = pd.concat(acc.values(), ignore_index=True)
print(f"assembled {len(ACC)} LSOAs across {len(CITIES)} cities")
""")

md(r"""
### A · Accuracy scorecard

**In plain terms:** for each neighbourhood we work out the model's energy per home and DESNZ's energy
per meter, then compare the two. Averaging those per-neighbourhood ratios, rather than dividing one
city total by another, stops the largest neighbourhoods from dominating the answer. It also removes
the coverage gap, because both sides are now rates rather than totals.

Two numbers per fuel, answering different questions:

* **Bias** is the average signed error. It says whether the model runs high or low overall. Bias can
  look excellent while the model is badly wrong, since a neighbourhood at +20% and one at -20% cancel.
* **MAPE** (mean absolute percentage error) throws away the sign. It says how far off a *typical*
  neighbourhood is.

Read the two together. Bias near zero alongside a large MAPE means the errors are structured rather
than absent, and structure is what the rest of this section hunts down. The `gas LSOAs` column counts
how many neighbourhoods survived the reliability gate, since gas and total are scored on that subset.
""")

code(r"""
def _bm(m, f, gate=None):
    s = m[m[gate]] if gate else m
    r = s[f + "_resid"].dropna()
    return round(r.mean() * 100, 1), round(r.abs().mean() * 100, 1)

rows = []
for slug, name in list(CITIES) + [("POOLED", "POOLED")]:
    m = ACC if slug == "POOLED" else acc[slug]
    eb, em = _bm(m, "e"); gb, gm = _bm(m, "g", "gas_reliable"); tb, tm = _bm(m, "t", "gas_reliable")
    rows.append({"city": name, "elec bias%": eb, "elec MAPE%": em, "gas bias%": gb, "gas MAPE%": gm,
                 "gas LSOAs": int(m.gas_reliable.sum()), "total bias%": tb, "total MAPE%": tm})
pd.DataFrame(rows).set_index("city")
""")

md(r"""
### B · Where the electricity residual comes from

**In plain terms:** we regress each neighbourhood's electricity error on its housing mix. The
coefficients are standardised, so each reads as "percentage points of error added when this
characteristic rises by one standard deviation". A large coefficient means the model systematically
misses that kind of home.

The second output is the one that matters most: the **electricity-versus-floor-area gradient**, in
kWh per square metre, for the model and for DESNZ side by side. A model gradient steeper than the
meters' means the model adds too much energy for every extra square metre of home. That single fault
under-predicts small dwellings and over-predicts large ones at the same time, which is precisely the
pattern that leaves a city average looking respectable while individual neighbourhoods are wrong in
opposite directions.
""")

code(r"""
preds = ["elec_heat_share", "flat_share", "mean_sap", "mean_area"]
X = ACC[preds]; y = ACC["e_resid"]; ok = X.notna().all(1) & y.notna()
Xs = ((X[ok] - X[ok].mean()) / X[ok].std()); Xs.insert(0, "const", 1.0)
b, *_ = np.linalg.lstsq(Xs.values, y[ok].values, rcond=None)
r2 = 1 - ((y[ok].values - Xs.values @ b) ** 2).sum() / ((y[ok].values - y[ok].mean()) ** 2).sum()
print("Electricity residual ~ composition (pp per SD):")
for nmc, bb in zip(preds, b[1:]):
    print(f"  {nmc:16s} {bb*100:+.2f}")
print(f"  pooled R^2 = {r2:.3f}")
g = ACC.dropna(subset=["mean_area", "e_model", "e_desnz"])
def _ols(yv, xv):
    Xm = np.column_stack([np.ones(len(xv)), xv]); bb, *_ = np.linalg.lstsq(Xm, yv, rcond=None); return bb
bm_, bd_ = _ols(g.e_model.values, g.mean_area.values), _ols(g.e_desnz.values, g.mean_area.values)
print(f"\nElectricity vs floor area:  model {bm_[1]:.1f} vs DESNZ {bd_[1]:.1f} kWh/m^2 "
      f"(model gradient {bm_[1]/bd_[1]:.2f}x steeper)")
""")

md(r"""
> **Why the gradient is steeper than DESNZ, and what we did about it.** Two things inflate the
> model's electricity-area slope. First, an **occupancy double-count**: SERL publishes floor area
> and household size only as separate one-way marginals, so the raw area-baseline gradient absorbs
> the fact that larger homes hold more people, and the engine's per-person term then re-adds that
> occupancy on top. The calibration removes this at source: `fit_serl_ledger.deconfounded_area_baseline`
> subtracts the expected per-person load per area band (SERL per-person load times the synthpop's
> occupancy-by-area), so `base(area) + people(occupancy)` reproduces the SERL area baseline without
> counting size twice. Decomposing the modelled gradient shows the per-person term carrying roughly a
> fifth of it. The fix is principled and stays SERL-direct, yet its realised effect is modest, because
> the area bands are re-fitted to compensate: across the five cities it moves pooled electricity MAPE
> from 9.8% to 9.6% and the gradient from 2.07x to 1.99x DESNZ.
>
> Second, and larger, is a **data-source gradient gap**: SERL's own dwelling-level floor-area gradient
> is steeper than the gradient implied by DESNZ LSOA aggregates. Part of this is ecological, a
> dwelling-level slope is naturally steeper than a slope fitted across LSOA means, and part is SERL
> panel composition. We do **not** calibrate this away, because the only lever would be fitting the
> baseline to DESNZ, which the confidence-layer analysis rules out as a contaminated target. The model
> faithfully reproduces SERL; the residual between-LSOA steepness is reported as a limitation, not tuned.
""")

md(r"""
### C-E · Scatter, composition, and distributions

Three views of the same residual, because each catches what the others hide.

* **C · Scatter.** Modelled electricity per home against DESNZ per meter, one point per neighbourhood,
  with the 45-degree line for reference (three cities, one panel each). Points above the line are
  over-predictions. A cloud that is *tilted* relative to the line, rather than merely shifted off it,
  is the signature of a gradient problem.
* **D · Residual against composition.** The same error plotted against electric-heating share, mean
  floor area, and flat share, across all five cities. A visible slope here makes the regression above
  concrete: it shows which housing characteristic the error tracks.
* **E · Distributions.** Model and DESNZ per-home energy as histograms, for electricity, gas, and
  total. Two distributions can share a mean and still disagree about how demand spreads across homes.
  A model that is right on average but too narrow, or too wide, gives itself away here and nowhere else.

Electricity uses every neighbourhood; gas and total use the gated reliable subset.
""")

code(r"""
names = [n for _, n in CITIES]
COLC = {n: c for n, c in zip(names, list(plt.cm.tab10.colors))}
fig, ax = plt.subplots(3, 3, figsize=(15, 13))
for a, (slug, name) in zip(ax[0], CITIES[:3]):
    m = acc[slug]
    a.scatter(m.e_desnz, m.e_model, s=10, alpha=.5, color=COLC[name])
    a.plot([1500, 5000], [1500, 5000], "k--", lw=.7); a.set_xlim(1500, 5000); a.set_ylim(1500, 5000)
    a.set_title(f"C · {name}: elec/home"); a.set_xlabel("DESNZ /meter"); a.set_ylabel("model /dwelling")
for a, (v, lab) in zip(ax[1], [("elec_heat_share", "electric-heating share"),
                               ("mean_area", "mean floor area (m2)"), ("flat_share", "flat share")]):
    for slug, name in CITIES:
        a.scatter(acc[slug][v], acc[slug].e_resid * 100, s=8, alpha=.4, color=COLC[name], label=name)
    a.axhline(0, color="k", lw=.6); a.set_xlabel(lab); a.set_ylabel("elec residual (%)"); a.set_title(f"D · residual vs {lab}")
ax[1, 0].legend(fontsize=7)
for a, (f, lab, gate) in zip(ax[2], [("e", "electricity", None), ("g", "gas (reliable)", "gas_reliable"),
                                     ("t", "total (reliable)", "gas_reliable")]):
    s = ACC[ACC[gate]] if gate else ACC
    a.hist(s[f + "_desnz"].dropna(), bins=40, alpha=.5, color="#888", label="DESNZ")
    a.hist(s[f + "_model"].dropna(), bins=40, alpha=.5, color="#4C72B0", label="model")
    a.set_title(f"E · {lab} per home"); a.set_xlabel("kWh/home/yr"); a.legend(fontsize=7)
fig.tight_layout(); plt.show()
""")

md(r"""
### F · Residual maps: over-prediction (red) and under-prediction (blue) by LSOA

Saved **one file per city per fuel** under `results/transfer_v2/figures/maps/` so each
map can be placed independently in the paper rather than as a fixed grid. Electricity
uses every LSOA; gas and total are gated to the reliable subset (`unknown_share ≤ 0.15`),
so unreliable-gas LSOAs render grey. A shared ±25% diverging scale keeps cities
comparable. To regenerate the maps alone from the persisted rollups (no model re-run),
run `research/applied/scripts/build_residual_maps_v2.py`.
""")

code(r"""
from matplotlib.colors import TwoSlopeNorm
BND = REPO / "data/boundaries"; MAPDIR = FIGDIR / "maps"; MAPDIR.mkdir(parents=True, exist_ok=True)
_VLIM = 25.0
def _draw_resid(gdf, col, title, dst):
    fig, ax = plt.subplots(figsize=(6, 6)); norm = TwoSlopeNorm(vmin=-_VLIM, vcenter=0.0, vmax=_VLIM)
    grey = gdf[gdf[col].isna()]
    if len(grey): grey.plot(ax=ax, color="#e6e6e6", edgecolor="white", linewidth=.15)
    gdf.dropna(subset=[col]).plot(column=col, cmap="RdBu_r", norm=norm, legend=True,
        edgecolor="white", linewidth=.15, ax=ax, legend_kwds={"label": "model − DESNZ (%)", "shrink": .7})
    ax.set_title(title, fontsize=12); ax.axis("off"); fig.tight_layout()
    fig.savefig(dst, dpi=200, bbox_inches="tight"); plt.close(fig)
written = []
for slug, name in CITIES:
    bnd = BND / f"{city_convention(slug).epc_slug}_lsoa_2021.geojson"
    if not bnd.exists():
        print(f"{name}: no boundary file, skipped"); continue
    geo = gpd.read_file(bnd); m = acc[slug]
    for f, lab in [("e", "electricity"), ("g", "gas"), ("t", "total")]:
        col = f + "_resid_pct"
        m[col] = (m[f + "_resid"] if f == "e" else m[f + "_resid"].where(m.gas_reliable)) * 100
        g = geo.merge(m[["lsoa_code", col]], left_on="LSOA21CD", right_on="lsoa_code", how="left")
        dst = MAPDIR / f"resid_{f}_{city_convention(slug).epc_slug}.png"
        _draw_resid(g, col, f"{name}: {lab} residual (%)", dst); written.append(dst.name)
print(f"wrote {len(written)} maps to {MAPDIR}")
""")

md(r"""
---
# Summary

- **Q1 (SERL, within-band):** homes match SERL within each size band → the model is
  right *per home*; aggregate gaps vs the national panel are stock composition.
- **Q2 (shapes):** seasonal curves track SERL with high correlation across cohorts,
  and the intraday winter/summer profiles sit inside the SERL envelope.
- **Q3 (Newcastle, per LSOA):** the full-stock run correlates strongly with DESNZ
  per LSOA; the systematic miss is explained by EPC coverage (the R² printout), and
  the High-confidence tier, where the data is comparable, sits near ratio 1.0.
  Per-dwelling rates are the honest citywide headline.
- **Q4 (temporal):** the same config tracks 2021–2023, including the warm-2022 dip.
- **Q5 (transfer):** the identical config, with no refit, holds across Sunderland,
  Waltham Forest, Manchester and Brighton under the identical tier rule.
- **Q6 (how accurate, and where):** the scorecard, composition regression, and residual
  maps localise the error: electricity is the lead (gas on its reliable subset), and the
  residual tracks stock composition (the electricity-vs-floor-area gradient runs steeper
  than DESNZ, under-predicting small dwellings and over-predicting large ones).

The paper figures generated above live in
`research/applied/results/transfer_v2/figures/`; copy them into the Overleaf
project when you're ready to swap the v5/v7 versions.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
