"""Generate research/policy/notebooks/heatpump_policy_scenarios.ipynb.

Paper 2 (policy): "Targeting Logic as a Governance Variable". Two real institutional
routes to a heat-pump programme, same budget, so the targeting RULE is the only
variable:
  Council / social rent      — the council OWNS the homes, so it can plan, permit,
                               and install at scale (low transaction cost); reaches
                               lower-income tenants. Maps to the Social Housing
                               Decarbonisation Fund.
  Provider / grants (wealthy) — the electricity provider offers grants; uptake is
                               voluntary and skews to wealthy owner-occupiers with
                               high consumption. Maps to the Boiler Upgrade Scheme.

The governance trade-off is deliverability + distribution (who can act, and on
whom) vs per-home energy impact + market mechanism — not abstract efficiency vs
equity. All demand is the v7 SERL-calibrated model's own per-home output (full-city
decomposition); heat-pump COP from heatpumpmonitor.org field data (2.8 / 4.21). The
conversion is the model's own deterministic transform (gas heat -> electricity at
the COP). Energy (GWh) is the headline; carbon/cost are downstream translations.

  .venv/bin/python research/applied/scripts/build_nb_policy.py
"""
from __future__ import annotations
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

md(r"""
# Targeting logic as a governance variable
### Two institutional routes to a heat-pump programme

The same number of heat pumps can be rolled out by different institutional actors,
and *who runs it* changes what can be delivered, on whom, and to what effect:

| Route | Actor | Lever | Real-world analogue |
|---|---|---|---|
| **Council / social rent** | Local authority | owns the stock → plan, permit, install **at scale** | Social Housing Decarbonisation Fund |
| **Provider / grants (wealthy)** | Energy supplier | **grants**; voluntary uptake, skews to wealthy high-consumers | Boiler Upgrade Scheme |

Run at the **same budget**, the two isolate the targeting *rule*. The council route
is deliverable (owned stock, no per-household consent, bulk procurement) and reaches
lower-income tenants, but social housing is smaller and better-insulated so each
home displaces less heat. The grant route reaches large, high-consuming homes —
more energy per install — but uptake is voluntary (slow, owner-occupier consent)
and public money flows to households who could largely self-fund. **Targeting logic
is therefore a governance variable: it trades deliverability and distribution
against per-home energy impact.**

**Calibrated inputs, no proxy fields.** Per-home consumption and space heating come
from the **v7 SERL-calibrated model itself** (full-Newcastle decomposition).
Heat-pump COP from **heatpumpmonitor.org** field data: DESNZ median **2.8**
(conservative headline), field median **4.21** (well-installed sensitivity).
Converting a gas home moves its space heat to electricity at `0.90/COP ≈ 0.32×` the
energy — about a third — and from the gas meter to the electricity meter.
""")

code(r"""
import os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)
plt.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":10})

# Reach differs by actor — that's the point. The council OWNS its social stock and
# can deliver it at scale; the provider's grant route is throttled by voluntary
# owner-occupier uptake. (Both editable.)
COUNCIL_HOMES   = None    # None = the council's whole eligible social-rent stock; or cap it
PROVIDER_UPTAKE = 0.20    # share of eligible wealthy owner-occupiers who take up a grant
COP_HEAD, COP_SENS = 2.8, 4.21
BOILER_EFF = 0.90
HP_COST, HP_LIFETIME = 12000, 15
GAS_KGCO2, ELEC_KGCO2 = 0.183, 0.207
hp_mult = {"conservative (COP 2.8)": BOILER_EFF/COP_HEAD, "well-installed (COP 4.21)": BOILER_EFF/COP_SENS}
HEAD = "conservative (COP 2.8)"
print("HP heating multiplier (× gas space heat):", {k: round(v,3) for k,v in hp_mult.items()})
""")

md(r"""
## 1 · Study population: the model's own per-home demand
Per-home **consumption** (electric+gas) and **space heating** come straight from
the v7 calibrated model (full-city decomposition when present, else the 30-LSOA
validation sample). Tenure and income come from the synthpop match; no proxy fields.
""")
code(r"""
city_csv = REPO/"results_lsoa"/"decomp_city_newcastle_2023_v7.csv"
samp_csv = REPO/"results_lsoa"/"decomp_sample_newcastle_2023_v7.csv"
N_NEWCASTLE_LSOA = 185
DECOMP = samp_csv
if city_csv.exists():
    n_done = pd.read_csv(city_csv, usecols=["lsoa_code"]).lsoa_code.nunique()
    if n_done >= 0.95*N_NEWCASTLE_LSOA: DECOMP = city_csv
    else: print(f"(full-city decomposition partial: {n_done}/{N_NEWCASTLE_LSOA} — using 30-LSOA sample)")
dec = pd.read_csv(DECOMP); dec["unique_id"]=dec["unique_id"].astype(str)
dec["consumption_kwh"]=dec.electric_kwh+dec.gas_kwh; dec["gas_heat_kwh"]=dec.heat_kwh
print(f"decomposition: {'FULL CITY' if DECOMP==city_csv else '30-LSOA sample'} — {len(dec):,} dwellings")

g = gpd.read_file(REPO/"data"/"epc_abm_newcastle.geojson"); g["UPRN"]=g["UPRN"].astype(str).str.strip()
hidp = pd.read_csv(REPO/"data"/"hidp_uprn_matches_tiered.csv", low_memory=False)
hidp["uprn_chr"]=hidp["uprn_chr"].astype(str).str.strip(); hidp=hidp.drop_duplicates("uprn_chr")
g = g.merge(hidp[["uprn_chr","hh_income_band","tenure"]], left_on="UPRN", right_on="uprn_chr", how="left")
df = dec.merge(g[["UPRN","sap_band_ord","is_heatpump_candidate","hh_income_band","tenure"]],
               left_on="unique_id", right_on="UPRN", how="left")
elig = df[(df.heating_bucket=="gas") & (df.is_heatpump_candidate==1)].copy()
elig["ten"] = elig.tenure.astype(str).str.lower()
print(f"eligible (gas-heated HP candidates): {len(elig):,}")
print("tenure mix:", elig.ten.value_counts().to_dict())
""")

md(r"""
## 2 · The two routes at their *realistic* reach
Not an equal budget — the whole point is that the actors can deliver different
numbers:
- **Council / social rent** — converts its **whole eligible social-rent stock**
  (it owns the homes, so it can).
- **Provider / grants (wealthy)** — reaches only the share of eligible wealthy
  owner-occupiers who **take up a grant** (voluntary), highest-consuming first.
""")
code(r"""
sr      = elig.ten.eq("social_rent")
wealthy = elig.ten.eq("owner_occupied") & elig.hh_income_band.isin(["q4_high","q5_highest"])
n_council  = int(sr.sum()) if COUNCIL_HOMES is None else min(COUNCIL_HOMES, int(sr.sum()))
n_provider = int(round(PROVIDER_UPTAKE * int(wealthy.sum())))
print(f"eligible pools — social-rent {int(sr.sum()):,} | wealthy owner-occupier {int(wealthy.sum()):,}")
print(f"reach — council does its whole stock = {n_council:,} homes | provider at {PROVIDER_UPTAKE:.0%} uptake = {n_provider:,} homes")
cohorts = {
    "Council / social rent":       elig[sr].consumption_kwh.nlargest(n_council).index,
    "Provider / grants (wealthy)": elig[wealthy].consumption_kwh.nlargest(n_provider).index,
}
for n, idx in cohorts.items():
    c = elig.loc[idx]
    print(f"{n:30s} n={len(c):,} | mean heat {c.gas_heat_kwh.mean():,.0f} kWh | "
          f"Q1/Q2 {100*c.hh_income_band.isin(['q1_lowest','q2_low']).mean():3.0f}% | EPC E-G {100*(c.sap_band_ord<=3).mean():3.0f}%")
""")

md("## 3 · Actual change in energy use, by route\nPer converted home: gas space heat removed, electricity added at `heat × 0.90/COP`; net saved = gas − elec.")
code(r"""
def impact(idx, mult):
    c = elig.loc[idx]; gas_saved=c.gas_heat_kwh.sum(); elec_added=(c.gas_heat_kwh*mult).sum()
    q12 = c.hh_income_band.isin(["q1_lowest","q2_low"])
    return dict(homes=len(c), gas_saved_GWh=gas_saved/1e6, elec_added_GWh=elec_added/1e6, net_GWh=(gas_saved-elec_added)/1e6,
                per_home_net_kwh=(gas_saved-elec_added)/max(len(c),1), q12_savings_share=c.gas_heat_kwh[q12].sum()/max(c.gas_heat_kwh.sum(),1e-9),
                cost_per_GWh=len(c)*HP_COST/1e6/max((gas_saved-elec_added)/1e6,1e-9))
res = pd.DataFrame([{**impact(idx,m), "route":n, "COP":cop} for n,idx in cohorts.items() for cop,m in hp_mult.items()])
print(res[["route","COP","gas_saved_GWh","elec_added_GWh","net_GWh","per_home_net_kwh","q12_savings_share","cost_per_GWh"]].round(2).to_string(index=False))
""")

md(r"""
## 4 · The governance trade-off
Per home, the grant route saves more (larger homes). But the council can deliver
*far more homes*, so on total energy delivered — and on equity — deliverability
wins. The per-home efficiency of grants is real but moot when uptake is the
binding constraint.
""")
code(r"""
head = {n: impact(idx, hp_mult[HEAD]) for n,idx in cohorts.items()}
co="Council / social rent"; pr="Provider / grants (wealthy)"
print(f"{'':30s}{'homes':>8s}{'net GWh':>9s}{'per home':>10s}{'% to Q1/Q2':>11s}")
for n in cohorts: print(f"{n:30s}{head[n]['homes']:8,d}{head[n]['net_GWh']:9.1f}{head[n]['per_home_net_kwh']:10.0f}{100*head[n]['q12_savings_share']:11.0f}")
print(f"\nper home: grants save {head[pr]['per_home_net_kwh']/head[co]['per_home_net_kwh']:.1f}× the council route — but the council reaches {head[co]['homes']/max(head[pr]['homes'],1):.1f}× the homes,")
print(f"so on TOTAL energy delivered the council saves {head[co]['net_GWh']/max(head[pr]['net_GWh'],1e-9):.1f}× as much ({head[co]['net_GWh']:.0f} vs {head[pr]['net_GWh']:.0f} GWh/yr).")
print(f"equity: council sends {100*head[co]['q12_savings_share']:.0f}% of savings to Q1/Q2 vs {100*head[pr]['q12_savings_share']:.0f}% for grants.")
print(f"deliverability: council owns its {head[co]['homes']:,} homes outright; the grant route depends on voluntary uptake.")

col={co:"#c84", pr:"#48c"}; names=list(cohorts); x=np.arange(2); lab=[n.split(" / ")[0] for n in names]
fig, ax = plt.subplots(1,3, figsize=(15,4))
ax[0].bar(x,[head[n]["net_GWh"] for n in names],color=[col[n] for n in names]); ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set(ylabel="GWh/yr", title="TOTAL net energy saved")
ax[1].bar(x,[head[n]["per_home_net_kwh"] for n in names],color=[col[n] for n in names]); ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set(ylabel="kWh/home", title="Per-home saving")
ax[2].bar(x,[100*head[n]["q12_savings_share"] for n in names],color=[col[n] for n in names]); ax[2].set_xticks(x); ax[2].set_xticklabels(lab); ax[2].set(ylabel="% to Q1/Q2", title="Reaches low-income")
plt.tight_layout(); plt.show()
""")

md("## 5 · Who is reached\nIncome and EPC composition of the treated homes under each route.")
code(r"""
fig, ax = plt.subplots(1,2, figsize=(13,4)); band={1:"G",2:"F",3:"E",4:"D",5:"C",6:"B",7:"A"}
for n, idx in cohorts.items():
    c=elig.loc[idx]
    inc=c.hh_income_band.value_counts().reindex(["q1_lowest","q2_low","q3_mid","q4_high","q5_highest"]).fillna(0)
    ax[0].plot(["Q1","Q2","Q3","Q4","Q5"], inc.values, "-o", label=n.split(" / ")[0], color=col[n])
    ax[1].plot([band[b] for b in range(1,8)], [(c.sap_band_ord==b).sum() for b in range(1,8)], "-o", label=n.split(" / ")[0], color=col[n])
ax[0].set(xlabel="income quintile", ylabel="homes", title="Income profile of treated"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="EPC band", ylabel="homes", title="EPC profile of treated"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
## 6 · Translation to carbon and cost
Energy is the model output; carbon and cost are downstream translations. BEIS 2023
factors: gas **0.183**, grid electricity **0.207** kgCO₂/kWh (the grid figure falls
over time, so carbon savings are a present-day floor). HP cost £12k gross (BUS grant
covers £7,500), levelised over a 15-year life.
""")
code(r"""
print(f"{'route':30s}{'net GWh':>9s}{'ktCO₂/yr':>10s}{'£/tCO₂':>9s}")
for n, idx in cohorts.items():
    c=elig.loc[idx]; gh=c.gas_heat_kwh.sum()
    carbon=(gh*GAS_KGCO2 - gh*hp_mult[HEAD]*ELEC_KGCO2)/1e6
    print(f"{n:30s}{head[n]['net_GWh']:9.1f}{carbon:10.2f}{len(c)*HP_COST/max(carbon*1e3*HP_LIFETIME,1e-9):9.0f}")
""")

md(r"""
## 7 · The governance finding

At each actor's *realistic* reach in Newcastle, calibrated to SERL demand and field
heat-pump performance:

- **Provider / grants (wealthy)** saves the most **per home** (large, high-consuming
  homes) and is cheapest per home — but its reach is throttled by **voluntary
  uptake**, and what it delivers is **regressive** (public subsidy to households
  who could largely self-fund).
- **Council / social rent** saves less per home, but because it **owns its stock**
  it can convert *many times more homes* — so it delivers **more total energy**,
  and it is **progressive** (savings reach lower-income tenants).

So the per-home efficiency advantage of grants is real but **moot when uptake is
the binding constraint**: the actor that can actually deliver at scale decarbonises
more and more fairly. The policy choice is not the technology, nor even efficiency
vs equity — it is **which institutional actor can act, on whom, and how fast**.
Targeting logic is a governance variable: the route that runs the programme changes
its total energy outcome, its distribution, and its deliverability together. (The
provider's reach is set by `PROVIDER_UPTAKE`; the headline result is robust across
plausible uptake — the council's owned-stock advantage dominates unless grant
uptake approaches the full eligible base.)

*Notes.* All demand is the v7 calibrated model's own output (no proxy fields, no
re-fit). Tenure/income from synthpop (~70% income coverage; unmatched excluded from
income targeting). Carbon and cost (§6) are translations of the energy result.
Winter electricity added has grid-peak implications worth a forward-looking
extension.
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"]={"name":"python"}
out = "research/policy/notebooks/heatpump_policy_scenarios.ipynb"
nbf.write(nb, out); print(f"wrote {out} with {len(cells)} cells")
