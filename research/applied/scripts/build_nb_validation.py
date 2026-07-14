"""Generate "research/applied/notebooks/3_validation.ipynb".
Reads the calibrated config (from notebook 1), runs the model, and validates per
cohort (vs SERL), per month (vs SERL), and per unit (vs DESNZ, held out).
"""
from __future__ import annotations
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

md(r"""
# 3 · Validation

Modelled demand vs independent data, in three views that isolate different failure
modes: **per-cohort split** vs SERL (the calibration target), **seasonal shape** vs
SERL, and **per unit** vs DESNZ (held out — never a calibration target). Consumes
the config from `1_calibration.ipynb`.
""")
code(r"""
import sys, os, subprocess
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt
REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)
SCRIPTS = REPO / "research" / "applied" / "scripts"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCRIPTS))
DATA = REPO / "data"; RES = REPO / "results_lsoa"
# FAST: 30-LSOA sample (cached). RUN_LIVE re-runs the model with the v7 config.
RUN_LIVE = False
TAG = "v7"; SAMPLE_N = 30
plt.rcParams.update({"figure.dpi":110,"axes.grid":True,"grid.alpha":0.3,"font.size":10})
DAYS = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

_d = pd.read_csv(DATA / "serl_8963_targets" / "daily_targets.csv")
def _mon(hf, qty):
    s=_d[(_d.period_type=="monthly")&(_d.year==2023)&(_d.weekday_weekend=="both")&
         (_d.heating_fuel==hf)&(_d.has_pv.isin(["No","All"]))&(_d.seg3_var=="none")&(_d.quantity==qty)]
    return pd.to_numeric(s["mean"],errors="coerce").groupby(s["month"]).mean().reindex(range(1,13))
serl_gg=_mon("Gas","Gas"); serl_ge=_mon("Gas","Electricity imports"); serl_ee=_mon("Electric","Electricity imports")
_floor=serl_ee[[6,7,8]].mean()
SERL={"gas_heated_elec":float(sum(serl_ge[m]*DAYS[m] for m in range(1,13))),
      "elec_heated_baseline":float(_floor*365),
      "elec_heated_heating":float(sum((serl_ee-_floor).clip(lower=0)[m]*DAYS[m] for m in range(1,13)))}
SERL["elec_heated_total"]=SERL["elec_heated_baseline"]+SERL["elec_heated_heating"]

CONFIG = REPO / "results" / "calibration_v7_cohort" / "calibrated_config.yaml"   # produced by notebook 1
DEC = RES / f"decomp_sample_newcastle_2023_{TAG}.csv"; MON = RES / f"monthly_cohort_newcastle_2023_{TAG}.csv"
if RUN_LIVE or not DEC.exists():
    subprocess.run([sys.executable, str(SCRIPTS/"run_monthly_retune.py"), TAG, str(SAMPLE_N), str(CONFIG)],
                   cwd=str(REPO), check=True, env={**os.environ, "PYTHONPATH": f"{REPO}:{SCRIPTS}"})
dec = pd.read_csv(DEC); mon = pd.read_csv(MON)
print(f"validation sample: {len(dec)} dwellings | electric-heated {(dec.heating_bucket=='electric').mean():.1%}")
""")

md("## 3.1 Per-cohort split vs SERL\nElectricity split into non-heating and heating, by heating cohort. A correct calibration reproduces both the split and the total.")
code(r"""
e = dec[dec.heating_bucket=="electric"]; g = dec[dec.heating_bucket=="gas"]
rows = [["electric-heated","non-heating",e.base_kwh.mean()+e.spike_kwh.mean(),SERL["elec_heated_baseline"]],
        ["electric-heated","heating",e.heat_kwh.mean(),SERL["elec_heated_heating"]],
        ["electric-heated","TOTAL",e.electric_kwh.mean(),SERL["elec_heated_total"]],
        ["gas-heated","electricity",g.electric_kwh.mean(),SERL["gas_heated_elec"]]]
t = pd.DataFrame(rows, columns=["cohort","component","model","SERL"]); t["ratio"]=(t.model/t.SERL).round(3)
t.model=t.model.round(0); t.SERL=t.SERL.round(0); print(t.to_string(index=False))
fig, ax = plt.subplots(figsize=(7,3.8)); x=np.arange(3); w=0.35
mod=[e.base_kwh.mean()+e.spike_kwh.mean(), e.heat_kwh.mean(), g.electric_kwh.mean()]
ser=[SERL["elec_heated_baseline"], SERL["elec_heated_heating"], SERL["gas_heated_elec"]]
ax.bar(x-w/2,mod,w,label="model"); ax.bar(x+w/2,ser,w,label="SERL")
ax.set_xticks(x); ax.set_xticklabels(["elec-heated\nnon-heat","elec-heated\nheating","gas-heated\nelec"])
ax.set(ylabel="kWh/yr/dwelling", title="Per-cohort electricity: model vs SERL"); ax.legend(); plt.show()
""")

md("## 3.2 Seasonal shape vs SERL")
code(r"""
ref = pd.DataFrame({"gas_gas":serl_gg*pd.Series(DAYS),"gas_elec":serl_ge*pd.Series(DAYS),"electric_elec":serl_ee*pd.Series(DAYS)})
pe = mon.pivot(index="month",columns="cohort",values="elec_kwh_per_dw"); pg = mon.pivot(index="month",columns="cohort",values="gas_kwh_per_dw")
M=["J","F","M","A","M","J","J","A","S","O","N","D"]
series=[("Gas-heated: GAS",pg["gas"],ref["gas_gas"]),("Gas-heated: elec",pe["gas"],ref["gas_elec"]),("Electric-heated: elec",pe["electric"],ref["electric_elec"])]
fig, axes = plt.subplots(1,3, figsize=(15,3.8))
for ax,(n,mo,se) in zip(axes,series):
    ax.plot(range(1,13),[mo[m] for m in range(1,13)],"-o",ms=4,label="model"); ax.plot(range(1,13),[se[m] for m in range(1,13)],"--s",ms=4,label="SERL")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(M); ax.set_title(n); ax.set_ylabel("kWh/month/dw"); ax.legend()
plt.tight_layout(); plt.show()
for n,mo,se in series: print(f"{n:24s} annual model {mo.sum():6.0f} SERL {se.sum():6.0f} ratio {mo.sum()/se.sum():.2f}")
""")

md("## 3.3 Per unit vs DESNZ (held-out check)\nDESNZ is never a calibration target. The residual above 1.0 on electricity is the SERL-panel-vs-population difference, left honest.")
code(r"""
from utils import load_desnz
from run_decomp_sample import pick_sample
sample = pick_sample()[:SAMPLE_N]
des = load_desnz("newcastle", 2023); des["lsoa_code"]=des["lsoa_code"].astype(str); des=des[des.lsoa_code.isin(sample)]
elec_pm = des.total_kwh_elec.sum()/des.meters_elec.sum(); gas_pm = des.total_kwh_gas.sum()/des.meters_gas.sum()
m_e = dec.electric_kwh.mean(); m_g = dec[dec.gas_kwh>0].gas_kwh.mean()
print(f"electricity: model {m_e:.0f}/dwelling vs DESNZ {elec_pm:.0f}/meter → {m_e/elec_pm:.3f}")
print(f"gas        : model {m_g:.0f}/gas-dwelling vs DESNZ {gas_pm:.0f}/meter → {m_g/gas_pm:.3f}")
""")

md("## 3.4 Spatial / temporal transfer")
code(r"""
import glob
rows=[]
for f in sorted(glob.glob(str(REPO/"research"/"applied"/"transfer_confidence_*.csv"))):
    t=pd.read_csv(f); rows.append({"city_year":Path(f).stem.replace("transfer_confidence_",""),"n_lsoa":len(t),
        "median totals ratio":round(t.tot_ratio.median(),3) if "tot_ratio" in t else None,
        "High %":round((t.confidence=="High").mean()*100) if "confidence" in t else None})
print(pd.DataFrame(rows).to_string(index=False))
print("\n(transfer_confidence_*.csv are pre-v7; regenerate with transfer.py --config the v7 config for the final 5-city numbers.)")
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"]={"name":"python"}
out = "research/applied/notebooks/3_validation.ipynb"
nbf.write(nb, out); print(f"wrote {out} with {len(cells)} cells")
