"""Generate research/policy/notebooks/heatpump_cop_calibration.ipynb.

The heat-pump performance (COP) calibration that the policy scenarios depend on:
COP anchored to heatpumpmonitor.org field data (results/field_fits/heatpump_cop.yaml,
produced by research/policy/scripts/fit_heatpump_cop.py). Shows the dual anchors
(conservative DESNZ vs well-installed field), the Carnot temperature curve, the
self-selection caveat, and how it maps into the model (hp_effect_mult).

  .venv/bin/python research/policy/scripts/build_nb_heatpump_cop.py
"""
from __future__ import annotations
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

md(r"""
# Heat-pump performance (COP) calibration

Heat-pump efficiency was the last unsourced assumption on the heating path — SERL
cannot identify it, so it is anchored to **external field data** from
**heatpumpmonitor.org** (OpenEnergyMonitor's public monitoring of real UK
installs), via [`fit_heatpump_cop.py`](../scripts/fit_heatpump_cop.py) →
[`heatpump_cop.yaml`](../../../results/field_fits/heatpump_cop.yaml).

This COP is what the policy scenarios convert gas heat through: a converted home's
space heat moves to electricity at `boiler_efficiency / COP`. The calibration
supplies **two defensible anchors** — a conservative headline and a well-installed
sensitivity.
""")

code(r"""
import os
from pathlib import Path
import numpy as np, yaml
import matplotlib.pyplot as plt
REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)
plt.rcParams.update({"figure.dpi":120,"axes.grid":True,"grid.alpha":0.3,"font.size":10})
cop = yaml.safe_load((REPO/"results"/"field_fits"/"heatpump_cop.yaml").read_text())
print("anchors (space-heating COP):")
print(f"  conservative (DESNZ Electrification of Heat median): {cop['cop_representative']}")
print(f"  well-installed (field median, space):                {cop['cop_field_median_space']}  (n={cop['diagnostics']['space_cop']['n']})")
print(f"  combined heat+DHW (field median):                    {cop['cop_field_median_combined']}")
print(f"\nmaps to model: hp_effect_mult = boiler_efficiency / COP")
for label, c in [("conservative 2.8", cop['cop_representative']), ("well-installed 4.21", cop['cop_field_median_space'])]:
    print(f"  {label:18s}: 0.90 / {c} = {0.90/c:.3f}  (a converted home uses {0.90/c:.0%} of its gas heat as electricity)")
""")

md("## Field COP distribution and the Carnot temperature curve")
code(r"""
d = cop["diagnostics"]["space_cop"]
fig, ax = plt.subplots(1,2, figsize=(13,4))
# distribution percentiles
pcts = ["p10","p25","median","p75","p90"]; vals=[d[p] for p in pcts]
ax[0].plot(pcts, vals, "-o", color="#48c")
ax[0].axhline(cop["cop_representative"], color="#c44", ls="--", label=f"conservative anchor {cop['cop_representative']}")
ax[0].set(ylabel="space COP", title=f"Field space-COP distribution (n={d['n']})"); ax[0].legend()
# Carnot temperature curve
cc = np.array(cop["cop_curve"])
ax[1].plot(cc[:,0], cc[:,1], "-o", color="#4a4")
ax[1].set(xlabel="outdoor temperature (°C)", ylabel="COP", title="Carnot-fit COP vs outdoor temperature")
plt.tight_layout(); plt.show()
print("by HP type (space COP):", {k: round(v["median"],2) for k,v in cop["diagnostics"].get("by_hp_type_space_cop",{}).items()})
""")

md(r"""
## Why a conservative default
The field sample is **self-selected, well-commissioned owner-enthusiast systems**,
biased *above* the representative installed base. So the model **defaults to the
conservative DESNZ Electrification-of-Heat median (COP 2.8)** as the headline, and
uses the field median (4.21) only as a *well-installed* sensitivity. This keeps the
policy results from over-stating heat-pump savings — a reviewer-proof choice.

| anchor | COP | source | use |
|---|---|---|---|
| conservative | 2.8 | DESNZ EoH trial median (ASHP SPFH4) | **headline** |
| well-installed | 4.21 | heatpumpmonitor.org field median (space) | sensitivity |

Consumed by [`heatpump_policy_scenarios.ipynb`](heatpump_policy_scenarios.ipynb).
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"]={"name":"python"}
out = "research/policy/notebooks/heatpump_cop_calibration.ipynb"
nbf.write(nb, out); print(f"wrote {out} with {len(cells)} cells")
