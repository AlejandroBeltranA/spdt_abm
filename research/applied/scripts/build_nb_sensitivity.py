"""Generate "research/applied/notebooks/2_sensitivity.ipynb".
Reads the calibrated config (from notebook 1) and reports the Morris sensitivity
screen + cross-city robustness.
"""
from __future__ import annotations
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

md(r"""
# 2 · Sensitivity analysis

How much does each calibrated parameter move annual demand, and is that stable
across cities? A Morris elementary-effects screen
([`sa_morris.py`](../../../research/applied/scripts/sa_morris.py)) perturbs each
parameter within its published standard error. Consumes the config produced by
`1_calibration.ipynb`; produces the influence ranking.
""")
code(r"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
REPO = Path.cwd()
while not (REPO / "household_energy").exists() and REPO != REPO.parent:
    REPO = REPO.parent
os.chdir(REPO)
SA_DIR = REPO / "results" / "sensitivity_analysis"
# Re-run the Morris screen live (slow) vs read the saved results.
RUN_SA = False
plt.rcParams.update({"figure.dpi":110,"axes.grid":True,"grid.alpha":0.3,"font.size":10})
if RUN_SA:
    import subprocess
    subprocess.run([sys.executable, str(REPO/"research/applied/scripts/sa_morris.py"),
                    "--r","4","--label","newcastle"], cwd=str(REPO), check=True)
print("reading", SA_DIR.relative_to(REPO))
""")

md("## 2.1 Influence ranking (Newcastle 2023)")
code(r"""
sa = pd.read_csv(SA_DIR / "sa_morris_newcastle.csv").sort_values("mu_star_energy")
ptab = pd.read_csv(SA_DIR / "sa_param_table.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.barh(sa.knob, sa.mu_star_energy_pct_base, color="#48c")
ax.set(xlabel="μ* on annual energy (% of baseline)", title="Morris influence ranking"); plt.tight_layout(); plt.show()
print(f"largest single-parameter effect: {sa.mu_star_energy_pct_base.max():.1f}% per ±2·SE step\n")
print(ptab[["knob","central","uncertainty","source"]].head(10).to_string(index=False))
""")

md("## 2.2 Cross-city robustness\nThe ranking should hold across cities with different stock mixes.")
code(r"""
p5 = SA_DIR / "sa_morris_5city.csv"
if p5.exists():
    sa5 = pd.read_csv(p5)
    print(sa5[[c for c in sa5.columns if "mu_star" in c or c in ("knob","city")]].head(20).to_string(index=False))
else:
    print("sa_morris_5city.csv not found — run the 5-city screen to populate.")
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"]={"name":"python"}
out = "research/applied/notebooks/2_sensitivity.ipynb"
nbf.write(nb, out); print(f"wrote {out} with {len(cells)} cells")
