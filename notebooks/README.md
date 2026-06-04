# Notebook Index

This directory contains the full analysis pipeline for the paper **"Too Hot to Handle? Modelling Household Energy Demand and Climate Policy Responses"**. The notebooks are designed to be run in order, though each can be opened independently given the prerequisite data files.

---

## Paper section map

| Notebook | Paper section | What it produces |
|----------|---------------|-----------------|
| `01-climate-prep.ipynb` | Data & Methods — Climate | Hourly ERA5 temperature parquet for the model |
| `household_energy_abm_tutorial.ipynb` | Data & Methods — Model Architecture | End-to-end walkthrough of the MABM; establishes all key concepts |
| `serl_calibration_clean.ipynb` | Data & Methods — Calibration | Analytically derived model parameters from SERL Smart Meter data; produces `calibrated_config.yaml`. Emits the v4 fuel-specific electric base heating slope (`heating_slope_kWh_per_deg_electric`) and neutralizes the legacy electric `heating_slope_mult` |
| `energy-model-validation.ipynb` | Results — Validation | ABM vs DESNZ per-LSOA comparison (2020–2023); the key quantitative validation. Includes the coverage-aware confidence layer that tiers LSOAs (High/Medium/Low) by EPC-to-meter coverage rather than treating DESNZ as a calibration target |
| `policy_scenarios_detailed.ipynb` | Results — Policy Experiments | Step-by-step walkthrough of all five policy scenarios (A–E) with interpretation |
| `policy_scenarios_summary.ipynb` | Results — Figures | Parallel batch run of all scenarios; produces summary tables and the comparison map |
| `full_run_example.ipynb` | Results — Future Projections | 2020–2039 long-horizon run; baseline for climate trajectory analysis |
| `abm_end_to_end_tutorial.ipynb` | Supplementary / Funder demo | Full pipeline with embedded outputs; **do not strip outputs** |

---

## Recommended reading order

If you are new to the project, work through notebooks in this sequence:

1. **`household_energy_abm_tutorial.ipynb`** — start here. Covers the full MABM design: how dwellings are represented, how climate drives heating/cooling, how residents occupy space, and how outputs are collected. All key concepts used in later notebooks are introduced here.

2. **`01-climate-prep.ipynb`** — understand the ERA5 climate input. Shows how outdoor temperature maps to individual dwellings via nearest-neighbour grid assignment.

3. **`serl_calibration_clean.ipynb`** — understand how model parameters were derived from the SERL Smart Meter dataset. The output `calibrated_config.yaml` is what makes the model's energy numbers meaningful rather than illustrative. As of the v4 calibration (2026-06-04) electric-heated dwellings use a fuel-specific base heating slope instead of inheriting the (3.7× steeper) gas slope, which fixes the electricity overshoot in all-electric LSOAs.

4. **`energy-model-validation.ipynb`** — the independent validation. Compares model output against DESNZ subnational electricity and gas statistics for 2020–2023, per LSOA. DESNZ is treated as an independent benchmark (with known meter-vs-EPC coverage contamination), not a calibration target: the coverage-aware confidence layer reports *which* LSOA outputs to trust (138 High / 43 Medium / 4 Low) and exports `research/applied/lsoa_confidence_2023.html`.

5. **`policy_scenarios_detailed.ipynb`** — the core policy experiments. Works through each scenario (income-targeted heat pump grants, multi-year HP vs baseline, top-user targeting, kids vs elderly, social rent) with detailed commentary.

6. **`policy_scenarios_summary.ipynb`** — runs all scenarios in parallel and produces the side-by-side comparison map. This is the source of the summary figures.

7. **`full_run_example.ipynb`** — extends the run window to 2039, demonstrating the future-climate projection capability.

---

## Data prerequisites

All input files are git-ignored and must be present in `data/` before running any notebook. See `data/README.md` for the full list and sources.

**Minimum required for most notebooks:**
- `data/epc_abm_newcastle.geojson` — EPC-matched household polygons for Newcastle
- `data/ncc_2t_timeseries_2010_2039.parquet` — ERA5 hourly temperatures (2010–2039); used by all notebooks for simulation
- `data/ncc_2t_timeseries_2010_2026.parquet` — ERA5 hourly temperatures (2010–2026); used for DESNZ historical validation

**For socio-demographic policy scenarios:**
- `data/hidp_uprn_matches_tiered.csv` — HIDP enrichment table (income, education, tenure, household composition, schedule type)

**For DESNZ validation:**
- `data/LSOA_domestic_elec_2010-2024.xlsx`
- `data/LSOA_domestic_gas_2010-2024.xlsx`

**For SERL calibration:**
- `data/UKDA-8963-csv/` — SERL aggregated statistics (restricted access; see serl.ac.uk)
- `data/serl_8963_targets/` — pre-processed calibration targets (output of `scripts/build_serl_8963_targets.py`)

---

## Two result directories required by the funder demo

`abm_end_to_end_tutorial.ipynb` references specific pre-computed outputs. These are not git-tracked but must be present locally for the notebook to render fully:

- `results/calibration_20260316_161904/` — DESNZ city comparison and SERL validation outputs
- `notebooks/results/scenario_v2_1/` — scenario summary and comparison map

Both are reproducible from `energy-run-lsoa` and `policy_scenarios_summary.ipynb` respectively.

---

## Five policy scenarios

All scenario notebooks implement the same five experiment patterns. These map directly to the paper's policy analysis section:

| Scenario | Targeting logic | Policy action |
|----------|----------------|---------------|
| A — Targeted HP grants | Low income (`q1`/`q2`) + low education | Heat pump candidate (priority class, 100% adoption) |
| B — Multi-year HP vs baseline | Same mask as A | Annual comparison over 5 years |
| C — Top-user HPs | Top X% by baseline annual kWh | Heat pump candidate (priority) |
| D — Kids vs Elderly | Children flag / retired schedule | Kids: retrofit (loft + glazing); Elderly: smart meter + heating setback |
| E — Social rent HPs | `tenure == social_rent` | Heat pump candidate (priority, 100% adoption) |

Scenarios require HIDP enrichment for the cohort masks (income, education, tenure, children, schedule type). The `USE_ENRICHMENT` toggle in each notebook controls whether to load the HIDP CSV or fall back to EPC-only attributes.

---

## Notebook hygiene

- Outputs are stripped from all notebooks except `abm_end_to_end_tutorial.ipynb` (funder demo, kept with outputs)
- The `nbstripout` git hook prevents accidental output commits: `nbstripout --install`
- Run the smoke test before committing: `python scripts/notebook_smoke.py notebooks/full_run_example.ipynb`
