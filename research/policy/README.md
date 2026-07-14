# research/policy — heat-pump calibration + policy experiments

Paper 2 (*Policy Studies Journal*, Research Note): **"Targeting logic as a
governance variable."** Uses the calibrated model to ask not *whether* to roll out
heat pumps but *who* runs the programme and *on whom* — an institutional-design
argument, quantified in energy terms.

## Notebooks (`notebooks/`)

| notebook | does |
|---|---|
| `heatpump_cop_calibration.ipynb` | the heat-pump performance (COP) calibration the scenarios depend on — anchored to **heatpumpmonitor.org** field data: conservative DESNZ median **2.8** (headline), well-installed field median **4.21** (sensitivity) |
| `heatpump_policy_scenarios.ipynb` | two institutional routes at their *realistic* reach: **Council / social rent** (owns its stock → delivers at scale) vs **Provider / grants (wealthy)** (voluntary uptake) |

## The result
Per home, grants save more (large wealthy homes) and are cheaper per GWh — but the
council can convert far more homes (it owns them), so it delivers **more total
energy and more equity**: at default settings **49 GWh / 32% of savings to the
bottom-two income quintiles** (council) vs **27 GWh / 0%** (grants). Deliverability
dominates: per-home efficiency is moot when grant uptake is the binding constraint.

## Inputs (all calibrated, no proxy fields)
Per-home consumption + space heating come from the **v7 SERL-calibrated model**
(full-city decomposition `results_lsoa/decomp_city_newcastle_2023_v7.csv`; falls
back to the 30-LSOA sample if absent). Tenure/income from the synthpop match
(`data/hidp_uprn_matches_tiered.csv`). HP conversion is the model's own
`hp_effect_mult = boiler_efficiency / COP`. **Energy (GWh) is the headline; carbon
and cost are downstream translations.** `energy_cal_kwh` is deliberately *not* used.

## Running
Open and **Run All** — it's analytical (no live model run), so it's fast. Levers in
cell 1: `COUNCIL_HOMES` (default = whole eligible social stock), `PROVIDER_UPTAKE`
(default 0.20), `COP_HEAD`/`COP_SENS`, costs, carbon factors. To refresh the
full-city demand, run `research/applied/scripts/run_city_decomp.py` (resumable).

## Scripts (`scripts/`)
`fit_heatpump_cop.py` (pulls the OpenEnergyMonitor API → `results/field_fits/heatpump_cop.yaml`),
`build_nb_heatpump_cop.py`, `build_nb_policy.py` (regenerate the notebooks).
