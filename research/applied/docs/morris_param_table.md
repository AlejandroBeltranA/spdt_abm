# Morris Elementary Effects — Parameter Table

**Target paper:** Applied Energy methodology paper (Paper 1).
**Outcome of interest:** LSOA-level annualised household electricity + gas
demand prediction error (MAPE) — Newcastle calibrated, transferred to
Sunderland / Waltham Forest / [Manchester / Cornwall].

This table is the input space for the Morris elementary-effects sweep
(task #10) and the prior distribution for the Monte Carlo propagation
(task #11) for any parameter not covered by the SERL bootstrap (#4).

13 parameters, spanning the full baseline-demand pathway. Heat-pump and
behavioural-intervention parameters are excluded — they belong to Paper 2.

---

## Parameter table

| Parameter | Description | Calibrated / nominal value | Morris range (low, high) | Justification & source |
|---|---|---|---|---|
| `heating_trigger_temp_C` | Outdoor ambient °C below which space heating engages (HDD base temperature). **Not** an indoor comfort thermostat — verified at `agent.py:712` where the value is compared against `ambient_tempC`. | 13.0 °C (calibrated 2026-05) | (11.0, 17.0) | CIBSE convention for UK domestic gas heating uses 15.5 °C base temperature for HDD; UK building-physics literature spans 14–16 °C depending on dwelling efficiency (Lowe 2007; Hamilton 2013, implicit in stock SAP). Calibrated 13.0 sits at the efficient-stock end. Range ±2 °C around CIBSE bracket covers the published distribution. |
| `heating_slope_kWh_per_deg` | Marginal heating energy per °C of HDD per household per hour | 0.238 (gas), 0.067 (elec) — calibrated 2020–2022 | (0.15, 0.30) gas; (0.04, 0.10) elec | Satre-Meloy & Hampton (2024) report +8.2% gas / +3.4% electricity per +1 °C setpoint at UK mean gas of 35 kWh/day → ~0.20–0.29 kWh/HDD; SERL bootstrap CIs from task #4 to refine. |
| `baseline_anchor_gas_kwh_per_hour` | Non-heating gas baseload (hot water + cooking) | 0.368 (calibrated 2020–2022) | (0.20, 0.45) | SERL summer Jun–Aug mean for gas-heated homes. Hamilton (2013, Fig. 1): hot water + cooking ≈ 26% of UK residential delivered energy. SERL off-season range 5–10 kWh/day → 0.20–0.42 kWh/h. SERL bootstrap CIs from task #4 to refine. |
| `baseline_anchor_elec_kwh_per_hour` | Non-heating electricity baseload | 0.335 (calibrated 2020–2022) | (0.20, 0.45) | SERL daily electricity mean 11.13 kWh, SD 8.30 (Satre-Meloy & Hampton 2024); 5th–95th percentile of daily mean across stock ≈ 4.7–10.7 kWh ⇒ hourly 0.20–0.45. |
| `epc_band_coef` | Multiplicative factor per SAP-band step (A=1 best → G=worst) | 1.15 per band | (1.08, 1.25) | HEED median gas: pre-1900 = 18.95 vs post-1990 = 16.23 MWh/yr (Hamilton 2013, Table 10); cavity-walled vs solid-walled differences imply 10–20% gradient per efficiency category. SAP scale logarithmic 1–100 (BRE/DECC 2009). |
| `dwelling_age_band_coef` | Multiplicative factor per dwelling-age band step (pre-1900 → 1900–1929 → … → post-1990) | 1.04 per band | (1.00, 1.10) | Hamilton (2013, Tables 10–12): median annual gas across UK age bands varies pre-1900 = 18.95 → post-1990 = 16.23 MWh, ~14% spread across 5 bands ⇒ ~4% per band. EPC band partially captures this but not fully — older dwellings with same SAP still draw more per heating-degree. |
| `floor_area_coef_kWh_per_m2` | Slope of annual gas demand on dwelling floor area | 110 kWh/m²/yr | (80, 160) | Stock-average ~110 kWh/m² for gas-heated UK dwellings (Lowe 2007, SAP 2005 calculations for 80 m² semi); Hamilton (2013, Fig. 9–10) shows detached ≈ 2× flat per dwelling at ~2× floor area, consistent with ±35% spread. |
| `occupants_coef` | Multiplicative effect per additional occupant on total demand | 1.18 | (1.10, 1.28) | Satre-Meloy & Hampton (2024, Table 5): +23.5% electricity, +1.37 kWh/day gas per occupant (~+4%); composite ≈ +15–22%; 95% CI 1.18–1.24 electricity. |
| `bedroom_coef` | Multiplicative effect per additional bedroom (dwelling-size proxy) | 1.17 | (1.06, 1.22) | Satre-Meloy & Hampton (2024): +17.5% electricity, +6.56 kWh/day gas (~+19%) per bedroom; Hamilton (2013) reports +22% gas per bedroom — bounds cover both estimates. |
| `schedule_weekday_weight` | Fraction of occupied hours that follow a workday (vs retired/shift) schedule | 0.55 | (0.40, 0.75) | "Adjust heating for home working" coefficient = +1.6% elec, +98 W gas (Satre-Meloy & Hampton 2024 Table 5–6); ONS labour force participation ~75% × WFH fraction 25–40% (post-2021). Literature thin → use ±0.15 around nominal. |
| `weekend_uplift` | Demand multiplier applied on weekends/non-working days | 1.10 | (1.00, 1.25) | Few directly comparable UK estimates; SERL half-hourly studies (Webborn et al. 2021 dataset descriptor) suggest 5–25% weekend uplift in residential gas. **Literature gap** — recommend ±20% default range. |
| `hdd_base_temp_C` | Base temperature for HDD calculation in climate response | 15.5 °C | (13.0, 17.0) | CIBSE convention 15.5 °C for UK gas heating; building-physics literature also uses 14–16 °C depending on dwelling efficiency (Lowe 2007; Hamilton 2013 implicit in stock SAP). Range = ±1 SD across published UK HDD analyses. Distinct from `heating_trigger_temp_C` above: that's *when* heating switches on; this is *what reference temp* the HDD signal is constructed against. |
| `gas_share_of_heating` | Fraction of heating delivered by gas (vs electricity) at dwelling level | 0.78 | (0.65, 0.90) | BEIS 2022: 78% of UK dwellings use gas (Satre-Meloy & Hampton 2024, p. 7); regional variation in Newcastle/Sunderland gas grid coverage gives ±10 pp; SERL gas-heated share = 74%. |

13 parameters; ~600 words of justification.

---

## Ranges rationale

### How bounds were chosen

Where the literature provides directly comparable empirical distributions, we
use the 5th–95th percentile of the reported distribution (`heating_trigger_temp_C`,
`baseline_anchor_elec_kwh_per_hour`) or the published 95% confidence interval
on a regression coefficient (`occupants_coef`, `bedroom_coef`, `heating_slope_kWh_per_deg`
— all anchored on Satre-Meloy & Hampton 2024 Tables 5–6).

For dwelling-physics parameters (`floor_area_coef_kWh_per_m2`, `epc_band_coef`,
`dwelling_age_band_coef`), bounds reflect the spread of stock-segment means
in Hamilton et al. (2013, Tables 10–12).

For `hdd_base_temp_C` and `gas_share_of_heating` we use the convention ±1 SD
across reported UK values.

Two parameters — `schedule_weekday_weight` and `weekend_uplift` — sit in
genuine literature gaps; for these we apply the default ±20% / ±0.15 absolute
rule and flag this explicitly so reviewers can see the weaker provenance.

Morris factor levels (p=4 or p=6) will be sampled uniformly across each
[low, high] interval. The 4 calibrated parameters (`heating_slope_kWh_per_deg`,
both `baseline_anchor_*`) will additionally have their literature priors
overridden by the bootstrap parameter distribution (task #4) for the MC
propagation step (#11).

### Parameter naming sanity check

`heating_trigger_temp_C` was renamed from `heating_setpoint_C` on 2026-05-21
after an external agent flagged the calibrated value of 13 °C as below the
entire UK indoor-comfort distribution. Inspection of `agent.py:712` confirmed
the parameter is compared against *outdoor* ambient temperature, making it a
heating-engage threshold (≈ HDD base temperature), not an indoor setpoint.
The new name is accurate; the calibrated value of 13 °C is defensible as the
efficient-stock end of the literature distribution. The old config key
remains accepted with a `DeprecationWarning`.

### Literature gaps to disclose in the paper

The following parameters lack quantified UK estimates and rely on default
ranges; Morris will tell us whether they matter. If any shows a high
elementary effect, dedicated empirical work (e.g. SERL half-hourly subsample)
will be needed before final claims:

- `weekend_uplift`
- `schedule_weekday_weight`
- `dwelling_age_band_coef` — partially covered by Hamilton 2013 but not
  with the precision of a published CI

### Why setpoint and heating-trigger are kept separate

`heating_trigger_temp_C` (when heating engages) and `hdd_base_temp_C` (the
reference temp for the heating-degree signal) are conceptually distinct and
the Morris design treats them as independent factors. In practice they are
correlated — moving one tends to require adjusting the other to fit the same
calibration data. The OAT sensitivity in the original sensitivity_analysis.ipynb
noticed this redundancy; Morris will quantify it via the σ statistic
(parameter–parameter interactions inflate σ relative to μ*).

---

## References

- BRE/DECC (2009). *The Government's Standard Assessment Procedure for Energy
  Rating of Dwellings (SAP 2009).* Building Research Establishment.
- CIBSE (2006). *Guide A: Environmental design.* Chartered Institution of
  Building Services Engineers.
- Hamilton, I. G. et al. (2013). The significance of household
  characteristics in space heating energy demand. *Building and Environment.*
- Lowe, R. (2007). Technical options and strategies for decarbonizing UK
  housing. *Building Research and Information.*
- Satre-Meloy, A. & Hampton, H. (2024). Drivers of residential electricity
  and gas consumption: evidence from UK smart-meter data (SERL).
  *Energy and Buildings* (working title; reference to be finalised against
  the version archived in the literature folder).
- Webborn, E. et al. (2021). The SERL Observatory dataset: longitudinal
  smart meter electricity and gas data, survey, EPC and climate data for
  over 13,000 households in Great Britain. *Energies.*
