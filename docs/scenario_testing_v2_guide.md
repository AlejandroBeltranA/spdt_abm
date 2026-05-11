# Scenario Testing v2 – Implementation Guide (dashboard-ready)

This guide distills the `notebooks/policy_scenarios_detailed.ipynb` and `notebooks/policy_scenarios_summary.ipynb` workflows into the pieces you need to wire up a manual dashboard with simple controls/toggles. It covers expected inputs, key variables and categories, and how each scenario manipulates the GeoDataFrame before running the model.

---

## 1) Required inputs / base dataset
- **Household GeoJSON**: one row per dwelling with at least `UPRN`, `property_type`, `floor_area_m2`, `sap_rating`, `lsoa_code`/`ward_code`.
- **Hourly climate parquet**: aligns to the ABM (same grid/timestamps).
- **Optional HIDP enrichment CSV** (`hidp_uprn_matches_tiered.csv`): adds socio‑demographics:
  - `hh_n_people`, `hh_children` (bool), `hh_income_band`, `hh_edu_detail`,
  - `tenure`, `dwelling_bucket`, `size_band` (bedrooms 1–4),
  - `schedule_type`, `schedule_profile` (derived).

### Common filters (map to UI controls)
- **Area filter**: `AREA_COLUMN` (e.g., `ward_code` or `lsoa_code`) + `AREA_CODE` (string).
- **Time window**: hours or explicit `start_utc`/`end_utc`.
- **Agent collection**: enable/disable agent-level DataCollector (perf knob).

---

## 2) Core fields and categories for UI controls
- **Income band** (`hh_income_band`): `q1_lowest`, `q2_low`, `q3_mid`, `q4_high`, `q5_highest`.
- **Education** (`hh_edu_detail`): e.g., `education_other`, `upper_further`, `postgrad_or_degree`, etc.
- **Tenure** (`tenure`): `owner_occupied`, `private_rent`, `social_rent`, (others pass through).
- **Children flag** (`hh_children`): boolean.
- **Schedule type** (`schedule_type`): `retired_household`, `working_adult_household`, `dual_earner_household`, `family_with_children`, `single_parent_with_children`, `student_household`, `unemployed_or_inactive`, `retired_household`.
- **Dwelling bucket** (`dwelling_bucket`): project-specific typology labels.
- **Size band** (`size_band`): bedrooms 1–4 (int).
- **Heat‑pump flags**:
  - `is_heatpump_candidate` (0/1)
  - `heatpump_candidate_class`: `priority`, `possible`, `difficult`, `non-possible`
- **Energy outputs of interest**:
  - `total_energy` (model-level hourly)
  - `energy_consumption` (agent-level hourly)
  - Aggregates: daily/annual kWh per dwelling, by cohort masks.

---

## 3) Scenario patterns (how to implement in a dashboard)

Each scenario follows the same steps:
1. Start from a **baseline copy** of the enriched GeoDataFrame (`gdf`).
2. Build a **boolean mask** for the target cohort using the selected controls.
3. Set policy flags on a **policy copy** of `gdf` (usually heat‑pump eligibility/class).
4. Optionally set a **config override** (YAML) to adjust adoption rates, years, or caps.
5. Run baseline vs policy model for the chosen window; compare kWh totals/means.

### A) Targeted heat‑pump grants (income + education)
- Mask: `hh_income_band` ∈ {q1_lowest, q2_low} AND `hh_edu_detail` ∈ {education_other, upper_further}.
- Action: `is_heatpump_candidate = 1`; `heatpump_candidate_class = "priority"`; config override `heatpump_adoption_rate = 1.0`.
- Window: 24h (fast smoke) or multi‑year (ward-only for speed).

### B) Multi-year HP vs baseline (ward-level)
- Reuse the mask above (or any mask from the controls).
- Run 2 models over N years (default 5): baseline vs policy; aggregate annual kWh by cohort (with/without HP).
- Plot annual GWh lines for both groups.

### C) Top‑user heat‑pump targeting
- Baseline 1-year run (agent-level) to rank households by `energy_consumption` sum.
- Mask: top X% (`TOP_FRAC`) of households by annual kWh.
- Action: mark those as HP priority; run shorter multi‑year comparison (e.g., 2 years).

### D) Kids vs Elderly policies
- Kids mask: `hh_children == True` OR `schedule_type` in {family_with_children, single_parent_with_children}.
- Elderly mask: `schedule_type == retired_household`.
- Actions (example):
  - Kids: set retrofit flags (`loft_ins_flag=1`, `glazing_flag=1`).
  - Elderly: apply smart meters + heating setback (config override `heating_setpoint_C_delta = -1`).
- Compare ΔkWh over a one‑week window.

### E) Social rent heat‑pump scenario
- Mask: `tenure == social_rent`.
- Action: mark as HP candidates (`is_heatpump_candidate=1`, `heatpump_candidate_class="priority"`), set adoption to 1.0.
- Window: 24h.

---

## 4) Baseline & heating (what the model now assumes)
- Baseline is a **fixed meter anchor**: 0.40 kWh/h × mild area exponent (0.20, clipped 0.85–1.25) × property-type baseline multiplier (detached 1.20, semi 1.10, end 1.05, mid 1.00, flats 0.85–0.90), capped at 1.5 kWh/h.
- SAP/envelope/fuel do **not** scale the baseline; they affect **heating only**.
- Heating uses a duty-cycle model with sublinear area scaling and `pt_heat_mult`; hard total cap remains 20 kWh/h (per dwelling).
- Diagnostics per dwelling: `base_kwh`, `heat_kwh`, `spike_kwh`, `cap_clip_*`.

---

## 5) UI wiring tips
- Expose controls for each mask dimension:
  - Income bands (multi-select), education (multi-select), tenure, children flag, schedule_type, top‑X% slider, area filter.
- Policy toggles:
  - “Mark selected as heat‑pump candidates” (sets flags + class),
  - Adoption rate slider (0–1),
  - Retrofit flag toggles (kids scenario),
  - Setback toggle/slider (elderly scenario).
- Runtime sliders:
  - Window length (hours/days/years),
  - Agent-level collection on/off for speed,
  - `TOP_FRAC`, `RUN_YEARS`, `RUN_YEARS_SHORT`.
- Outputs to show:
  - Cohort sizes (baseline vs treated),
  - kWh totals and ΔkWh,
  - Per-dwelling averages,
  - Optional annual plot (GWh) for multi-year runs.

---

## 6) File paths referenced in the notebook
- `GEOJSON`: household polygons/points
- `CLIMATE`: hourly climate parquet
- `HIDP_CSV`: optional socio‑demographics enrichment
- `AREA_COLUMN` / `AREA_CODE`: filter
- `OUTDIR`: where to write optional parquet/CSV outputs

Keep these as dashboard-configurable text inputs or defaults.

---

## 7) Minimal run loop (pseudocode)
```python
# baseline
model_base = EnergyModel(gdf=gdf_base, climate_parquet=CLIMATE, ...)
for _ in range(hours): model_base.step()

# policy
gdf_pol = gdf_base.copy()
gdf_pol.loc[mask, "is_heatpump_candidate"] = 1
gdf_pol.loc[mask, "heatpump_candidate_class"] = "priority"
model_pol = EnergyModel(gdf=gdf_pol, climate_parquet=CLIMATE, ...)
for _ in range(hours): model_pol.step()

delta_kwh = model_pol.model_dc.get_model_vars_dataframe()["total_energy"].sum() \
          - model_base.model_dc.get_model_vars_dataframe()["total_energy"].sum()
```

---

Use this guide to reproduce the notebook’s scenarios in a dashboard: each scenario is just “build mask → set flags → run baseline vs policy → show ΔkWh”. All category labels above match the columns produced by the HIDP enrichment step.***
