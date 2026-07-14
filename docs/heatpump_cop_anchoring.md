# Heat-pump COP — parameter provenance

Provenance note for the heat-pump efficiency parameters in
`household_energy/config_defaults.yaml`. Companion to
`model_audit_and_equations.md`. Branch `v4-elec-slope-confidence`.

Reproduce with:

```bash
python research/applied/scripts/fit_heatpump_cop.py        # fetch live
# offline from cached pulls:
python research/applied/scripts/fit_heatpump_cop.py \
    --systems-json /tmp/hpm_systems.json --stats-json /tmp/hpm_stats365.json
# -> results/field_fits/heatpump_cop.yaml
```

---

## 1. What was wrong

Heat-pump efficiency was the last unsourced assumption on the heating path:

- `model.heatpump_cop_ref = 2.80` — a flat COP, no citation in code or config.
- `systems.heat_pump.cop_curve = [[15,3.2],[7,2.8],[0,2.4],[-5,2.0]]` — a
  hand-drawn temperature curve, and **dead config** (read by no `.py`).

Every other parameter on the heating and per-person electricity paths is either
SERL-fitted, literature-cited, or explicitly neutralised. COP was the exception.
SERL cannot identify it (the panel does not segment by heating technology at the
needed resolution), so it needs an *external* anchor.

## 2. Data source

**heatpumpmonitor.org** (OpenEnergyMonitor) — an open, continuously-updated
registry of field-monitored UK heat pumps. Each system reports measured
electricity and heat (class-2 heat meters) via emoncms. No-auth public JSON API:

| Endpoint | Content |
|---|---|
| `GET /system/list/public.json` | Per-system metadata (type, output, refrigerant, property, floor area, heat loss, age, flow temp) |
| `GET /system/stats/last365` | Per-system measured COP (space / combined / water), mean outdoor + flow temps, kWh totals, coverage length |

Licence: code AGPL-3.0; performance data published openly for comparison.

**Sample-bias caveat.** These are self-selected, well-commissioned
owner-enthusiast systems — the *good-install* end of the distribution. The field
COP reads high relative to the representative installed base. It is therefore
treated as an **upper, well-installed anchor**, not the baseline.

## 3. Which COP maps to the model

`hp_effect_mult = boiler_efficiency / heatpump_cop_ref` (agent.py) multiplies the
**space-heating** slope. The field metric that maps to it is therefore
**space-heating COP** (SPFH4 space), not the combined space+DHW SPF. The point
anchor uses space COP; combined is reported for context.

## 4. Results (last 365 days, ≥90-day coverage per system; 764 systems total)

| Metric | n | median | mean | p10–p90 |
|---|---|---|---|---|
| **Space-heating COP** (maps to model) | 464 | **4.21** | 4.21 | 3.37–5.03 |
| Combined SPF (space + DHW) | 690 | 3.90 | 3.90 | 3.17–4.61 |
| Water / DHW COP | 421 | 3.02 | 2.96 | 2.16–3.73 |

By technology (space COP): Air Source median **4.20** (n=452), Ground Source
**4.58** (n=12).

**Temperature curve.** Field mean-outdoor-temperatures span only ~3–14 °C, so a
raw linear fit cannot be trusted out to the −5 °C the curve needs. Instead a
constant **Carnot efficiency** is fitted — `η = COP·(T_flow − T_out)/T_flow`,
medianed across systems (η ≈ 0.298) — and `COP(T_out) = η·T_flow/(T_flow − T_out)`
is evaluated at a representative flow temperature (≈31 °C, the field median):

| Outdoor °C | 15 | 7 | 0 | −5 |
|---|---|---|---|---|
| COP | 5.69 | 3.79 | 2.93 | 2.52 |

This replaces the hand-drawn curve. (Both reflect the well-installed sample, so
both read high vs a representative stock; the curve remains unused by model code.)

## 5. The dual anchor (decision)

The config keeps a citable conservative default **and** carries the field anchor:

| Key | Value | Role |
|---|---|---|
| `heatpump_cop_ref` (config default) | **2.80** | Conservative/representative default — DESNZ Electrification of Heat trial median ASHP SPFH4 |
| `cop_representative` (YAML) | 2.80 | Same value, explicit citable fallback |
| `cop_field_median_space` (YAML) | 4.21 | Well-installed scenario anchor (this data) |
| `cop_field_median_combined` (YAML) | 3.90 | Context |

Rationale: for a policy ABM the defensible baseline is the representative trial
median, not the optimistic field median. 2.80 is retained as the default;
heatpumpmonitor provides an evidenced, reproducible **upper bound / scenario
lever** and independent corroboration that 2.80 is conservative rather than
arbitrary. Re-anchoring to 4.21 later is a one-line swap — but see §6 first.

## 6. Open issue — efficiency is double-counted

HP efficiency is applied twice (see `model_audit_and_equations.md`):

1. `hp_effect_mult = 0.90 / 2.80 ≈ 0.32` on the space-heating slope, and
2. `systems.heat_pump.heating_slope_mult = 0.70` on the same slope.

Combined `≈ 0.32 × 0.70 ≈ 0.225`, implying an **effective COP ≈ 4.0** — not the
stated 2.80. This must be resolved (make `heatpump_cop_ref` the single efficiency
source) **before** re-anchoring to the field value, or the "anchored" number will
not be what the model actually uses.

## 7. Methodology-section impact

`§3.4` currently states: *"Boiler efficiency (0.90) and heat-pump coefficient of
performance (2.8) are held at central UK Energy Saving Trust estimates."* This
can be upgraded: the HP COP is now anchored to the DESNZ Electrification of Heat
trial median (representative) and independently corroborated/bounded by
heatpumpmonitor.org field data (well-installed). It moves from "literature
default" toward the same provenance discipline as the SERL-fitted parameters.

Citation for the default value: Energy Systems Catapult (2023), *Electrification
of Heat Demonstration Project: Heat Pump Performance Data Analysis* (median ASHP
SPFH4 ≈ 2.80).
