# Model logic audit & paper-ready equations

Audit of `household_energy/agent.py` + `household_energy/model.py` (branch
`v4-elec-slope-confidence`, post Phase-1/2/3 cleanup), and a LaTeX equation set
describing the model **as implemented** — not as the config comments imply.

> **STATUS (2026-06-24): partly superseded.** The canonical paper formula is now the
> three-term equation in
> `docs/research/paper draft/too_hot_overleaf_v2/methodology_3_3_architecture_draft.md`
> (`E = min(B + H + S, E_max)`, order B, H, S). The five-term form below (with separate
> cooling and DHW terms) predates that re-scope: in the live v7 config DHW is neutralised
> to zero and cooling is a small structural default, so the paper folds both into the
> baseline and omits them from the headline equation. Other v7 deltas not yet reflected
> in the per-equation detail below: per-cohort baseline anchors with the recentring and
> the electric-heated multiplier removed, the electric heating slope wired in, and a
> mean-one monthly heating profile $\sigma_m$ that supersedes the hard heating-months
> gate (the within-day $\pi_h$ is identity in v7). Treat this file as an implementation
> audit, not the paper's formula; reconcile against §3.3/§3.4 before citing.

---

## Part 1 — Equations (as implemented)

Notation: dwelling $i$, hour $t$; $T_{i,t}$ = outdoor ambient temperature at the
dwelling's nearest climate grid point; $A_i$ = floor area (m²); $n_i$ = number
of residents; $O_{i,t} \in \{0,\dots,n_i\}$ = residents at home; $f_i \in
\{\text{gas}, \text{elec}, \text{other}\}$ = heating-fuel bucket;
$h(t) \in \{0..23\}$ = local hour of day.

### 1. Total hourly demand

$$
E_{i,t} \;=\; \min\!\Big( B_i \;+\; S_{i,t} \;+\; H_{i,t} \;+\; C_{i,t} \;+\; W_{i,t},\;\; E^{\max} \Big)
$$

baseline (non-climate) load, occupant activity load, space heating, space
cooling, domestic hot water; capped at $E^{\max}$ (config
`max_total_kwh_per_hour`). Each component is routed to electricity, gas, or
other fuel as described in §8.

### 2. Baseline load (constant, computed once)

$$
B_i \;=\; \min\!\Big( \bar{b}\cdot \mu^{\text{PT}}_{\text{base}}(p_i)\cdot a_i \cdot \lambda,\; B^{\max}\Big),
\qquad
a_i = \mathrm{clip}\!\left[\Big(\tfrac{A_i}{A_{\text{ref}}}\Big)^{\gamma};\; 0.85,\,1.25\right]
$$

with anchor $\bar b$ (kWh/h, meter-derived; `baseline_anchor_kwh_per_hour`),
property-type multiplier $\mu^{\text{PT}}_{\text{base}}$, global scale
$\lambda$ (`level_scale`), $A_{\text{ref}}=70$ m², $\gamma=0.20$. When separate
fuel anchors are configured, $B_i^{\text{elec}}$ and $B_i^{\text{gas}}$ are
computed independently with fuel-specific anchors and property-type maps, and
$B_i^{\text{gas}}=0$ unless $f_i=\text{gas}$.

### 3. Occupant activity load (per person, summed)

For each resident $p$ of dwelling $i$:

$$
s_{p,t} =
\begin{cases}
e_{\text{home}} \cdot \kappa_{\text{awake}} & \text{home, awake}\\
e_{\text{home}} \cdot \kappa_{\text{sleep}} & \text{home, asleep}\\
e_{\text{away}} & \text{away}
\end{cases}
\qquad
S_{i,t} = \sigma^{\text{SAP}}_i \sum_{p} s_{p,t}\, w_p
$$

where $w_p \in [0.75, 1.30]$ is a wealth-quintile multiplier and
$\sigma^{\text{SAP}}_i \in [0.80, 1.20]$ is a linear-in-SAP multiplier
(§4, eq. for $\mu_{\text{lin}}$ with spike bounds). Presence follows
deterministic leave/return schedules drawn from household-type archetypes with
±1 h jitter; awake/asleep follows per-archetype wake/sleep windows.

### 4. Space heating

**Heating-degree signal** (outdoor trigger, not an indoor thermostat):

$$
HD_{i,t} = \max\!\Big(0,\; \big(\tau - \delta\cdot \mathbb{1}[O_{i,t}=0]\big) - T_{i,t} - \Delta \Big)
$$

with trigger $\tau = 18.5$ °C, vacancy setback $\delta = 2$ °C, deadband
$\Delta = 0.5$ °C.

**Per-dwelling slope** (kWh per °C·h, computed once):

$$
k_i = \mathrm{clip}\Big( k_0(f_i)\cdot
\mu^{\text{SAP}}_{\text{step}} \cdot
\mu^{\text{PT}}_{\text{heat}}(p_i)\cdot
\mu^{\text{SAP}}_{\text{lin}}(SAP_i)\cdot
\mu^{A}(A_i)\cdot
\mu^{\text{env}}(r_i)\cdot
\mu^{\text{sys}}_i
;\; k_{\min},\, k_{\max}\Big)
$$

- $k_0(f)$: fuel-specific base slope — SERL-fitted, electric-heated homes get
  `heating_slope_kWh_per_deg_electric`, all others the shared gas slope.
- $\mu^{\text{SAP}}_{\text{step}}$: 1.10 if $SAP<50$, 0.90 if $SAP>80$, else 1. **[see finding A3]**
- $\mu^{\text{SAP}}_{\text{lin}}$: linear from 1.30 at $SAP=40$ to 0.70 at $SAP=90$ (clamped). **[A3]**
- $\mu^{A}$: SERL per-band slope multiplier lookup (5 floor-area bands,
  normalised to the 51–100 m² band); fallback
  $\mathrm{clip}[(A_i/75)^{0.811}; 0.5, 3.0]$.
- $\mu^{\text{env}} = 1 - 0.2\, r_i$ with retrofit envelope score $r_i\in[0,1]$.
- $\mu^{\text{sys}}$: heating-system multiplier (communal 0.85, heat pump 0.70, …). **[A2]**

**Hourly heating:**

$$
H_{i,t} = \underbrace{\min\!\Big( k_i \cdot \varphi_i \cdot m^{\text{SAP}}(f_i)\, m^{\text{age}}(f_i)\cdot HD_{i,t},\; K_i\Big)}_{\text{linear demand, capacity-capped}}
\;\cdot\; \omega(O_{i,t}) \;\cdot\; \pi_{h(t)}(f_i)\cdot \rho_{h(t)}
$$

- $\varphi_i = \eta_{\text{boiler}} / COP$ ($\approx 0.90/2.8 = 0.32$) if the
  dwelling has a heat pump, else 1. **[A2]**
- $m^{\text{SAP}}, m^{\text{age}}$: SERL-calibrated (v2) heating-side
  multipliers, looked up by the dwelling's EPC band and building-age band,
  fuel-specific; default 1.
- $K_i = \mathrm{clip}(0.10\,A_i;\, 4,\, K^{\max})$ kWh/h — physical heat-output
  cap (CIBSE sizing rule, ~never binds).
- $\omega(O) = \omega_0 + (1-\omega_0)\,O/n_i$, $\omega_0 = 0.5$ — continuous
  occupancy modulation. **[A4: interacts with the $\delta$ setback]**
- $\pi_{h}(f)$: fuel-specific 24-h heating profile, normalised to mean 1. **[A5]**
- $\rho_{h}$: AM/PM peak and winter-peak calibration multipliers
  (= 1 outside the 06–09 / 17–21 windows).

### 5. Space cooling

$$
C_{i,t} = c \cdot \max\!\big(0,\; T_{i,t} - \tau_c - \Delta\big)\cdot \omega(O_{i,t}),
\qquad \tau_c = 24\ \text{°C},\; c = 0.03\ \text{kWh/°C·h}
$$

applied to **all** dwellings, routed to electricity. **[B3: universal latent
cooling assumption]**

### 6. Domestic hot water

$$
W_{i,t} = \frac{d_{\text{home}} + d_{\text{pp}}\, n_i}{24}\;\cdot\; \psi_{h(t)}(f_i)\cdot \rho^{\text{dhw}}_{h(t)} \;\cdot\;\Big[\rho_0 + (1-\rho_0)\,\tfrac{O_{i,t}}{n_i}\Big]
$$

with daily DHW components $d_{\text{home}}, d_{\text{pp}}$ (kWh/day), 24-h DHW
profile $\psi_h$ (mean 1), away floor $\rho_0 = 0.5$. (Zero in the default
config — DHW is then implicit in the baseline anchor.)

### 7. SERL emulation layer (optional)

When enabled, per-fuel hourly totals are blended with empirical SERL diurnal ×
monthly shape multipliers by segment $g(i)$:

$$
E^{f}_{i,t} \leftarrow E^{f}_{i,t}\cdot\Big[(1-\alpha) + \alpha\, m^{f}_{\text{hour}}(g_i, h(t))\, m^{f}_{\text{month}}(g_i, mo(t))\Big]
$$

### 8. Fuel routing

Heating $H$ goes to the dwelling's bucket $f_i$ (explicit
fuel×system lookup maps with flag-based fallback); cooling is always electric;
baseline and activity loads are electric except a configurable gas share for
gas-heated homes (`gas_base_share`, `gas_spike_share`, 0 by default); DHW
follows $f_i$.

### Suggested paper framing

The model is a **bottom-up synthetic demand model**: hourly demand is generated
from structure (floor area, property type, EPC band, building age, heating
system/fuel) and occupancy schedules, with the temperature-response slopes and
band multipliers **fitted to SERL smart-meter panel data**; the per-dwelling
EPC/DESNZ annual figures are *not* used as anchors (see A7).

---

## Part 2 — Audit findings

### A. Inconsistencies / likely bugs (fix or explicitly justify before submission)

**A1 — Heat-pump adoption scenarios do not electrify demand.**
`_assign_heatpumps()` sets `has_heatpump=True`, which reduces heating via
$\varphi_i\approx0.32$ — but the fuel bucket is resolved from
`main_fuel_type`/`main_heating_system` through the explicit maps
(`mains gas|boiler → gas`), and the `has_heatpump` flag is only consulted in
the *fallback* branch ([agent.py:679](household_energy/agent.py:679)). The
cached bucket is also never refreshed after assignment. **Result: a converted
home's (reduced) heating stays on the gas meter — HP policy scenarios show gas
savings but no electricity uptake.** This is the single most consequential
issue for Paper 2 policy scenarios.

**A2 — Heat-pump efficiency is double-counted.** Homes with heat pumps get
both $\varphi_i = \eta/COP \approx 0.32$ in `apply_climate`
([agent.py:862](household_energy/agent.py:862)) *and* the systems-config slope
multiplier 0.70 in `_compute_heat_slope`
([agent.py:549](household_energy/agent.py:549)). Combined ≈ 0.225, i.e. an
implied COP of ~4.0 rather than the stated 2.8. Pick one mechanism.

**A3 — SAP enters the heating slope three times.** (i) the hard-coded step
(×1.10 below SAP 50 / ×0.90 above 80, [agent.py:488](household_energy/agent.py:488)),
(ii) the linear 1.30→0.70 `sap_scaling` multiplier
([agent.py:504](household_energy/agent.py:504)), and (iii) the v2 SERL
SAP-band multiplier $m^{\text{SAP}}$ in `apply_climate`. If the v2 band
multipliers are fitted as *total* SERL band ratios (not residual corrections
after (i)+(ii)), efficiency differences are double/triple-counted. Either drop
(i) and (ii) when v2 multipliers are present, or document that the v2 fit was
performed on top of the internal multipliers.

**A4 — Vacancy reduces heating twice.** An empty dwelling gets both the −2 °C
trigger setback (reduces $HD$) *and* $\omega(0)=0.5$. The two mechanisms
target the same behaviour (unoccupied setback); their product is an
unintended, temperature-dependent compound reduction. State one mechanism in
the paper (and ideally remove the other).

**A5 — The 24-h heating profile is not total-preserving.** $\pi_h$ is
normalised to mean 1 *over hours*, but heating demand is correlated with hour
of day (cold nights/evening peaks coincide with $\pi_h>1$), so
$\mathbb{E}[\pi_h \cdot HD] \neq \mathbb{E}[HD]$. This re-introduces exactly
the "simulator integrates to a different annual total than the fitted slope"
problem that the Phase-1 linearization removed. Quantify the bias (one run
with $\pi\equiv1$) or renormalise $\pi$ weighted by mean $HD_h$.

**A6 — The total cap breaks the fuel decomposition.** `_enforce_total_caps`
([model.py:1198](household_energy/model.py:1198)) clips
`energy_consumption` to 20 kWh/h but leaves `electric_kwh`/`gas_kwh`/`other_kwh`
unclipped, so `total_energy ≠ electric + gas + other` in any clipped hour, and
the annual *per-fuel* accumulators use the unclipped values while
`annual_kwh_by_year` uses the clipped one. Apply the pro-rata clip to the fuel
trackers too (the clip fractions `fb/fh/fs` are already computed but never
subtracted from anything).

**A7 — Calibrated annual demand is loaded but never used.** `annual_energy_kwh`
(`energy_cal_kwh` etc.) is carried as a static attribute and reported, but no
equation consumes it. The docstrings ("prefer calibrated demand") suggest
otherwise. Fine as a design choice — but the paper must describe the model as
fully synthetic bottom-up, validated against (not anchored to) meter data.

**A8 — Dead heterogeneity: the first slope computation is overwritten.**
`HouseholdAgent.__init__` computes a slope with archetype `ua_mult`, a retrofit
multiplier, and a ±10 % lognormal-ish noise term
([agent.py:344-349](household_energy/agent.py:344)), which is then *replaced*
by `_compute_heat_slope()` ([agent.py:391](household_energy/agent.py:391), and
again at [model.py:535](household_energy/model.py:535)). Consequences: (a) the
model has **no stochastic between-dwelling heterogeneity** in slopes beyond
observables — don't claim it; (b) the `archetypes:` `ua_mult` config section
is inert.

**A9 — Documented levers that do nothing.** Read in and reported, but with no
effect on any kWh: retrofit flags (`cwi/swi/loft/floor/glazing`),
`heating_controls`/`meter_type`, and the entire `envelope_levers`, `controls`,
`age_bands`, and `appliance_loads` config sections; `apply_structural_multipliers`;
`holiday_prob_daily`; the heat-pump `cop_curve` (a flat COP of 2.8 is used —
no temperature dependence, which also flatters HPs in cold snaps). The only
live envelope channel is `retrofit_envelope_score`. **Do not present
insulation flags or heating controls as model levers in the paper.**

### B. Limitations / assumptions to state in the paper

**B1 — Outdoor-trigger heating, no building thermal dynamics.** Heating
responds to *outdoor* temperature crossing 18.5 °C (≈18.0 °C net of deadband)
with no indoor-temperature state, thermal mass, or solar/internal gains. This
is a steady-state degree-hour model, equivalent to a per-dwelling
linear-response (PRISM-style) formulation at hourly resolution — defensible,
but say it; "setpoint" language must be avoided (the code already renamed it).

**B2 — Slope-based fits assume linearity in HD.** Consistent with the
SERL linear-OLS calibration (good — this was the point of Phase 1), but means
no saturation at extreme cold beyond the rarely-binding capacity cap $K_i$.

**B3 — Universal cooling.** Every dwelling cools above 24 °C at 0.03 kWh/°C·h.
UK residential AC penetration is low (~5–15 %); for Paper 3 (climate
exposure / overheating) either frame $C_{i,t}$ as *latent* cooling demand or
add an AC-ownership share. As-is, aggregate summer electricity response will
be overstated relative to today's stock.

**B4 — Occupancy and behaviour are deterministic archetypes.** Fixed
leave/return/wake/sleep hours with ±1 h jitter, identical every day: no
weekends, no holidays (`holiday_prob_daily` inert), no stochastic presence.
Diurnal variance will be too regular; this matters for peak-load statistics
more than for annual totals.

**B5 — Wealth and SAP scaling of appliance load are ad hoc.** Wealth
multipliers (0.75–1.30) are hard-coded and wealth is *randomly assigned* when
missing; appliance spikes scale with the building's SAP rating
($\sigma^{\text{SAP}}$), a building-fabric rating with no clear mechanism for
appliance use. Both are uncalibrated; consider dropping $\sigma^{\text{SAP}}$
or justifying it as a socioeconomic proxy.

**B6 — Timekeeping inconsistencies (small).** `local_hour()` uses a fixed
UTC-offset captured at $t_0$ — diurnal profiles drift by 1 h across DST
transitions, while monthly logic does proper tz conversion; and the
`heating_months` gate uses the **UTC** month while the winter-peak logic uses
the **local** month. Harmless in the UK but inconsistent.

**B7 — Cap-clip attribution recorded but components unrevised.** When the
SERL emulation layer or caps rescale fuels, component trackers
(`base_kwh`, `heat_kwh`, `spike_kwh`) are not rescaled — decomposition plots
from clipped/emulated runs are approximate.

**B8 — Initial-condition edge cases.** All residents start at home regardless
of start hour; presence flips only on exact hour equality, so jitter that
makes `leave == return` strands a person away for a full day. Negligible in
aggregate, worth a one-line robustness note.

### C. Priority recommendation order

1. **A1** (HP fuel routing) — invalidates HP scenario results.
2. **A6** (cap vs fuel split) — silent accounting error in validation outputs.
3. **A2, A3, A4** (double counting) — biases magnitudes; decide the canonical
   mechanism for each and delete the rest.
4. **A5** — quantify; likely a few % on annual heating.
5. **A8/A9** — delete dead code/config so the paper's parameter table matches
   the implementation.
6. State B1–B5 in the limitations section.

---

## Part 3 — Branch review (`v4-elec-slope-confidence` vs `main`)

The branch's core changes are sound and well-motivated: Phase 1 replaces the
saturating duty-cycle (`duty = loss/(loss+K)`, K a free knob) with linear
`slope × HD` capped at a physical capacity — correctly resolving the
functional-form mismatch with the linear-OLS SERL slope fit; Phase 2 swaps the
unsourced area power-law for the SERL per-band lookup; Phase 3 adds the
fuel-specific SERL SAP-band/age-band multipliers; the `_serl_age_band`
"post-YYYY" fix is correct. Branch-specific findings:

**D1 — A3 (SAP multiple-counting) is partly branch-introduced, partly
neutralised.** The linear `sap_scaling` slope multiplier
(`slope *= _sap_multiplier("slope")`) was **added on this branch** on top of
the pre-existing step multiplier. The uncommitted `calibrate_serl.py` change
emits `sap_scaling: {slope_mult_lo: 1.0, slope_mult_hi: 1.0}`, which
neutralises the *linear* term in calibrated configs — but the **step
multiplier (×1.10 / ×0.90) is hard-coded in `_compute_heat_slope` and not
config-controllable**, so it still compounds with the v2 SERL SAP-band
multipliers even in calibrated runs. Make the step config-controllable (or
delete it) and zero it in the calibration output alongside the linear term.

**D2 — `config_defaults.yaml` was not updated for the Phase 1/2 cleanup.**
It still ships `heat_slope_area_exp: 0.6` (the code comment calls 0.811 the
default, but the YAML — which always loads — wins, so non-calibrated runs use
0.6); the removed duty-cycle keys (`loss_to_duty_k`, `base_heat_capacity`,
`heat_capacity_area_exp`, `min_heat_capacity`) are still present (silently
ignored); and the cap defaults disagree with the code
(`max_heat_kwh_per_hour`: YAML 30 vs code 24; `max_total`: YAML 32 vs code
20 — YAML wins). The paper's parameter table should quote the YAML-effective
values.

**D3 — Default `heat_slope_max: 0.10` silently caps calibrated slopes.** SERL
gas slopes are ~0.21; `calibrate_serl.py` uncaps to 5.0 in emitted configs,
but any run that takes calibrated slope keys without the calibrated
`heat_slope_max` clips every dwelling at 0.10. Consider raising the package
default.

**D4 — Uncommitted work must land together.** The presence-spikes fit wiring
in `calibrate_serl.py` shells out to `fit_presence_spikes.py`, which is
currently **untracked** — committing the wiring without the script breaks the
pipeline. Same for `gas_slope_probe.py` / `diag_p3_vs_p4_hourly.py` if
referenced elsewhere.

**D5 — Behaviour changes vs main worth a regression note.** The branch also
(a) added the communal-system slope multiplier branch (0.85) ahead of the
heat-pump branch, (b) rebuilt `_compute_heat_capacity` without
SAP/property-type/system modifiers (good — documented rationale), and
(c) introduced the explicit fuel-bucket maps with cached routing — the cache
is what makes A1 (HP adoption not refreshing the bucket) bite.
