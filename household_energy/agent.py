"""
agent.py
========

Domain agents for the Household-Energy ABM:

* **HouseholdAgent** – one per building polygon / dwelling unit
* **PersonAgent**    – individual resident linked to a HouseholdAgent

Both inherit from Mesa / mesa-geo base classes 

The module also contains:

* a *PROPERTY_TYPE_MULTIPLIER* look-up table that scales energy consumption
  according to house archetype; and
* three schedule profiles (`Parent`, `Worker`, `Homebody`) that define when a
  resident leaves / returns home during a 24-h cycle.

This version adds light-weight climate hooks to HouseholdAgent:
- `clim_idx`: index of nearest climate point (set once by the model)
- `ambient_tempC`: last sampled outdoor temperature (°C)
- `apply_climate(...)`: converts ambient temp → kWh and adds it to the tick load
"""


from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import geopandas as gpd               # only used for typing / IDE hints
import mesa
import mesa_geo as mg
from shapely.geometry.base import BaseGeometry


# ────────────────────────────────────────────────────────────────────
#  Energy scaling by property archetype
#  Baseline vs heating multipliers are decoupled to avoid double-counting structure.
# ────────────────────────────────────────────────────────────────────

PROPERTY_TYPE_MULT_BASE: Dict[str, float] = {
    "mid-terraced house": 1.00,
    "semi-detached house": 1.00,
    "small block of flats/dwelling converted in to flats": 1.00,
    "large block of flats": 1.00,
    "block of flats": 1.00,
    "end-terraced house": 1.00,
    "detached house": 1.00,
    "flat in mixed use building": 1.00,
}

PROPERTY_TYPE_MULT_HEAT: Dict[str, float] = {
    "mid-terraced house": 1.00,
    "semi-detached house": 1.10,
    "small block of flats/dwelling converted in to flats": 0.90,
    "large block of flats": 0.85,
    "block of flats": 0.85,
    "end-terraced house": 1.05,
    "detached house": 1.20,
    "flat in mixed use building": 0.90,
}

# Backwards-compatible alias (used elsewhere); treat as the heating map.
PROPERTY_TYPE_MULTIPLIER: Dict[str, float] = PROPERTY_TYPE_MULT_HEAT

PROPERTY_TYPES: List[str] = list(PROPERTY_TYPE_MULT_BASE.keys())

# ────────────────────────────────────────────────────────────────────

SCHEDULE_PROFILES = [
    {"name": "Parent",    "leave":  7, "return": 15},
    {"name": "Worker",    "leave":  9, "return": 17},
    {"name": "Homebody",  "leave": None, "return": None},   # never leaves
]


# ────────────────────────────────────────────────────────────────────
#  HouseholdAgent
# ────────────────────────────────────────────────────────────────────

class HouseholdAgent(mg.GeoAgent):
    """Spatial agent representing one dwelling (building polygon or centroid)."""

    def __init__(
        self,
        unique_id: str,
        model: "mesa.Model",
        geometry: BaseGeometry,
        *,
        property_type: str = "unknown",
        sap_rating: float = 70,
        # NEW: prefer calibrated annual demand (DESNZ/LSOA adjusted)
        annual_energy_kwh: float = 10_000,  # NEW
        # ─── core drivers (plumb-through; optional) ─────────────────
        floor_area_m2: float | None = None,          # NEW
        property_age: str | None = None,             # NEW
        main_fuel_type: str | None = None,           # NEW
        main_heating_system: str | None = None,      # NEW
        retrofit_envelope_score: float | None = None,# NEW (0–1 expected)
        imd_decile: float | None = None,             # NEW
        is_heatpump_candidate: int | None = None,    # NEW
        heatpump_candidate_class: str | None = None, # NEW
        schedule_type: str | None = None,            # NEW
        # NEW: socio‑demographic and dwelling attributes (optional)
        hidp: str | None = None,                     # NEW
        hh_n_people: int | None = None,              # NEW
        hh_children: bool | None = None,             # NEW
        hh_income: float | None = None,              # NEW
        hh_income_band: str | None = None,           # NEW
        hh_edu_detail: str | None = None,            # NEW
        dwelling_bucket: str | None = None,          # NEW
        tenure: str | None = None,                   # NEW
        size_band: int | None = None,                # NEW (bedrooms, capped at 4)
        # ─── policy levers & context (optional) ────────────────────
        heating_controls: str | None = None,         # NEW
        meter_type: str | None = None,               # NEW
        cwi_flag: int | None = None,                 # NEW
        swi_flag: int | None = None,                 # NEW
        loft_ins_flag: int | None = None,            # NEW
        floor_ins_flag: int | None = None,           # NEW
        glazing_flag: int | None = None,             # NEW
        is_electric_heating: int | None = None,      # NEW
        is_gas: int | None = None,                   # NEW
        is_oil: int | None = None,                   # NEW
        is_solid_fuel: int | None = None,            # NEW
        is_off_gas: int | None = None,               # NEW
        crs: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, geometry=geometry, crs=crs)

        # identity & static attributes
        self.unique_id: str = unique_id
        self.property_type: str = property_type.strip().lower()
        self.sap_rating: float = sap_rating
        # track whether this dwelling came with a calibrated annual value
        self.has_calibrated_energy: bool = bool(getattr(self, "annual_energy_kwh", None))  # may be overridden in model.py

        # NEW: household identifiers / demographics
        # Robust HIDP: allow NaN/float/None and fall back to unique_id
        if isinstance(hidp, str) and hidp.strip():
            self.hidp: Optional[str] = hidp.strip()
        else:
            try:
                self.hidp = str(int(hidp)).strip()
            except Exception:
                self.hidp = str(unique_id)
        try:
            self.hh_n_people: Optional[int] = int(hh_n_people) if hh_n_people not in (None, "", float("nan")) else None       # NEW
        except Exception:
            self.hh_n_people = None
        if hh_children in (None, ""):                                                                       # NEW
            self.hh_children: Optional[bool] = None                                                         # NEW
        else:                                                                                               # NEW
            val = str(hh_children).strip().lower()                                                          # NEW
            self.hh_children = val in ("true", "1", "yes", "y", "t")                                        # NEW
        try:
            self.hh_income: Optional[float] = float(hh_income) if hh_income not in (None, "") else None        # NEW
        except Exception:
            self.hh_income = None
        self.hh_income_band: Optional[str] = (hh_income_band or None)                                        # NEW
        self.hh_edu_detail: Optional[str] = (hh_edu_detail or None)                                          # NEW
        self.dwelling_bucket: Optional[str] = (dwelling_bucket or None)                                      # NEW
        self.tenure: Optional[str] = (tenure or None)                                                        # NEW
        if isinstance(schedule_type, str) and schedule_type.strip():
            self.schedule_type: Optional[str] = schedule_type.strip()
        else:
            self.schedule_type = None                                                                       # NEW
        try:
            self.size_band: Optional[int] = int(size_band) if size_band not in (None, "") else None         # NEW
        except Exception:
            self.size_band = None

        # NEW: prefer calibrated annual kWh; keep legacy alias for compatibility
        self.annual_energy_kwh: float = float(annual_energy_kwh)  # NEW
        self.energy_demand: float = self.annual_energy_kwh        # NEW (legacy alias)

        # per-tick state – cleared by model.step()
        self.energy_consumption: float = 0.0

        # residents
        self.residents: List["PersonAgent"] = []

        # --- climate state (populated/used by the model) -----------
        self.clim_idx: Optional[int] = None
        self.ambient_tempC: float = float("nan")
        self.climate_heating_kWh: float = 0.0
        self.climate_cooling_kWh: float = 0.0

        # NEW: attach core drivers (kept raw; used in calc/reporters)
        self.floor_area_m2 = None if floor_area_m2 is None else float(floor_area_m2)    # NEW
        self.property_age  = (property_age or "").strip().lower() if property_age else None  # NEW
        self.main_fuel_type = (main_fuel_type or "").strip().lower() if main_fuel_type else None  # NEW
        self.main_heating_system = (main_heating_system or "").strip().lower() if main_heating_system else None  # NEW
        self.is_heatpump_candidate = 1 if (is_heatpump_candidate or 0) else 0  # NEW
        self.heatpump_candidate_class = (
            (heatpump_candidate_class or "").strip().lower() if heatpump_candidate_class else None
        )  # NEW
        self.has_heatpump = "heat pump" in (self.main_heating_system or "").lower()
        self.was_heatpump_initial = bool(self.has_heatpump)   # <-- NEW
        self.retrofit_envelope_score = None if retrofit_envelope_score is None else float(retrofit_envelope_score)  # NEW
        self.imd_decile = None if imd_decile is None else float(imd_decile)  # NEW
        # -----------------------------------------------------------
        # --- per-home climate sensitivity (heating slope) ---  # NEW
        cfg = getattr(self.model, "config", None)
        cfg_arche = cfg.archetypes if cfg else {}
        heat_loss_default = {
            "detached house": 1.30, "semi-detached house": 1.15,
            "end-terraced house": 1.10, "mid-terraced house": 1.00,
            "small block of flats/dwelling converted in to flats": 0.85,
            "large block of flats": 0.75, "block of flats": 0.85,
            "flat in mixed use building": 0.90,
        }
        ptype = (self.property_type or "").strip().lower()
        sap   = float(self.sap_rating or 70.0)
        retro = float(self.retrofit_envelope_score or 0.5)
        fa    = float(self.floor_area_m2 or 90.0)

        p_mult  = cfg_arche.get(ptype, {}).get("ua_mult") if ptype in cfg_arche else None
        if p_mult is None:
            p_mult = heat_loss_default.get(ptype, 1.0)
        sap_mult   = 1.15 if sap < 55 else (0.90 if sap > 80 else 1.00)
        retro_mult = 1.10 - 0.20 * max(0.0, min(1.0, retro))
        area_mult  = max(0.7, min(1.6, fa / 90.0))
        rng = __import__("random").Random(hash(str(self.unique_id)) & 0xFFFFFFFF)
        noise_mult = max(0.75, min(1.25, rng.normalvariate(1.0, 0.10)))

        self.heat_slope_kWh_per_deg = (
            self.model.heating_slope_kWh_per_deg * p_mult * sap_mult * retro_mult * area_mult * noise_mult
        )

        # heat-pump effectiveness vs boiler (simple, deterministic)
        self.hp_effect_mult = getattr(self.model, "boiler_efficiency", 0.90) / getattr(self.model, "heatpump_cop_ref", 2.8)
        # -----------------------------------------------------------


        # NEW: policy levers (coerce to 0/1 where appropriate)
        def _b(v):  # NEW
            try:
                return int(v) if v is not None else 0
            except Exception:
                return 0

        self.heating_controls = (heating_controls or "").strip().lower() if heating_controls else None  # NEW
        self.meter_type = (meter_type or "").strip().lower() if meter_type else None  # NEW
        self.cwi_flag = _b(cwi_flag)                # NEW
        self.swi_flag = _b(swi_flag)                # NEW
        self.loft_ins_flag = _b(loft_ins_flag)      # NEW
        self.floor_ins_flag = _b(floor_ins_flag)    # NEW
        self.glazing_flag = _b(glazing_flag)        # NEW
        self.is_electric_heating = _b(is_electric_heating)  # NEW
        self.is_gas = _b(is_gas)                    # NEW
        self.is_oil = _b(is_oil)                    # NEW
        self.is_solid_fuel = _b(is_solid_fuel)      # NEW
        self.is_off_gas = _b(is_off_gas)            # NEW

        # NEW: fast occupancy counter (maintained by PersonAgent.step)
        self.occupancy_count: int = 0  # NEW

        # NEW: precompute hourly base once (big speed win)
        self._hourly_base_kwh: float = self._compute_hourly_base_kwh()  # NEW
        # NEW: per-household heat slope (kWh per degC-hour) for climate response
        self.heat_slope_kWh_per_deg: float = self._compute_heat_slope(getattr(model, "heating_slope_kWh_per_deg", 0.05))
        # NEW: per-household heating capacity (kWh/h) for duty-cycle model
        self.heat_capacity_kWh_per_hour: float = self._compute_heat_capacity()

    # NEW: compute static hourly base from structure/levers (called once)
    def _compute_hourly_base_kwh(self) -> float:  # NEW
        """Non-climate baseline (small, meter-anchored, year-round).

        Intent:
        - Do NOT rescale baseline by EPC/SAP/envelope/fuel.
        - Keep a modest fixed load that remains in summer.
        - Property features shape heating only (handled elsewhere).
        """
        cfg = getattr(self.model, "config", None)
        level_scale = getattr(self.model, "level_scale", 1.0)
        # Baseline anchored on meter data, not EPC annual
        base = float(getattr(self.model, "baseline_anchor_kwh_per_hour", 0.4))

        # property-type multiplier (baseline map from config if present)
        pt_mult = PROPERTY_TYPE_MULT_BASE.get(self.property_type, 1.0)
        if cfg and isinstance(cfg.model, dict):
            pt_mult = cfg.model.get("property_type_mult_base", {}).get(self.property_type, pt_mult)
        base *= pt_mult

        # floor area weak sublinear scaling
        fa = self.floor_area_m2
        if fa is not None and fa > 0:
            ref = float(getattr(self.model, "baseline_area_ref_m2", 70.0))
            exp = float(getattr(self.model, "baseline_area_exp", 0.20))
            lo, hi = getattr(self.model, "baseline_area_clip", (0.85, 1.25))
            area_mult = max(lo, min(hi, (fa / ref) ** exp))
            base *= area_mult

        hourly = base * level_scale
        # cap baseline to avoid unrealistic power draw
        max_base = getattr(self.model, "max_base_kwh_per_hour", None)
        if max_base is not None:
            hourly = min(hourly, float(max_base))
        return hourly

    def _compute_heat_slope(self, base_slope: float) -> float:
        """Per-household temperature sensitivity (heating slope). Structure affects slope, not annual anchor."""
        slope = float(base_slope)
        cfg = getattr(self.model, "config", None)

        # SAP: gentle modulation
        if self.sap_rating < 50:
            slope *= 1.10
        elif self.sap_rating > 80:
            slope *= 0.90

        # Property type multiplier (bounded, from config if present)
        pt_mult = PROPERTY_TYPE_MULT_HEAT.get(self.property_type, 1.0)
        if cfg and isinstance(cfg.model, dict):
            pt_mult = cfg.model.get("pt_heat_mult", {}).get(self.property_type, cfg.model.get("pt_heat_mult", {}).get("default", pt_mult))
        slope *= pt_mult

        # Floor area sublinear scaling
        area_exp = getattr(self.model, "heat_slope_area_exp", 0.6)
        if self.floor_area_m2 is not None and self.floor_area_m2 > 0:
            slope *= max(0.7, min(1.6, (self.floor_area_m2 / 90.0) ** area_exp))
        elif self.size_band is not None and (self.floor_area_m2 is None or self.floor_area_m2 <= 0):
            try:
                sb = int(self.size_band)
                if cfg:
                    slope *= float(cfg.households.get("bedroom_multiplier", {}).get(sb, 1.0))
            except Exception:
                pass

        # Envelope quality: better envelope lowers slope (up to 20%)
        if self.retrofit_envelope_score is not None:
            env_mult = 1.0 - 0.20 * max(0.0, min(1.0, self.retrofit_envelope_score))
            slope *= env_mult

        # Heating system nudges
        fuel = (self.main_fuel_type or "")
        heat = (self.main_heating_system or "")
        systems_cfg = cfg.systems if cfg else {}
        sys_mult = None
        if "heat pump" in heat and "heat_pump" in systems_cfg:
            sys_mult = systems_cfg.get("heat_pump", {}).get("heating_slope_mult", 0.70)
        elif "electric" in fuel and "electric_heating" in systems_cfg:
            sys_mult = systems_cfg.get("electric_heating", {}).get("heating_slope_mult", 1.00)
        elif "gas" in fuel and "gas_boiler" in systems_cfg:
            sys_mult = systems_cfg.get("gas_boiler", {}).get("heating_slope_mult", 1.00)
        if sys_mult is not None:
            slope *= sys_mult

        # Clamp
        s_min = getattr(self.model, "heat_slope_min", 0.0)
        s_max = getattr(self.model, "heat_slope_max", 0.10)
        return max(s_min, min(s_max, slope))

    def _compute_heat_capacity(self) -> float:
        """Compute per-dwelling heating capacity (kWh/h), sublinear in area, bounded."""
        cfg = getattr(self.model, "config", None)
        base_cap = float(getattr(self.model, "base_heat_capacity", 8.0))
        cap = base_cap

        # property type multiplier (reuse pt_heat_mult where available)
        if cfg and isinstance(cfg.model, dict):
            cap *= cfg.model.get("pt_heat_mult", {}).get(self.property_type, cfg.model.get("pt_heat_mult", {}).get("default", 1.0))
        else:
            cap *= PROPERTY_TYPE_MULT_HEAT.get(self.property_type, 1.0)

        # floor area scaling (sublinear)
        area_exp = getattr(self.model, "heat_capacity_area_exp", 0.5)
        if self.floor_area_m2 is not None and self.floor_area_m2 > 0:
            cap *= max(0.7, min(1.6, (self.floor_area_m2 / 90.0) ** area_exp))

        # system type nudges
        heat = (self.main_heating_system or "").lower()
        fuel = (self.main_fuel_type or "").lower()
        if "heat pump" in heat:
            cap *= 0.9
        elif "electric" in fuel:
            cap *= 0.9
        elif "oil" in fuel:
            cap *= 1.0
        elif "solid" in fuel:
            cap *= 1.1

        # bounds
        min_cap = float(getattr(self.model, "min_heat_capacity", 4.0))
        max_cap = float(getattr(self.model, "max_heat_kwh_per_hour", 20.0))
        return max(min_cap, min(max_cap, cap))

    def refresh_hourly_base(self) -> None:  # NEW: call if levers change mid-run
        self._hourly_base_kwh = self._compute_hourly_base_kwh()  # NEW
        self.heat_capacity_kWh_per_hour = self._compute_heat_capacity()

    # ------------------------------------------------------------------
    #  Convenience helpers used by the model each tick
    # ------------------------------------------------------------------

    def reset_energy(self) -> None:
        self.energy_consumption = 0.0
        self.climate_heating_kWh = 0.0
        self.climate_cooling_kWh = 0.0
        self.base_kwh = 0.0
        self.heat_kwh = 0.0
        self.spike_kwh = 0.0
        self.cap_clip_total = 0.0
        self.cap_clip_base = 0.0
        self.cap_clip_heat = 0.0
        self.cap_clip_spike = 0.0

    def calc_base_energy(self) -> float:
        # NEW: return cached hourly base (computed once)
        return self._hourly_base_kwh  # NEW

    # ------------------------------------------------------------------
    #  Climate integration – called by the model
    # ------------------------------------------------------------------

    def set_climate_index(self, idx: int) -> None:
        self.clim_idx = int(idx)

    def apply_climate(
        self,
        tempC: float,
        *,
        heating_setpoint: float,
        cooling_threshold: float,
        heat_slope: Optional[float],
        cool_slope: float,
        occupancy: Optional[int] = None,) -> None:
        self.ambient_tempC = float(tempC)
        if not math.isfinite(self.ambient_tempC):
            self.climate_heating_kWh = 0.0
            self.climate_cooling_kWh = 0.0
            self.heat_kwh = 0.0
            return

        db = 0.5  # NEW: thermostat deadband (°C)
        hd = max(0.0, (heating_setpoint - self.ambient_tempC) - db)
        cd = max(0.0, (self.ambient_tempC - cooling_threshold) - db)

        # Demand severity (dimensionless) based on slope and temperature gap
        base_slope = float(heat_slope) if heat_slope is not None else float(self.heat_slope_kWh_per_deg)
        eff_heat_slope = base_slope * (self.hp_effect_mult if self.has_heatpump else 1.0)
        loss_index = hd * eff_heat_slope

        # Duty cycle (0-1), soft saturation
        K = float(getattr(self.model, "loss_to_duty_k", 3.0))
        duty = loss_index / (loss_index + K) if loss_index > 0 else 0.0
        duty = max(0.0, min(1.0, duty))

        # Capacity-based heating (kWh/h)
        heat_capacity = self.heat_capacity_kWh_per_hour
        heat = duty * heat_capacity

        cool = cd * float(cool_slope)

        # Hard cap as safety net
        max_heat = getattr(self.model, "max_heat_kwh_per_hour", None)
        if max_heat is not None:
            heat = min(heat, float(max_heat))

        # Optional: dampen when nobody is home (simple heuristic)
        if occupancy is not None and occupancy <= 0:
            heat *= 0.5
            cool *= 0.5

        self.heat_kwh = heat
        self.climate_heating_kWh = heat
        self.climate_cooling_kWh = cool
        self.energy_consumption += heat + cool


# ────────────────────────────────────────────────────────────────────
#  PersonAgent
# ────────────────────────────────────────────────────────────────────

class PersonAgent(mesa.Agent):
    """Individual resident whose presence drives stochastic load spikes."""

    def __init__(
        self,
        unique_id: str,
        model: "mesa.Model",
        home: HouseholdAgent,
        *,
        schedule_profile: str = "unknown",
        leave_hour: Optional[int] = None,
        return_hour: Optional[int] = None,
        wealth: Optional[str] = None,
        sap: Optional[float] = None,
    ) -> None:
        super().__init__(model=model)

        self.unique_id: str = unique_id
        self.home: HouseholdAgent = home

        self.schedule_profile: str = schedule_profile
        self.leave_hour: Optional[int] = leave_hour
        self.return_hour: Optional[int] = return_hour
        self.at_home: bool = True   # updated each tick

        self.wealth: str = wealth or "medium"
        self.sap: float = sap if sap is not None else home.sap_rating

        self.energy: float = 0.0

    def step(self) -> None:
        """Update presence status and add corresponding kWh to household."""
        hour = self.model.local_hour() if hasattr(self.model, "local_hour") else self.model.current_hour % 24

        # presence logic with occupancy counter updates  # NEW
        if self.leave_hour is None or self.return_hour is None:
            self.at_home = True
        else:
            if self.at_home and hour == self.leave_hour:
                self.at_home = False
                self.home.occupancy_count -= 1   # NEW
            elif (not self.at_home) and hour == self.return_hour:
                self.at_home = True
                self.home.occupancy_count += 1   # NEW

        # energy spike
        base_spike = self.model.energy_per_person_home
        if self.wealth == "high":
            base_spike *= 1.3
        elif self.wealth == "low":
            base_spike *= 0.8
        if self.sap < 50:
            base_spike *= 1.2
        elif self.sap > 80:
            base_spike *= 0.8

        if self.at_home:
            self.home.spike_kwh += base_spike
            self.home.energy_consumption += base_spike
            self.energy = base_spike
        else:
            standby = self.model.energy_per_person_away
            self.home.spike_kwh += standby
            self.home.energy_consumption += standby
            self.energy = standby
