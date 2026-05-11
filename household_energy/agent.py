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
from typing import Dict, List, Optional

import mesa
import mesa_geo as mg
import pandas as pd
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
        # mesa-geo API differs across versions:
        # - older: GeoAgent(model=..., geometry=..., crs=...)
        # - newer: GeoAgent(unique_id=..., model=..., geometry=..., crs=...)
        try:
            super().__init__(unique_id=unique_id, model=model, geometry=geometry, crs=crs)
        except TypeError:
            super().__init__(model=model, geometry=geometry, crs=crs)

        def _clean_text(v: object) -> Optional[str]:
            if v is None or pd.isna(v):
                return None
            s = str(v).strip().lower()
            return s or None

        # identity & static attributes
        self.unique_id: str = unique_id
        self.property_type: str = _clean_text(property_type) or ""
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
        self.hh_income_band: Optional[str] = _clean_text(hh_income_band)                                     # NEW
        self.hh_edu_detail: Optional[str] = _clean_text(hh_edu_detail)                                       # NEW
        self.dwelling_bucket: Optional[str] = _clean_text(dwelling_bucket)                                   # NEW
        self.tenure: Optional[str] = _clean_text(tenure)                                                     # NEW
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
        # Annual rollups tracked by the model each tick (avoids huge agent_dc frames)
        self.annual_kwh_by_year: dict[int, float] = {}

        # residents
        self.residents: List["PersonAgent"] = []

        # --- climate state (populated/used by the model) -----------
        self.clim_idx: Optional[int] = None
        self.ambient_tempC: float = float("nan")
        self.climate_heating_kWh: float = 0.0
        self.climate_cooling_kWh: float = 0.0

        # NEW: attach core drivers (kept raw; used in calc/reporters)
        self.floor_area_m2 = None if floor_area_m2 is None else float(floor_area_m2)    # NEW
        self.property_age = _clean_text(property_age)  # NEW
        self.main_fuel_type = _clean_text(main_fuel_type)  # NEW
        self.main_heating_system = _clean_text(main_heating_system)  # NEW
        self.is_heatpump_candidate = 1 if (is_heatpump_candidate or 0) else 0  # NEW
        self.heatpump_candidate_class = _clean_text(heatpump_candidate_class)  # NEW
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
        sap_mult   = self._sap_multiplier(kind="slope", sap_value=sap)
        retro_mult = 1.10 - 0.20 * max(0.0, min(1.0, retro))
        area_mult  = max(0.7, min(1.6, fa / 90.0))
        rng = __import__("random").Random(hash(str(self.unique_id)) & 0xFFFFFFFF)
        noise_mult = max(0.75, min(1.25, rng.normalvariate(1.0, 0.10)))

        self.heat_slope_kWh_per_deg = (
            self.model.heating_slope_kWh_per_deg * p_mult * sap_mult * retro_mult * area_mult * noise_mult
        )

        # heat-pump effectiveness vs boiler (simple, deterministic)
        self.hp_effect_mult = getattr(self.model, "boiler_efficiency", 0.90) / getattr(self.model, "heatpump_cop_ref", 2.8)

        # Precompute SAP helpers
        self._sap_params_cache = self._sap_params()
        self.sap_idx = self._sap_index(self.sap_rating)
        self.sap_spike_mult = self._sap_multiplier(kind="spike", sap_value=self.sap_rating)
        # -----------------------------------------------------------


        # NEW: policy levers (coerce to 0/1 where appropriate)
        def _b(v):  # NEW
            try:
                return int(v) if v is not None else 0
            except Exception:
                return 0

        self.heating_controls = _clean_text(heating_controls)  # NEW
        self.meter_type = _clean_text(meter_type)  # NEW
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
        self._hourly_base_electric_kwh, self._hourly_base_gas_kwh = self._compute_hourly_base_components()
        self._hourly_base_kwh: float = self._hourly_base_electric_kwh + self._hourly_base_gas_kwh
        # NEW: per-household heat slope (kWh per degC-hour) for climate response
        self.heat_slope_kWh_per_deg: float = self._compute_heat_slope(getattr(model, "heating_slope_kWh_per_deg", 0.05))
        # NEW: per-household heating capacity (kWh/h) for duty-cycle model
        self.heat_capacity_kWh_per_hour: float = self._compute_heat_capacity()
        # Cache fuel-split routing once; this is used in person-hour hot paths.
        self._refresh_fuel_split_cache()

    def _baseline_area_multiplier(self) -> float:
        """Weak sublinear baseline area scaling shared by fuel baselines."""
        fa = self.floor_area_m2
        if fa is None or fa <= 0:
            return 1.0
        ref = float(getattr(self.model, "baseline_area_ref_m2", 70.0))
        exp = float(getattr(self.model, "baseline_area_exp", 0.20))
        lo, hi = getattr(self.model, "baseline_area_clip", (0.85, 1.25))
        return max(lo, min(hi, (fa / ref) ** exp))

    # NEW: compute static hourly baseline components from structure/levers (called once)
    def _compute_hourly_base_components(self) -> tuple[float, float]:
        """Return baseline components: (electric_kWh/h, gas_kWh/h).

        Intent:
        - Do NOT rescale baseline by EPC/SAP/envelope/fuel.
        - Keep a modest fixed load that remains in summer.
        - Property features shape heating only (handled elsewhere).
        """
        level_scale = getattr(self.model, "level_scale", 1.0)
        pt_mult_map = getattr(self.model, "property_type_mult_base", PROPERTY_TYPE_MULT_BASE)
        pt_mult_map_e = getattr(self.model, "property_type_mult_base_electric", pt_mult_map)
        pt_mult_map_g = getattr(self.model, "property_type_mult_base_gas", pt_mult_map)
        pt_mult = pt_mult_map.get(self.property_type, pt_mult_map.get("default", 1.0))
        pt_mult_e = pt_mult_map_e.get(self.property_type, pt_mult_map_e.get("default", pt_mult))
        pt_mult_g = pt_mult_map_g.get(self.property_type, pt_mult_map_g.get("default", pt_mult))
        area_mult = self._baseline_area_multiplier()

        if not bool(getattr(self.model, "use_separate_fuel_baseline_anchors", False)):
            base_total = float(getattr(self.model, "baseline_anchor_kwh_per_hour", 0.4))
            base_total *= pt_mult * area_mult * level_scale
            gas_share = self._resolve_base_gas_share()
            gas_kwh = base_total * gas_share
            elec_kwh = base_total - gas_kwh
        else:
            base_elec = float(getattr(self.model, "baseline_anchor_elec_kwh_per_hour", 0.0))
            base_gas = float(getattr(self.model, "baseline_anchor_gas_kwh_per_hour", 0.0))
            elec_kwh = base_elec * pt_mult_e * area_mult * level_scale
            gas_kwh = (
                base_gas * pt_mult_g * area_mult * level_scale
                if self._resolve_heating_fuel_bucket() == "gas"
                else 0.0
            )

        hourly = elec_kwh + gas_kwh
        max_base = getattr(self.model, "max_base_kwh_per_hour", None)
        if max_base is not None:
            max_base = float(max_base)
            if hourly > max_base and hourly > 0:
                scale = max_base / hourly
                elec_kwh *= scale
                gas_kwh *= scale
        return max(0.0, elec_kwh), max(0.0, gas_kwh)

    def _compute_hourly_base_kwh(self) -> float:
        elec_kwh, gas_kwh = self._compute_hourly_base_components()
        return elec_kwh + gas_kwh

    def _compute_heat_slope(self, base_slope: float) -> float:
        """Per-household temperature sensitivity (heating slope). Structure affects slope, not annual anchor."""
        slope = float(base_slope)
        cfg = getattr(self.model, "config", None)

        def _ptype_archetype(ptype: str | None) -> str:
            t = (ptype or "").strip().lower()
            if "detached" in t and "semi" not in t:
                return "detached"
            if "semi-detached" in t or "semi detached" in t:
                return "semi-detached"
            if "terraced" in t:
                return "terraced"
            if "flat" in t or "flats" in t:
                return "flat"
            return "default"

        # SAP: gentle modulation
        if self.sap_rating < 50:
            slope *= 1.10
        elif self.sap_rating > 80:
            slope *= 0.90

        # Property type multiplier (bounded, prefer model-config override)
        pt_mult_map = getattr(self.model, "property_type_mult_heat", PROPERTY_TYPE_MULT_HEAT)
        arche = _ptype_archetype(self.property_type)
        pt_mult = pt_mult_map.get(self.property_type)
        if pt_mult is None:
            pt_mult = pt_mult_map.get(arche)
        if pt_mult is None:
            pt_mult = pt_mult_map.get("default", 1.0)
        slope *= pt_mult

        # SAP scaling (linear)
        slope *= self._sap_multiplier(kind="slope", sap_value=self.sap_rating)

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
        if self._is_communal_system():
            sys_mult = systems_cfg.get("communal", {}).get("heating_slope_mult", 0.85)
        elif "heat pump" in heat and "heat_pump" in systems_cfg:
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
        pt_mult_map = getattr(self.model, "property_type_mult_heat", PROPERTY_TYPE_MULT_HEAT)
        t = (self.property_type or "").strip().lower()
        if "detached" in t and "semi" not in t:
            arche = "detached"
        elif "semi-detached" in t or "semi detached" in t:
            arche = "semi-detached"
        elif "terraced" in t:
            arche = "terraced"
        elif "flat" in t or "flats" in t:
            arche = "flat"
        else:
            arche = "default"
        pt_mult = pt_mult_map.get(self.property_type)
        if pt_mult is None:
            pt_mult = pt_mult_map.get(arche)
        if pt_mult is None:
            pt_mult = pt_mult_map.get("default", 1.0)
        cap *= pt_mult

        # SAP scaling (linear)
        cap *= self._sap_multiplier(kind="cap", sap_value=self.sap_rating)

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

    # ------------------------------------------------------------------
    #  SAP helpers
    # ------------------------------------------------------------------
    def _sap_params(self):
        defaults = {
            "sap_lo": 40.0, "sap_hi": 90.0,
            "slope_mult_hi": 1.30, "slope_mult_lo": 0.70,
            "cap_mult_hi": 1.15, "cap_mult_lo": 0.85,
            "spike_mult_hi": 1.20, "spike_mult_lo": 0.80,
        }
        if hasattr(self, "_sap_params_cache"):
            return self._sap_params_cache
        cfg = getattr(self.model, "config", None)
        params = dict(defaults)
        if cfg and isinstance(getattr(cfg, "model", {}), dict):
            params.update(cfg.model.get("sap_scaling", {}))
        self._sap_params_cache = params
        return params

    def _sap_index(self, sap_value) -> float:
        params = self._sap_params()
        sap = float(sap_value) if sap_value is not None else 70.0
        sap = max(params["sap_lo"], min(params["sap_hi"], sap))
        return (sap - params["sap_lo"]) / (params["sap_hi"] - params["sap_lo"])

    def _sap_multiplier(self, kind: str, sap_value=None) -> float:
        params = self._sap_params()
        idx = self._sap_index(sap_value)
        if kind == "slope":
            hi, lo = params["slope_mult_hi"], params["slope_mult_lo"]
        elif kind == "cap":
            hi, lo = params["cap_mult_hi"], params["cap_mult_lo"]
        elif kind == "spike":
            hi, lo = params["spike_mult_hi"], params["spike_mult_lo"]
        else:
            hi = lo = 1.0
        return hi + (lo - hi) * idx

    def refresh_hourly_base(self) -> None:  # NEW: call if levers change mid-run
        self._hourly_base_electric_kwh, self._hourly_base_gas_kwh = self._compute_hourly_base_components()
        self._hourly_base_kwh = self._hourly_base_electric_kwh + self._hourly_base_gas_kwh
        self.heat_capacity_kWh_per_hour = self._compute_heat_capacity()
        self._refresh_fuel_split_cache()

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
        # Fuel-split tracking (additive; does not replace energy_consumption)
        self.electric_kwh = 0.0
        self.gas_kwh = 0.0
        self.other_kwh = 0.0
        self.cap_clip_total = 0.0
        self.cap_clip_base = 0.0
        self.cap_clip_heat = 0.0
        self.cap_clip_spike = 0.0

    @staticmethod
    def _norm_token(value) -> str:
        return str(value or "").strip().lower()

    def _model_cfg(self) -> dict:
        cfg = getattr(self.model, "config", None)
        return cfg.model if cfg else {}

    def _is_communal_system(self) -> bool:
        model_cfg = self._model_cfg()
        labels = model_cfg.get("communal_system_labels", ["communal", "district"])
        label_set = {self._norm_token(v) for v in labels}
        return self._norm_token(self.main_heating_system) in label_set

    def _resolve_heating_fuel_bucket(self) -> str:
        """Classify heating energy bucket via explicit maps: electric, gas, or other."""
        model_cfg = self._model_cfg()
        fuel = self._norm_token(self.main_fuel_type)
        heat = self._norm_token(self.main_heating_system)

        combo_map = {self._norm_token(k): self._norm_token(v) for k, v in model_cfg.get("heating_fuel_combo_map", {}).items()}
        system_map = {self._norm_token(k): self._norm_token(v) for k, v in model_cfg.get("heating_system_bucket_map", {}).items()}
        fuel_map = {self._norm_token(k): self._norm_token(v) for k, v in model_cfg.get("fuel_type_bucket_map", {}).items()}

        bucket = combo_map.get(f"{fuel}|{heat}")
        if bucket is None:
            bucket = system_map.get(heat)
        if bucket is None:
            bucket = fuel_map.get(fuel)

        # Deterministic fallback only when no explicit mapping is provided.
        if bucket is None:
            if getattr(self, "has_heatpump", False) or getattr(self, "is_electric_heating", 0):
                bucket = "electric"
            elif getattr(self, "is_gas", 0):
                bucket = "gas"
            elif getattr(self, "is_oil", 0) or getattr(self, "is_solid_fuel", 0):
                bucket = "other"
            else:
                bucket = "other"

        if bucket not in {"electric", "gas", "other"}:
            bucket = "other"
        return bucket

    def _resolve_base_gas_share(self) -> float:
        """Share of baseline load allocated to gas for gas-heated homes."""
        if bool(getattr(self.model, "use_separate_fuel_baseline_anchors", False)):
            return 0.0
        if self._resolve_heating_fuel_bucket() != "gas":
            return 0.0
        model_cfg = self._model_cfg()
        share = float(model_cfg.get("gas_base_share", 0.0))
        if self._is_communal_system():
            share = float(model_cfg.get("gas_base_share_communal", share))
        return max(0.0, min(1.0, share))

    def _resolve_gas_spike_share(self) -> float:
        """Share of spikes allocated to gas for gas-heated homes (hot water usage)."""
        if self._resolve_heating_fuel_bucket() != "gas":
            return 0.0
        model_cfg = self._model_cfg()
        share = float(model_cfg.get("gas_spike_share", 0.0))
        if self._is_communal_system():
            share = float(model_cfg.get("gas_spike_share_communal", share))
        # clamp to [0,1]
        return max(0.0, min(1.0, share))

    def _refresh_fuel_split_cache(self) -> None:
        bucket = self._resolve_heating_fuel_bucket()
        self._cached_heating_bucket = bucket
        if bucket != "gas":
            self._cached_base_gas_share = 0.0
            self._cached_gas_spike_share = 0.0
            return

        model_cfg = self._model_cfg()
        is_communal = self._is_communal_system()

        if bool(getattr(self.model, "use_separate_fuel_baseline_anchors", False)):
            base_share = 0.0
        else:
            base_share = float(model_cfg.get("gas_base_share", 0.0))
        spike_share = float(model_cfg.get("gas_spike_share", 0.0))
        if is_communal:
            if not bool(getattr(self.model, "use_separate_fuel_baseline_anchors", False)):
                base_share = float(model_cfg.get("gas_base_share_communal", base_share))
            spike_share = float(model_cfg.get("gas_spike_share_communal", spike_share))

        self._cached_base_gas_share = max(0.0, min(1.0, base_share))
        self._cached_gas_spike_share = max(0.0, min(1.0, spike_share))

    def _heating_fuel_bucket(self) -> str:
        bucket = getattr(self, "_cached_heating_bucket", None)
        if bucket is None:
            self._refresh_fuel_split_cache()
            bucket = self._cached_heating_bucket
        return bucket

    def _base_gas_share(self) -> float:
        share = getattr(self, "_cached_base_gas_share", None)
        if share is None:
            self._refresh_fuel_split_cache()
            share = self._cached_base_gas_share
        return float(share)

    def _gas_spike_share(self) -> float:
        share = getattr(self, "_cached_gas_spike_share", None)
        if share is None:
            self._refresh_fuel_split_cache()
            share = self._cached_gas_spike_share
        return float(share)
    def calc_base_energy(self) -> float:
        # NEW: return cached hourly base (computed once)
        return self._hourly_base_kwh  # NEW

    def calc_base_electric_energy(self) -> float:
        return float(getattr(self, "_hourly_base_electric_kwh", 0.0))

    def calc_base_gas_energy(self) -> float:
        return float(getattr(self, "_hourly_base_gas_kwh", 0.0))

    def add_person_load(self, wealth: str, at_home: bool, awake: bool = True) -> float:
        """Apply one person's occupancy-driven load and route it by fuel bucket."""
        if at_home:
            load = float(getattr(self.model, "energy_per_person_home", 0.06))
            if awake:
                load *= float(getattr(self.model, "awake_home_spike_mult", 1.0))
            else:
                load *= float(getattr(self.model, "sleep_home_spike_mult", 0.3))
        else:
            load = float(getattr(self.model, "energy_per_person_away", 0.01))

        wealth_mult = {
            "very_low": 0.75,
            "low": 0.9,
            "mid": 1.0,
            "high": 1.15,
            "very_high": 1.3,
        }.get(wealth, 1.0)
        load *= wealth_mult

        if hasattr(self, "sap_spike_mult"):
            load *= self.sap_spike_mult

        self.spike_kwh += load
        self.energy_consumption += load
        share = self._gas_spike_share()
        gas_add = load * share
        self.gas_kwh += gas_add
        self.electric_kwh += (load - gas_add)
        return load

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

        # Temperature gaps with a small deadband
        db = 0.5  # thermostat deadband (°C)
        hd = max(0.0, (heating_setpoint - self.ambient_tempC) - db)
        cd = max(0.0, (self.ambient_tempC - cooling_threshold) - db)

        # Slope and duty cycle
        base_slope = float(heat_slope) if heat_slope is not None else float(self.heat_slope_kWh_per_deg)
        eff_heat_slope = base_slope * (self.hp_effect_mult if self.has_heatpump else 1.0)
        loss_index = hd * eff_heat_slope
        K = float(getattr(self.model, "loss_to_duty_k", 3.0))
        duty = loss_index / (loss_index + K) if loss_index > 0 else 0.0
        duty = max(0.0, min(1.0, duty))

        # Capacity-based heating & cooling
        heat = duty * self.heat_capacity_kWh_per_hour
        cool = cd * float(cool_slope)

        # Safety caps
        max_heat = getattr(self.model, "max_heat_kwh_per_hour", None)
        if max_heat is not None:
            heat = min(heat, float(max_heat))

        if occupancy is not None:
            n_residents = max(1, len(self.residents))
            occ_share = max(0.0, min(1.0, float(occupancy) / float(n_residents)))
            away_floor = float(getattr(self.model, "heating_occupancy_away_mult", 0.5))
            away_floor = max(0.0, min(1.0, away_floor))
            occ_mult = away_floor + (1.0 - away_floor) * occ_share
            heat *= occ_mult
            cool *= occ_mult

        self.heat_kwh = heat
        self.climate_heating_kWh = heat
        self.climate_cooling_kWh = cool
        self.energy_consumption += heat + cool
        # Assign heating kWh to fuel bucket (cooling treated as electric by default)
        bucket = self._heating_fuel_bucket()
        if bucket == "electric":
            self.electric_kwh += heat + cool
        elif bucket == "gas":
            self.gas_kwh += heat
            self.electric_kwh += cool
        else:
            self.other_kwh += heat
            self.electric_kwh += cool


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
        wake_hour: Optional[int] = None,
        sleep_hour: Optional[int] = None,
        wealth: Optional[str] = None,
        sap: Optional[float] = None,
    ) -> None:
        # mesa Agent API also differs by version.
        try:
            super().__init__(unique_id=unique_id, model=model)
        except TypeError:
            super().__init__(model=model)

        self.unique_id: str = unique_id
        self.home: HouseholdAgent = home

        self.schedule_profile: str = schedule_profile
        self.leave_hour: Optional[int] = leave_hour
        self.return_hour: Optional[int] = return_hour
        self.wake_hour: Optional[int] = wake_hour
        self.sleep_hour: Optional[int] = sleep_hour
        self.at_home: bool = True   # updated each tick
        self.awake: bool = True

        self.wealth: str = wealth or "medium"
        self.sap: float = sap if sap is not None else home.sap_rating

        self.energy: float = 0.0

    @staticmethod
    def _is_awake_at_hour(hour: int, wake_hour: Optional[int], sleep_hour: Optional[int]) -> bool:
        if wake_hour is None or sleep_hour is None:
            return True
        w = int(wake_hour) % 24
        s = int(sleep_hour) % 24
        h = int(hour) % 24
        if w == s:
            return True
        if w < s:
            return (h >= w) and (h < s)
        return (h >= w) or (h < s)

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

        self.awake = self._is_awake_at_hour(hour, self.wake_hour, self.sleep_hour)
        self.energy = self.home.add_person_load(self.wealth, self.at_home, self.awake)
