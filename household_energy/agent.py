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
# ────────────────────────────────────────────────────────────────────

PROPERTY_TYPE_MULTIPLIER: Dict[str, float] = {
    "mid-terraced house": 1.00,
    "semi-detached house": 1.25,
    "small block of flats/dwelling converted in to flats": 0.80,
    "large block of flats": 0.70,
    "block of flats": 0.70,
    "end-terraced house": 1.05,
    "detached house": 1.80,
    "flat in mixed use building": 0.85,
}

PROPERTY_TYPES: List[str] = list(PROPERTY_TYPE_MULTIPLIER.keys())

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
        self.retrofit_envelope_score = None if retrofit_envelope_score is None else float(retrofit_envelope_score)  # NEW
        self.imd_decile = None if imd_decile is None else float(imd_decile)  # NEW

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

    # NEW: compute static hourly base from structure/levers (called once)
    def _compute_hourly_base_kwh(self) -> float:  # NEW
        base = float(getattr(self, "annual_energy_kwh", self.energy_demand))
        # SAP adjustment
        if self.sap_rating < 50:
            base *= 1.20
        elif self.sap_rating > 80:
            base *= 0.80
        # archetype
        base *= PROPERTY_TYPE_MULTIPLIER.get(self.property_type, 1.0)
        # floor area scaling
        if self.floor_area_m2 is not None and self.floor_area_m2 > 0:
            scale = max(0.6, min(2.0, self.floor_area_m2 / 90.0))
            base *= scale
        # envelope quality (0–1 → up to -20%)
        if self.retrofit_envelope_score is not None:
            env_mult = 1.0 - 0.20 * max(0.0, min(1.0, self.retrofit_envelope_score))
            base *= env_mult
        # heating system / fuel nudges
        fuel = (self.main_fuel_type or "")
        heat = (self.main_heating_system or "")
        if "electric" in fuel:
            base *= 1.05
        if "heat pump" in heat:
            base *= 0.85
        # policy levers (stackable)
        lever_mult = 1.0
        if self.cwi_flag:      lever_mult *= 0.92
        if self.swi_flag:      lever_mult *= 0.90
        if self.loft_ins_flag: lever_mult *= 0.95
        if self.floor_ins_flag:lever_mult *= 0.96
        if self.glazing_flag:  lever_mult *= 0.96
        hc = (self.heating_controls or "")
        if "programmer and thermostat" in hc:
            lever_mult *= 0.97
        elif "programmer only" in hc:
            lever_mult *= 0.99
        if (self.meter_type or "").startswith("smart"):
            lever_mult *= 0.98
        if self.is_off_gas:
            lever_mult *= 1.05
        base *= lever_mult
        return base / 365 / 24

    def refresh_hourly_base(self) -> None:  # NEW: call if levers change mid-run
        self._hourly_base_kwh = self._compute_hourly_base_kwh()  # NEW

    # ------------------------------------------------------------------
    #  Convenience helpers used by the model each tick
    # ------------------------------------------------------------------

    def reset_energy(self) -> None:
        self.energy_consumption = 0.0
        self.climate_heating_kWh = 0.0
        self.climate_cooling_kWh = 0.0

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
        heat_slope: float,
        cool_slope: float,
        occupancy: Optional[int] = None,) -> None:
        self.ambient_tempC = float(tempC)
        if not math.isfinite(self.ambient_tempC):
            self.climate_heating_kWh = 0.0
            self.climate_cooling_kWh = 0.0
            return

        hd = max(0.0, heating_setpoint - self.ambient_tempC)
        cd = max(0.0, self.ambient_tempC - cooling_threshold)

        heat = hd * float(heat_slope)
        cool = cd * float(cool_slope)

        # Optional: dampen when nobody is home (simple heuristic)
        if occupancy is not None and occupancy <= 0:
            heat *= 0.5
            cool *= 0.5

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
            self.home.energy_consumption += base_spike
            self.energy = base_spike
        else:
            standby = self.model.energy_per_person_away
            self.home.energy_consumption += standby
            self.energy = standby
