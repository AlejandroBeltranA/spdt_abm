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

#: Multiplicative factor applied to *annual energy demand* to reflect
#: differences among EST property archetypes.
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

#: Convenience list (UI components, plots, etc.)
PROPERTY_TYPES: List[str] = list(PROPERTY_TYPE_MULTIPLIER.keys())

# ────────────────────────────────────────────────────────────────────
#  Schedule profiles (leave + return hour in local time)
# ────────────────────────────────────────────────────────────────────

#: Coarse stereotypes used to create residents with different daily routines.
SCHEDULE_PROFILES = [
    {"name": "Parent",    "leave":  7, "return": 15},
    {"name": "Worker",    "leave":  9, "return": 17},
    {"name": "Homebody",  "leave": None, "return": None},   # never leaves
]


# ────────────────────────────────────────────────────────────────────
#  HouseholdAgent
# ────────────────────────────────────────────────────────────────────

class HouseholdAgent(mg.GeoAgent):
    """Spatial agent representing one dwelling (building polygon or centroid).

    Attributes
    ----------
    energy_demand : float
        Annual kWh demand taken from EST sample (before multipliers).
    energy_consumption : float
        Running total for the *current* simulation step (hour).
        Reset to zero at the start of each tick by the model.
    residents : list[PersonAgent]
        Back-references to the PersonAgents who live here.

    Climate hooks (set/used by the model each tick)
    -----------------------------------------------
    clim_idx : Optional[int]
        Index of the nearest climate grid point (assigned once by the model).
    ambient_tempC : float
        Latest sampled outdoor temperature (°C) for this dwelling.
    climate_heating_kWh / climate_cooling_kWh : float
        Per-tick kWh added due to heating/cooling degree-hours.
    """

    def __init__(
        self,
        unique_id: str,
        model: "mesa.Model",
        geometry: BaseGeometry,
        *,
        property_type: str = "unknown",
        sap_rating: float = 70,
        energy_demand: float = 10_000,
        crs: Optional[str] = None,
    ) -> None:
        # initialise GeoAgent first (handles geometry + spatial index)
        super().__init__(model=model, geometry=geometry, crs=crs)

        # domain attributes
        self.unique_id: str = unique_id
        self.property_type: str = property_type.strip().lower()
        self.sap_rating: float = sap_rating
        self.energy_demand: float = energy_demand

        # per-tick state – cleared by model.step()
        self.energy_consumption: float = 0.0

        # residents
        self.residents: List["PersonAgent"] = []

        # --- climate state (populated/used by the model) -----------
        self.clim_idx: Optional[int] = None
        self.ambient_tempC: float = float("nan")
        self.climate_heating_kWh: float = 0.0
        self.climate_cooling_kWh: float = 0.0

    # ------------------------------------------------------------------
    #  Convenience helpers used by the model each tick
    # ------------------------------------------------------------------

    def reset_energy(self) -> None:
        """Clear the per-hour accumulator before a new model step."""
        self.energy_consumption = 0.0
        # reset climate contributions (ambient_tempC is overwritten later)
        self.climate_heating_kWh = 0.0
        self.climate_cooling_kWh = 0.0

    def calc_base_energy(self) -> float:
        """Return *hourly* base load in kWh for this property.

        Adjusts the raw EST annual demand by:
        1. SAP rating (penalise < 50  | bonus > 80)
        2. archetype multiplier (terrace vs detached …)
        """
        base = float(self.energy_demand)

        # SAP adjustment
        if self.sap_rating < 50:
            base *= 1.20
        elif self.sap_rating > 80:
            base *= 0.80

        # archetype multiplier
        base *= PROPERTY_TYPE_MULTIPLIER.get(self.property_type, 1.0)

        # convert annual kWh  →  hourly kWh
        return base / 365 / 24

    # ------------------------------------------------------------------
    #  Climate integration – called by the model
    # ------------------------------------------------------------------

    def set_climate_index(self, idx: int) -> None:
        """Remember which climate grid point this dwelling uses."""
        self.clim_idx = int(idx)

    def apply_climate(
        self,
        tempC: float,
        *,
        heating_setpoint: float,
        cooling_threshold: float,
        heat_slope: float,
        cool_slope: float,
        occupancy: Optional[int] = None,
    ) -> None:
        """Update ambient temp and add climate-driven kWh to this dwelling.

        Parameters
        ----------
        tempC : float
            Ambient outdoor temperature for this dwelling (°C).
        heating_setpoint : float
            Comfort baseline where heating demand begins (°C).
        cooling_threshold : float
            Temperature above which cooling demand begins (°C).
        heat_slope : float
            kWh added per °C of heating degree-hours.
        cool_slope : float
            kWh added per °C of cooling degree-hours.
        occupancy : Optional[int]
            Number of residents at home (if provided, can downweight loads).
        """
        self.ambient_tempC = float(tempC)
        if not math.isfinite(self.ambient_tempC):
            self.climate_heating_kWh = 0.0
            self.climate_cooling_kWh = 0.0
            return

        # degree-hours
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
    """Individual resident whose presence drives stochastic load spikes.

    Energy is *added to the household* each hour:

    * at_home  →  ``energy_per_person_home``  (scaled by wealth + SAP)
    * away     →  ``energy_per_person_away``  (standby baseline)

    The model, not the agent, aggregates these per-person spikes.
    """

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

        # identity and environment
        self.unique_id: str = unique_id
        self.home: HouseholdAgent = home

        # daily routine
        self.schedule_profile: str = schedule_profile
        self.leave_hour: Optional[int] = leave_hour
        self.return_hour: Optional[int] = return_hour
        self.at_home: bool = True   # updated each tick

        # socio-economic factors
        self.wealth: str = wealth or "medium"
        self.sap: float = sap if sap is not None else home.sap_rating

        # per-hour energy contribution (saved by DataCollector)
        self.energy: float = 0.0

    # ------------------------------------------------------------------
    #  Agent behaviour – called once per simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Update presence status and add corresponding kWh to household."""
        # Allow model to provide a local clock if available; fall back to modulo.
        hour = self.model.local_hour() if hasattr(self.model, "local_hour") else self.model.current_hour % 24

        # ---------------- presence logic ----------------
        if self.leave_hour is None or self.return_hour is None:
            self.at_home = True  # Homebody stays inside
        else:
            if self.at_home and hour == self.leave_hour:
                self.at_home = False
            elif (not self.at_home) and hour == self.return_hour:
                self.at_home = True

        # ---------------- energy spike ------------------
        base_spike = self.model.energy_per_person_home

        # wealth factor
        if self.wealth == "high":
            base_spike *= 1.3
        elif self.wealth == "low":
            base_spike *= 0.8

        # interplay with dwelling SAP (very efficient homes temper spikes)
        if self.sap < 50:
            base_spike *= 1.2
        elif self.sap > 80:
            base_spike *= 0.8

        # add to household & record own consumption
        if self.at_home:
            self.home.energy_consumption += base_spike
            self.energy = base_spike
        else:
            standby = self.model.energy_per_person_away
            self.home.energy_consumption += standby
            self.energy = standby