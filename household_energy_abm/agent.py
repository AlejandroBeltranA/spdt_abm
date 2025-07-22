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

"""

from __future__ import annotations

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

        # list populated when PersonAgents are instantiated
        self.residents: List["PersonAgent"] = []

    # ------------------------------------------------------------------
    #  Convenience helpers used by the model each tick
    # ------------------------------------------------------------------

    def reset_energy(self) -> None:
        """Clear the per-hour accumulator before a new model step."""
        self.energy_consumption = 0.0

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
        hour = self.model.current_hour % 24

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
