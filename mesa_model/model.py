"""
model.py
========

Core Mesa/mesa-geo **EnergyModel** that coordinates:

1. A *HouseholdAgent* for every building polygon in the supplied GeoJSON
2. A configurable number of *PersonAgent* residents per household

Each tick represents **one hour**.  The model:

* resets base load for each dwelling,
* steps every PersonAgent (in/out, adds load spikes),
* aggregates results by property type and wealth group,
* records per-step metrics via Mesa’s DataCollector.

Outputs can be live-streamed to Solara widgets or written to Parquet/CSV for
offline analysis (see `run.py` and `analyze.py`).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List

import geopandas as gpd
import mesa
import mesa_geo as mg

from agent import (
    HouseholdAgent,
    PersonAgent,
    PROPERTY_TYPES,
    SCHEDULE_PROFILES,
)

# ────────────────────────────────────────────────────────────────────
#  EnergyModel
# ────────────────────────────────────────────────────────────────────


class EnergyModel(mesa.Model):
    """Agent-based model of hourly residential energy demand.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Buildings layer.  Must contain at least *fid* and *geometry* columns.
    n_residents_func : Callable[[HouseholdAgent], int], default ``lambda h: 2``
        Strategy that determines how many residents are created for each
        dwelling.  Allows heterogeneous population if needed.

    Public attributes
    -----------------
    household_agents : list[HouseholdAgent]
    person_agents    : list[PersonAgent]
    current_hour     : int   (running counter, 0 → …)
    cumulative_energy: float (Σ over all hours and dwellings)
    """

    # optional: used by Mesa-Geo visualisation if GeoJSON was not supplied
    geojson_regions: str = "data/ncc_neighborhood.geojson"

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        gdf= None,
        *,
        n_residents_func: Callable[[HouseholdAgent], int] = lambda _h: 2,
    ):
        super().__init__()

        # Tick counter (0, 1, …) → hour of simulation
        self.current_hour: int = 0

        # Containers for easy iteration
        self.household_agents: List[HouseholdAgent] = []
        self.person_agents: List[PersonAgent] = []

        # Spatial index for visualisation / spatial queries (mesa-geo)
        self.space = mg.GeoSpace(crs=gdf.crs)

        # Global parameters (can be tuned in dashboards)
        self.energy_per_person_home: float = 1.5  # kWh/h if person is home
        self.energy_per_person_away: float = 0.5  # kWh/h standby while away

        # Per-category accumulators (reset every tick by step())
        self.energy_by_type: Dict[str, float] = {t: 0.0 for t in PROPERTY_TYPES}
        self.energy_by_wealth: Dict[str, float] = dict.fromkeys(
            ["high", "medium", "low"], 0.0
        )

        # Total across *all* hours – useful for dashboards / KPIs
        self.cumulative_energy: float = 0.0

        # ------------- 1. instantiate households --------------------
        for _, row in gdf.iterrows():
            house = HouseholdAgent(
                unique_id=row["fid"],
                model=self,
                geometry=row["geometry"],
                property_type=row.get("property_type", ""),
                sap_rating=row.get("sap_rating", 70),
                energy_demand=row.get("energy_demand", 10_000),
                crs=gdf.crs,
            )
            self.household_agents.append(house)
            self.space.add_agents([house])

        # ------------- 2. instantiate residents ---------------------
        uid_counter = 0
        for house in self.household_agents:
            # Ensure geometry is valid & reduce to centroid for visual clarity
            house.geometry = house.geometry.buffer(0).centroid

            for _ in range(n_residents_func(house)):
                profile = random.choice(SCHEDULE_PROFILES)
                wealth  = random.choice(["high", "medium", "low"])

                person = PersonAgent(
                    unique_id=f"{house.unique_id}_{uid_counter}",
                    model=self,
                    home=house,
                    schedule_profile=profile["name"],
                    leave_hour=profile["leave"],
                    return_hour=profile["return"],
                    wealth=wealth,
                    sap=house.sap_rating,
                )
                self.person_agents.append(person)
                house.residents.append(person)
                uid_counter += 1

        # ------------- 3. DataCollector set-up ----------------------
        # helper closures so we don’t capture loop vars late
        make_type_getter = lambda p: (lambda m: m.energy_by_type.get(p, 0))
        make_wealth_getter = lambda grp: (lambda m: m.energy_by_wealth.get(grp, 0))

        model_reporters = {
            **{t: make_type_getter(t) for t in PROPERTY_TYPES},
            **{w: make_wealth_getter(w) for w in ["high", "medium", "low"]},
            "total_energy": lambda m: sum(
                h.energy_consumption for h in m.household_agents
            ),
            "cumulative_energy": lambda m: m.cumulative_energy,
        }

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters={
                "energy": lambda a: getattr(a, "energy", 0.0),
                "energy_consumption": lambda a: getattr(a, "energy_consumption", 0.0),
            },
        )

        # initial snapshot (t = 0)
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    #  Per-tick update
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance simulation by one hour."""
        self.current_hour += 1

        # --- 1. reset + add base load for each dwelling ------------
        for h in self.household_agents:
            h.reset_energy()
            h.energy_consumption += h.calc_base_energy()

        # --- 2. update every resident (presence + load spikes) -----
        for p in self.person_agents:
            p.step()

        # --- 3. aggregate by property type + wealth group ----------
        self.energy_by_type = {t: 0.0 for t in PROPERTY_TYPES}
        for h in self.household_agents:
            ptype = getattr(h, "property_type", "")
            if ptype in self.energy_by_type:
                self.energy_by_type[ptype] += h.energy_consumption

        self.energy_by_wealth = dict.fromkeys(["high", "medium", "low"], 0.0)
        for p in self.person_agents:
            self.energy_by_wealth[p.wealth] += p.energy

        # --- 4. cumulative total ----------------------------------
        tick_total = sum(h.energy_consumption for h in self.household_agents)
        self.cumulative_energy += tick_total

        # --- 5. record everything ---------------------------------
        self.datacollector.collect(self)
