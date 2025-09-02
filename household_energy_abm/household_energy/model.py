"""
model.py
========

Core Mesa/mesa-geo EnergyModel coordinating:
- one HouseholdAgent per building polygon
- multiple PersonAgents per household

Each tick = 1 hour. The model:
* resets base load for each dwelling,
* steps every PersonAgent,
* samples ambient temperature and applies climate-driven kWh at each dwelling,
* aggregates by property type and wealth group,
* records per-step metrics via Mesa’s DataCollector.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import geopandas as gpd
import mesa
import mesa_geo as mg
import numpy as np
import pandas as pd  # for timezone handling

from .climate import ClimateField
from .agent import HouseholdAgent, PersonAgent, PROPERTY_TYPES, SCHEDULE_PROFILES


class EnergyModel(mesa.Model):
    """Agent-based model of hourly residential energy demand."""

    # optional: used by Mesa-Geo visualisation if GeoJSON was not supplied
    geojson_regions: str = "data/ncc_neighborhood.geojson"

    def __init__(
        self,
        gdf: gpd.GeoDataFrame | None = None,
        *,
        n_residents_func: Callable[[HouseholdAgent], int] = lambda _h: 2,
        climate_parquet: Optional[str] = None,
        climate_start: str | np.datetime64 | pd.Timestamp | None = None,
        local_tz: str = "Europe/London",
        level_scale: float = 1.0,
        collect_agent_level: bool = True
    ):
        super().__init__()

        # Tick counter (0, 1, …) → hour of simulation
        self.current_hour: int = 0

        # Containers for easy iteration
        self.household_agents: List[HouseholdAgent] = []
        self.person_agents: List[PersonAgent] = []

        # Spatial index for visualisation / spatial queries (mesa-geo)
        if gdf is None:
            raise ValueError("EnergyModel requires a GeoDataFrame `gdf`.")
        self.space = mg.GeoSpace(crs=gdf.crs)

        # Global parameters (tunable)
        self.energy_per_person_home: float = 1.5  # kWh/h if person is home
        self.energy_per_person_away: float = 0.5  # kWh/h standby while away

        # Simple climate→load coupling (tunable)
        self.heating_setpoint_C: float = 18.5
        self.cooling_threshold_C: float = 24.0
        self.heating_slope_kWh_per_deg: float = 0.10
        self.cooling_slope_kWh_per_deg: float = 0.08

        # Per-category accumulators (reset every tick by step())
        self.energy_by_type: Dict[str, float] = {t: 0.0 for t in PROPERTY_TYPES}
        self.energy_by_wealth: Dict[str, float] = dict.fromkeys(
            ["high", "medium", "low"], 0.0
        )

        # Total across *all* hours – useful for dashboards / KPIs
        self.cumulative_energy: float = 0.0

        # --- climate hook ---------------------------------
        self.climate: Optional[ClimateField] = None
        self._clim_idx_per_house: Optional[np.ndarray] = None
        self._t0: int = 0

        if climate_parquet:
            self.climate = ClimateField(climate_parquet)  # loads T×P arrays

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

        # Ensure geometry is centroided for clarity (and consistent mapping)
        for h in self.household_agents:
            h.geometry = h.geometry.buffer(0).centroid

        # ✅ Map climate ONCE (after houses exist) and assign per-house index
        if self.climate is not None:
            lats = np.fromiter(
                (h.geometry.y for h in self.household_agents),
                dtype=np.float32,
                count=len(self.household_agents),
            )
            lons = np.fromiter(
                (h.geometry.x for h in self.household_agents),
                dtype=np.float32,
                count=len(self.household_agents),
            )
            self._clim_idx_per_house = self.climate.map_households(lats, lons)
            for h, idx in zip(self.household_agents, self._clim_idx_per_house):
                h.set_climate_index(idx)

            # align start time
            if climate_start is None:
                climate_start = self.climate.times[0]
            self._t0 = self.climate.time_index_for(climate_start)

            # initialise ambient field
            for h in self.household_agents:
                h.ambient_tempC = float("nan")

        # Optional: local clock (used by PersonAgent if available)
        self._local_tz = local_tz
        self._clock0 = 0
        if climate_start is not None:
            ts0 = pd.to_datetime(climate_start, utc=True)
            self._clock0 = ts0.tz_convert(self._local_tz).hour

        # ------------- 2. instantiate residents ---------------------
        uid_counter = 0
        for house in self.household_agents:
            for _ in range(n_residents_func(house)):
                profile = random.choice(SCHEDULE_PROFILES)
                wealth = random.choice(["high", "medium", "low"])

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
        make_type_getter = lambda p: (lambda m: m.energy_by_type.get(p, 0))
        make_wealth_getter = lambda grp: (lambda m: m.energy_by_wealth.get(grp, 0))

        model_reporters = {
            **{t: make_type_getter(t) for t in PROPERTY_TYPES},
            **{w: make_wealth_getter(w) for w in ["high", "medium", "low"]},
            "total_energy": lambda m: sum(h.energy_consumption for h in m.household_agents),
            "cumulative_energy": lambda m: m.cumulative_energy,
            "ambient_mean_tempC": lambda m: float(
                np.nanmean([getattr(h, "ambient_tempC", np.nan) for h in m.household_agents])
            ),
            "climate_hour_index": lambda m: m.current_hour,
        }

        agent_reporters = {} if not collect_agent_level else {
            "agent_type": lambda a: "household" if isinstance(a, HouseholdAgent) else "person",
            "energy": lambda a: getattr(a, "energy", 0.0),
            "energy_consumption": lambda a: getattr(a, "energy_consumption", 0.0),
            "ambient_tempC": lambda a: getattr(a, "ambient_tempC", float("nan")),
            "climate_heating_kWh": lambda a: getattr(a, "climate_heating_kWh", 0.0),
            "climate_cooling_kWh": lambda a: getattr(a, "climate_cooling_kWh", 0.0),
        }

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )


        # initial snapshot (t = 0)
        self.datacollector.collect(self)

    # Convenience: local wall-clock hour (respects climate_start + local_tz)
    def local_hour(self) -> int:
        return int((self._clock0 + self.current_hour) % 24)

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

        # --- 3. climate sampling + apply per dwelling --------------
        if self.climate is not None and self._clim_idx_per_house is not None:
            t = self._t0 + (self.current_hour - 1)  # -1 because we collected at init
            if 0 <= t < len(self.climate.times):
                vecP = self.climate.temps_at_index(t)  # shape [P]
                for h in self.household_agents:
                    idx = h.clim_idx
                    tempC = float(vecP[idx]) if idx is not None else float("nan")
                    # simple occupancy count (optional dampening inside apply_climate)
                    occ = sum(1 for r in h.residents if getattr(r, "at_home", False))
                    h.apply_climate(
                        tempC,
                        heating_setpoint=self.heating_setpoint_C,
                        cooling_threshold=self.cooling_threshold_C,
                        heat_slope=self.heating_slope_kWh_per_deg,
                        cool_slope=self.cooling_slope_kWh_per_deg,
                        occupancy=occ,
                    )
            else:
                # out of climate range; mark as NaN and add no climate kWh
                for h in self.household_agents:
                    h.apply_climate(
                        float("nan"),
                        heating_setpoint=self.heating_setpoint_C,
                        cooling_threshold=self.cooling_threshold_C,
                        heat_slope=self.heating_slope_kWh_per_deg,
                        cool_slope=self.cooling_slope_kWh_per_deg,
                    )

        # --- 4. aggregate by property type + wealth group ----------
        self.energy_by_type = {t: 0.0 for t in PROPERTY_TYPES}
        for h in self.household_agents:
            ptype = getattr(h, "property_type", "")
            if ptype in self.energy_by_type:
                self.energy_by_type[ptype] += h.energy_consumption

        self.energy_by_wealth = dict.fromkeys(["high", "medium", "low"], 0.0)
        for p in self.person_agents:
            self.energy_by_wealth[p.wealth] += p.energy

        # --- 5. cumulative total ----------------------------------
        tick_total = sum(h.energy_consumption for h in self.household_agents)
        self.cumulative_energy += tick_total

        # --- 6. record everything ---------------------------------
        self.datacollector.collect(self)
