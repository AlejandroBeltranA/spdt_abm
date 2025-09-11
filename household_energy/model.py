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
        collect_agent_level: bool = True,
        agent_collect_every: int = 24,  # NEW: downsample agent collection (hours)
    ):
        super().__init__()

        self.current_hour: int = 0
        self.household_agents: List[HouseholdAgent] = []
        self.person_agents: List[PersonAgent] = []

        if gdf is None:
            raise ValueError("EnergyModel requires a GeoDataFrame `gdf`.")
        self.space = mg.GeoSpace(crs=gdf.crs)

        self.energy_per_person_home: float = 0.06
        self.energy_per_person_away: float = 0.01

        self.heating_setpoint_C: float = 18.5
        self.cooling_threshold_C: float = 24.0
        self.heating_slope_kWh_per_deg: float = 0.10
        self.cooling_slope_kWh_per_deg: float = 0.08

        self.energy_by_type: Dict[str, float] = {t: 0.0 for t in PROPERTY_TYPES}
        self.energy_by_wealth: Dict[str, float] = dict.fromkeys(["high", "medium", "low"], 0.0)
        self.cumulative_energy: float = 0.0

        self.climate: Optional[ClimateField] = None
        self._clim_idx_per_house: Optional[np.ndarray] = None
        self._t0: int = 0

        if climate_parquet:
            self.climate = ClimateField(climate_parquet)

        # ------------- 1. instantiate households --------------------
        for _, row in gdf.iterrows():
            house = HouseholdAgent(
                unique_id=str(row.get("UPRN", row.get("uprn", row.get("fid")))),  # NEW: UPRN-friendly
                model=self,
                geometry=row["geometry"],
                # core
                property_type=row.get("property_type", ""),
                sap_rating=row.get("sap_rating", 70),
                # prefer calibrated demand; fallback to legacy if missing
                annual_energy_kwh=row.get("energy_cal_kwh", row.get("energy_demand", 10_000)),
                # drivers
                floor_area_m2=row.get("floor_area_m2"),
                property_age=row.get("property_age"),
                main_fuel_type=row.get("main_fuel_type"),
                main_heating_system=row.get("main_heating_system"),
                retrofit_envelope_score=row.get("retrofit_envelope_score"),
                imd_decile=row.get("imd_decile"),
                # levers / context
                heating_controls=row.get("heating_controls"),
                meter_type=row.get("meter_type"),
                cwi_flag=row.get("cwi_flag"),
                swi_flag=row.get("swi_flag"),
                loft_ins_flag=row.get("loft_ins_flag"),
                floor_ins_flag=row.get("floor_ins_flag"),
                glazing_flag=row.get("glazing_flag"),
                is_electric_heating=row.get("is_electric_heating"),
                is_gas=row.get("is_gas"),
                is_oil=row.get("is_oil"),
                is_solid_fuel=row.get("is_solid_fuel"),
                is_off_gas=row.get("is_off_gas"),
                crs=gdf.crs,
            )
            self.household_agents.append(house)
            self.space.add_agents([house])

        # Ensure geometry is centroided for clarity (and consistent mapping)
        for h in self.household_agents:
            g = getattr(h, "geometry", None)
            if g is None or g.is_empty:
                continue
            if g.geom_type == "Point":
                continue
            try:
                gg = g.buffer(0)
            except Exception:
                gg = g
            if gg.is_empty:
                gg = g.representative_point()
                h.geometry = gg
            else:
                h.geometry = gg.centroid

        # ✅ Map climate ONCE (after houses exist) and assign per-house index
        if self.climate is not None:
            valid_houses = [h for h in self.household_agents
                            if getattr(h, "geometry", None) is not None and not h.geometry.is_empty]
            lats = np.fromiter((h.geometry.y for h in valid_houses), dtype=np.float32, count=len(valid_houses))
            lons = np.fromiter((h.geometry.x for h in valid_houses), dtype=np.float32, count=len(valid_houses))
            if len(valid_houses) > 0:
                self._clim_idx_per_house = self.climate.map_households(lats, lons)
                for h, idx in zip(valid_houses, self._clim_idx_per_house):
                    h.set_climate_index(idx)

            if climate_start is None:
                climate_start = self.climate.times[0]
            self._t0 = self.climate.time_index_for(climate_start)

            for h in self.household_agents:
                h.ambient_tempC = float("nan")

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
                # NEW: initial occupancy count (Homebody or pre-first-leave)
                if getattr(person, "at_home", True):  # NEW
                    house.occupancy_count += 1         # NEW
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
            # static attributes for analysis
            "property_type": lambda a: getattr(a, "property_type", None),
            "sap_rating": lambda a: getattr(a, "sap_rating", None),
            "annual_energy_kwh": lambda a: getattr(a, "annual_energy_kwh", None),
            "floor_area_m2": lambda a: getattr(a, "floor_area_m2", None),
            "property_age": lambda a: getattr(a, "property_age", None),
            "main_fuel_type": lambda a: getattr(a, "main_fuel_type", None),
            "main_heating_system": lambda a: getattr(a, "main_heating_system", None),
            "retrofit_envelope_score": lambda a: getattr(a, "retrofit_envelope_score", None),
            "imd_decile": lambda a: getattr(a, "imd_decile", None),
            "heating_controls": lambda a: getattr(a, "heating_controls", None),
            "meter_type": lambda a: getattr(a, "meter_type", None),
            "cwi_flag": lambda a: getattr(a, "cwi_flag", None),
            "swi_flag": lambda a: getattr(a, "swi_flag", None),
            "loft_ins_flag": lambda a: getattr(a, "loft_ins_flag", None),
            "floor_ins_flag": lambda a: getattr(a, "floor_ins_flag", None),
            "glazing_flag": lambda a: getattr(a, "glazing_flag", None),
            "is_off_gas": lambda a: getattr(a, "is_off_gas", None),
            "is_electric_heating": lambda a: getattr(a, "is_electric_heating", None),
            "is_gas": lambda a: getattr(a, "is_gas", None),
            "is_oil": lambda a: getattr(a, "is_oil", None),
            "is_solid_fuel": lambda a: getattr(a, "is_solid_fuel", None),
        }

        # NEW: split collectors → model every step; agent downsampled
        self.model_dc = mesa.DataCollector(model_reporters=model_reporters)  # NEW
        self.agent_dc = None  # NEW
        if collect_agent_level:  # NEW
            self.agent_dc = mesa.DataCollector(agent_reporters=agent_reporters)  # NEW
        self.agent_collect_every = max(1, int(agent_collect_every))  # NEW

        # NEW: backward-compat alias (so existing code referencing .datacollector still works for model-level)
        self.datacollector = self.model_dc  # NEW

        # initial snapshot (t = 0)
        self.model_dc.collect(self)
        if self.agent_dc is not None and (self.current_hour % self.agent_collect_every == 0):  # NEW
            self.agent_dc.collect(self)  # NEW

    def local_hour(self) -> int:
        return int((self._clock0 + self.current_hour) % 24)

    # ------------------------------------------------------------------
    #  Per-tick update
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance simulation by one hour."""
        self.current_hour += 1

        # 1) reset + add precomputed base load
        for h in self.household_agents:
            h.reset_energy()
            h.energy_consumption += h.calc_base_energy()

        # 2) update residents
        for p in self.person_agents:
            p.step()

        # 3) climate sampling + apply per dwelling
        if self.climate is not None and self._clim_idx_per_house is not None:
            t = self._t0 + (self.current_hour - 1)
            if 0 <= t < len(self.climate.times):
                vecP = self.climate.temps_at_index(t)  # shape [P]
                for h in self.household_agents:
                    idx = h.clim_idx
                    tempC = float(vecP[idx]) if idx is not None else float("nan")
                    occ = h.occupancy_count  # NEW: fast counter (no per-hour loop)
                    h.apply_climate(
                        tempC,
                        heating_setpoint=self.heating_setpoint_C,
                        cooling_threshold=self.cooling_threshold_C,
                        heat_slope=self.heating_slope_kWh_per_deg,
                        cool_slope=self.cooling_slope_kWh_per_deg,
                        occupancy=occ,
                    )
            else:
                for h in self.household_agents:
                    h.apply_climate(
                        float("nan"),
                        heating_setpoint=self.heating_setpoint_C,
                        cooling_threshold=self.cooling_threshold_C,
                        heat_slope=self.heating_slope_kWh_per_deg,
                        cool_slope=self.cooling_slope_kWh_per_deg,
                    )

        # 4) aggregate by property type + wealth group
        self.energy_by_type = {t: 0.0 for t in PROPERTY_TYPES}
        for h in self.household_agents:
            ptype = getattr(h, "property_type", "")
            if ptype in self.energy_by_type:
                self.energy_by_type[ptype] += h.energy_consumption

        self.energy_by_wealth = dict.fromkeys(["high", "medium", "low"], 0.0)
        for p in self.person_agents:
            self.energy_by_wealth[p.wealth] += p.energy

        # 5) cumulative total
        tick_total = sum(h.energy_consumption for h in self.household_agents)
        self.cumulative_energy += tick_total

        # 6) collect
        self.model_dc.collect(self)
        if self.agent_dc is not None and (self.current_hour % self.agent_collect_every == 0):  # NEW
            self.agent_dc.collect(self)  # NEW
