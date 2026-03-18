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
import math

from .climate import ClimateField
from .agent import HouseholdAgent, PersonAgent, PROPERTY_TYPES, SCHEDULE_PROFILES
from .config import load_config, ModelConfig

# ------------------------------------------------------------------
# Schedule archetypes (hour-level, simple) used when schedule_type
# is provided in the household CSV. Leave/return are hours 0–23;
# None means always at home.
# ------------------------------------------------------------------
SCHEDULE_DEFS: Dict[str, tuple[Optional[int], Optional[int]]] = {
    "HOME_ALLDAY":  (None, None),
    "WORK_STD":     (9, 17),
    "WORK_EARLY":   (6, 15),
    "WORK_LATE":    (12, 21),
    "PART_TIME_AM": (8, 13),
    "PART_TIME_PM": (13, 18),
    "SCHOOL_RUN":   (9, 15),
    "STUDENT":      (10, 16),
    "OUT_LONG":     (7, 20),
}

# ------------------------------------------------------------------
# Schedule archetypes (hour-level, simple) used when schedule_type
# is provided in the household CSV. Leave/return are hours 0–23;
# None means always at home.
# ------------------------------------------------------------------
SCHEDULE_DEFS: Dict[str, tuple[Optional[int], Optional[int]]] = {
    "HOME_ALLDAY":  (None, None),
    "WORK_STD":     (9, 17),
    "WORK_EARLY":   (6, 15),
    "WORK_LATE":    (12, 21),
    "PART_TIME_AM": (8, 13),
    "PART_TIME_PM": (13, 18),
    "SCHOOL_RUN":   (9, 15),
    "STUDENT":      (10, 16),
    "OUT_LONG":     (7, 20),
}


class EnergyModel(mesa.Model):
    """Agent-based model of hourly residential energy demand."""

    geojson_regions: str = "data/ncc_neighborhood.geojson"

    def __init__(
        self,
        gdf: gpd.GeoDataFrame | None = None,
        *,
        n_residents_func: Callable[[HouseholdAgent], int] | None = None,
        climate_parquet: Optional[str] = None,
        climate_start: str | np.datetime64 | pd.Timestamp | None = None,
        local_tz: str = "Europe/London",
        level_scale: float = 1.0,
        collect_agent_level: bool = True,
        agent_collect_every: int = 24,  # NEW: downsample agent collection (hours)
        config_path: str | None = None,
    ):
        super().__init__()

        self.current_hour: int = 0
        self.household_agents: List[HouseholdAgent] = []
        self.person_agents: List[PersonAgent] = []

        if gdf is None:
            raise ValueError("EnergyModel requires a GeoDataFrame `gdf`.")
        self.space = mg.GeoSpace(crs=gdf.crs)

        # Scenario/config (externalizable)
        self.config: ModelConfig = load_config(config_path)

        self.energy_per_person_home: float = float(self.config.model.get("energy_per_person_home", 0.06))
        self.energy_per_person_away: float = float(self.config.model.get("energy_per_person_away", 0.01))

        self.heating_setpoint_C: float = float(self.config.model.get("heating_setpoint_C", 18.5))
        self.cooling_threshold_C: float = float(self.config.model.get("cooling_threshold_C", 24.0))
        self.heating_slope_kWh_per_deg: float = float(self.config.model.get("heating_slope_kWh_per_deg", 0.05))
        self.cooling_slope_kWh_per_deg: float = float(self.config.model.get("cooling_slope_kWh_per_deg", 0.03))
        self.apply_structural_multipliers: bool = bool(self.config.model.get("apply_structural_multipliers", True))
        # heat-slope shaping / caps
        self.heat_slope_area_exp: float = float(self.config.model.get("heat_slope_area_exp", 0.6))
        self.heat_slope_min: float = float(self.config.model.get("heat_slope_min", 0.0))
        self.heat_slope_max: float = float(self.config.model.get("heat_slope_max", 0.10))
        self.max_heat_kwh_per_hour: float = float(self.config.model.get("max_heat_kwh_per_hour", 20.0))
        self.max_total_kwh_per_hour: float = float(self.config.model.get("max_total_kwh_per_hour", 20.0))
        self.max_base_kwh_per_hour: float = float(self.config.model.get("max_base_kwh_per_hour", 1.5))
        self.loss_to_duty_k: float = float(self.config.model.get("loss_to_duty_k", 3.0))
        self.base_heat_capacity: float = float(self.config.model.get("base_heat_capacity", 8.0))
        self.heat_capacity_area_exp: float = float(self.config.model.get("heat_capacity_area_exp", 0.5))
        self.min_heat_capacity: float = float(self.config.model.get("min_heat_capacity", 4.0))
        # Baseline anchor params
        # Baseline is now a small meter-derived constant; structural multipliers are
        # applied to heating only (see _compute_hourly_base_kwh in agent.py).
        self.use_epc_for_baseline: bool = bool(self.config.model.get("use_epc_for_baseline", False))
        self.baseline_anchor_kwh_per_hour: float = float(self.config.model.get("baseline_anchor_kwh_per_hour", 0.4))
        self.baseline_area_ref_m2: float = float(self.config.model.get("baseline_area_ref_m2", 70.0))
        self.baseline_area_exp: float = float(self.config.model.get("baseline_area_exp", 0.20))
        self.baseline_area_clip = tuple(self.config.model.get("baseline_area_clip", (0.85, 1.25)))
        self.property_type_mult_base: Dict[str, float] = self.config.model.get("property_type_mult_base", {})

        # --------------- NEW: heat pump params --------------------
        self.boiler_efficiency = 0.90       # for hp effectiveness (boiler η)
        self.heatpump_cop_ref  = 2.8        # simple, flat COP for now
        self.heatpump_adoption_rate = 0.0   # 0..1 of eligible homes (or dict per class)
        self.heatpump_class_weight = {
            "priority": 1.4, "possible": 1.0, "difficult": 0.6,
            "non-possible": 0.0, None: 1.0,
        }

        self.energy_by_type: Dict[str, float] = {t: 0.0 for t in PROPERTY_TYPES}
        self.energy_by_wealth: Dict[str, float] = dict.fromkeys(["high", "medium", "low"], 0.0)
        self.cumulative_energy: float = 0.0

        self.climate: Optional[ClimateField] = None
        self._clim_idx_per_house: Optional[np.ndarray] = None
        self._t0: int = 0

        # Config metadata (propagated to outputs for traceability)
        self.config_name: str = self.config.name
        self.config_date: str = self.config.date

        if climate_parquet:
            self.climate = ClimateField(climate_parquet)

        # ------------- 1. instantiate households --------------------
        resident_cap = int(self.config.households.get("resident_cap", 10))
        bedroom_mult = self.config.households.get("bedroom_multiplier", {})
        default_residents = int(self.config.households.get("n_residents_default", 2))

        def _default_residents(h: HouseholdAgent) -> int:
            n = getattr(h, "hh_n_people", None)
            if n is None:
                return default_residents
            try:
                n = int(n)
            except Exception:
                return default_residents
            return max(1, min(resident_cap, n))

        if n_residents_func is None:
            n_residents_func = _default_residents

        for _, row in gdf.iterrows():
            has_calibrated_energy = any(
                pd.notna(row.get(k))
                for k in ("energy_cal_kwh", "energy_demand_kwh", "energy_demand")
            )
            house = HouseholdAgent(
                unique_id=str(row.get("UPRN", row.get("uprn", row.get("fid")))),  # NEW: UPRN-friendly
                model=self,
                geometry=row["geometry"],
                # core
                property_type=row.get("property_type", ""),
                sap_rating=row.get("sap_rating", 70),
                # prefer calibrated demand; fallback to legacy if missing
                annual_energy_kwh=row.get(
                    "energy_cal_kwh",
                    row.get("energy_demand_kwh", row.get("energy_demand", 10_000)),
                ),
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
                # NEW: heat pump candidate inputs (from your DataFrame)
                is_heatpump_candidate=row.get("is_heatpump_candidate"),
                heatpump_candidate_class=row.get("heatpump_candidate_class"),
                # NEW: socio‑demo / dwelling inputs (optional)
                hidp=row.get("hidp"),
                hh_n_people=row.get("hh_n_people"),
                hh_children=row.get("hh_children"),
                hh_income=row.get("hh_income"),
                hh_income_band=row.get("hh_income_band"),
                hh_edu_detail=row.get("hh_edu_detail"),
                dwelling_bucket=row.get("dwelling_bucket"),
                tenure=row.get("tenure"),
                size_band=row.get("size_band"),
                schedule_type=row.get("schedule_type"),
                crs=gdf.crs,
            )
            house.has_calibrated_energy = has_calibrated_energy
            # recompute heat slope in case config differs from default
            house.heat_slope_kWh_per_deg = house._compute_heat_slope(self.heating_slope_kWh_per_deg)
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

        # --- assign heat pumps according to policy (runs once) ---  NEW
        self.boiler_efficiency = float(self.config.model.get("boiler_efficiency", self.boiler_efficiency))
        self.heatpump_cop_ref = float(self.config.model.get("heatpump_cop_ref", self.heatpump_cop_ref))
        self.heatpump_adoption_rate = self.config.model.get("heatpump_adoption_rate", self.heatpump_adoption_rate)
        self.heatpump_class_weight.update(self.config.model.get("heatpump_class_weight", {}))
        self._assign_heatpumps()

        self._local_tz = local_tz
        self._clock0 = 0
        if climate_start is not None:
            ts0 = pd.to_datetime(climate_start, utc=True)
            self._clock0 = ts0.tz_convert(self._local_tz).hour

        # ------------- 2. instantiate residents ---------------------
        uid_counter = 0
        legacy_profiles = self.config.schedules.get("default_profiles") or SCHEDULE_PROFILES

        rng_sched = random.Random(12345)  # deterministic schedule jitter per run

        def _jitter(hr: Optional[int]) -> Optional[int]:
            if hr is None:
                return None
            j = rng_sched.randint(-1, 1)
            return int(max(0, min(23, hr + j)))

        def _schedule_tuple(tag: str) -> tuple[Optional[int], Optional[int]]:
            leave, ret = SCHEDULE_DEFS.get(tag, (None, None))
            return _jitter(leave), _jitter(ret)

        # Map household-level schedule_type (if present) to per-person leave/return.
        # Falls back to legacy Parent/Worker/Homebody when schedule_type is missing/unknown.
        def _assign_household_schedules(h: HouseholdAgent, n_people: int) -> list[dict]:
            stype_raw = getattr(h, "schedule_type", None)
            stype = stype_raw.strip().lower() if isinstance(stype_raw, str) else ""
            children_flag = getattr(h, "hh_children", None)
            n_children = 0
            if children_flag is True:
                n_children = 1
            if stype in ("family_with_children", "single_parent_with_children"):
                n_children = max(n_children, 1)
            if stype == "family_with_children" and n_people > 3:
                n_children = max(n_children, min(2, n_people - 2))
            n_children = min(n_children, max(0, n_people - 1))
            n_children = max(0, n_children)
            n_adults = max(1, n_people - n_children)

            people: list[dict] = []

            if stype == "retired_household":
                for _ in range(n_adults):
                    leave, ret = _schedule_tuple("HOME_ALLDAY")
                    people.append({"role": "adult", "schedule_profile": "HOME_ALLDAY", "leave": leave, "return": ret})
            elif stype == "unemployed_or_inactive":
                for _ in range(n_adults):
                    tag = "PART_TIME_PM" if rng_sched.random() < 0.3 else "HOME_ALLDAY"
                    leave, ret = _schedule_tuple(tag)
                    people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})
            elif stype == "working_adult_household":
                for _ in range(n_adults):
                    tag = "WORK_STD"
                    leave, ret = _schedule_tuple(tag)
                    people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})
            elif stype == "dual_earner_household":
                for i in range(n_adults):
                    tag = "WORK_STD" if i == 0 else ("WORK_EARLY" if rng_sched.random() < 0.5 else "WORK_LATE")
                    leave, ret = _schedule_tuple(tag)
                    people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})
            elif stype == "student_household":
                for _ in range(n_adults):
                    tag = "STUDENT"
                    leave, ret = _schedule_tuple(tag)
                    people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})
            elif stype == "family_with_children":
                # adults
                for i in range(n_adults):
                    tag = "SCHOOL_RUN" if i == 0 else "WORK_STD"
                    leave, ret = _schedule_tuple(tag)
                    people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})
            elif stype == "single_parent_with_children":
                for _ in range(n_adults):
                    tag = "PART_TIME_AM" if rng_sched.random() < 0.6 else "SCHOOL_RUN"
                    leave, ret = _schedule_tuple(tag)
                    people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})
            else:
                # Fallback to legacy profiles
                for _ in range(n_people):
                    prof = random.choice(legacy_profiles)
                    people.append(
                        {
                            "role": "adult",
                            "schedule_profile": prof["name"],
                            "leave": prof["leave"],
                            "return": prof["return"],
                        }
                    )
                return people

            # add children schedules
            for _ in range(n_children):
                tag = "SCHOOL_RUN"
                leave, ret = _schedule_tuple(tag)
                people.append({"role": "child", "schedule_profile": tag, "leave": leave, "return": ret})

            return people

        for house in self.household_agents:
            n_people = n_residents_func(house)
            scheds = _assign_household_schedules(house, n_people)
            for sched in scheds:
                wealth = random.choice(["high", "medium", "low"])
                person = PersonAgent(
                    unique_id=f"{house.unique_id}_{uid_counter}",
                    model=self,
                    home=house,
                    schedule_profile=sched["schedule_profile"],
                    leave_hour=sched.get("leave"),
                    return_hour=sched.get("return"),
                    wealth=wealth,
                    sap=house.sap_rating,
                )
                self.person_agents.append(person)
                house.residents.append(person)
                if getattr(person, "at_home", True):
                    house.occupancy_count += 1
                uid_counter += 1

        # ------------- 3. DataCollector set-up ----------------------
        make_type_getter = lambda p: (lambda m: m.energy_by_type.get(p, 0))
        make_wealth_getter = lambda grp: (lambda m: m.energy_by_wealth.get(grp, 0))

        def _mean_ambient_temp(m) -> float:
            vals = [getattr(h, "ambient_tempC", np.nan) for h in m.household_agents]
            if not vals:
                return float("nan")
            arr = np.array(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return float("nan")
            return float(np.nanmean(arr))

        model_reporters = {
            **{t: make_type_getter(t) for t in PROPERTY_TYPES},
            **{w: make_wealth_getter(w) for w in ["high", "medium", "low"]},
            "total_energy": lambda m: sum(h.energy_consumption for h in m.household_agents),
            "cumulative_energy": lambda m: m.cumulative_energy,
            "ambient_mean_tempC": _mean_ambient_temp,
            "climate_hour_index": lambda m: m.current_hour,
            "config_name": lambda m: getattr(m, "config_name", ""),
            "config_date": lambda m: getattr(m, "config_date", ""),
            "config_notes": lambda m: getattr(m.config, "notes", ""),
        }

        agent_reporters = {} if not collect_agent_level else {
            "agent_type": lambda a: "household" if isinstance(a, HouseholdAgent) else "person",
            "energy": lambda a: getattr(a, "energy", 0.0),
            "energy_consumption": lambda a: getattr(a, "energy_consumption", 0.0),
            "occupancy_count": lambda a: getattr(a, "occupancy_count", None) if isinstance(a, HouseholdAgent) else None,
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
            # socio‑demo / household additions (optional)
            "hidp": lambda a: getattr(a, "hidp", None),
            "hh_n_people": lambda a: getattr(a, "hh_n_people", None),
            "hh_children": lambda a: getattr(a, "hh_children", None),
            "hh_income_band": lambda a: getattr(a, "hh_income_band", None),
            "hh_edu_detail": lambda a: getattr(a, "hh_edu_detail", None),
            "dwelling_bucket": lambda a: getattr(a, "dwelling_bucket", None),
            "tenure": lambda a: getattr(a, "tenure", None),
            "size_band": lambda a: getattr(a, "size_band", None),
            "schedule_type": lambda a: getattr(a, "schedule_type", None),
            "schedule_profile": lambda a: getattr(a, "schedule_profile", None),
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
            h.base_kwh = h.calc_base_energy()
            h.energy_consumption += h.base_kwh

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
                        heat_slope=getattr(h, "heat_slope_kWh_per_deg", self.heating_slope_kWh_per_deg),  # CHANGED
                        cool_slope=self.cooling_slope_kWh_per_deg,
                        occupancy=occ,
                    )

            else:
                for h in self.household_agents:
                    h.apply_climate(
                        float("nan"),
                        heating_setpoint=self.heating_setpoint_C,
                        cooling_threshold=self.cooling_threshold_C,
                        heat_slope=getattr(h, "heat_slope_kWh_per_deg", self.heating_slope_kWh_per_deg),  # CHANGED
                        cool_slope=self.cooling_slope_kWh_per_deg,
                    )

        # 4) aggregate by property type + wealth group
        # enforce total hourly cap per dwelling after all components (base + climate + spikes)
        max_total = getattr(self, "max_total_kwh_per_hour", None)
        if max_total is not None:
            for h in self.household_agents:
                if h.energy_consumption > max_total:
                    pre = h.energy_consumption
                    clip = pre - max_total
                    # proportional attribution for diagnostics
                    base = getattr(h, "base_kwh", 0.0)
                    heat = getattr(h, "heat_kwh", 0.0)
                    spike = getattr(h, "spike_kwh", pre - base - heat)
                    denom = base + heat + spike
                    if denom <= 0:
                        fb = fh = fs = 0.0
                    else:
                        fb = clip * (base / denom)
                        fh = clip * (heat / denom)
                        fs = clip * (spike / denom)
                    h.cap_clip_total = clip
                    h.cap_clip_base = fb
                    h.cap_clip_heat = fh
                    h.cap_clip_spike = fs
                    h.energy_consumption = max_total

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
    def _assign_heatpumps(self) -> None:
        """Assign heat pumps to top X% of eligible candidates (or per-class shares).
        Scoring uses expected kWh reduction from lowering the heating slope via HP.
        Deterministic (ties broken by object id). Skips homes that already had a HP.
        """
        rate = self.heatpump_adoption_rate
        if not rate:
            return

        def hp_score(h: HouseholdAgent) -> float:
            if not getattr(h, "is_heatpump_candidate", 0):
                return -1.0
            cls = getattr(h, "heatpump_candidate_class", "non-possible")
            if cls == "non-possible":
                return -1.0
            # expected gain ~ slope * (1 - hp_effect_mult) * class_weight
            slope = getattr(h, "heat_slope_kWh_per_deg", self.heating_slope_kWh_per_deg)
            hp_mult = getattr(h, "hp_effect_mult", self.boiler_efficiency / self.heatpump_cop_ref)
            gain = max(0.0, slope * (1.0 - hp_mult))
            w = self.heatpump_class_weight.get(cls, 1.0)
            return gain * w

        # Eligible (and not already HP at baseline)
        elig = [
            h for h in self.household_agents
            if getattr(h, "is_heatpump_candidate", 0) == 1
            and getattr(h, "heatpump_candidate_class", "non-possible") != "non-possible"
            and not getattr(h, "was_heatpump_initial", False)
        ]
        if not elig:
            return

        # Case A: single global fraction
        if isinstance(rate, (int, float)):
            ranked = sorted(elig, key=lambda h: (-hp_score(h), id(h)))
            n_take = int(len(ranked) * float(rate) + 1e-9)
            for h in ranked[:n_take]:
                h.has_heatpump = True
            return

        # Case B: per-class fractions, e.g. {"priority":0.45, "possible":0.20, "difficult":0.05}
        if isinstance(rate, dict):
            by_class = {"priority": [], "possible": [], "difficult": []}
            for h in elig:
                c = getattr(h, "heatpump_candidate_class", None)
                if c in by_class:
                    by_class[c].append(h)
            for c, homes in by_class.items():
                homes.sort(key=lambda h: (-hp_score(h), id(h)))
                frac = float(rate.get(c, 0.0))
                n_take = int(len(homes) * frac + 1e-9)
                for h in homes[:n_take]:
                    h.has_heatpump = True
