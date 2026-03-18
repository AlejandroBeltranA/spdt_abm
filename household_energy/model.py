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
from pathlib import Path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import mesa
import mesa_geo as mg
import numpy as np
import pandas as pd  # for timezone handling

from .climate import ClimateField
from .agent import HouseholdAgent, PersonAgent, PROPERTY_TYPES, PROPERTY_TYPE_MULT_BASE, PROPERTY_TYPE_MULT_HEAT, SCHEDULE_PROFILES
from .config import load_config, ModelConfig

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd

WEALTH_BUCKETS = ["very_low", "low", "mid", "high", "very_high"]

# ------------------------------------------------------------------
# Schedule archetypes (hour-level, simple) used when schedule_type
# is provided in the household CSV. Leave/return are hours 0–23;
# None means always at home.
# ------------------------------------------------------------------
DEFAULT_SCHEDULE_DEFS: Dict[str, tuple[Optional[int], Optional[int]]] = {
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
        gdf: "gpd.GeoDataFrame" | None = None,
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
        self._start_ts_utc: pd.Timestamp | None = None

        if gdf is None:
            raise ValueError("EnergyModel requires a GeoDataFrame `gdf`.")
        self.space = mg.GeoSpace(crs=gdf.crs)

        # Scenario/config (externalizable)
        self.config: ModelConfig = load_config(config_path)

        self.energy_per_person_home: float = float(self.config.model.get("energy_per_person_home", 0.06))
        self.energy_per_person_away: float = float(self.config.model.get("energy_per_person_away", 0.01))

        self.heating_setpoint_C: float = float(self.config.model.get("heating_setpoint_C", 18.5))
        self.cooling_threshold_C: float = float(self.config.model.get("cooling_threshold_C", 24.0))
        self.setpoint_setback_C: float = float(self.config.model.get("setpoint_setback_C", 2.0))
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
        # Intraday structure: explicit profiles for heating and DHW.
        # Profiles are 24-element arrays normalized to mean=1.0.
        def _profile24(raw: object, default: list[float]) -> np.ndarray:
            vals = default
            if isinstance(raw, (list, tuple)) and len(raw) == 24:
                try:
                    vals = [float(x) for x in raw]
                except Exception:
                    vals = default
            arr = np.asarray(vals, dtype=float)
            arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
            if np.isnan(arr).any():
                arr = np.asarray(default, dtype=float)
            m = float(np.mean(arr))
            if not np.isfinite(m) or m <= 0:
                return np.ones(24, dtype=float)
            return arr / m

        self.heating_profile_24h: np.ndarray = _profile24(
            self.config.model.get("heating_profile_24h"),
            [1.20, 1.15, 1.10, 1.00, 0.95, 1.00, 1.10, 1.20, 1.05, 0.95, 0.90, 0.90,
             0.90, 0.90, 0.90, 0.95, 1.00, 1.10, 1.25, 1.30, 1.25, 1.20, 1.15, 1.10],
        )
        self.dhw_profile_24h: np.ndarray = _profile24(
            self.config.model.get("dhw_profile_24h"),
            [0.55, 0.50, 0.45, 0.40, 0.45, 0.75, 1.20, 1.35, 1.10, 0.90, 0.85, 0.85,
             0.85, 0.85, 0.90, 1.00, 1.10, 1.25, 1.35, 1.30, 1.20, 1.00, 0.80, 0.65],
        )
        self.dhw_daily_kwh_per_home: float = float(self.config.model.get("dhw_daily_kwh_per_home", 0.0))
        self.dhw_daily_kwh_per_person: float = float(self.config.model.get("dhw_daily_kwh_per_person", 0.0))
        self.dhw_away_mult: float = float(self.config.model.get("dhw_away_mult", 0.5))
        self.heating_occupancy_away_mult: float = float(self.config.model.get("heating_occupancy_away_mult", 0.5))
        # AM/PM peak controls used by calibration (S2/W1 stages).
        self.heating_peak_morning_mult: float = float(self.config.model.get("heating_peak_morning_mult", 1.0))
        self.heating_peak_evening_mult: float = float(self.config.model.get("heating_peak_evening_mult", 1.0))
        self.heating_winter_morning_mult: float = float(self.config.model.get("heating_winter_morning_mult", 1.0))
        self.heating_winter_evening_mult: float = float(self.config.model.get("heating_winter_evening_mult", 1.0))
        self.dhw_peak_morning_mult: float = float(self.config.model.get("dhw_peak_morning_mult", 1.0))
        self.dhw_peak_evening_mult: float = float(self.config.model.get("dhw_peak_evening_mult", 1.0))
        # Baseline anchor params
        # Baseline is now a small meter-derived constant; structural multipliers are
        # applied to heating only (see _compute_hourly_base_kwh in agent.py).
        self.use_epc_for_baseline: bool = bool(self.config.model.get("use_epc_for_baseline", False))
        self.baseline_anchor_kwh_per_hour: float = float(self.config.model.get("baseline_anchor_kwh_per_hour", 0.4))
        self.baseline_anchor_elec_kwh_per_hour: float = float(
            self.config.model.get("baseline_anchor_elec_kwh_per_hour", self.baseline_anchor_kwh_per_hour)
        )
        self.baseline_anchor_gas_kwh_per_hour: float = float(
            self.config.model.get("baseline_anchor_gas_kwh_per_hour", 0.0)
        )
        self.use_separate_fuel_baseline_anchors: bool = bool(
            self.config.model.get(
                "use_separate_fuel_baseline_anchors",
                ("baseline_anchor_elec_kwh_per_hour" in self.config.model)
                or ("baseline_anchor_gas_kwh_per_hour" in self.config.model),
            )
        )
        self.baseline_area_ref_m2: float = float(self.config.model.get("baseline_area_ref_m2", 70.0))
        self.baseline_area_exp: float = float(self.config.model.get("baseline_area_exp", 0.20))
        self.baseline_area_clip = tuple(self.config.model.get("baseline_area_clip", (0.85, 1.25)))
        self.property_type_mult_base: Dict[str, float] = self.config.model.get("property_type_mult_base", PROPERTY_TYPE_MULT_BASE)
        # Heating multipliers: support both legacy `pt_heat_mult` and explicit `property_type_mult_heat`.
        heat_mult_cfg = self.config.model.get("property_type_mult_heat", None)
        if heat_mult_cfg is None:
            heat_mult_cfg = self.config.model.get("pt_heat_mult", None)
        self.property_type_mult_heat: Dict[str, float] = heat_mult_cfg or PROPERTY_TYPE_MULT_HEAT

        # Schedules (tunable): schedule archetype defs + WFH + jitter.
        # - `schedule_defs` maps archetype tag -> (leave_hour, return_hour)
        # - `type_map` maps incoming household schedule_type -> {leave, return} (direct override)
        schedules_cfg = getattr(self.config, "schedules", {}) or {}
        self.schedule_type_map: Dict[str, dict] = schedules_cfg.get("type_map", {}) or {}
        self.wfh_share: float = float(schedules_cfg.get("wfh_share", 0.0) or 0.0)
        self.schedule_jitter_hours: int = int(schedules_cfg.get("jitter_hours", 1) or 1)

        def _coerce_hour(v: object) -> Optional[int]:
            if v is None:
                return None
            try:
                i = int(v)
            except Exception:
                return None
            return max(0, min(23, i))

        def _parse_schedule_defs(raw: object) -> Optional[Dict[str, tuple[Optional[int], Optional[int]]]]:
            if raw is None or not isinstance(raw, dict):
                return None
            out: Dict[str, tuple[Optional[int], Optional[int]]] = {}
            for k, v in raw.items():
                if v is None:
                    out[str(k)] = (None, None)
                    continue
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    out[str(k)] = (_coerce_hour(v[0]), _coerce_hour(v[1]))
                    continue
                if isinstance(v, dict):
                    out[str(k)] = (_coerce_hour(v.get("leave")), _coerce_hour(v.get("return")))
                    continue
            return out or None

        sched_defs_raw = (
            schedules_cfg.get("schedule_defs")
            or schedules_cfg.get("archetypes")
            or None
        )
        self.schedule_defs = _parse_schedule_defs(sched_defs_raw) or DEFAULT_SCHEDULE_DEFS

        # --------------- NEW: heat pump params --------------------
        self.boiler_efficiency = 0.90       # for hp effectiveness (boiler η)
        self.heatpump_cop_ref  = 2.8        # simple, flat COP for now
        self.heatpump_adoption_rate = 0.0   # 0..1 of eligible homes (or dict per class)
        self.heatpump_class_weight = {
            "priority": 1.4, "possible": 1.0, "difficult": 0.6,
            "non-possible": 0.0, None: 1.0,
        }

        self.energy_by_type: Dict[str, float] = {t: 0.0 for t in PROPERTY_TYPES}
        self.energy_by_wealth: Dict[str, float] = dict.fromkeys(WEALTH_BUCKETS, 0.0)
        self.cumulative_energy: float = 0.0
        self.total_energy: float = 0.0
        self.total_electric_kwh: float = 0.0
        self.total_gas_kwh: float = 0.0
        self.total_other_kwh: float = 0.0
        self.ambient_mean_tempC: float = float("nan")

        self.climate: Optional[ClimateField] = None
        self._clim_idx_per_house: Optional[np.ndarray] = None
        self._t0: int = 0

        # ── Optional: SERL-derived hourly/monthly multipliers by seg3 (emulation layer)
        # Purpose: allow importing empirical shapes (diurnal + seasonality) by segment
        # and blending them into the ABM’s per-hour electric/gas outputs.
        self.serl_profiles_enabled: bool = bool(self.config.raw.get("serl_profiles", {}).get("enabled", False))
        self.serl_seg3_column: str = str(self.config.raw.get("serl_profiles", {}).get("seg3_column", "none"))
        self.serl_fallback_value: str = str(self.config.raw.get("serl_profiles", {}).get("fallback_value", "none"))
        self.serl_profiles_csv: Optional[str] = self.config.raw.get("serl_profiles", {}).get("profiles_csv")
        self.serl_profiles_alpha: float = float(self.config.raw.get("serl_profiles", {}).get("alpha", 1.0))
        self.serl_use_hourly: bool = bool(self.config.raw.get("serl_profiles", {}).get("use_hourly", True))
        self.serl_use_monthly: bool = bool(self.config.raw.get("serl_profiles", {}).get("use_monthly", True))
        # Internal lookup: (kind, fuel, seg3_value, idx) -> mult
        self._serl_mult: dict[tuple[str, str, str, int], float] = {}

        if self.serl_profiles_enabled:
            if not self.serl_profiles_csv:
                raise ValueError("serl_profiles.enabled=true but serl_profiles.profiles_csv is missing.")
            self._load_serl_profiles(Path(self.serl_profiles_csv))

        # ── Optional: lightweight per-hour aggregates by segmentation(s) for calibration.
        calib_cfg = self.config.raw.get("calibration", {}) or {}
        self.calibration_segmentations: list[dict] = calib_cfg.get("segmentations", []) or []
        # stored as list of per-hour records (long form)
        self._seg_records: list[dict] = []

        # Config metadata (propagated to outputs for traceability)
        self.config_name: str = self.config.name
        self.config_date: str = self.config.date

        if climate_parquet:
            self.climate = ClimateField(climate_parquet)

        # ------------- 1. instantiate households --------------------
        resident_cap = int(self.config.households.get("resident_cap", 10))
        bedroom_mult = self.config.households.get("bedroom_multiplier", {})
        default_residents = int(self.config.households.get("n_residents_default", 2))
        default_residents_flat = int(self.config.households.get("n_residents_default_flat", default_residents))
        default_residents_detached = int(self.config.households.get("n_residents_default_detached", default_residents))
        default_residents_house = int(self.config.households.get("n_residents_default_house", default_residents))

        def _default_residents(h: HouseholdAgent) -> int:
            n = getattr(h, "hh_n_people", None)
            if n is None:
                sb = getattr(h, "size_band", None)
                try:
                    sb_i = int(sb) if sb is not None else None
                except Exception:
                    sb_i = None
                if sb_i is not None and sb_i > 0:
                    if sb_i <= 1:
                        return 1
                    if sb_i == 2:
                        return 2
                    if sb_i == 3:
                        return 3
                    return 4

                ptype = (getattr(h, "property_type", "") or "").strip().lower()
                if "flat" in ptype or "flats" in ptype:
                    return default_residents_flat
                if "detached" in ptype and "semi" not in ptype:
                    return default_residents_detached
                return default_residents_house
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
            # Attach SERL segmentation label (if configured) for profile multipliers.
            if self.serl_profiles_enabled and self.serl_seg3_column and self.serl_seg3_column != "none":
                try:
                    v = row.get(self.serl_seg3_column, self.serl_fallback_value)
                except Exception:
                    v = self.serl_fallback_value
                v = self.serl_fallback_value if v is None else str(v).strip()
                if v == "" or v.lower() in ("nan", "<na>"):
                    v = self.serl_fallback_value
                setattr(house, "serl_seg3_value", v)

            # Attach configured calibration segmentation attributes (verbatim), so the model
            # can compute per-hour aggregates without enabling agent-level DataCollector.
            for seg in self.calibration_segmentations:
                try:
                    attr = str(seg.get("attr") or "").strip()
                except Exception:
                    attr = ""
                if not attr:
                    continue
                try:
                    vv = row.get(attr)
                except Exception:
                    vv = None
                setattr(house, attr, vv)
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
            self._start_ts_utc = pd.to_datetime(climate_start, utc=True)

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

        def _jitter(hr: Optional[int], rng_local: random.Random) -> Optional[int]:
            if hr is None:
                return None
            jmax = max(0, int(getattr(self, "schedule_jitter_hours", 1)))
            if jmax <= 0:
                return int(max(0, min(23, hr)))
            j = rng_local.randint(-jmax, jmax)
            return int(max(0, min(23, hr + j)))

        def _schedule_tuple(tag: str, rng_local: random.Random) -> tuple[Optional[int], Optional[int]]:
            leave, ret = self.schedule_defs.get(tag, (None, None))
            return _jitter(leave, rng_local), _jitter(ret, rng_local)

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

            rng_local = random.Random(hash(str(getattr(h, "unique_id", id(h)))) & 0xFFFFFFFF)

            def _add(tag: str):
                # Optional work-from-home: convert some working profiles to HOME_ALLDAY.
                if tag in ("WORK_STD", "WORK_EARLY", "WORK_LATE", "PART_TIME_AM", "PART_TIME_PM", "OUT_LONG"):
                    if float(getattr(self, "wfh_share", 0.0)) > 0 and rng_local.random() < float(getattr(self, "wfh_share", 0.0)):
                        tag = "HOME_ALLDAY"
                leave, ret = _schedule_tuple(tag, rng_local)
                people.append({"role": "adult", "schedule_profile": tag, "leave": leave, "return": ret})

            # Direct override by schedule_type → hours (from config `schedules.type_map`)
            if stype and stype in getattr(self, "schedule_type_map", {}):
                m = getattr(self, "schedule_type_map", {}).get(stype) or {}
                leave = _jitter(_coerce_hour(m.get("leave")), rng_local)
                ret = _jitter(_coerce_hour(m.get("return")), rng_local)
                for _ in range(n_adults):
                    people.append({"role": "adult", "schedule_profile": stype, "leave": leave, "return": ret})
                for _ in range(n_children):
                    tag = "SCHOOL_RUN"
                    c_leave, c_ret = _schedule_tuple(tag, rng_local)
                    people.append({"role": "child", "schedule_profile": tag, "leave": c_leave, "return": c_ret})
                return people

            handlers = {
                "retired_household": lambda: [_add("HOME_ALLDAY") for _ in range(n_adults)],
                "unemployed_or_inactive": lambda: [
                    _add("PART_TIME_PM" if rng_local.random() < 0.3 else "HOME_ALLDAY") for _ in range(n_adults)
                ],
                "working_adult_household": lambda: [_add("WORK_STD") for _ in range(n_adults)],
                "dual_earner_household": lambda: [
                    _add("WORK_STD" if i == 0 else ("WORK_EARLY" if rng_local.random() < 0.5 else "WORK_LATE"))
                    for i in range(n_adults)
                ],
                "student_household": lambda: [_add("STUDENT") for _ in range(n_adults)],
                "family_with_children": lambda: [_add("SCHOOL_RUN" if i == 0 else "WORK_STD") for i in range(n_adults)],
                "single_parent_with_children": lambda: [
                    _add("PART_TIME_AM" if rng_local.random() < 0.6 else "SCHOOL_RUN") for _ in range(n_adults)
                ],
            }

            if stype in handlers:
                handlers[stype]()
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
                leave, ret = _schedule_tuple(tag, rng_local)
                people.append({"role": "child", "schedule_profile": tag, "leave": leave, "return": ret})

            return people

        for house in self.household_agents:
            n_people = n_residents_func(house)
            scheds = _assign_household_schedules(house, n_people)
            for sched in scheds:
                w = getattr(house, "wealth_bucket", None)
                if w is None:
                    rng_w = random.Random(hash(str(house.unique_id)) & 0xFFFFFFFF)
                    w = rng_w.choice(["very_low", "low", "mid", "high", "very_high"])
                person = PersonAgent(
                    unique_id=f"{house.unique_id}_{uid_counter}",
                    model=self,
                    home=house,
                    schedule_profile=sched["schedule_profile"],
                    leave_hour=sched.get("leave"),
                    return_hour=sched.get("return"),
                    wealth=w,
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
            return float(getattr(m, "ambient_mean_tempC", float("nan")))

        model_reporters = {
            **{t: make_type_getter(t) for t in PROPERTY_TYPES},
            **{w: make_wealth_getter(w) for w in WEALTH_BUCKETS},
            "total_energy": lambda m: float(getattr(m, "total_energy", 0.0)),
            "total_electric_kwh": lambda m: float(getattr(m, "total_electric_kwh", 0.0)),
            "total_gas_kwh": lambda m: float(getattr(m, "total_gas_kwh", 0.0)),
            "total_other_kwh": lambda m: float(getattr(m, "total_other_kwh", 0.0)),
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
            "base_kwh": lambda a: getattr(a, "base_kwh", 0.0),
            "heat_kwh": lambda a: getattr(a, "heat_kwh", 0.0),
            "spike_kwh": lambda a: getattr(a, "spike_kwh", 0.0),
            "electric_kwh": lambda a: getattr(a, "electric_kwh", 0.0),
            "gas_kwh": lambda a: getattr(a, "gas_kwh", 0.0),
            "other_kwh": lambda a: getattr(a, "other_kwh", 0.0),
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

    def _current_local_month(self) -> Optional[int]:
        if self._start_ts_utc is None:
            return None
        hour_start_utc = self._start_ts_utc + pd.to_timedelta(self.current_hour - 1, unit="h")
        try:
            return int(hour_start_utc.tz_convert(self._local_tz).month)
        except Exception:
            return None

    def _load_serl_profiles(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"SERL profiles CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        req = {"kind", "fuel", "seg3_value", "idx", "mult"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"SERL profiles CSV missing columns: {sorted(missing)} (found {sorted(df.columns)})")
        for _, r in df.iterrows():
            kind = str(r["kind"]).strip().lower()
            fuel = str(r["fuel"]).strip().lower()
            seg3 = str(r["seg3_value"]).strip()
            try:
                idx = int(r["idx"])
            except Exception:
                continue
            try:
                mult = float(r["mult"])
            except Exception:
                continue
            if not np.isfinite(mult) or mult <= 0:
                continue
            self._serl_mult[(kind, fuel, seg3, idx)] = mult

    def _serl_multiplier(self, *, kind: str, fuel: str, seg3_value: str, idx: int) -> float:
        """Return SERL multiplier for (kind,fuel,seg3,idx) with fallback to baseline and 1.0."""
        key = (kind, fuel, seg3_value, idx)
        if key in self._serl_mult:
            return float(self._serl_mult[key])
        # Fallback to baseline seg3_value if present
        key2 = (kind, fuel, self.serl_fallback_value, idx)
        if key2 in self._serl_mult:
            return float(self._serl_mult[key2])
        return 1.0

    def _apply_serl_profile_multipliers(self) -> None:
        """Scale per-household electric/gas kWh for this hour using SERL-derived multipliers."""
        if not self.serl_profiles_enabled:
            return
        alpha = float(max(0.0, min(1.0, self.serl_profiles_alpha)))
        if alpha <= 0:
            return
        h = self.local_hour()
        m = self._current_local_month()
        for house in self.household_agents:
            seg3 = str(getattr(house, "serl_seg3_value", self.serl_fallback_value) or self.serl_fallback_value)
            me = 1.0
            mg = 1.0
            if self.serl_use_hourly:
                me *= self._serl_multiplier(kind="hourly", fuel="electric", seg3_value=seg3, idx=h)
                mg *= self._serl_multiplier(kind="hourly", fuel="gas", seg3_value=seg3, idx=h)
            if self.serl_use_monthly and m is not None:
                me *= self._serl_multiplier(kind="monthly", fuel="electric", seg3_value=seg3, idx=int(m))
                mg *= self._serl_multiplier(kind="monthly", fuel="gas", seg3_value=seg3, idx=int(m))

            # Blend: alpha=0 -> no change, alpha=1 -> full SERL multiplier
            me = (1.0 - alpha) + alpha * float(me)
            mg = (1.0 - alpha) + alpha * float(mg)

            house.electric_kwh = float(getattr(house, "electric_kwh", 0.0)) * me
            house.gas_kwh = float(getattr(house, "gas_kwh", 0.0)) * mg
            # keep other_kwh untouched
            house.energy_consumption = float(house.electric_kwh) + float(house.gas_kwh) + float(getattr(house, "other_kwh", 0.0))

    def _apply_intraday_profiles(self) -> None:
        """Apply explicit 24h structure to heating and DHW.

        - Heating: modulate climate-driven `heat_kwh` by hourly profile.
        - DHW: inject additional time-of-day load routed by heating fuel bucket.
        """
        h_idx = self.local_hour()
        month = self._current_local_month()
        is_winter = month in {11, 12, 1, 2, 3}

        heat_mult = float(self.heating_profile_24h[h_idx]) if 0 <= h_idx < 24 else 1.0
        dhw_share = float(self.dhw_profile_24h[h_idx]) if 0 <= h_idx < 24 else 1.0

        # Split AM/PM peaks explicitly for calibration leverage.
        if 6 <= h_idx <= 9:
            heat_mult *= float(self.heating_peak_morning_mult)
            dhw_share *= float(self.dhw_peak_morning_mult)
            if is_winter:
                heat_mult *= float(self.heating_winter_morning_mult)
        elif 17 <= h_idx <= 21:
            heat_mult *= float(self.heating_peak_evening_mult)
            dhw_share *= float(self.dhw_peak_evening_mult)
            if is_winter:
                heat_mult *= float(self.heating_winter_evening_mult)

        for house in self.household_agents:
            old_heat = float(getattr(house, "heat_kwh", 0.0))
            if old_heat > 0:
                new_heat = old_heat * heat_mult
                delta = new_heat - old_heat
                if abs(delta) > 1e-12:
                    bucket = house._heating_fuel_bucket() if hasattr(house, "_heating_fuel_bucket") else "other"
                    if bucket == "electric":
                        house.electric_kwh += delta
                    elif bucket == "gas":
                        house.gas_kwh += delta
                    else:
                        house.other_kwh += delta
                    house.heat_kwh = new_heat
                    house.climate_heating_kWh = new_heat
                    house.energy_consumption += delta

            # Add DHW load with reduced away consumption.
            dhw_home = float(self.dhw_daily_kwh_per_home)
            dhw_per_person = float(self.dhw_daily_kwh_per_person)
            n_residents = max(1, len(getattr(house, "residents", []) or []))
            dhw_daily = dhw_home + (dhw_per_person * n_residents)

            if dhw_daily > 0:
                occ = int(getattr(house, "occupancy_count", 0) or 0)
                occ_share = max(0.0, min(1.0, float(occ) / float(n_residents)))
                away_floor = max(0.0, min(1.0, float(self.dhw_away_mult)))
                occ_mult = away_floor + (1.0 - away_floor) * occ_share
                dhw = max(0.0, (dhw_daily * dhw_share / 24.0) * occ_mult)
                if dhw > 0:
                    bucket = house._heating_fuel_bucket() if hasattr(house, "_heating_fuel_bucket") else "other"
                    if bucket == "electric":
                        house.electric_kwh += dhw
                    elif bucket == "gas":
                        house.gas_kwh += dhw
                    else:
                        house.other_kwh += dhw
                    house.energy_consumption += dhw

    def _collect_segmentation_aggregates(self) -> None:
        """Collect per-hour aggregates by configured segmentations.

        Records long-form rows:
          timestamp_utc, segmentation, value, n_homes, electric_kwh, gas_kwh
        where electric_kwh/gas_kwh are totals for that group (this hour).
        """
        if not self.calibration_segmentations:
            return
        if self._start_ts_utc is None:
            return
        ts = self._start_ts_utc + pd.to_timedelta(self.current_hour - 1, unit="h")
        ts = pd.to_datetime(ts, utc=True)

        for seg in self.calibration_segmentations:
            name = str(seg.get("name") or seg.get("attr") or "").strip() or "seg"
            attr = str(seg.get("attr") or "").strip()
            if not attr:
                continue
            fallback = str(seg.get("fallback_value") or "none")
            acc: dict[str, dict[str, float]] = {}
            for h in self.household_agents:
                v = getattr(h, attr, None)
                if v is None:
                    key = fallback
                else:
                    key = str(v).strip()
                    if key == "" or key.lower() in ("nan", "<na>"):
                        key = fallback
                if key not in acc:
                    acc[key] = {"n": 0.0, "e": 0.0, "g": 0.0}
                acc[key]["n"] += 1.0
                acc[key]["e"] += float(getattr(h, "electric_kwh", 0.0))
                acc[key]["g"] += float(getattr(h, "gas_kwh", 0.0))

            for v, a in acc.items():
                self._seg_records.append(
                    {
                        "timestamp_utc": ts,
                        "segmentation": name,
                        "value": v,
                        "n_homes": int(a["n"]),
                        "electric_kwh": float(a["e"]),
                        "gas_kwh": float(a["g"]),
                    }
                )

    def get_segmentation_timeseries(self) -> pd.DataFrame:
        """Return collected segmentation aggregates (long form)."""
        if not self._seg_records:
            return pd.DataFrame(columns=["timestamp_utc", "segmentation", "value", "n_homes", "electric_kwh", "gas_kwh"])
        return pd.DataFrame(self._seg_records)

    # ------------------------------------------------------------------
    #  Per-tick update
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance simulation by one hour."""
        self.current_hour += 1
        self._reset_base_loads()
        self._update_residents()
        self._apply_climate_tick()
        self._apply_intraday_profiles()
        self._apply_serl_profile_multipliers()
        self._enforce_total_caps()
        self._aggregate_hour()
        self._collect_segmentation_aggregates()
        self._accumulate_annual_kwh()
        self._collect()

    def _accumulate_annual_kwh(self) -> None:
        """Accumulate per-household annual kWh by UTC year.

        This avoids materializing `agent_dc.get_agent_vars_dataframe()` for common
        analyses like "top consumers in YEAR".
        """
        if self._start_ts_utc is None:
            return
        hour_start_utc = self._start_ts_utc + pd.to_timedelta(self.current_hour - 1, unit="h")
        year = int(hour_start_utc.year)
        for h in self.household_agents:
            try:
                h.annual_kwh_by_year[year] = float(h.annual_kwh_by_year.get(year, 0.0)) + float(getattr(h, "energy_consumption", 0.0))
            except Exception:
                continue

    def _reset_base_loads(self) -> None:
        for h in self.household_agents:
            h.reset_energy()
            if hasattr(h, "calc_base_electric_energy") and hasattr(h, "calc_base_gas_energy"):
                base_e = float(h.calc_base_electric_energy())
                base_g = float(h.calc_base_gas_energy())
                h.base_kwh = base_e + base_g
                h.electric_kwh += base_e
                h.gas_kwh += base_g
            else:
                h.base_kwh = h.calc_base_energy()
                gas_share = h._base_gas_share() if hasattr(h, "_base_gas_share") else 0.0
                gas_add = h.base_kwh * gas_share
                h.gas_kwh += gas_add
                h.electric_kwh += (h.base_kwh - gas_add)
            h.energy_consumption += h.base_kwh

    def _update_residents(self) -> None:
        for p in self.person_agents:
            p.step()

    def _apply_climate_tick(self) -> None:
        if self.climate is None or self._clim_idx_per_house is None:
            self.ambient_mean_tempC = float("nan")
            return
        t = self._t0 + (self.current_hour - 1)
        heat_slope_default = self.heating_slope_kWh_per_deg
        cool_slope_default = self.cooling_slope_kWh_per_deg
        temp_sum = 0.0
        temp_n = 0
        if 0 <= t < len(self.climate.times):
            vecP = self.climate.temps_at_index(t)  # shape [P]
            for h in self.household_agents:
                idx = h.clim_idx
                tempC = float(vecP[idx]) if idx is not None else float("nan")
                if np.isfinite(tempC):
                    temp_sum += tempC
                    temp_n += 1
                occ = h.occupancy_count
                setp = self.heating_setpoint_C
                if occ is not None and occ <= 0:
                    setp = float(self.heating_setpoint_C) - float(getattr(self, "setpoint_setback_C", 0.0))
                h.apply_climate(
                    tempC,
                    heating_setpoint=setp,
                    cooling_threshold=self.cooling_threshold_C,
                    heat_slope=getattr(h, "heat_slope_kWh_per_deg", heat_slope_default),
                    cool_slope=cool_slope_default,
                    occupancy=occ,
                )
        else:
            for h in self.household_agents:
                h.apply_climate(
                    float("nan"),
                    heating_setpoint=self.heating_setpoint_C,
                    cooling_threshold=self.cooling_threshold_C,
                    heat_slope=getattr(h, "heat_slope_kWh_per_deg", heat_slope_default),
                    cool_slope=cool_slope_default,
                )
        self.ambient_mean_tempC = (temp_sum / temp_n) if temp_n > 0 else float("nan")

    def _enforce_total_caps(self) -> None:
        max_total = getattr(self, "max_total_kwh_per_hour", None)
        if max_total is None:
            return
        for h in self.household_agents:
            if h.energy_consumption <= max_total:
                continue
            pre = h.energy_consumption
            clip = pre - max_total
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

    def _aggregate_hour(self) -> None:
        self.energy_by_type = {t: 0.0 for t in PROPERTY_TYPES}
        total = 0.0
        total_electric = 0.0
        total_gas = 0.0
        total_other = 0.0
        for h in self.household_agents:
            ptype = getattr(h, "property_type", "")
            if ptype in self.energy_by_type:
                self.energy_by_type[ptype] += h.energy_consumption
            total += float(getattr(h, "energy_consumption", 0.0))
            total_electric += float(getattr(h, "electric_kwh", 0.0))
            total_gas += float(getattr(h, "gas_kwh", 0.0))
            total_other += float(getattr(h, "other_kwh", 0.0))

        self.energy_by_wealth = dict.fromkeys(WEALTH_BUCKETS, 0.0)
        for p in self.person_agents:
            if p.wealth not in self.energy_by_wealth:
                self.energy_by_wealth[p.wealth] = 0.0
            self.energy_by_wealth[p.wealth] += p.energy

        self.total_energy = total
        self.total_electric_kwh = total_electric
        self.total_gas_kwh = total_gas
        self.total_other_kwh = total_other
        self.cumulative_energy += total

    def _collect(self) -> None:
        self.model_dc.collect(self)
        if self.agent_dc is not None and (self.current_hour % self.agent_collect_every == 0):
            self.agent_dc.collect(self)
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
