"""
household_energy — Agent-based model for residential energy demand.

Simulates hour-by-hour energy consumption for every dwelling in a GeoJSON
neighbourhood. Built on Mesa 3 + mesa-geo, with built-in policy levers for
heat pumps, retrofits, and socio-demographic targeting.

Quickstart
----------
>>> from household_energy import EnergyModel, ClimateField, load_config
>>> model = EnergyModel(gdf=my_geodataframe, climate_parquet="data/climate.parquet")
>>> for _ in range(168):  # one week
...     model.step()
>>> df = model.model_dc.get_model_vars_dataframe()

CLI entry points (installed via pip install -e .)
-------------------------------------------------
  energy-run          headless runner (single GeoJSON)
  energy-run-lsoa     per-LSOA batch runner
  energy-analyze      post-run plots and maps
  energy-server       Solara interactive dashboard
"""

from household_energy.model import EnergyModel
from household_energy.agent import HouseholdAgent, PersonAgent
from household_energy.climate import ClimateField
from household_energy.config import ModelConfig, load_config

__all__ = [
    "EnergyModel",
    "HouseholdAgent",
    "PersonAgent",
    "ClimateField",
    "ModelConfig",
    "load_config",
]
