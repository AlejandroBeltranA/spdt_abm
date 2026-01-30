"""Configuration loader for household_energy.

Provides a simple way to externalize model parameters into a YAML/JSON file.
Defaults live in ``config_defaults.yaml``; users can pass ``config_path`` to override.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_PATH = Path(__file__).with_name("config_defaults.yaml")


@dataclass
class ModelConfig:
    """Lightweight container for model-level parameters.

    We keep this intentionally loose; unknown fields are preserved in the raw dict.
    """

    raw: Dict[str, Any]

    @property
    def meta(self) -> Dict[str, Any]:
        return self.raw.get("meta", {})

    @property
    def name(self) -> str:
        return str(self.meta.get("name", "default"))

    @property
    def date(self) -> str:
        return str(self.meta.get("date", ""))

    @property
    def notes(self) -> str:
        return str(self.meta.get("notes", ""))

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def archetypes(self) -> Dict[str, Any]:
        return self.raw.get("archetypes", {})

    @property
    def controls(self) -> Dict[str, Any]:
        return self.raw.get("controls", {})

    @property
    def envelope_levers(self) -> Dict[str, Any]:
        return self.raw.get("envelope_levers", {})

    @property
    def systems(self) -> Dict[str, Any]:
        return self.raw.get("systems", {})

    @property
    def schedules(self) -> Dict[str, Any]:
        return self.raw.get("schedules", {})

    @property
    def meters(self) -> Dict[str, Any]:
        return self.raw.get("meters", {})

    @property
    def households(self) -> Dict[str, Any]:
        """Household-level options (optional hidp CSV, bedroom multipliers, etc.)."""
        return self.raw.get("households", {})


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Optional[str | Path] = None) -> ModelConfig:
    """Load defaults and merge optional overrides from ``config_path``."""

    base = _load_yaml(DEFAULT_PATH)
    if config_path:
        override_path = Path(config_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Config override not found: {override_path}")
        overrides = _load_yaml(override_path)
        base = _deep_merge(base, overrides)
    return ModelConfig(raw=base)


__all__ = ["ModelConfig", "load_config", "DEFAULT_PATH"]
