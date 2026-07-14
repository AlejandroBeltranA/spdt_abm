"""Dated run snapshots for the research pipeline.

A *run* is a dated folder ``results/runs/<YYYY-MM-DD>/`` holding the run's
calibrated config, a manifest, and the small summary tables the notebooks produce
(validation, sensitivity, policy). Large decompositions stay as the version-tagged
"latest" CSVs and are referenced by date in the manifest rather than copied.

The most recent run date is recorded in ``results/runs/latest.txt``, so downstream
notebooks read the latest run by default (``RUN_DATE=None``) without coordinating
clocks. Set ``RUN_DATE="2026-06-24"`` in a notebook to read a specific snapshot.

The existing v7 paths (``results/calibration_v7_cohort/``, ``results_lsoa/*_v7.csv``)
remain the mutable "latest" working copies the chain reads when no run dir exists.
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "results" / "runs"
LATEST_CONFIG = REPO / "results" / "calibration_v7_cohort" / "calibrated_config.yaml"


def today() -> str:
    return datetime.date.today().isoformat()


def _latest() -> str | None:
    p = RUNS / "latest.txt"
    return p.read_text().strip() if p.exists() else None


def start_run(date: str | None = None) -> Path:
    """Begin (or reopen) a dated run and record it as latest. Called by notebook 1."""
    d = date or today()
    p = RUNS / d
    p.mkdir(parents=True, exist_ok=True)
    (RUNS / "latest.txt").write_text(d)
    return p


def run_dir(date: str | None = None, create: bool = False) -> Path:
    """Resolve a run dir: explicit date, else the latest run, else today's."""
    d = date or _latest() or today()
    p = RUNS / d
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def config_for(date: str | None = None) -> Path:
    """The config a downstream notebook should read: the run's frozen copy if it
    exists, else the mutable latest working copy."""
    c = run_dir(date) / "calibrated_config.yaml"
    return c if c.exists() else LATEST_CONFIG


def write_manifest(run: Path, **fields) -> None:
    (run / "manifest.json").write_text(
        json.dumps({"run_date": run.name, **fields}, indent=2, default=str)
    )
