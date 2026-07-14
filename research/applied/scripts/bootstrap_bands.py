#!/usr/bin/env python3
"""
Shared parametric-bootstrap helper for the SERL fit scripts.

Why parametric, not a household resample
----------------------------------------
SERL 8963 is the *aggregated* release: we never see household-level rows, only
disclosure-controlled per-segment summaries (``mean``, ``sd``, ``se``,
``n_rounded``). So the honest uncertainty band on a fitted parameter is *not*
a resample over households (we don't have them) but a Monte-Carlo propagation
of SERL's own published sampling error:

    for b in 1..B:
        mean*_band ~ Normal(mean_band, se_band)        # one draw per band row
        refit on the perturbed band means
    SE(param)  = std over the B refits
    CI(param)  = 2.5 / 97.5 percentiles over the B refits

This treats each fit as an independent SA factor (Morris assumes independent
factors anyway), so an independent per-script bootstrap is exactly the right
granularity — no need to share a single perturbation draw across fits.

Each fit script supplies:
  - a loaded DataFrame that carries the value column it fits on plus a column
    (or derived array) of published standard errors aligned to those rows, and
  - a ``leaf_fn(perturbed_df) -> dict[str, float]`` that recomputes *only* the
    config-bound leaf parameters (the band multipliers, the setpoint, the
    per-person slope — whatever ends up in calibrated_config.yaml).

``bootstrap_leaves`` then returns ``{leaf_name: {se, ci_lo, ci_hi, n_boot_ok}}``
which the script merges into its YAML under a ``bootstrap`` block, alongside the
point estimates. The SA parameter table (build_sa_param_table.py) reads those
blocks; nothing in the model run path depends on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_se(
    data: pd.DataFrame,
    *,
    value_col: str,
    se_col: str | None = "se",
    sd_col: str = "sd",
    n_col: str = "n_rounded",
) -> np.ndarray:
    """Published SE per row, with sensible fallbacks.

    Preference order, per row:
      1. ``se_col`` if present and finite.
      2. ``sd_col`` / sqrt(``n_col``) — the textbook SE of the mean.
      3. 0.0 — a suppressed/edge band contributes no perturbation (conservative:
         it keeps that band pinned at its point value rather than inventing a
         spread).

    Returns a float array aligned to ``data`` rows.
    """
    n = len(data)
    out = np.zeros(n, dtype=float)

    se = (
        pd.to_numeric(data[se_col], errors="coerce").to_numpy(float)
        if (se_col is not None and se_col in data.columns)
        else np.full(n, np.nan)
    )
    sd = (
        pd.to_numeric(data[sd_col], errors="coerce").to_numpy(float)
        if sd_col in data.columns
        else np.full(n, np.nan)
    )
    nn = (
        pd.to_numeric(data[n_col], errors="coerce").to_numpy(float)
        if n_col in data.columns
        else np.full(n, np.nan)
    )

    use_se = np.isfinite(se) & (se >= 0)
    out[use_se] = se[use_se]

    need = ~use_se & np.isfinite(sd) & np.isfinite(nn) & (nn > 0)
    out[need] = sd[need] / np.sqrt(nn[need])

    return out


def bootstrap_leaves(
    data: pd.DataFrame,
    leaf_fn,
    *,
    value_col: str,
    se: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Parametric bootstrap of a fit's leaf parameters.

    Parameters
    ----------
    data
        The fit's loaded band table. Not mutated.
    leaf_fn
        ``perturbed_df -> dict[str, float]``. Must recompute the leaf params
        from the (perturbed) value column. Raising inside ``leaf_fn`` drops
        that draw rather than aborting the bootstrap (e.g. a draw that makes a
        reference-band slope non-positive).
    value_col
        Column perturbed each draw (``mean`` for daily fits, ``mean_kwh`` for
        diurnal).
    se
        Per-row standard errors aligned to ``data`` (use :func:`derive_se`).
    n_boot, seed
        Draw count and RNG seed (deterministic — no Date/random globals).

    Returns
    -------
    ``{leaf_name: {"se", "ci_lo", "ci_hi", "n_boot_ok"}}``
    """
    rng = np.random.default_rng(seed)
    base = pd.to_numeric(data[value_col], errors="coerce").to_numpy(float)
    se = np.asarray(se, dtype=float)
    se = np.where(np.isfinite(se), se, 0.0)
    n = len(base)

    acc: dict[str, list[float]] = {}
    n_ok = 0
    for _ in range(int(n_boot)):
        d = data.copy()
        d[value_col] = base + se * rng.standard_normal(n)
        try:
            leaves = leaf_fn(d)
        except Exception:
            continue
        n_ok += 1
        for k, v in leaves.items():
            fv = float(v)
            if np.isfinite(fv):
                acc.setdefault(k, []).append(fv)

    out: dict[str, dict] = {}
    for k, vals in acc.items():
        a = np.asarray(vals, dtype=float)
        out[k] = {
            "se":        float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "ci_lo":     float(np.percentile(a, 2.5)),
            "ci_hi":     float(np.percentile(a, 97.5)),
            "n_boot_ok": int(n_ok),
        }
    return out


def merge_point_and_bands(point: dict, bands: dict) -> dict:
    """Combine ``{leaf: value}`` point estimates with the bootstrap bands.

    Returns ``{leaf: {value, se, ci_lo, ci_hi, n_boot_ok}}`` — the shape the SA
    parameter table consumes. Leaves present in ``point`` but absent from
    ``bands`` (e.g. a leaf fixed by construction) get a zero band.
    """
    merged: dict[str, dict] = {}
    for leaf, value in point.items():
        b = bands.get(leaf, {})
        merged[leaf] = {
            "value":     float(value),
            "se":        float(b.get("se", 0.0)),
            "ci_lo":     float(b.get("ci_lo", value)),
            "ci_hi":     float(b.get("ci_hi", value)),
            "n_boot_ok": int(b.get("n_boot_ok", 0)),
        }
    return merged
