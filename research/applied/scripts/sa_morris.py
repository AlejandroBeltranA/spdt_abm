#!/usr/bin/env python3
"""
Morris elementary-effects sensitivity screen for the v5 ABM on Newcastle.

Method
------
Method of Morris (Morris 1991; Campolongo et al. 2007 μ* refinement). For k
factors we build r trajectories; each trajectory is k+1 model runs in which one
factor moves by ±Δ at a time, giving one elementary effect (EE) per factor per
trajectory. We report, per factor:

  μ*  = mean(|EE|)  — overall influence on the output (the screening ranking)
  σ   = std(EE)     — non-linearity / interaction with other factors

Total model runs = r·(k+1). Cheap relative to Sobol, and enough to *screen*
which parameters matter before any expensive variance decomposition.

What is perturbed
-----------------
The knobs from build_sa_param_table.py. SERL-fitted knobs move over their
design coordinate u∈[0,1] mapped to central ± k_sigma·SE (the parametric
bootstrap of SERL published sampling error). The literature knob (night
setback) moves over its cited [low, high]. Band-lookup knobs move coherently:
a single u shifts every band by the same number of its own SEs.

Two output metrics, both on a tier-stratified Newcastle LSOA subsample run
through the production transfer path (full-year, no window approximation):

  energy  : citywide annual energy, Σ abm_kwh over the subsample (headline)
  r2_cov  : R²(tot_ratio ~ coverage) — does spatial structure survive the
            perturbation? (secondary)

A subsample is used because r·(k+1) full-city runs are infeasible; the per-LSOA
energy response to a parameter perturbation is highly correlated across LSOAs,
so a tier-stratified subsample's mean response is a faithful proxy for the
citywide sensitivity *ranking* (it is not a recalibration). The subsample size
and trajectory count are logged so the coverage is never silently capped.

Usage:
  # smoke test (fast, proves the pipeline)
  python research/applied/scripts/sa_morris.py --r 1 --n-lsoa 6 --label smoke
  # full screen
  python research/applied/scripts/sa_morris.py --r 8 --n-lsoa 18 --max-procs 9
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(REPO))

from transfer import _city_paths  # noqa: E402
from household_energy.run_lsoa_batch import RunConfig, _run_single_lsoa  # noqa: E402
from utils import compute_confidence_tiers  # noqa: E402

LSOA_COL = "lsoa_code"


# ─────────────────────────────────────────────────────────────────────────────
# Config mutation
# ─────────────────────────────────────────────────────────────────────────────

def apply_design(base_cfg: dict, knobs: list[dict], u: np.ndarray) -> dict:
    """Map a Morris design row u∈[0,1]^k to a full config dict."""
    cfg = copy.deepcopy(base_cfg)
    m = cfg["model"]
    for knob, ui in zip(knobs, u):
        kind = knob["kind"]
        if kind == "scalar":
            val = knob["central"] + (2 * ui - 1) * knob["k_sigma"] * knob["se"]
            for key in knob["config_keys"]:
                m[key] = float(val)
        elif kind == "scalar_literature":
            val = knob["low"] + ui * (knob["high"] - knob["low"])
            for key in knob["config_keys"]:
                m[key] = float(val)
        elif kind == "lookup":
            z = (2 * ui - 1) * knob["k_sigma"]
            d = dict(m.get(knob["config_key"], {}))
            for b, cen in knob["central"].items():
                d[b] = float(cen + z * knob["se"][b])
            m[knob["config_key"]] = d
        else:
            raise ValueError(f"unknown knob kind {kind!r}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Morris sampling (standard B* trajectory construction)
# ─────────────────────────────────────────────────────────────────────────────

def morris_trajectories(k: int, r: int, p: int, rng: np.random.Generator):
    """Return an (r, k+1, k) array of design points in [0,1] and an (r, k)
    array giving, per trajectory, the order in which factors are perturbed.
    """
    delta = p / (2.0 * (p - 1.0))
    grid = np.arange(p) / (p - 1.0)
    feasible = grid[grid <= 1.0 - delta]  # so x*+Δ stays in [0,1]
    B = np.tril(np.ones((k + 1, k)), -1)
    J = np.ones((k + 1, k))

    trajs = np.empty((r, k + 1, k))
    orders = np.empty((r, k), dtype=int)
    for t in range(r):
        xstar = rng.choice(feasible, size=k)
        Dstar = np.diag(rng.choice([-1.0, 1.0], size=k))
        perm = rng.permutation(k)
        Pstar = np.eye(k)[perm]
        Bstar = (xstar[None, :] + (delta / 2.0) * ((2 * B - J) @ Dstar + J)) @ Pstar
        trajs[t] = np.clip(Bstar, 0.0, 1.0)
        orders[t] = perm
    return trajs, delta


def elementary_effects(traj_pts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """EE per factor for one trajectory. ``traj_pts`` is (k+1, k); ``y`` is
    (k+1,). Between consecutive rows exactly one coordinate changes by ±Δ;
    EE for that coordinate = Δy / Δx (sign handled by the coordinate delta).
    """
    k = traj_pts.shape[1]
    ee = np.full(k, np.nan)
    for j in range(k):
        dx = traj_pts[j + 1] - traj_pts[j]
        i = int(np.argmax(np.abs(dx)))
        if dx[i] != 0 and np.isfinite(y[j]) and np.isfinite(y[j + 1]):
            ee[i] = (y[j + 1] - y[j]) / dx[i]
    return ee


# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation on a stratified LSOA subsample
# ─────────────────────────────────────────────────────────────────────────────

def stratified_lsoas(city: str, year: int, n: int) -> list[str]:
    """Tier-stratified Newcastle LSOA subsample (deterministic, evenly spaced
    within each confidence tier) so R²(coverage) keeps its spread."""
    conf = REPO / "research/applied" / f"transfer_confidence_{city}_{year}.csv"
    c = pd.read_csv(conf)
    out: list[str] = []
    tiers = ["High", "Medium", "Low"]
    counts = c["confidence"].value_counts()
    total = sum(counts.get(t, 0) for t in tiers)
    for t in tiers:
        sub = c[c["confidence"] == t].sort_values(LSOA_COL)[LSOA_COL].tolist()
        if not sub:
            continue
        take = max(1, round(n * len(sub) / total))
        idx = np.linspace(0, len(sub) - 1, num=min(take, len(sub))).round().astype(int)
        out.extend(sub[i] for i in dict.fromkeys(idx))
    return sorted(dict.fromkeys(out))


def evaluate(cfg_dict: dict, lsoas: list[str], city: str, year: int,
             paths, max_procs: int) -> tuple[float, float]:
    """Run the subsample through the production transfer path; return
    (citywide annual energy Σabm_kwh, R²(tot_ratio~coverage))."""
    import multiprocessing as mp

    conv, geo, climate, hidp = paths
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        cfg_path = Path(tmp.name)

    rc = RunConfig(
        geojson=geo, climate=climate, hidp_csv=hidp,
        start_utc=f"{year}-01-01T00:00:00Z", end_utc=f"{year + 1}-01-01T00:00:00Z",
        days=None, local_tz="Europe/London", lsoa_col=LSOA_COL,
        outdir=REPO / "results_lsoa" / f"sa_{conv.epc_slug}",
        agent_collect_every=1, stamp=f"sa_{conv.epc_slug}_{year}",
        config_path=cfg_path, save_model_timeseries=False,
    )
    tasks = [(code, rc, i + 1, len(lsoas)) for i, code in enumerate(lsoas)]
    if max_procs == 1:
        rows = [_run_single_lsoa(*t) for t in tasks]
    else:
        with mp.Pool(processes=max(1, max_procs)) as pool:
            rows = pool.starmap(_run_single_lsoa, tasks)
    cfg_path.unlink(missing_ok=True)

    abm = pd.concat([r for r in rows if r is not None], ignore_index=True)
    energy = float(abm["abm_kwh"].sum())

    c = compute_confidence_tiers(abm, city, year=year)
    v = c[["tot_ratio", "coverage"]].replace([np.inf, -np.inf], np.nan).dropna()
    r2 = _ols_r2(v["tot_ratio"].to_numpy(), v["coverage"].to_numpy()) if len(v) >= 3 else float("nan")
    return energy, r2


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

# Readable parameter names for the figure (config keys are unwieldy).
KNOB_LABELS = {
    "heating_setpoint_C": "heating setpoint",
    "building_age_mult_heating_gas": "building-age mult.",
    "sap_band_mult_heating_gas": "SAP-band mult.",
    "heat_slope_area_bands": "floor-area mult.",
    "setpoint_setback_C": "unoccupied setback",
    "heating_slope_kWh_per_deg": "heating slope",
    "energy_per_person_home": "per-person load",
    "baseline_elec_area_bands": "elec. baseline area",
    "baseline_anchor_gas_kwh_per_hour": "gas baseline anchor",
    "baseline_anchor_elec_kwh_per_hour": "elec. baseline anchor",
}


def plot_morris(summary: pd.DataFrame, out_path: Path) -> None:
    """Two ranked panels: influence on citywide energy, and on the coverage fit.

    The σ (non-linearity) diagnostic is reported in the text, not plotted: σ is
    two orders of magnitude below μ* for every parameter, so a μ*–σ scatter sits
    flat on the axis and reads as empty. The second panel instead shows each
    parameter's effect on the per-LSOA coverage fit (R²), the structure the
    reliability layer rests on.
    """
    s = summary.sort_values("mu_star_energy_pct_base")
    labels = [KNOB_LABELS.get(k, k) for k in s["knob"]]
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)

    axes[0].barh(y, s["mu_star_energy_pct_base"], color="#4C72B0")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("μ*  (% of citywide energy, per ±2 SE move)", fontsize=9)
    axes[0].set_title("A  Influence on citywide energy", fontsize=11, loc="left", fontweight="bold")
    axes[0].tick_params(labelsize=8)

    axes[1].barh(y, s["mu_star_r2cov"], color="#55A868")
    axes[1].set_xlabel("μ*  (effect on per-LSOA R² of the coverage fit)", fontsize=9)
    axes[1].set_title("B  Influence on the coverage fit", fontsize=11, loc="left", fontweight="bold")
    axes[1].tick_params(labelsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", default="newcastle")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--calib-dir", type=Path, default=REPO / "results/serl_ledger")
    p.add_argument("--param-table", type=Path,
                   default=REPO / "results/sensitivity_analysis/sa_param_table_v2.yaml")
    p.add_argument("--knobs", default=None,
                   help="Comma-separated knob names to screen (subset of the "
                        "param table); the rest stay at their central config "
                        "value. Default: all. Use for the Phase-D top-N "
                        "cross-city robustness check.")
    p.add_argument("--r", type=int, default=8, help="Morris trajectories")
    p.add_argument("--levels", type=int, default=4, help="Morris grid levels p")
    p.add_argument("--n-lsoa", type=int, default=18, help="stratified LSOA subsample size")
    p.add_argument("--max-procs", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="newcastle")
    p.add_argument("--out-dir", type=Path, default=REPO / "results/sensitivity_analysis")
    p.add_argument("--replot", action="store_true",
                   help="Skip the screen; re-render the figure from the saved "
                        "sa_morris_<label>.csv (use after editing plot_morris).")
    args = p.parse_args(argv)

    if args.replot:
        summary = pd.read_csv(args.out_dir / f"sa_morris_{args.label}.csv")
        plot_morris(summary, args.out_dir / f"sa_morris_{args.label}.png")
        print(f"[SA] re-rendered sa_morris_{args.label}.png from saved CSV")
        return 0

    base_cfg = yaml.safe_load((args.calib_dir / "calibrated_config.yaml").read_text())
    knobs = yaml.safe_load(args.param_table.read_text())["knobs"]
    if args.knobs:
        want = [s.strip() for s in args.knobs.split(",") if s.strip()]
        by_name = {kn["name"]: kn for kn in knobs}
        missing = [w for w in want if w not in by_name]
        if missing:
            raise SystemExit(f"--knobs names not in param table: {missing}")
        knobs = [by_name[w] for w in want]
    k = len(knobs)
    rng = np.random.default_rng(args.seed)

    lsoas = stratified_lsoas(args.city, args.year, args.n_lsoa)
    paths = _city_paths(args.city)
    trajs, delta = morris_trajectories(k, args.r, args.levels, rng)
    n_runs = args.r * (k + 1)
    print(f"[SA] {args.city} | {k} knobs | r={args.r} | {len(lsoas)} LSOAs "
          f"| {n_runs} model runs | Δ={delta:.3f}", flush=True)
    print(f"[SA] subsample LSOAs: {lsoas}", flush=True)

    # Evaluate every trajectory point.
    energy = np.full((args.r, k + 1), np.nan)
    r2 = np.full((args.r, k + 1), np.nan)
    run_i = 0
    for t in range(args.r):
        for j in range(k + 1):
            cfg = apply_design(base_cfg, knobs, trajs[t, j])
            e, rr = evaluate(cfg, lsoas, args.city, args.year, paths, args.max_procs)
            energy[t, j], r2[t, j] = e, rr
            run_i += 1
            print(f"[SA] run {run_i}/{n_runs} (traj {t} pt {j}): "
                  f"energy={e:,.0f} kWh  r2_cov={rr:.3f}", flush=True)

    # Elementary effects → μ*, σ.
    ee_e = np.vstack([elementary_effects(trajs[t], energy[t]) for t in range(args.r)])
    ee_r = np.vstack([elementary_effects(trajs[t], r2[t]) for t in range(args.r)])
    names = [kn["name"] for kn in knobs]
    summary = pd.DataFrame({
        "knob": names,
        "mu_star_energy": np.nanmean(np.abs(ee_e), axis=0),
        "mu_energy":      np.nanmean(ee_e, axis=0),
        "sigma_energy":   np.nanstd(ee_e, axis=0),
        "mu_star_r2cov":  np.nanmean(np.abs(ee_r), axis=0),
        "sigma_r2cov":    np.nanstd(ee_r, axis=0),
    }).sort_values("mu_star_energy", ascending=False).reset_index(drop=True)

    base_energy = float(np.nanmedian(energy))
    summary["mu_star_energy_pct_base"] = summary["mu_star_energy"] / base_energy * 100.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / f"sa_morris_{args.label}.csv", index=False)
    np.savez(args.out_dir / f"sa_morris_{args.label}_raw.npz",
             trajs=trajs, energy=energy, r2=r2, names=np.array(names))
    with open(args.out_dir / f"sa_morris_{args.label}_meta.json", "w") as f:
        json.dump({
            "city": args.city, "year": args.year, "k": k, "r": args.r,
            "levels": args.levels, "delta": delta, "n_lsoa": len(lsoas),
            "lsoas": lsoas, "n_runs": n_runs, "base_energy_kwh": base_energy,
            "calib_dir": str(args.calib_dir),
        }, f, indent=2)
    plot_morris(summary, args.out_dir / f"sa_morris_{args.label}.png")

    print("\n[SA] μ* ranking on citywide energy (Newcastle subsample):")
    print(summary[["knob", "mu_star_energy_pct_base", "sigma_energy",
                   "mu_star_r2cov"]].to_string(index=False))
    print(f"\n[SA] wrote sa_morris_{args.label}.{{csv,png,npz}} + meta to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
