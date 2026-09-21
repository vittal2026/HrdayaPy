#!/usr/bin/env python3
"""
run_experiment4.py
===================================================================
Experiment 4 -- PMJ capture threshold and propagation delay, studied
SEPARATELY for the orthodromic (Purkinje -> myocardium) and antidromic
(myocardium -> Purkinje) directions, on the same coupled Purkinje-fibre
+ myocardium slab geometry used in Experiments 2/3 (Section 2.11.4 /
3.4 of the manuscript). Thin sweep driver on top of experiment_common.py
(see that module's docstring); this script owns the sweep strategy only.

TWO-PHASE (COARSE-THEN-FINE) SWEEP
-----------------------------------
A single fixed c_pmj grid is a poor choice here because we don't know in
advance where the capture/block boundary sits, and it can sit in very
different places for different n_pmj -- our first pass showed
n_pmj in {1, 2} never achieving propagating capture anywhere in a
{0.5, 1, 2}x (and then {0.1..4}x) bracket, i.e. that range was entirely
wasted for those rows, while it may be massively oversampling a range
that's already deep in the "always captures" regime for larger n_pmj.

So instead:

  Phase 1 (coarse) : a WIDE, sparse, log-spaced c_pmj grid
                      (C_PMJ_SCALES_COARSE, default spanning 4 decades,
                      0.01x-100x) run for every (direction, n_pmj) cell.
                      Cheap, and its only job is to locate roughly where
                      the transition from block to capture happens (or
                      establish that a cell is always-captures or
                      never-captures over that whole range).
  Phase 2 (fine)   : for each cell where phase 1 found an actual
                      block->capture transition (a bracket:
                      [last non-capturing scale, first capturing scale]),
                      run N_FINE_POINTS additional log-spaced points
                      STRICTLY INSIDE that bracket. Cells that were
                      always-capture or never-capture over the whole
                      coarse range are left alone (see the printed
                      warnings -- widen --c_pmj_scales_coarse and rerun
                      if a cell needs a wider bracket).

Total simulations = (coarse grid size) + sum of N_FINE_POINTS over cells
that actually had a transition -- typically far fewer than a single
uniform fine grid over the whole range would need, and concentrated
where the answer is actually uncertain.

ANISOTROPIC PORT
-----------------------------------
experiment_common.run_directional() and base_values.py were already ported
to the anisotropic solver (sigma_l/sigma_t + a per-voxel fibre field
uniform along Z, taken from niederer_values.py -- see those modules'
docstrings). This script's only anisotropy-related change is in run_cell():
it now passes sigma_l=ec.SIGMA_L, sigma_t=ec.SIGMA_T instead of the retired
sigma_M=ec.SIGMA_M. Per the manuscript's Section 2.11.4 scope, Experiment 4
still sweeps only c_pmj and n_pmj -- (sigma_l, sigma_t) are fixed at
base_values.py's defaults, not swept here.

Run with:  python3 run_experiment4.py
Optional:  --dt 0.05 --dx_p 0.3152 --dx_m 0.3152 --T_base 180
           --c_pmj_scales_coarse 0.01 0.1 1 10 100 --n_fine 4
           --n_pmj_values 1 2 4 8 --directions orthodromic antidromic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# experiment_common.py / base_values.py now live ALONGSIDE this script
# (both directly in this same directory), not one level up as before.
# Python already puts the running script's own directory on sys.path
# automatically (as sys.path[0]) when invoked as `python3
# run_experiment4.py`, so this insert is technically redundant for that
# invocation style -- it's kept explicit anyway as a safety net for
# other invocation methods (e.g. `python3 -m`, or a test runner) that
# don't guarantee that behaviour.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_common as ec

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Sweep defaults
# =============================================================================
C_PMJ_SCALES_COARSE = [0.0001, 0.001, 0.01, 0.1, 1.0]
N_PMJ_VALUES = [1, 2, 4]
DIRECTIONS = ["orthodromic", "antidromic"]
N_FINE_POINTS = 4   # additional log-spaced points per detected bracket


# =============================================================================
# Single-cell run + bookkeeping
# =============================================================================
def run_cell(direction: str, n_pmj: int, scale: float, phase: str,
             dt: float, dx_p: float, dx_m: float, T: float) -> dict:
    c_pmj = ec.C_PMJ * scale
    tag = (f"exp4_{phase}_{direction}_npmj{n_pmj}_cpmjscale{scale:g}"
           f"_dt{dt:g}_dxp{dx_p:g}_dxm{dx_m:g}")
    print(f"\n--- [{phase}] {direction}  n_pmj={n_pmj}  c_pmj scale={scale:g} "
          f"(c_pmj={c_pmj:.4g} mS) ---")
    out = ec.run_directional(
        sigma_l=ec.SIGMA_L, sigma_t=ec.SIGMA_T, sigma_P=ec.SIGMA_P,
        dt=dt, dx_p=dx_p, dx_m=dx_m, T=T, tag=tag,
        c_pmj=c_pmj, n_pmj=n_pmj, direction=direction,
        out_dir=OUT_DIR,
    )
    delay_str = (f"{out['pmj_delay_ms']:.3f} ms"
                 if out["pmj_propagating_capture"] and np.isfinite(out["pmj_delay_ms"])
                 else "-- (no propagating capture)")
    print(f"    junction_capture_fraction = {out['pmj_junction_capture_fraction']:.3f}   "
          f"propagating_capture = {out['pmj_propagating_capture']}   "
          f"n_myo_activated = {out['pmj_n_myo_activated']}   delay = {delay_str}")

    # Observed Purkinje conduction velocity for this run, via the same
    # restricted-region fit Experiment 3 uses (ec.estimate_purkinje_cv).
    cv_p = ec.estimate_purkinje_cv(out["p_arc_length"], out["p_act"])
    print(f"    Purkinje CV = {cv_p['cv_mm_per_ms']:.2f} mm/ms "
          f"(n={cv_p['n_points']}, R^2={cv_p['r2']:.4f})")
    return dict(direction=direction, n_pmj=n_pmj, phase=phase,
                c_pmj_scale=scale, c_pmj=c_pmj,
                junction_capture_fraction=out["pmj_junction_capture_fraction"],
                propagating_capture=bool(out["pmj_propagating_capture"]),
                n_myo_activated=out["pmj_n_myo_activated"],
                pmj_delay_ms=out["pmj_delay_ms"])


def run_jobs(jobs: list[tuple[str, int, float, str]],
             dt: float, dx_p: float, dx_m: float, T: float) -> list[dict]:
    return [run_cell(direction, n_pmj, scale, phase, dt, dx_p, dx_m, T)
            for direction, n_pmj, scale, phase in jobs]


# =============================================================================
# Bracket detection -- assumes capture is monotonic non-decreasing in c_pmj
# at fixed n_pmj (physically expected: more coupling conductance should
# never make capture LESS likely). Under that assumption, the transition
# point is "smallest scale from which every larger tested scale also
# captures" -- robust to isolated noise near the boundary, unlike a naive
# first-True search.
# =============================================================================
def find_transition(results: list, direction: str, n_pmj: int) -> dict:
    """
    Returns one of:
      {"status": "always_capture", "hi": <smallest tested scale>}
          -- capture already holds at the smallest scale tested; the true
             threshold is <= that value, not localised further.
      {"status": "never_capture", "lo": <largest tested scale>}
          -- capture never holds anywhere in the tested range; the true
             threshold is > that value -- widen --c_pmj_scales_coarse.
      {"status": "bracketed", "lo": <last non-capturing>, "hi": <first
             sustained-capturing>}
          -- the interesting case: a genuine transition was found between
             these two tested scales.
    """
    rows = sorted([r for r in results if r["direction"] == direction and r["n_pmj"] == n_pmj],
                   key=lambda r: r["c_pmj_scale"])
    if not rows:
        return dict(status="no_data")
    caps = [bool(r["propagating_capture"]) for r in rows]
    scales = [r["c_pmj_scale"] for r in rows]

    # smallest i such that caps[i:] are all True
    i_sustained = next((i for i in range(len(caps)) if all(caps[i:])), None)

    if i_sustained == 0:
        return dict(status="always_capture", hi=scales[0])
    if i_sustained is None:
        return dict(status="never_capture", lo=scales[-1])
    return dict(status="bracketed", lo=scales[i_sustained - 1], hi=scales[i_sustained])


def make_fine_jobs(coarse_results: list, directions: list, n_pmj_values: list,
                    n_fine: int) -> tuple[list, dict]:
    fine_jobs = []
    transitions = {}
    for direction in directions:
        for n_pmj in n_pmj_values:
            t = find_transition(coarse_results, direction, n_pmj)
            transitions[(direction, n_pmj)] = t
            if t["status"] == "bracketed":
                fine_scales = np.geomspace(t["lo"], t["hi"], num=n_fine + 2)[1:-1]
                fine_jobs += [(direction, n_pmj, float(s), "fine") for s in fine_scales]
    return fine_jobs, transitions


# =============================================================================
# Reporting
# =============================================================================
def print_transition_summary(transitions: dict, coarse_range: tuple):
    lo_c, hi_c = coarse_range
    print("\n" + "=" * 72)
    print("Capture/block transition summary (after coarse pass)")
    print("=" * 72)
    for (direction, n_pmj), t in sorted(transitions.items()):
        if t["status"] == "always_capture":
            msg = f"always captures over tested range (<= {t['hi']:g}x) -- lower --c_pmj_scales_coarse to localise further"
        elif t["status"] == "never_capture":
            msg = f"never captures over tested range (> {t['lo']:g}x) -- RAISE --c_pmj_scales_coarse (max tested was {hi_c:g}x)"
        elif t["status"] == "bracketed":
            msg = f"bracketed in [{t['lo']:g}x, {t['hi']:g}x] -- refining"
        else:
            msg = "no data"
        print(f"  {direction:>11} n_pmj={n_pmj}: {msg}")


def print_final_thresholds(all_results: list, directions: list, n_pmj_values: list):
    print("\n" + "=" * 72)
    print("Final localised capture thresholds (after coarse+fine)")
    print("=" * 72)
    for direction in directions:
        for n_pmj in n_pmj_values:
            t = find_transition(all_results, direction, n_pmj)
            if t["status"] == "bracketed":
                width_decades = np.log10(t["hi"] / t["lo"])
                print(f"  {direction:>11} n_pmj={n_pmj}: threshold in "
                      f"[{t['lo']:.4g}x, {t['hi']:.4g}x]  "
                      f"(bracket width: {width_decades:.2f} decades)")
            elif t["status"] == "always_capture":
                print(f"  {direction:>11} n_pmj={n_pmj}: always captures (<= {t['hi']:g}x tested)")
            elif t["status"] == "never_capture":
                print(f"  {direction:>11} n_pmj={n_pmj}: never captures (> {t['lo']:g}x tested)")


def print_delay_table(results: list, direction: str, fixed_n_pmj: int):
    print(f"\nPMJ propagation summary vs. c_pmj ({direction}, n_pmj={fixed_n_pmj} fixed):")
    print(f"{'phase':>6} {'c_pmj scale':>12} {'c_pmj (mS)':>12} {'junction cap.':>13} "
          f"{'propagating':>12} {'n_myo_act':>10} {'delay (ms)':>11}")
    for r in sorted([r for r in results
                      if r["direction"] == direction and r["n_pmj"] == fixed_n_pmj],
                     key=lambda r: r["c_pmj_scale"]):
        delay_str = (f"{r['pmj_delay_ms']:.3f}"
                     if r["propagating_capture"] and np.isfinite(r["pmj_delay_ms"]) else "--")
        print(f"{r['phase']:>6} {r['c_pmj_scale']:>12.3g} {r['c_pmj']:>12.4g} "
              f"{r['junction_capture_fraction']:>13.3f} "
              f"{'yes' if r['propagating_capture'] else 'no':>12} "
              f"{r['n_myo_activated']:>10d} {delay_str:>11}")


def make_capture_map(all_results: list, directions: list, n_pmj_values: list, out_path: Path):
    """
    Scatter/step plot of propagating_capture (Y/N) vs. c_pmj scale (log x),
    one row per n_pmj, one panel per direction. Replaces the fixed-grid
    heatmap from the single-pass version -- coarse+fine sampling is
    irregular (different points tested for different n_pmj), which a
    heatmap can't represent honestly.
    """
    fig, axes = plt.subplots(1, len(directions), figsize=(7 * len(directions), 4.5), squeeze=False)
    axes = axes[0]
    default_scale = 1.0
    default_n_pmj = ec.N_PMJ

    for ax, direction in zip(axes, directions):
        for n_pmj in n_pmj_values:
            rows = sorted([r for r in all_results
                            if r["direction"] == direction and r["n_pmj"] == n_pmj],
                           key=lambda r: r["c_pmj_scale"])
            if not rows:
                continue
            xs = [r["c_pmj_scale"] for r in rows]
            ys = [n_pmj] * len(rows)
            colors = ["#2ca02c" if r["propagating_capture"] else "#d62728" for r in rows]
            markers_coarse = [r["phase"] == "coarse" for r in rows]
            for x, y, c, is_coarse in zip(xs, ys, colors, markers_coarse):
                ax.scatter(x, y, color=c, marker="o" if is_coarse else "D",
                           s=70 if is_coarse else 45, zorder=3,
                           edgecolors="black", linewidths=0.5)
            t = find_transition(all_results, direction, n_pmj)
            if t["status"] == "bracketed":
                mid = np.sqrt(t["lo"] * t["hi"])   # geometric mean, i.e. midpoint in log space
                # short vertical tick centred on this row's y-position, in
                # log-y data coordinates (robust regardless of whether
                # n_pmj_values happens to be evenly log-spaced)
                y_lo, y_hi = n_pmj / 1.3, n_pmj * 1.3
                ax.plot([mid, mid], [y_lo, y_hi], color="black", linewidth=1.0, alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_yticks(n_pmj_values)
        ax.set_yticklabels([str(n) for n in n_pmj_values])
        ax.set_xlabel(r"$c_{PMJ}$ ($\times$ default)")
        ax.set_ylabel(r"$n_{PMJ}$")
        ax.set_title(f"{direction.capitalize()}: propagating capture "
                      "(green=yes, red=no; o=coarse, D=fine)")
        ax.axvline(default_scale, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_results(all_results: list, args, out_dir: Path):
    json_path = out_dir / "exp4_pmj_results.json"
    with open(json_path, "w") as f:
        json.dump(dict(
            results=all_results,
            discretisation=dict(dt=args.dt, dx_p=args.dx_p, dx_m=args.dx_m, T=args.T_base),
            c_pmj_scales_coarse=args.c_pmj_scales_coarse, n_fine=args.n_fine,
            n_pmj_values=args.n_pmj_values,
            production_default=dict(c_pmj=ec.C_PMJ, n_pmj=ec.N_PMJ),
            propagation_check=dict(myo_radius_mm=ec.PROPAGATION_CHECK_RADIUS_MM,
                                    purkinje_arc_mm=ec.PURKINJE_PROPAGATION_CHECK_MM),
            anti_stim=dict(target_mm=ec.ANTI_STIM_TARGET_MM.tolist(),
                           radius_mm=ec.ANTI_STIM_RADIUS_MM),
        ), f, indent=2)

    npz_path = out_dir / "exp4_pmj_results.npz"
    np.savez(
        npz_path,
        direction=np.array([r["direction"] for r in all_results]),
        n_pmj=np.array([r["n_pmj"] for r in all_results]),
        phase=np.array([r["phase"] for r in all_results]),
        c_pmj_scale=np.array([r["c_pmj_scale"] for r in all_results]),
        c_pmj=np.array([r["c_pmj"] for r in all_results]),
        junction_capture_fraction=np.array([r["junction_capture_fraction"] for r in all_results]),
        propagating_capture=np.array([r["propagating_capture"] for r in all_results]),
        n_myo_activated=np.array([r["n_myo_activated"] for r in all_results]),
        delay_ms=np.array([r["pmj_delay_ms"] for r in all_results]),
    )
    return json_path, npz_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dt", type=float, default=ec.REF_DT, help="solver time step [ms]")
    ap.add_argument("--dx_p", type=float, default=ec.REF_DX_P, help="Purkinje segment length [mm]")
    ap.add_argument("--dx_m", type=float, default=ec.REF_DX_M, help="myocardial voxel size [mm]")
    ap.add_argument("--T_base", type=float, default=180.0,
                     help="simulated time [ms] for every run")
    ap.add_argument("--c_pmj_scales_coarse", type=float, nargs="+", default=C_PMJ_SCALES_COARSE,
                     help="wide, sparse log-spaced c_pmj scan used to bracket the "
                          "capture/block transition for each (direction, n_pmj) cell")
    ap.add_argument("--n_fine", type=int, default=N_FINE_POINTS,
                     help="number of additional log-spaced points to sample strictly "
                          "inside each detected bracket")
    ap.add_argument("--n_pmj_values", type=int, nargs="+", default=N_PMJ_VALUES,
                     help="myocardial nodes coupled per PMJ terminal")
    ap.add_argument("--directions", type=str, nargs="+", default=DIRECTIONS,
                     choices=list(ec.DIRECTIONS), help="which direction(s) to run")
    ap.add_argument("--skip_fine", action="store_true",
                     help="run the coarse pass only (for a quick first look)")
    args = ap.parse_args()

    n_coarse = len(args.c_pmj_scales_coarse) * len(args.n_pmj_values) * len(args.directions)
    print("=" * 72)
    print("Experiment 4: PMJ capture threshold and propagation delay (coarse-then-fine)")
    ec.print_base_values()
    print(f"Discretisation fixed at dt={args.dt} ms, dx_p={args.dx_p} mm, dx_m={args.dx_m} mm, "
          f"T={args.T_base} ms")
    print(f"Phase 1 (coarse): c_pmj scales {args.c_pmj_scales_coarse} x "
          f"n_pmj {args.n_pmj_values} x directions {args.directions}  "
          f"= {n_coarse} simulations")
    print("=" * 72)

    coarse_jobs = [(direction, n_pmj, scale, "coarse")
                   for direction in args.directions
                   for n_pmj in args.n_pmj_values
                   for scale in args.c_pmj_scales_coarse]
    coarse_results = run_jobs(coarse_jobs, args.dt, args.dx_p, args.dx_m, args.T_base)

    coarse_range = (min(args.c_pmj_scales_coarse), max(args.c_pmj_scales_coarse))
    fine_jobs, transitions = make_fine_jobs(coarse_results, args.directions,
                                             args.n_pmj_values, args.n_fine)
    print_transition_summary(transitions, coarse_range)

    fine_results = []
    if args.skip_fine:
        print("\n--skip_fine set: skipping the refinement pass.")
    elif fine_jobs:
        print(f"\nPhase 2 (fine): {len(fine_jobs)} additional simulations "
              f"across {len(set((d, n) for d, n, _, _ in fine_jobs))} bracketed cell(s)")
        fine_results = run_jobs(fine_jobs, args.dt, args.dx_p, args.dx_m, args.T_base)
    else:
        print("\nNo bracketed transitions found -- nothing to refine "
              "(check the always/never-capture warnings above).")

    all_results = coarse_results + fine_results
    json_path, npz_path = save_results(all_results, args, OUT_DIR)

    fig_path = OUT_DIR / "exp4_pmj_capture_map.png"
    make_capture_map(all_results, args.directions, args.n_pmj_values, fig_path)

    for direction in args.directions:
        print_delay_table(all_results, direction, fixed_n_pmj=ec.N_PMJ)

    print_final_thresholds(all_results, args.directions, args.n_pmj_values)

    print(f"\nTotal simulations run: {len(all_results)} "
          f"({len(coarse_results)} coarse + {len(fine_results)} fine)")
    print(f"Saved combined results to {npz_path} and {json_path}")
    print(f"Saved capture map to {fig_path}")


if __name__ == "__main__":
    main()
