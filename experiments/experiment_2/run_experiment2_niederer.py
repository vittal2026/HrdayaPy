#!/usr/bin/env python3
"""
run_experiment2_niederer.py
===================================================================
Reproduction of the Niederer et al. (2011) N-version benchmark on the
ANISOTROPIC solver, at the paper's OWN three tested spatial resolutions
(dx=0.1, 0.2, 0.5mm, all at dt=0.01ms -- see niederer_values.LEVELS).

WHY THIS ISN'T A CONVERGENCE SWEEP LIKE run_experiment2.py
-------------------------------------------------------------------------
run_experiment2.py's multi-level refinement sweep exists to answer "does
refining our own grid change our own answer" -- there, the only available
ground truth is internal self-consistency across levels. Here, each level
compares against an INDEPENDENT, EXACT literature reference row (the
paper's own 11-code results at that same dx/dt), so this is an absolute-
accuracy check at three separate points, not a self-consistency check
across them. There's no manifest-based "skip already-converged levels"
convergence-tolerance logic here for the same reason -- each level is a
separate pass/fail against its own ground truth, not a step in a
refinement ladder.

The manifest/checkpointing below exists purely so a crash partway through
(e.g. the expensive dx=0.1mm level) doesn't lose already-completed,
already-expensive levels -- same practical rationale as
run_experiment2.py's checkpointing, different reason for existing.

WHAT THIS DOES NOT VALIDATE (see prior discussion)
-------------------------------------------------------------------------
Fibre is uniform and aligned with the long (X) grid axis throughout, per
the standard benchmark's own geometry, at all three levels. That means
this exercises the sigma_l/sigma_t plumbing and the anisotropic-CV-vs-
propagation-angle behaviour (propagation along the P1->P8 diagonal is
oblique to the fibre, a real and useful check) -- but the FIBRE itself is
never oblique to the voxel grid, so this does not exercise the
18-connectivity edge-diagonal stencil at all. A separate rotated-fibre
variant is still needed for that.

Run with:  python3 run_experiment2_niederer.py
Optional:  --levels 0.2 0.5        (run a subset, e.g. skip the expensive
                                     dx=0.1mm level for a quick check)
           --start_level 0.2       (skip levels already in the manifest
                                     up to and including this one)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import niederer_values as nv
import niederer_common as nc

OUT_DIR = Path(__file__).resolve().parent / "outputs_niederer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def compare_to_reference(dx_m: float, point_times: dict) -> dict:
    """
    Exact match now (not an interpolated envelope): each level's dx_m has
    its own reference row in niederer_values.REFERENCE_ACTIVATION_TIMES_MS.
    "Within envelope" here means within reference mean +/- 2*SD -- a
    literal literature-derived acceptance band, not an interpolated one.
    """
    ref = nv.REFERENCE_ACTIVATION_TIMES_MS[dx_m]
    out = {}
    for p in POINT_ORDER:
        ours = point_times[p]["t_ms"]
        r = ref[p]
        within = abs(ours - r["mean_ms"]) <= 2 * r["sd_ms"]
        out[p] = dict(ours_ms=ours, ref=r, within_2sd=bool(within))
    return out


def print_comparison_table(dx_m: float, point_times: dict, comparison: dict):
    print(f"\n  dx={dx_m}mm  {'point':>6} {'ours (ms)':>10} {'literature (ms)':>18} "
          f"{'dist(mm)':>9}  status")
    for p in POINT_ORDER:
        c = comparison[p]
        r = c["ref"]
        flag = "OK" if c["within_2sd"] else "<-- OUTSIDE mean+/-2SD"
        print(f"          {p:>6} {c['ours_ms']:>10.2f} "
              f"{r['mean_ms']:>10.2f} +/- {r['sd_ms']:<6.2f} "
              f"{point_times[p]['nearest_dist_mm']:>9.3f}  {flag}")


def make_comparison_plot(all_results: dict, out_path: Path):
    """all_results: {dx_m: dict(point_times, comparison, diagonal)}"""
    dx_levels = sorted(all_results.keys(), reverse=True)   # coarsest first, matches Fig.2's blue/green/red ordering
    colors = {0.1: "C3", 0.2: "C2", 0.5: "C0"}   # matches the paper's own Figure 2
                                                    # convention exactly: red=0.1mm (finest),
                                                    # green=0.2mm, blue=0.5mm (coarsest)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel 1: P1-P9 point comparison, one column of points per level ----
    ax = axes[0]
    x = np.arange(len(POINT_ORDER))
    width = 0.8 / (len(dx_levels) + 1)
    for i, dx_m in enumerate(dx_levels):
        comparison = all_results[dx_m]["comparison"]
        ref_mean = [comparison[p]["ref"]["mean_ms"] for p in POINT_ORDER]
        ref_sd   = [comparison[p]["ref"]["sd_ms"]   for p in POINT_ORDER]
        ours     = [comparison[p]["ours_ms"]        for p in POINT_ORDER]
        c = colors.get(dx_m, f"C{i+4}")
        offset = (i - (len(dx_levels)-1)/2) * width
        ax.errorbar(x + offset, ref_mean, yerr=ref_sd, fmt="s", color=c,
                    alpha=0.5, capsize=2, label=f"literature, dx={dx_m}mm")
        ax.plot(x + offset, ours, "o", color=c, markersize=7, zorder=5,
                 markeredgecolor="black", markeredgewidth=0.5, label=f"ours, dx={dx_m}mm")
    ax.set_xticks(x); ax.set_xticklabels(POINT_ORDER)
    ax.set_ylabel("Activation time (ms)")
    ax.set_title("Point-wise activation times, P1-P9")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # ---- Panel 2: P1->P8 diagonal curves, one per level (mirrors the
    #      paper's own Figure 2 red/green/blue = 0.1/0.2/0.5mm layout) ----
    ax = axes[1]
    diag_len = np.linalg.norm(nv.corner_points_mm()["P8"] - nv.corner_points_mm()["P1"])
    for dx_m in dx_levels:
        diagonal = all_results[dx_m]["diagonal"]
        c = colors.get(dx_m, "C5")
        ax.plot(diagonal["arc_length_mm"], diagonal["t_ms"], "-", color=c,
                 label=f"ours, dx={dx_m}mm")
        comparison = all_results[dx_m]["comparison"]
        ax.plot(0.0,      comparison["P1"]["ref"]["mean_ms"], "s", color=c, zorder=4)
        ax.plot(diag_len, comparison["P8"]["ref"]["mean_ms"], "s", color=c, zorder=4)
    ax.set_xlabel("Arc length along P1->P8 diagonal (mm)")
    ax.set_ylabel("Activation time (ms)")
    ax.set_title("Activation along the P1->P8 diagonal\n(squares = literature mean at P1/P8)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                     help="subset of dx_m values to run (default: all of niederer_values.LEVELS)")
    ap.add_argument("--start_level", type=float, default=None,
                     help="skip levels with dx_m >= this value if already present in the "
                          "manifest (coarser levels are cheaper and run first below)")
    args = ap.parse_args()

    levels = nv.LEVELS if args.levels is None else \
        [lv for lv in nv.LEVELS if lv["dx_m"] in args.levels]
    if not levels:
        sys.exit(f"No matching levels for --levels {args.levels} "
                  f"(available: {[lv['dx_m'] for lv in nv.LEVELS]})")

    # Coarsest (cheapest) first -- so a crash on the expensive dx=0.1mm
    # level still leaves the cheaper levels' results checkpointed.
    levels = sorted(levels, key=lambda lv: -lv["dx_m"])

    print("=" * 72)
    print("Niederer benchmark reproduction (anisotropic solver, paper's own 3 levels)")
    nv.print_niederer_values()
    print("=" * 72)

    manifest_path = OUT_DIR / "exp2_niederer_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    all_results = {}
    for lv in levels:
        dx_m, dt = lv["dx_m"], lv["dt"]
        key = f"{dx_m:g}"
        tag = f"niederer_dx{dx_m:g}_dt{dt:g}"

        if args.start_level is not None and dx_m >= args.start_level and key in manifest:
            print(f"\n--- dx={dx_m}mm dt={dt}ms --- (skipped, loaded from {manifest_path.name})")
            cached = manifest[key]
            point_times = cached["point_times"]
            diagonal = dict(arc_length_mm=np.array(cached["diagonal"]["arc_length_mm"]),
                             t_ms=np.array(cached["diagonal"]["t_ms"]))
        else:
            print(f"\n--- dx={dx_m}mm dt={dt}ms ---")
            out = nc.run_niederer(dx_m=dx_m, dt=dt, T=nv.T, tag=tag, out_dir=OUT_DIR)
            point_times = nc.extract_point_activation_times(out["myo_coords"], out["myo_act"], dx_m)
            diagonal = nc.extract_diagonal_curve(out["myo_coords"], out["myo_act"], dx_m)

        comparison = compare_to_reference(dx_m, point_times)
        print_comparison_table(dx_m, point_times, comparison)
        all_results[dx_m] = dict(point_times=point_times, comparison=comparison, diagonal=diagonal)

        # Checkpoint immediately, same rationale as run_experiment2.py: a
        # crash on a later (possibly much more expensive) level shouldn't
        # lose this one's already-completed solve.
        manifest[key] = dict(
            point_times=point_times,
            diagonal=dict(arc_length_mm=np.asarray(diagonal["arc_length_mm"]).tolist(),
                          t_ms=np.asarray(diagonal["t_ms"]).tolist()),
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\n" + "=" * 72)
    print("Summary across all levels run this session:")
    for dx_m in sorted(all_results.keys()):
        comparison = all_results[dx_m]["comparison"]
        n_outside = sum(1 for p in POINT_ORDER if not comparison[p]["within_2sd"])
        print(f"  dx={dx_m}mm: {len(POINT_ORDER)-n_outside}/{len(POINT_ORDER)} points "
              f"within literature mean +/- 2SD"
              + ("  (wide band expected at this resolution -- see niederer_values.py)"
                 if dx_m == 0.5 else ""))

    results_path = OUT_DIR / "exp2_niederer_results.json"
    with open(results_path, "w") as f:
        json.dump(dict(
            levels=[dict(dx_m=k, **{kk: vv for kk, vv in v.items() if kk != "diagonal"},
                        diagonal=dict(arc_length_mm=np.asarray(v["diagonal"]["arc_length_mm"]).tolist(),
                                      t_ms=np.asarray(v["diagonal"]["t_ms"]).tolist()))
                    for k, v in all_results.items()],
            sigma_l=nv.SIGMA_L, sigma_t=nv.SIGMA_T,
        ), f, indent=2)

    fig_path = OUT_DIR / "exp2_niederer_comparison.png"
    make_comparison_plot(all_results, fig_path)

    print(f"\nSaved results to {results_path}")
    print(f"Saved comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
