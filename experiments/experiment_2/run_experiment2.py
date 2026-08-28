bsmse2@MedicalScience:~/CarEP/experiments/experiment_2$ cat run_experiment2.py
#!/usr/bin/env python3
"""
run_experiment2.py
===================================================================
Experiment 2 -- conduction-velocity convergence under joint grid
refinement, on the coupled 1D-Purkinje / 3D-myocardium geometry
(Section 2.11.2 / 3.8 of the manuscript).

This is a thin sweep driver on top of experiment_common.py, which
owns the geometry, the single-run solve wrapper (run_one), and the
restricted-region CV estimators (estimate_myocardial_cv /
estimate_purkinje_cv). This script adds only the resolution-level
loop and the convergence reporting/plot.

NOTE ON MANUSCRIPT METHODS: this deliberately does NOT reproduce the
Niederer et al. N-version benchmark cable/slab setup described in the
current Methods Section 2.6.2 (a bare cable / homogeneous slab with a
direct planar stimulus). Instead it measures CV convergence on a
coupled Purkinje-fibre + myocardium geometry, so that the same run
gives us both a Purkinje-domain CV and a myocardial-domain CV, driven
physiologically (Purkinje stimulus -> PMJ -> tissue) rather than by a
direct tissue stimulus. The Methods text will need to be rewritten to
match once this design is finalised -- flagging here rather than
silently diverging from what's currently written.

WHAT IS SWEPT
-------------
A single "resolution level" scales dt, dx_p and dx_m jointly, rather
than sweeping each axis independently (the independent-axis design
would need 3x the runs and doesn't isolate anything extra for this
non-anatomical benchmark geometry, where the same numerical scheme
underlies both spatial axes and the time axis). Conductivities
(sigma_M, sigma_P) are held fixed at the base_values.py defaults
(experiment_common.SIGMA_M / SIGMA_P, sourced from base_values.py)
throughout -- only discretisation is swept.

Level 0 is the coarsest ("base") grid, fixed at base_values.py's
REF_DX_P / REF_DX_M / REF_DT (0.3152 mm / 0.3152 mm / 0.0500 ms --
see that module's docstring for why this, and not 0.5 mm, is the
canonical level-0 grid); each subsequent level refines dx_p and dx_m
by a REFINEMENT_RATIO of 1.5 (i.e. dx_p and dx_m are divided by 1.5,
equivalently multiplied by DX_REFINE_FACTOR = 1/1.5 = 0.6667, at each
step) and dt by DT_REFINE_FACTOR = 0.5. N_REFINEMENTS = 2 refinements
are run, giving three levels in total:

    level   dx_p (mm)   dx_m (mm)   dt (ms)
      0       0.3152      0.3152     0.0500
      1       0.2101      0.2101     0.0250
      2       0.1401      0.1401     0.0125

CONVERGENCE CRITERION
----------------------
Convergence between successive levels is judged on the *absolute*
change in CV, not a relative/percentage change: the run is flagged
"converged" once |CV(level i+1) - CV(level i)| < CONVERGENCE_ABS_TOL
(0.05 mm/ms), for both the myocardial and Purkinje CV estimates.

Run with:  python3 run_experiment2.py
Optional:  --n_refinements 2 --refinement_ratio 1.5 --dt_refine_factor 0.5
           --base_dx_p 0.3152 --base_dx_m 0.3152 --base_dt 0.05 --T 180
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

# experiment_common.py / base_values.py now live one level up, in
# experiments_pyonly/, shared by Experiments 2/3/4 -- see that module's
# docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiment_common as ec

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Refinement-schedule defaults
# =============================================================================
N_REFINEMENTS     = 2       # -> 3 levels total (level 0 = base, + 2 refinements)
REFINEMENT_RATIO  = 1.5     # dx_p, dx_m divided by this factor at each refinement
DX_REFINE_FACTOR  = 1.0 / REFINEMENT_RATIO   # applied jointly to dx_p and dx_m
DT_REFINE_FACTOR  = 0.5     # applied to dt at each refinement

# Absolute-change threshold (mm/ms) used to flag whether successive levels
# have converged: a level is flagged "converged" once the CV changes by
# less than this amount from the previous (coarser) level. Applied to CV
# in mm/ms (not the cm/s display units) so the tolerance is fixed
# regardless of unit choice downstream.
CONVERGENCE_ABS_TOL_MM_PER_MS = 0.05   # mm/ms


def build_levels(base_dt: float, base_dx_p: float, base_dx_m: float,
                  n_refinements: int, dx_factor: float, dt_factor: float) -> list[dict]:
    """Level 0 = base (coarsest); each subsequent level multiplies dx_p and
    dx_m by dx_factor and dt by dt_factor, relative to the previous level."""
    levels = []
    dt, dx_p, dx_m = base_dt, base_dx_p, base_dx_m
    for level in range(n_refinements + 1):
        levels.append(dict(level=level, dt=dt, dx_p=dx_p, dx_m=dx_m))
        dt *= dt_factor
        dx_p *= dx_factor
        dx_m *= dx_factor
    return levels


def make_convergence_plot(results_myo: list, results_pkj: list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def panel(ax, results, label):
        dx_m = np.array([r["dx_m"] for r in results])
        cvs = np.array([r["cv_cm_per_s"] for r in results])
        order = np.argsort(-dx_m)   # coarsest (largest dx_m) first
        dx_m, cvs = dx_m[order], cvs[order]

        ax.plot(dx_m, cvs, "o-", color="C0", zorder=3, label="measured")
        if np.isfinite(cvs[-1]):
            ax.axhline(cvs[-1], color="k", linestyle="--", alpha=0.6,
                       label=f"finest level ({cvs[-1]:.2f} cm/s)")
        ax.invert_xaxis()   # so "refining" reads left -> right
        ax.set_xlabel(r"$dx_m$ (mm)")
        ax.set_ylabel("Conduction velocity (cm/s)")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)

    panel(axes[0], results_myo, "Myocardial domain")
    panel(axes[1], results_pkj, "Purkinje domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_convergence_table(title: str, results: list) -> None:
    print(f"\n{title}")
    print(f"{'level':>6} {'dx_p':>8} {'dx_m':>8} {'dt':>8} "
          f"{'CV (mm/ms)':>12} {'CV (cm/s)':>12} {'n_pts':>8} {'R^2':>8} {'abs. change (mm/ms)':>22}")
    prev_cv_mm_ms = None
    for r in results:
        if prev_cv_mm_ms is None or not (np.isfinite(prev_cv_mm_ms) and np.isfinite(r["cv_mm_per_ms"])):
            abs_str = "          --"
        else:
            abs_change = abs(r["cv_mm_per_ms"] - prev_cv_mm_ms)
            flag = "  <-- not yet converged" if abs_change > CONVERGENCE_ABS_TOL_MM_PER_MS else "  <-- converged"
            abs_str = f"{abs_change:>10.4f}{flag}"
        flag_r2 = "  <-- low R^2" if (np.isfinite(r["r2"]) and r["r2"] < ec.R2_WARN_THRESHOLD) else ""
        print(f"{r['level']:>6d} {r['dx_p']:>8.4g} {r['dx_m']:>8.4g} {r['dt']:>8.4g} "
              f"{r['cv_mm_per_ms']:>12.4f} {r['cv_cm_per_s']:>12.2f} {r['n_points']:>8d} "
              f"{r['r2']:>8.4f}{flag_r2} {abs_str}")
        prev_cv_mm_ms = r["cv_mm_per_ms"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dt",   type=float, default=ec.REF_DT,
                     help="level-0 (coarsest) time step [ms]")
    ap.add_argument("--base_dx_p", type=float, default=ec.REF_DX_P,
                     help="level-0 (coarsest) Purkinje segment length [mm]")
    ap.add_argument("--base_dx_m", type=float, default=ec.REF_DX_M,
                     help="level-0 (coarsest) myocardial voxel size [mm]")
    ap.add_argument("--n_refinements", type=int, default=N_REFINEMENTS,
                     help="number of refinements after level 0 (total levels = this + 1)")
    ap.add_argument("--refinement_ratio", type=float, default=REFINEMENT_RATIO,
                     help="factor by which dx_p and dx_m are DIVIDED at each refinement "
                          "(e.g. 1.5 means each level's dx is the previous level's dx / 1.5)")
    ap.add_argument("--dt_refine_factor", type=float, default=DT_REFINE_FACTOR,
                     help="multiplicative factor applied to dt per refinement")
    ap.add_argument("--T", type=float, default=180.0,
                     help="total simulated time [ms] (same physics at every level, so fixed)")
    ap.add_argument("--start_level", type=int, default=0,
                     help="skip solving levels below this one, reusing cached results "
                          "from the manifest file if present (see exp2_manifest.json)")
    args = ap.parse_args()

    dx_refine_factor = 1.0 / args.refinement_ratio
    levels = build_levels(args.base_dt, args.base_dx_p, args.base_dx_m,
                           args.n_refinements, dx_refine_factor, args.dt_refine_factor)

    manifest_path = OUT_DIR / "exp2_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    print("=" * 72)
    print("Experiment 2: joint grid-resolution refinement")
    ec.print_base_values()
    print(f"sigma_M={ec.SIGMA_M:.4g} mS/mm, sigma_P={ec.SIGMA_P:.4g} mS/mm (fixed, base_values.py defaults)")
    print("=" * 72)
    for lv in levels:
        print(f"  level {lv['level']}: dt={lv['dt']:.4g} ms, dx_p={lv['dx_p']:.4g} mm, "
              f"dx_m={lv['dx_m']:.4g} mm")

    results_myo, results_pkj = [], []
    for lv in levels:
        level, dt, dx_p, dx_m = lv["level"], lv["dt"], lv["dx_p"], lv["dx_m"]
        tag = f"resSweep_level{level}_dt{dt:g}_dxp{dx_p:g}_dxm{dx_m:g}"
        key = str(level)

        if level < args.start_level and key in manifest:
            print(f"\n--- level={level}  dt={dt:.4g} ms  dx_p={dx_p:.4g} mm  dx_m={dx_m:.4g} mm "
                  f"--- (skipped, loaded from {manifest_path.name})")
            results_myo.append(manifest[key]["myo"])
            results_pkj.append(manifest[key]["pkj"])
            print(f"    myocardial CV = {manifest[key]['myo']['cv_cm_per_s']:.2f} cm/s (cached)")
            print(f"    Purkinje   CV = {manifest[key]['pkj']['cv_cm_per_s']:.2f} cm/s (cached)")
            continue

        print(f"\n--- level={level}  dt={dt:.4g} ms  dx_p={dx_p:.4g} mm  dx_m={dx_m:.4g} mm ---")
        out = ec.run_one(ec.SIGMA_M, ec.SIGMA_P, dt, dx_p, dx_m, args.T, tag, out_dir=OUT_DIR)

        cv_m = ec.estimate_myocardial_cv(out["myo_coords"], out["myo_act"], dx_m)
        cv_p = ec.estimate_purkinje_cv(out["p_arc_length"], out["p_act"])
        print(f"    myocardial CV = {cv_m['cv_cm_per_s']:.2f} cm/s "
              f"(n={cv_m['n_points']}, R^2={cv_m['r2']:.4f})")
        print(f"    Purkinje   CV = {cv_p['cv_cm_per_s']:.2f} cm/s "
              f"(n={cv_p['n_points']}, R^2={cv_p['r2']:.4f})")

        myo_entry = dict(level=level, dt=dt, dx_p=dx_p, dx_m=dx_m,
                          cv_mm_per_ms=cv_m["cv_mm_per_ms"], cv_cm_per_s=cv_m["cv_cm_per_s"],
                          n_points=cv_m["n_points"], r2=cv_m["r2"])
        pkj_entry = dict(level=level, dt=dt, dx_p=dx_p, dx_m=dx_m,
                          cv_mm_per_ms=cv_p["cv_mm_per_ms"], cv_cm_per_s=cv_p["cv_cm_per_s"],
                          n_points=cv_p["n_points"], r2=cv_p["r2"])
        results_myo.append(myo_entry)
        results_pkj.append(pkj_entry)

        # Checkpoint immediately -- so a crash on a later level (e.g. the
        # Vm-snapshot OOM at fine resolutions) doesn't lose this level's
        # already-completed solve + CV fit.
        manifest[key] = dict(myo=myo_entry, pkj=pkj_entry)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ---- Save combined results ----
    npz_path = OUT_DIR / "exp2_cv_results.npz"
    np.savez(
        npz_path,
        level=np.array([r["level"] for r in results_myo]),
        dt=np.array([r["dt"] for r in results_myo]),
        dx_p=np.array([r["dx_p"] for r in results_myo]),
        dx_m=np.array([r["dx_m"] for r in results_myo]),
        myo_cv_cm_per_s=np.array([r["cv_cm_per_s"] for r in results_myo]),
        myo_n_points=np.array([r["n_points"] for r in results_myo]),
        myo_r2=np.array([r["r2"] for r in results_myo]),
        pkj_cv_cm_per_s=np.array([r["cv_cm_per_s"] for r in results_pkj]),
        pkj_n_points=np.array([r["n_points"] for r in results_pkj]),
        pkj_r2=np.array([r["r2"] for r in results_pkj]),
    )
    json_path = OUT_DIR / "exp2_cv_results.json"
    with open(json_path, "w") as f:
        json.dump(dict(myocardial_sweep=results_myo, purkinje_sweep=results_pkj,
                        refinement_schedule=dict(
                            n_refinements=args.n_refinements,
                            refinement_ratio=args.refinement_ratio,
                            dx_refine_factor=dx_refine_factor,
                            dt_refine_factor=args.dt_refine_factor,
                            base_dt=args.base_dt, base_dx_p=args.base_dx_p, base_dx_m=args.base_dx_m),
                        convergence_abs_tol_mm_per_ms=CONVERGENCE_ABS_TOL_MM_PER_MS,
                        exclusion_buffers_mm=dict(
                            myo_lateral=ec.MYO_LATERAL_BUFFER_MM, myo_entry=ec.MYO_ENTRY_BUFFER_MM,
                            myo_exit=ec.MYO_EXIT_BUFFER_MM, myo_pmj_radius=ec.PMJ_RADIUS_EXCLUDE_MM,
                            purkinje_root=ec.PURKINJE_ROOT_BUFFER_MM,
                            purkinje_pmj=ec.PURKINJE_PMJ_BUFFER_MM)),
                  f, indent=2)

    fig_path = OUT_DIR / "exp2_cv_convergence.png"
    make_convergence_plot(results_myo, results_pkj, fig_path)

    print_convergence_table("Myocardial domain -- CV vs. resolution level:", results_myo)
    print_convergence_table("Purkinje domain -- CV vs. resolution level:", results_pkj)
    print(f"\nConvergence criterion: absolute change in CV between successive levels "
          f"< {CONVERGENCE_ABS_TOL_MM_PER_MS:.2f} mm/ms")
    print(f"\nSaved combined results to {npz_path} and {json_path}")
    print(f"Saved CV-vs-resolution convergence figure to {fig_path}")


if __name__ == "__main__":
    main()
bsmse2@MedicalScience:~/CarEP/experiments/experiment_2$
