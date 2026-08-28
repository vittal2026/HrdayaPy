"""
run_experiment3.py
===================================================================
Experiment 3 -- conduction-velocity dependence on conductivity,
myocardial and Purkinje domains scaled TOGETHER (a single joint
sweep), on the same coupled Purkinje-fibre + myocardium geometry used
in Experiment 2 (Section 2.11.3 / 3.3 of the manuscript).

This is a thin sweep driver on top of experiment_common.py, which
owns the geometry, the single-run solve wrapper (run_one), and the
restricted-region CV estimators (estimate_myocardial_cv /
estimate_purkinje_cv) -- see that module's docstring for the rationale
behind the restricted fit region. This script adds only the
conductivity-sweep loop, the scaling plot, the sweep-specific result
tables, and (see CACHING below) an optional disk-cache path that
reuses a prior run_one() call's already-saved
activation_maps_{tag}.npz / coupled_{tag}.npz instead of re-solving.

Discretisation (dt, dx_p, dx_m) is held FIXED throughout this
experiment, at Experiment 2's converged, finest-refinement grid
(dt = 0.0125 ms, dx_p = dx_m = 0.140089 mm -- Experiment 2's level 2,
Table~tab:cv_convergence, the level at which the absolute-change
convergence criterion of Section~sec:methods-exp2 is met in both
domains) unless overridden via --dt/--dx_p/--dx_m.

WHAT IS SWEPT
-------------
A single JOINT sweep, following cable theory's CV ~ sqrt(sigma /
(Cm*Am)) prediction: sigma_M and sigma_P are scaled TOGETHER by the
same factor at each step, sigma_M *= scale and sigma_P *= scale
simultaneously, for scale in {0.5, 1.0, 2.0}.

Because lower conductivity means slower conduction in BOTH domains at
once, T is extended automatically (with a safety margin) for scales
below 1.0 so the wavefront still reaches the far end of the 50 mm slab
within the simulated window; override with --T_sweep if a run does
not fully activate.

PMJ COUPLING OVERRIDE
----------------------
At scale=2.0, the base_values.py production default c_pmj (5e-2 mS)
and n_pmj (4 nodes) were too small relative to the doubled sigma_M to
launch a propagating wavefront out of the myocardium. Raising both
c_pmj (to 5e-1 mS) and n_pmj (to 8 nodes) together resolves this.

Both overrides are applied to EVERY point in this sweep (not just the
scale=2.0 point that required them), via EXP3_C_PMJ_OVERRIDE /
EXP3_N_PMJ_OVERRIDE below, for one consistent PMJ coupling protocol
across the whole sweep -- WITHOUT changing base_values.py's shared
C_PMJ / N_PMJ, which remain Experiment 2's and Experiment 4's
defaults. Because of this, scale=0.5 and scale=1.0 CVs here are NOT
measured under the same PMJ coupling as Experiment 2's baseline (0.05
mS / 4 nodes vs. 0.5 mS / 8 nodes here); treat them as provisional
until cross-checked against Experiment 2's numbers for consistency.

CACHING
-------
run_one() already checkpoints two files per call into out_dir:
  activation_maps_{tag}.npz   (myo_coords, myo_act -- small, ~MB)
  coupled_{tag}.npz           (comp_nodes, branch_map, branch_frames_p,
                               times -- used to rebuild Purkinje CV;
                               can be large, GB-scale at fine grids)
`tag` is built from scale/dt/dx_p/dx_m ONLY -- it does NOT encode
c_pmj/n_pmj. That means two runs made under different PMJ-coupling
settings but the same (scale, dt, dx_p, dx_m) collide on the same
tag and silently overwrite each other's files on disk; there is no
way to tell from the tag alone which coupling produced a given cached
file.

To close that gap, when --use_cache is passed this script writes (on
every fresh solve) and reads (on every cache hit) a small sidecar
manifest_{tag}.json recording exactly which c_pmj/n_pmj/dt/dx_p/dx_m
were used to produce that tag's files. A cache hit is only accepted
if the manifest exists AND its recorded c_pmj/n_pmj match this run's
EXP3_C_PMJ_OVERRIDE/EXP3_N_PMJ_OVERRIDE -- if the manifest is missing
(pre-existing files from before this provision was added) or the
values don't match, the script refuses the cache hit and re-solves,
rather than silently trusting an unverifiable file. Use
--trust_unverified_cache to force acceptance of a cache hit with no
manifest (only if you have separately confirmed, e.g. from console
logs, what coupling produced those files) -- off by default.

Run with:  python3 run_experiment3.py
Optional:  --dt 0.0125 --dx_p 0.140089 --dx_m 0.140089 --T_base 180
           --scales 0.5 1.0 2.0
           --use_cache               (reuse prior solves where verified)
           --force_rerun             (ignore any cache, always re-solve)
           --trust_unverified_cache  (accept cache hits with no manifest)
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
# Discretisation defaults -- Experiment 2's CONVERGED grid
# =============================================================================
VALIDATED_DT   = 0.0125        # ms  -- Experiment 2 level 2
VALIDATED_DX_P = 0.140088889   # mm  -- Experiment 2 level 2 (Purkinje segment length)
VALIDATED_DX_M = 0.140088889   # mm  -- Experiment 2 level 2 (myocardial voxel size)

# =============================================================================
# PMJ coupling override -- LOCAL to Experiment 3 only, applied to EVERY point
# in the sweep (see PMJ COUPLING OVERRIDE section of the module docstring).
# =============================================================================
EXP3_C_PMJ_OVERRIDE = 5e-1   # mS    (base_values.py default is 5e-2 mS)
EXP3_N_PMJ_OVERRIDE = 8      # nodes (base_values.py default is 4)

# =============================================================================
# Sweep defaults
# =============================================================================
SCALES = [0.5, 1.0, 2.0]


# =============================================================================
# Caching: manifest + loader (see CACHING in module docstring)
# =============================================================================
def _manifest_path(out_dir: Path, tag: str) -> Path:
    return out_dir / f"manifest_{tag}.json"


def _write_manifest(out_dir: Path, tag: str, *, dt: float, dx_p: float, dx_m: float,
                     c_pmj: float, n_pmj: int, sigma_m: float, sigma_p: float, T: float) -> None:
    manifest = dict(tag=tag, dt=dt, dx_p=dx_p, dx_m=dx_m,
                     c_pmj=c_pmj, n_pmj=n_pmj,
                     sigma_m=sigma_m, sigma_p=sigma_p, T=T)
    with open(_manifest_path(out_dir, tag), "w") as f:
        json.dump(manifest, f, indent=2)


def _load_one_from_disk(tag: str, out_dir: Path) -> dict:
    """
    Reconstruct run_one()'s return dict (myo_coords, myo_act, p_arc_length,
    p_act) from activation_maps_{tag}.npz / coupled_{tag}.npz already saved
    by a prior run_one() call, without re-running compute_coupled or
    compute_activation_maps.

    NOTE: this function does NOT itself check the manifest / PMJ-coupling
    provenance -- that check happens in the caller (see try_load_cached
    below) BEFORE this is invoked, so this function only has to worry about
    whether the files exist and parse.
    """
    am_path = out_dir / f"activation_maps_{tag}.npz"
    coupled_path = out_dir / f"coupled_{tag}.npz"
    if not am_path.exists() or not coupled_path.exists():
        missing = [p.name for p in (am_path, coupled_path) if not p.exists()]
        raise FileNotFoundError(f"Missing cached output(s) for tag={tag!r}: {missing}")

    am = np.load(am_path, allow_pickle=True)
    myo_coords = am["coords_mm"]
    activation_times = am["activation_times"]
    if activation_times.shape[1] == 0:
        myo_act = np.full(myo_coords.shape[0], np.nan, dtype=np.float32)
    else:
        myo_act = activation_times[:, 0]

    coupled = np.load(coupled_path, allow_pickle=True)
    results = {k: coupled[k] for k in coupled.files}
    p_coords, p_arc_length, p_Vm, p_t = ec.purkinje_ordered_trace(results)
    p_act = ec.first_crossing_times(p_Vm, p_t, ec.ACT_THRESHOLD)

    return dict(myo_coords=myo_coords, myo_act=myo_act,
                p_arc_length=p_arc_length, p_act=p_act)


def try_load_cached(tag: str, out_dir: Path, *, dt: float, dx_p: float, dx_m: float,
                     c_pmj: float, n_pmj: int, trust_unverified: bool) -> dict | None:
    """
    Returns a run_one()-shaped dict on a VERIFIED cache hit, or None if no
    usable cache exists (caller should then call ec.run_one() as normal).
    Never silently returns data whose coupling provenance is unconfirmed
    unless trust_unverified=True is explicitly passed.
    """
    am_path = out_dir / f"activation_maps_{tag}.npz"
    coupled_path = out_dir / f"coupled_{tag}.npz"
    if not (am_path.exists() and coupled_path.exists()):
        return None

    manifest_path = _manifest_path(out_dir, tag)
    if not manifest_path.exists():
        if trust_unverified:
            print(f"    [cache] WARNING: no manifest for tag={tag!r}; "
                  f"--trust_unverified_cache set, loading anyway. "
                  f"Coupling used to produce this file is NOT confirmed by this script.")
            return _load_one_from_disk(tag, out_dir)
        print(f"    [cache] files exist for tag={tag!r} but no manifest_{tag}.json found -- "
              f"cannot verify what c_pmj/n_pmj produced them. Re-solving. "
              f"(Pass --trust_unverified_cache to accept unverified files instead.)")
        return None

    with open(manifest_path) as f:
        m = json.load(f)
    mismatches = []
    for key, expected in (("dt", dt), ("dx_p", dx_p), ("dx_m", dx_m),
                           ("c_pmj", c_pmj), ("n_pmj", n_pmj)):
        if not np.isclose(m.get(key, np.nan), expected):
            mismatches.append(f"{key}: manifest={m.get(key)!r} vs requested={expected!r}")
    if mismatches:
        print(f"    [cache] manifest for tag={tag!r} does NOT match requested run "
              f"parameters -- re-solving. Mismatches: {'; '.join(mismatches)}")
        return None

    print(f"    [cache hit, verified] loading {tag} from existing "
          f"activation_maps_{tag}.npz / coupled_{tag}.npz "
          f"(manifest confirms c_pmj={m['c_pmj']}, n_pmj={m['n_pmj']})")
    return _load_one_from_disk(tag, out_dir)


def make_scaling_plot(results: list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def panel(ax, cv_key, label):
        scales = np.array([r["scale"] for r in results])
        cvs = np.array([r[cv_key] for r in results])
        order = np.argsort(scales)
        scales, cvs = scales[order], cvs[order]
        sqrt_scale = np.sqrt(scales)

        if np.any(np.isclose(scales, 1.0)):
            i0 = int(np.where(np.isclose(scales, 1.0))[0][0])
        else:
            i0 = len(scales) // 2
        is_anchor = np.zeros(len(scales), dtype=bool)
        is_anchor[i0] = True

        ax.scatter(sqrt_scale[~is_anchor], cvs[~is_anchor], color="C0", zorder=3,
                   s=70, label="measured (test)")
        ax.scatter(sqrt_scale[is_anchor], cvs[is_anchor], color="C3", marker="*",
                   s=220, zorder=4, edgecolors="black", linewidths=0.8,
                   label="reference (anchor, not a test)")

        if np.isfinite(cvs[i0]):
            cv0, s0 = cvs[i0], sqrt_scale[i0]
            xs = np.linspace(sqrt_scale.min() * 0.9, sqrt_scale.max() * 1.1, 50)
            ax.plot(xs, cv0 * (xs / s0), "k--", zorder=2,
                    label=r"predicted $CV \propto \sqrt{\sigma}$ (anchored at ref.)")

            for j in range(len(scales)):
                if is_anchor[j] or not np.isfinite(cvs[j]):
                    continue
                cv_pred = cv0 * (sqrt_scale[j] / s0)
                pct_err = 100.0 * (cvs[j] - cv_pred) / cv_pred
                ax.annotate(f"{pct_err:+.1f}%",
                            (sqrt_scale[j], cvs[j]),
                            textcoords="offset points", xytext=(8, -12),
                            fontsize=9, color="C0")

        ax.set_xlabel(r"$\sqrt{\sigma / \sigma_{default}}$")
        ax.set_ylabel("Conduction velocity (cm/s)")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    panel(axes[0], "cv_m_cm_per_s", "Myocardial domain")
    panel(axes[1], "cv_p_cm_per_s", "Purkinje domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_table(title: str, results: list):
    print(f"\n{title}")
    print(f"{'scale':>8} {'sigma_M':>12} {'sigma_P':>12} {'CV_m (cm/s)':>12} {'R^2_m':>8} "
          f"{'CV_p (cm/s)':>12} {'R^2_p':>8}")
    for r in results:
        flag_m = "  <-- low R^2_m" if (np.isfinite(r["r2_m"]) and r["r2_m"] < ec.R2_WARN_THRESHOLD) else ""
        flag_p = "  <-- low R^2_p" if (np.isfinite(r["r2_p"]) and r["r2_p"] < ec.R2_WARN_THRESHOLD) else ""
        print(f"{r['scale']:>8.2g} {r['sigma_m']:>12.4g} {r['sigma_p']:>12.4g} "
              f"{r['cv_m_cm_per_s']:>12.2f} {r['r2_m']:>8.4f} "
              f"{r['cv_p_cm_per_s']:>12.2f} {r['r2_p']:>8.4f}{flag_m}{flag_p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dt", type=float, default=VALIDATED_DT, help="solver time step [ms]")
    ap.add_argument("--dx_p", type=float, default=VALIDATED_DX_P, help="Purkinje segment length [mm]")
    ap.add_argument("--dx_m", type=float, default=VALIDATED_DX_M, help="myocardial voxel size [mm]")
    ap.add_argument("--T_base", type=float, default=180.0,
                     help="base simulated time [ms] at default (1.0x) conductivity")
    ap.add_argument("--T_sweep", type=float, default=None,
                     help="override: fixed T [ms] to use for every sweep run "
                          "(otherwise auto-scaled from --T_base)")
    ap.add_argument("--scales", type=float, nargs="+", default=SCALES,
                     help="joint conductivity scale factors, applied to sigma_M AND sigma_P "
                          "simultaneously (relative to base_values.py defaults)")
    ap.add_argument("--use_cache", action="store_true",
                     help="reuse a prior solve's activation_maps_{tag}.npz/coupled_{tag}.npz "
                          "instead of re-running compute_coupled, when a manifest confirms "
                          "the cached files were produced under the same "
                          "(dt, dx_p, dx_m, c_pmj, n_pmj). Off by default -- without this "
                          "flag every scale point is always solved fresh, as before.")
    ap.add_argument("--force_rerun", action="store_true",
                     help="ignore any cache even if --use_cache is set; always re-solve. "
                          "Useful to refresh the cache/manifest after changing the PMJ "
                          "override values.")
    ap.add_argument("--trust_unverified_cache", action="store_true",
                     help="with --use_cache, accept a cache hit even when no "
                          "manifest_{tag}.json is present (e.g. files produced before this "
                          "provision existed) -- only use this if you have separately "
                          "confirmed what coupling produced those files.")
    args = ap.parse_args()

    if args.force_rerun and args.trust_unverified_cache:
        print("Note: --force_rerun overrides --trust_unverified_cache; every point will "
              "be re-solved regardless.")

    results = []
    ec.print_base_values()

    print("=" * 72)
    print("Joint sweep: sigma_M and sigma_P scaled together by the same factor")
    print(f"Discretisation fixed at dt={args.dt} ms, dx_p={args.dx_p} mm, dx_m={args.dx_m} mm")
    print(f"PMJ coupling: c_pmj={EXP3_C_PMJ_OVERRIDE} mS, n_pmj={EXP3_N_PMJ_OVERRIDE} "
          f"(LOCAL override, applied to every point in this sweep; "
          f"base_values.py default is c_pmj={ec.C_PMJ} mS, n_pmj={ec.N_PMJ}, "
          f"unchanged for Exp. 2/4)")
    print(f"Caching: {'ENABLED (--use_cache)' if args.use_cache else 'disabled'}"
          f"{', but --force_rerun set so every point re-solves anyway' if args.force_rerun else ''}")
    print("=" * 72)

    for scale in args.scales:
        sigma_M = ec.SIGMA_M * scale
        sigma_P = ec.SIGMA_P * scale
        if args.T_sweep is not None:
            T = args.T_sweep
        else:
            T = args.T_base / np.sqrt(min(scale, 1.0)) * 1.25 if scale < 1.0 else args.T_base

        tag = f"jointSweep_scale{scale:g}_dt{args.dt:g}_dxp{args.dx_p:g}_dxm{args.dx_m:g}"
        print(f"\n--- scale={scale:g}  (sigma_M={sigma_M:.4g} mS/mm, "
              f"sigma_P={sigma_P:.4g} mS/mm)  T={T:.0f} ms  "
              f"c_pmj={EXP3_C_PMJ_OVERRIDE} mS  n_pmj={EXP3_N_PMJ_OVERRIDE} ---")

        out = None
        if args.use_cache and not args.force_rerun:
            out = try_load_cached(tag, OUT_DIR, dt=args.dt, dx_p=args.dx_p, dx_m=args.dx_m,
                                   c_pmj=EXP3_C_PMJ_OVERRIDE, n_pmj=EXP3_N_PMJ_OVERRIDE,
                                   trust_unverified=args.trust_unverified_cache)

        if out is None:
            out = ec.run_one(sigma_M, sigma_P, args.dt, args.dx_p, args.dx_m, T, tag,
                              out_dir=OUT_DIR,
                              c_pmj=EXP3_C_PMJ_OVERRIDE, n_pmj=EXP3_N_PMJ_OVERRIDE)
            _write_manifest(OUT_DIR, tag, dt=args.dt, dx_p=args.dx_p, dx_m=args.dx_m,
                             c_pmj=EXP3_C_PMJ_OVERRIDE, n_pmj=EXP3_N_PMJ_OVERRIDE,
                             sigma_m=sigma_M, sigma_p=sigma_P, T=T)

        cv_m = ec.estimate_myocardial_cv(out["myo_coords"], out["myo_act"], args.dx_m)
        cv_p = ec.estimate_purkinje_cv(out["p_arc_length"], out["p_act"])
        print(f"    myocardial CV = {cv_m['cv_cm_per_s']:.2f} cm/s "
              f"(n={cv_m['n_points']}, R^2={cv_m['r2']:.4f})   "
              f"Purkinje CV = {cv_p['cv_cm_per_s']:.2f} cm/s "
              f"(n={cv_p['n_points']}, R^2={cv_p['r2']:.4f})")
        results.append(dict(
            scale=scale, sigma_m=sigma_M, sigma_p=sigma_P,
            c_pmj=EXP3_C_PMJ_OVERRIDE, n_pmj=EXP3_N_PMJ_OVERRIDE,
            cv_m_mm_per_ms=cv_m["cv_mm_per_ms"], cv_m_cm_per_s=cv_m["cv_cm_per_s"],
            n_points_m=cv_m["n_points"], r2_m=cv_m["r2"],
            cv_p_mm_per_ms=cv_p["cv_mm_per_ms"], cv_p_cm_per_s=cv_p["cv_cm_per_s"],
            n_points_p=cv_p["n_points"], r2_p=cv_p["r2"],
        ))

    # ---- Save results ----
    npz_path = OUT_DIR / "exp3_cv_results.npz"
    np.savez(
        npz_path,
        scale=np.array([r["scale"] for r in results]),
        sigma_m=np.array([r["sigma_m"] for r in results]),
        sigma_p=np.array([r["sigma_p"] for r in results]),
        c_pmj=np.array([r["c_pmj"] for r in results]),
        n_pmj=np.array([r["n_pmj"] for r in results]),
        cv_m_cm_per_s=np.array([r["cv_m_cm_per_s"] for r in results]),
        n_points_m=np.array([r["n_points_m"] for r in results]),
        r2_m=np.array([r["r2_m"] for r in results]),
        cv_p_cm_per_s=np.array([r["cv_p_cm_per_s"] for r in results]),
        n_points_p=np.array([r["n_points_p"] for r in results]),
        r2_p=np.array([r["r2_p"] for r in results]),
    )
    json_path = OUT_DIR / "exp3_cv_results.json"
    with open(json_path, "w") as f:
        json.dump(dict(joint_sweep=results,
                        discretisation=dict(dt=args.dt, dx_p=args.dx_p, dx_m=args.dx_m),
                        pmj_coupling=dict(
                            base_default_c_pmj=ec.C_PMJ,
                            base_default_n_pmj=ec.N_PMJ,
                            override_c_pmj=EXP3_C_PMJ_OVERRIDE,
                            override_n_pmj=EXP3_N_PMJ_OVERRIDE,
                            note="overrides applied to EVERY point in this sweep "
                                 "(0.5, 1.0, and 2.0); base_values.py's defaults are "
                                 "NOT used anywhere in this experiment's runs, so "
                                 "scale=0.5/1.0 CVs here are not directly comparable "
                                 "to Experiment 2's baseline-coupling numbers"),
                        exclusion_buffers_mm=dict(
                            myo_lateral=ec.MYO_LATERAL_BUFFER_MM, myo_entry=ec.MYO_ENTRY_BUFFER_MM,
                            myo_exit=ec.MYO_EXIT_BUFFER_MM, myo_pmj_radius=ec.PMJ_RADIUS_EXCLUDE_MM,
                            purkinje_root=ec.PURKINJE_ROOT_BUFFER_MM,
                            purkinje_pmj=ec.PURKINJE_PMJ_BUFFER_MM)),
                  f, indent=2)

    fig_path = OUT_DIR / "exp3_cv_scaling.png"
    make_scaling_plot(results, fig_path)

    print_table("Joint sweep (sigma_M and sigma_P scaled together):", results)
    print(f"\nSaved combined results to {npz_path} and {json_path}")
    print(f"Saved CV-vs-sqrt(sigma) scaling figure to {fig_path}")


if __name__ == "__main__":
    main()
