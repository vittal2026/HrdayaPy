"""
inspect_pipeline/functions/register_torso.py
==============================================
Rigid (no-scale) ICP registration of the dataset's torso surface mesh onto
the simulated BSPM's torso surface point cloud.

Why this should converge to (near) identity
---------------------------------------------
The simulated torso surface (bspm.npz -> surface_xyz) is a resampled,
coarse-grid boundary of TORSO_NRRD_PATH — and that NRRD was itself
voxelised (multi_label_nrrd.py, in the dataset scripts) directly from this
same patient's torso mesh (P###_torso.mat). So source and target already
live in the same physical mm frame; ICP here is a sanity check / small
correction for voxelisation and resampling error, not a big alignment
problem. A poor fit (large residual RMSE) is a signal that TORSO_NRRD_PATH
and GT_PATIENT_ID in config.py don't actually refer to the same patient.

Algorithm
---------
Point-to-point ICP (Kabsch/Umeyama rotation, no scaling):
  1. Subsample the target cloud for KD-tree speed.
  2. Repeat: find nearest target point for each source point, solve the
     best rigid transform mapping matched pairs, apply it, check RMSE
     convergence.

Result convention
------------------
    sim_coords ≈ R @ dataset_coords.T).T + t

i.e. apply_torso_registration() maps points FROM dataset space INTO the
simulated BSPM's frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Rigid-transform fit (Kabsch, no scaling)
# ---------------------------------------------------------------------------

def _kabsch_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best R (3x3), t (3,) mapping src -> dst (paired, same length), no scale."""
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    src_c = src - c_src
    dst_c = dst - c_dst

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = c_dst - R @ c_src
    return R, t


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------

def icp_rigid(
    source_pts:      np.ndarray,
    target_pts:      np.ndarray,
    max_iterations:  int   = 60,
    tol_mm:          float = 1e-4,
    target_subsample: int  = 20_000,
    verbose:         bool  = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Rigid point-to-point ICP: source_pts -> target_pts.

    Returns
    -------
    R, t   : rigid transform, `registered = (R @ source_pts.T).T + t`
    stats  : dict with pre/post RMSE, mean/max NN distance, n_iter
    """
    rng = np.random.default_rng(0)
    if len(target_pts) > target_subsample:
        sub_idx = rng.choice(len(target_pts), target_subsample, replace=False)
        tgt = target_pts[sub_idx]
    else:
        tgt = target_pts

    tree = cKDTree(tgt)

    # Baseline (pre-registration) residual, for reporting.
    d0, _ = tree.query(source_pts, k=1)
    pre_rmse = float(np.sqrt(np.mean(d0 ** 2)))
    pre_mean = float(d0.mean())

    src = source_pts.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_rmse = None
    n_iter = 0

    for it in range(max_iterations):
        n_iter = it + 1
        dists, nn_idx = tree.query(src, k=1)
        matched = tgt[nn_idx]

        R, t = _kabsch_rigid(src, matched)
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        rmse = float(np.sqrt(np.mean(dists ** 2)))
        if verbose:
            print(f"    iter {it:2d}   rmse = {rmse:7.3f} mm   "
                  f"mean = {dists.mean():7.3f} mm   max = {dists.max():7.3f} mm")

        if prev_rmse is not None and abs(prev_rmse - rmse) < tol_mm:
            break
        prev_rmse = rmse

    final_dists, _ = tree.query(src, k=1)
    stats = dict(
        pre_rmse   = pre_rmse,
        pre_mean   = pre_mean,
        rmse       = float(np.sqrt(np.mean(final_dists ** 2))),
        mean_dist  = float(final_dists.mean()),
        max_dist   = float(final_dists.max()),
        n_iter     = n_iter,
    )
    return R_total, t_total, stats


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def apply_torso_registration(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map points from dataset space into the simulated BSPM frame."""
    return (R @ coords.T).T + t


# ---------------------------------------------------------------------------
# Public entry point (called by inspect_pipeline/torso_registration.py)
# ---------------------------------------------------------------------------

def run_torso_registration(
    bspm_npz_path:     Path,
    gt_torso_mat_path: Path,
    max_iterations:    int   = 60,
    tol_mm:            float = 1e-4,
    target_subsample:  int   = 20_000,
    verbose:           bool  = True,
) -> dict:
    """
    Load inputs, run ICP, return the result dict.

    Returns
    -------
    dict with keys: R (3,3), t (3,), pre_rmse, pre_mean, rmse, mean_dist,
    max_dist, n_iter
    """
    from .load_ground_truth import load_gt_torso_mesh

    nodes, _faces, _leadlinks = load_gt_torso_mesh(gt_torso_mat_path)

    bspm = np.load(str(bspm_npz_path))
    surface_xyz = bspm["surface_xyz"].astype(np.float64)

    if verbose:
        print(f"  Dataset torso nodes   : {len(nodes):,}")
        print(f"  Simulated surface pts : {len(surface_xyz):,}")

    R, t, stats = icp_rigid(
        nodes, surface_xyz,
        max_iterations   = max_iterations,
        tol_mm           = tol_mm,
        target_subsample = target_subsample,
        verbose          = verbose,
    )

    if verbose:
        print(f"\n  Pre-registration  RMSE = {stats['pre_rmse']:.3f} mm  "
              f"(mean {stats['pre_mean']:.3f} mm)")
        print(f"  Post-registration RMSE = {stats['rmse']:.3f} mm  "
              f"(mean {stats['mean_dist']:.3f} mm, max {stats['max_dist']:.3f} mm) "
              f"after {stats['n_iter']} iterations")
        if stats["rmse"] > 15.0:
            print("  [warn] RMSE is large for 'derived from the same anatomy' data — "
                  "check that GT_PATIENT_ID matches the patient used to build "
                  "TORSO_NRRD_PATH.")

    result = dict(R=R, t=t, **stats)
    return result
