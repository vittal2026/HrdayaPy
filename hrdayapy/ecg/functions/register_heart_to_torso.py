"""
ecg/functions/register_heart_to_torso.py
=========================================
Rigid registration of the simulation heart point cloud (from vm_snapshots.npz)
onto the myocardium region of the full torso NRRD segmentation.

Algorithm
---------
1. Extract physical coordinates (mm) of all label-2 voxels from the NRRD
   — these are the registration *target*.
2. Load coords_mm from vm_snapshots.npz — the registration *source*.
3. Compute an isotropic scale factor from the ratio of bounding-box spans
   (median across axes).
4. Align bounding-box centres (translation, scale-aware).
5. Brute-force search all 48 signed permutation matrices of the cube group
   (every axis permutation combined with every independent choice of
   per-axis sign: the 24 proper rotations, det = +1, plus the 24
   reflections, det = -1); pick the one that maximises voxel Dice.

The result is a rigid similarity transform:

    coord_torso_mm = R @ (coord_heart_mm * scale) + t

where R is one of the 48 signed permutation matrices, scale is a
scalar, and t is the translation into NRRD physical space.

Outputs saved to PATH_HEART_REGISTRATION (heart_registration.npz):
    R       (3, 3)  transform matrix (signed permutation matrix)
    t       (3,)    translation  [mm]
    scale   scalar  isotropic scale factor
    dice    scalar  voxel Dice after registration (quality metric)
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# NRRD loading
# ---------------------------------------------------------------------------

def _load_nrrd_heart_pts(nrrd_path: Path, heart_label) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the torso NRRD and return physical coordinates (mm) of all voxels
    whose label is in *heart_label*.

    Parameters
    ----------
    heart_label : int or sequence of int
        NRRD label value(s) that make up the heart (e.g. a single
        myocardium label, or (atria_label, ventricles_label)).

    Returns
    -------
    pts      : (N, 3) float64  physical mm coords of heart voxels
    spacing  : (3,)   float64
    origin   : (3,)   float64
    """
    try:
        import nrrd
    except ImportError as e:
        raise ImportError("pynrrd is required:  pip install pynrrd") from e

    volume, header = nrrd.read(str(nrrd_path))
    volume = np.asarray(volume)

    if "space directions" in header:
        sd      = np.asarray(header["space directions"], dtype=np.float64)
        spacing = np.array([np.linalg.norm(sd[i]) for i in range(3)])
        origin  = np.asarray(header.get("space origin", [0., 0., 0.]), np.float64)
    elif "spacings" in header:
        spacing = np.asarray(header["spacings"], dtype=np.float64)
        origin  = np.zeros(3, dtype=np.float64)
    else:
        spacing = np.ones(3, dtype=np.float64)
        origin  = np.zeros(3, dtype=np.float64)

    ijk = np.argwhere(np.isin(volume, np.atleast_1d(heart_label))).astype(np.float64)
    pts = ijk * spacing[np.newaxis, :] + origin[np.newaxis, :]
    return pts, spacing, origin


# ---------------------------------------------------------------------------
# Rotation search helpers
# ---------------------------------------------------------------------------

def _all_signed_permutation_matrices() -> list[np.ndarray]:
    """
    All 48 signed permutation matrices of the cube group: every axis
    permutation combined with every independent choice of per-axis sign.

    24 of these have det = +1 (proper rotations) and 24 have det = -1
    (improper rotations / reflections). Both are included -- see the
    "Note on det(R) = -1" section of this module's docstring for why
    reflections need to be in the search space here.
    """
    mats = []
    for perm in permutations([0, 1, 2]):
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    R = np.zeros((3, 3))
                    for i, (p, s) in enumerate(zip(perm, (s0, s1, s2))):
                        R[i, p] = s
                    mats.append(R)
    return mats  # exactly 48


def _voxel_dice(pts_a: np.ndarray,
                pts_b: np.ndarray,
                vox_mm: float) -> float:
    """Voxel-level Dice coefficient between two point clouds."""
    all_pts = np.vstack([pts_a, pts_b])
    mn      = all_pts.min(axis=0)

    def voxelise(pts: np.ndarray) -> np.ndarray:
        idx   = np.floor((pts - mn) / vox_mm).astype(np.int32)
        shape = tuple(idx.max(axis=0) + 2)
        vol   = np.zeros(shape, dtype=np.bool_)
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return vol

    va = voxelise(pts_a)
    vb = voxelise(pts_b)
    sh = tuple(max(va.shape[d], vb.shape[d]) for d in range(3))

    def pad(v: np.ndarray) -> np.ndarray:
        out = np.zeros(sh, dtype=np.bool_)
        out[:v.shape[0], :v.shape[1], :v.shape[2]] = v
        return out

    va, vb = pad(va), pad(vb)
    inter  = (va & vb).sum()
    denom  = va.sum() + vb.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Core registration
# ---------------------------------------------------------------------------

def register_heart_to_torso(
    src_pts:    np.ndarray,
    tgt_pts:    np.ndarray,
    dice_vox_mm: float = 3.0,
    verbose:    bool   = True,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute R, t, scale such that:

        (R @ (src_pts * scale).T).T + t  ≈  tgt_pts

    Parameters
    ----------
    src_pts      : (N, 3) coords_mm from vm_snapshots  [heart grid mm]
    tgt_pts      : (M, 3) physical mm coords of NRRD label-2 voxels
    dice_vox_mm  : voxel size used during rotation search Dice evaluation
    verbose      : print progress

    Returns
    -------
    R     : (3, 3) float64  one of the 48 signed permutation matrices
                            (24 proper rotations + 24 reflections)
    t     : (3,)   float64  translation  [mm]
    scale : float           isotropic scale factor (tgt_span / src_span)
    dice  : float           voxel Dice of registered cloud vs target
    """
    # ── 1. Isotropic scale from bounding-box spans ────────────────────────
    src_span     = src_pts.max(axis=0) - src_pts.min(axis=0)
    tgt_span     = tgt_pts.max(axis=0) - tgt_pts.min(axis=0)
    per_axis     = tgt_span / src_span
    scale        = float(np.median(per_axis))

    if verbose:
        print(f"  Snapshot bbox spans : {np.round(src_span, 1)} mm")
        print(f"  NRRD myo bbox spans : {np.round(tgt_span, 1)} mm")
        print(f"  Per-axis scale      : {np.round(per_axis, 4)}")
        print(f"  Isotropic scale     : {scale:.4f}")

    # ── 2. Scale source, then align bounding-box centres ─────────────────
    src_scaled = src_pts * scale
    c_src      = (src_scaled.max(axis=0) + src_scaled.min(axis=0)) / 2.0
    c_tgt      = (tgt_pts.max(axis=0)   + tgt_pts.min(axis=0))    / 2.0
    src_c      = src_scaled - c_src   # centred scaled source

    # ── 3. Subsample target for speed during rotation search ──────────────
    rng   = np.random.default_rng(0)
    n_sub = min(len(tgt_pts), 50_000)
    tgt_sub = tgt_pts[rng.choice(len(tgt_pts), n_sub, replace=False)]

    if verbose:
        print(f"\n  Searching 48 signed permutation matrices "
              f"(24 rotations + 24 reflections; "
              f"Dice vox = {dice_vox_mm} mm, "
              f"target subsample = {n_sub:,}) ...")

    # ── 4. Brute-force search over rotations + reflections ────────────────
    best_dice, best_R = -1.0, np.eye(3)
    for R in _all_signed_permutation_matrices():
        candidate = (R @ src_c.T).T + c_tgt
        d = _voxel_dice(candidate, tgt_sub, dice_vox_mm)
        if d > best_dice:
            best_dice, best_R = d, R

    # ── 5. Final translation with best R ─────────────────────────────────
    t = c_tgt - best_R @ c_src

    if verbose:
        print(f"  Best Dice : {best_dice:.4f}")
        print(f"  R =\n{np.round(best_R, 4)}")
        print(f"  t = {np.round(t, 2)} mm")
        print(f"  R det = {np.linalg.det(best_R):.4f}")
        if best_dice < 0.5:
            print("  [warn] Dice < 0.5 — inspect registration visually.")

    return best_R, t, scale, best_dice


# ---------------------------------------------------------------------------
# Public entry point (called by ecg/registration.py)
# ---------------------------------------------------------------------------

def run_registration(
    vm_snapshots_path: Path,
    torso_nrrd_path:   Path,
    heart_label       = 2,
    dice_vox_mm:       float = 3.0,
    verbose:           bool  = True,
) -> dict:
    """
    Load inputs, run registration, return result dict.

    Parameters
    ----------
    heart_label : int or sequence of int
        NRRD label value(s) that make up the heart.

    Returns
    -------
    dict with keys: R, t, scale, dice
    """
    # Load snapshot point cloud
    snap      = np.load(str(vm_snapshots_path))
    src_pts   = snap["coords_mm"].astype(np.float64)

    if verbose:
        print(f"  Snapshot nodes  : {len(src_pts):,}")
        print(f"  x [{src_pts[:,0].min():.1f}, {src_pts[:,0].max():.1f}]  "
              f"y [{src_pts[:,1].min():.1f}, {src_pts[:,1].max():.1f}]  "
              f"z [{src_pts[:,2].min():.1f}, {src_pts[:,2].max():.1f}] mm")

    # Load NRRD heart voxels
    if verbose:
        print(f"\n  Loading NRRD heart (label {heart_label}) from {torso_nrrd_path.name} ...")
    tgt_pts, spacing, origin = _load_nrrd_heart_pts(torso_nrrd_path, heart_label)

    if verbose:
        tgt_span = tgt_pts.max(axis=0) - tgt_pts.min(axis=0)
        print(f"  NRRD myo voxels : {len(tgt_pts):,}")
        print(f"  NRRD spacing    : {np.round(spacing, 3)} mm")
        print(f"  NRRD origin     : {np.round(origin, 2)} mm")

    # Register
    R, t, scale, dice = register_heart_to_torso(
        src_pts, tgt_pts,
        dice_vox_mm=dice_vox_mm,
        verbose=verbose,
    )

    return dict(R=R, t=t, scale=scale, dice=dice)


def apply_registration(coords_mm: np.ndarray,
                        R: np.ndarray,
                        t: np.ndarray,
                        scale: float) -> np.ndarray:
    """
    Map heart-grid coords into NRRD physical space.

        torso_coords = R @ (coords_mm * scale) + t

    Parameters
    ----------
    coords_mm : (N, 3)  coords in heart-grid mm  (e.g. from vm_snapshots)
    R         : (3, 3)  rotation
    t         : (3,)    translation
    scale     : float   isotropic scale

    Returns
    -------
    (N, 3) float64  physical coordinates in NRRD space [mm]
    """
    return (R @ (coords_mm * scale).T).T + t
