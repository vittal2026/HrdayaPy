"""
generate_biventricular_coordinate.py
=====================================
Wraps split_lv_rv into a continuous-style coordinate field chi, matching
the array convention used by generate_transmural_coordinate (phi) and
generate_apicobasal_coordinate (psi):

    chi = 0.0   ->  LV myocardium
    chi = 1.0   ->  RV myocardium
    chi = NaN   ->  outside the myocardium mask (or unreached by the
                     watershed -- see the "unassigned" warning below)

Unlike phi/psi this is not a Laplace solve -- it is the binary chamber
membership produced by split_lv_rv, reshaped into the same NaN-outside
array convention so it slots into the rest of the pipeline (and into
ectopic-stimulus location lookups) exactly the way phi/psi do.

Public API
----------
    chi, labels, info = generate_biventricular_coordinate(S, axis, base_idx, ...)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .split_lv_rv import split_lv_rv


def generate_biventricular_coordinate(
    S: np.ndarray,
    axis: int | None = None,
    base_idx: int | None = None,
    connectivity: int = 1,
    save_path: str | Path | None = None,
    labels_save_path: str | Path | None = None,
    lv_seed: np.ndarray | None = None,
    rv_seed: np.ndarray | None = None,
):
    """
    Parameters
    ----------
    S : (Nx,Ny,Nz) ndarray, bool or {0,1}
        Binary myocardium mask (LV+RV fused, no cavities).
    axis, base_idx : legacy path only, as returned by
        find_apico_basal_axis / compute_basal_plane.
    connectivity : int, default 1
        Passed through to split_lv_rv (legacy path only).
    lv_seed, rv_seed : (Nx,Ny,Nz) bool, optional
        LV / RV endocardial surface-voxel masks (preferred path, e.g.
        from mesh_labelling.label_ventricle_mesh). See split_lv_rv for
        why this is preferred over the axis-based legacy path.
    save_path : if given, cache `chi` to this .npy path; subsequent
        calls load from disk instead of recomputing (requires
        `labels_save_path` to also exist, or be omitted).
    labels_save_path : if given, cache the underlying integer `labels`
        (0=background, 1=LV, 2=RV) to this .npy path alongside chi.

    Returns
    -------
    chi    : (Nx,Ny,Nz) float32   0.0=LV, 1.0=RV, NaN outside mask
    labels : (Nx,Ny,Nz) uint8     0=background, 1=LV, 2=RV (from split_lv_rv)
    info   : dict                 diagnostics from split_lv_rv (empty
                                   dict if loaded from cache)
    """
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
    if labels_save_path is not None:
        labels_save_path = Path(labels_save_path).with_suffix(".npy")

    if (save_path is not None and save_path.exists()
            and (labels_save_path is None or labels_save_path.exists())):
        print(f"  [biventricular] Loading cached chi from {save_path}")
        chi = np.load(save_path)
        labels = np.load(labels_save_path) if labels_save_path is not None else None
        return chi, labels, {}

    S = np.asarray(S).astype(bool)
    print(f"  [biventricular] Splitting {int(S.sum()):,} myocardial "
          f"voxels into LV / RV ...")
    labels, info = split_lv_rv(S, axis=axis, base_idx=base_idx,
                                connectivity=connectivity,
                                lv_seed=lv_seed, rv_seed=rv_seed)

    n_lv = int((labels == 1).sum())
    n_rv = int((labels == 2).sum())
    n_unassigned = int(S.sum()) - n_lv - n_rv
    print(f"  [biventricular] LV: {n_lv:,}  |  RV: {n_rv:,}  |  "
          f"unassigned: {n_unassigned:,}")
    if n_unassigned > 0:
        print("  [biventricular] WARNING: some myocardial voxels were not "
              "reached by the watershed (disconnected from both seeds) -- "
              "they will be NaN in chi. Check mask connectivity.")

    chi = np.full(S.shape, np.nan, dtype=np.float32)
    chi[labels == 1] = 0.0
    chi[labels == 2] = 1.0

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, chi)
        print(f"  [biventricular] Saved chi -> {save_path}")
    if labels_save_path is not None:
        labels_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_save_path, labels)
        print(f"  [biventricular] Saved labels -> {labels_save_path}")

    return chi, labels, info
