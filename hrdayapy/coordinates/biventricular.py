"""
coordinates/biventricular.py
=============================
Compute/load pair for the biventricular coordinate chi (1 = LV, 2 = RV).

compute_chi(...) needs the LV/RV endocardial seed masks that come out of
compute_geometry's anatomy dict (anatomy["lv_endo_voxels"], anatomy["rv_endo_voxels"]).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import generate_biventricular_coordinate


def compute_chi(
    S,
    lv_seed,
    rv_seed,
    *,
    connectivity: int = 1,
    save_path_chi=None,
    save_path_chi_labels=None,
):
    """
    Split the fused myocardium mask into LV / RV regions.

    Parameters
    ----------
    S                 : (Nx,Ny,Nz) bool -- myocardium mask, from compute_geometry
    lv_seed, rv_seed  : (Nx,Ny,Nz) bool -- LV/RV endocardial voxels, i.e.
                         anatomy["lv_endo_voxels"] / anatomy["rv_endo_voxels"].
                         Once psi is available, prefer the anatomy dict
                         returned by coordinates.refine_geometry_with_psi
                         (S, anatomy, psi) over the raw compute_geometry
                         output here: the Stage-1 split runs before psi
                         exists and falls back to a hull-volume heuristic
                         that can merge or misassign LV/RV endocardium
                         wherever they're topologically connected through
                         an open/cut base, whereas refine_geometry_with_psi
                         uses the psi-based basal cut (ported from
                         test_psi_basal_cut.py) to split them correctly.
    connectivity      : voxel connectivity used for the region-growing split
    save_path_chi        : if given, chi is written here as .npy
    save_path_chi_labels : if given, the integer label volume is written
                            here as .npy

    Always recomputes: this always runs the full computation and
    overwrites save_path_chi / save_path_chi_labels, even if a file is
    already sitting there. (generate_biventricular_coordinate itself has
    a "load if it already exists" shortcut for its own save_path, but
    this wrapper routes around it via a temp path so that flag -- True
    in the calling script -- reliably means "compute", full stop.)

    Returns
    -------
    chi    : (Nx,Ny,Nz) float  -- continuous biventricular coordinate
    labels : (Nx,Ny,Nz) int    -- 1 = LV, 2 = RV, 0 = background
    """
    tmp_path_chi = None
    if save_path_chi is not None:
        save_path_chi = Path(save_path_chi).with_suffix(".npy")
        save_path_chi.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_chi = save_path_chi.with_name(save_path_chi.stem + "_tmp_compute.npy")
        if tmp_path_chi.exists():
            tmp_path_chi.unlink()

    tmp_path_labels = None
    if save_path_chi_labels is not None:
        save_path_chi_labels = Path(save_path_chi_labels).with_suffix(".npy")
        save_path_chi_labels.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_labels = save_path_chi_labels.with_name(
            save_path_chi_labels.stem + "_tmp_compute.npy"
        )
        if tmp_path_labels.exists():
            tmp_path_labels.unlink()

    chi, labels, info = generate_biventricular_coordinate(
        S=S,
        lv_seed=lv_seed,
        rv_seed=rv_seed,
        connectivity=connectivity,
        save_path=tmp_path_chi,
        labels_save_path=tmp_path_labels,
    )

    if tmp_path_chi is not None:
        tmp_path_chi.replace(save_path_chi)
    if tmp_path_labels is not None:
        tmp_path_labels.replace(save_path_chi_labels)

    return chi, labels


def load_chi(path_chi, path_chi_labels):
    """Load a previously saved (chi, labels) pair. No recomputation."""
    path_chi = Path(path_chi)
    path_chi_labels = Path(path_chi_labels)
    if not path_chi.exists():
        raise FileNotFoundError(f"No saved chi at {path_chi}")
    if not path_chi_labels.exists():
        raise FileNotFoundError(f"No saved chi_labels at {path_chi_labels}")
    return np.load(path_chi), np.load(path_chi_labels)
