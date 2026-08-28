"""
coordinates/rotational.py
============================
Compute/load pair for the rotational coordinate theta (0 to 2*pi around
the long axis). This is interactive the first time (you pick the septal
seam's apex/AV-node split and the anterior-ridge tip); after that, the
septal ridge and anterior vertex are cached so re-running with the same
ridge_save_path / anterior_vertex_save_path skips the picking step.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import generate_rotational_coordinate


def compute_theta(
    S,
    chi_labels,
    psi,
    long_axis,
    *,
    mesh_step: int = 2,
    n_bins: int = 100,
    lv_inner_frac: float = 0.35,
    rv_inner_frac: float = 0.0,
    smooth_sigma: float = 2.0,
    psi_apex_threshold: float = 0.05,
    ridge_save_path=None,
    anterior_vertex_save_path=None,
    save_path=None,
):
    """
    Solve for the rotational coordinate theta.

    Parameters
    ----------
    S          : (Nx,Ny,Nz) bool -- myocardium mask
    chi_labels : (Nx,Ny,Nz) int  -- 0=bg, 1=LV, 2=RV, from compute_chi
    psi        : (Nx,Ny,Nz) float -- apicobasal coordinate, from compute_psi
    long_axis  : (3,) unit vector, apex -> base, from anatomy["long_axis"]
    mesh_step  : marching-cubes step size for the picking surface
    n_bins     : psi-stratification bins for the per-bin reference frame
    lv_inner_frac, rv_inner_frac : innermost voxel fraction used for each
                 ventricle's ring centroid
    smooth_sigma        : Gaussian smoothing along psi for the frames
    psi_apex_threshold  : voxels below this psi get theta = NaN
    ridge_save_path             : .npz cache for the septal seam + arcs
    anterior_vertex_save_path   : .npy cache for the anterior-ridge tip
    save_path                   : if given, theta is written here as .npy

    Always recomputes theta itself: this always runs the full theta
    solve and overwrites save_path, even if a file is already sitting
    there. (generate_rotational_coordinate has a "load if it already
    exists" shortcut for its own save_path, but this wrapper routes
    around it via a temp path so that flag -- True in the calling
    script -- reliably means "compute", full stop.)

    Note this does NOT touch ridge_save_path / anterior_vertex_save_path
    -- those cache the interactively-picked septal seam / anterior
    vertex on purpose, so re-running doesn't force you to re-click them;
    delete those files yourself if you actually want to re-pick.

    Returns
    -------
    theta : (Nx,Ny,Nz) float, in [0, 2*pi) inside the myocardium, NaN
            elsewhere
    """
    tmp_path = None
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = save_path.with_name(save_path.stem + "_tmp_compute.npy")
        if tmp_path.exists():
            tmp_path.unlink()

    theta = generate_rotational_coordinate(
        S=S, chi_labels=chi_labels, psi=psi, long_axis=long_axis,
        mesh_step=mesh_step, n_bins=n_bins,
        lv_inner_frac=lv_inner_frac, rv_inner_frac=rv_inner_frac,
        smooth_sigma=smooth_sigma, psi_apex_threshold=psi_apex_threshold,
        ridge_save_path=ridge_save_path,
        anterior_vertex_save_path=anterior_vertex_save_path,
        save_path=tmp_path,
    )

    if tmp_path is not None:
        tmp_path.replace(save_path)

    return theta


def load_theta(path):
    """Load a previously saved theta field. No recomputation, no picker."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved theta at {path}")
    return np.load(path)
