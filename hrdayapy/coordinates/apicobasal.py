"""
coordinates/apicobasal.py
============================
Compute/load pair for the apicobasal coordinate psi (0 = apex, 1 = base).
Solves a Laplace equation between the apex and basal landmark regions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import generate_apicobasal_coordinate


def compute_psi(
    S,
    apex_mask,
    basal_mask,
    *,
    apex_percent: float = 0.10,
    max_iter: int = 40,
    tol: float = 1e-5,
    device=None,
    save_path=None,
):
    """
    Solve for the apicobasal coordinate psi.

    Parameters
    ----------
    S                    : (Nx,Ny,Nz) bool -- myocardium mask
    apex_mask, basal_mask: (Nx,Ny,Nz) bool -- landmarks["apex_voxels"] /
                            landmarks["basal_voxels"] from compute_landmarks
    apex_percent         : fallback apex-region size if apex_mask is coarse
    max_iter, tol        : solver iteration cap / tolerance
    device               : "cuda", "cpu", or None (auto)
    save_path            : if given, psi is written here as .npy

    Always recomputes: this always runs the full solve and overwrites
    save_path, even if a file is already sitting there.
    (generate_apicobasal_coordinate itself has a "load if it already
    exists" shortcut for its own save_path, but this wrapper routes
    around it via a temp path so that flag -- True in the calling
    script -- reliably means "compute", full stop.)

    Returns
    -------
    psi : (Nx,Ny,Nz) float, in [0, 1] inside the myocardium, NaN elsewhere
    """
    tmp_path = None
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = save_path.with_name(save_path.stem + "_tmp_compute.npy")
        if tmp_path.exists():
            tmp_path.unlink()

    psi = generate_apicobasal_coordinate(
        S=S, apex_mask=apex_mask, basal_mask=basal_mask,
        apex_percent=apex_percent, max_iter=max_iter, tol=tol,
        device=device, save_path=tmp_path,
    )

    if tmp_path is not None:
        tmp_path.replace(save_path)

    return psi


def load_psi(path):
    """Load a previously saved psi field. No recomputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved psi at {path}")
    return np.load(path)
