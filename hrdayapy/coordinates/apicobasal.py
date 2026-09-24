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
from .functions._multires import coarsen_mask, upsample_field


def compute_psi(
    S,
    apex_mask,
    basal_mask,
    *,
    apex_percent: float = 0.10,
    max_iter: int = 400,
    tol: float = 1e-5,
    device=None,
    save_path=None,
    coarsen_factor: int = 1,
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
    coarsen_factor       : int, default 1 (no coarsening). If > 1, solves
                            the Laplace equation on S downsampled by this
                            integer factor per axis, then trilinearly
                            upsamples back to S's full resolution (exact
                            apex/basal Dirichlet values re-applied
                            afterward at the true fine-resolution masks).
                            psi is smooth/low-frequency, so a coarse solve
                            captures its large-scale shape well -- use
                            this when S's own resolution is finer than the
                            solve needs to be accurate at, e.g. because S
                            was chosen for the physics simulation's
                            resolution needs, not psi's. See
                            functions/_multires.py for the tradeoff this
                            makes. Use 1 whenever a full-resolution solve
                            is actually affordable.

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

    if coarsen_factor > 1:
        S_bool = S.astype(bool)
        coarse_shape = tuple(-(-s // coarsen_factor) for s in S_bool.shape)
        print(f"  [apicobasal] coarsen_factor={coarsen_factor}: solving on "
              f"a ~{coarse_shape} grid, then upsampling to {S_bool.shape}")
        S_c     = coarsen_mask(S_bool, coarsen_factor)
        apex_c  = coarsen_mask(apex_mask, coarsen_factor) & S_c
        basal_c = coarsen_mask(basal_mask, coarsen_factor) & S_c

        psi_c = generate_apicobasal_coordinate(
            S=S_c, apex_mask=apex_c, basal_mask=basal_c,
            apex_percent=apex_percent, max_iter=max_iter, tol=tol,
            device=device, save_path=None,
        )
        psi = upsample_field(
            psi_c, S_bool.shape, fine_mask=S_bool,
            dirichlet_fine=[
                (apex_mask.astype(bool) & S_bool, 1e-6),
                (basal_mask.astype(bool) & S_bool, 1.0),
            ],
        )
        if tmp_path is not None:
            np.save(tmp_path, psi)
            tmp_path.replace(save_path)
        return psi

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
