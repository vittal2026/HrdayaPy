"""
ecg/torso_grid.py
====================
Compute/load pair for the coarse torso finite-difference grid (Step 10).

Resamples the torso NRRD to dx_coarse_mm, assigns conductivities from
sigma_map, assembles and factorises the FDM stiffness matrix, and
identifies surface nodes -- everything compute_bspm needs to run the
per-frame Poisson solves in Step 11.

Note: the returned dict's "solve_fn" (bound to the factorised matrix)
and "A_scipy" (raw sparse stiffness matrix) are NOT saved to disk --
only the geometry/conductivity/surface metadata is. load_torso_grid
therefore still re-assembles and re-factorises the stiffness matrix on
every call (fast: ~10-30s), it just skips the NRRD resampling step.
It needs the same sigma_map / dx_coarse_mm compute_torso_grid was
called with, not just save_path.
"""

from __future__ import annotations

from .functions import build_torso_grid as _build_torso_grid
from .functions import load_torso_grid as _load_torso_grid


def compute_torso_grid(
    torso_nrrd_path,
    sigma_map: dict,
    *,
    dx_coarse_mm: float = 2.0,
    verbose: bool = True,
    save_path=None,
    use_sbm: bool = False,
    sbm_interface_width_mm: float = 3.0,
    sbm_psi_cutoff: float = 1e-3,
):
    """
    Build, factorise, and save the coarse torso FDM grid.

    Parameters
    ----------
    torso_nrrd_path : path to torso.seg.nrrd
    sigma_map       : dict  label (int) -> conductivity (S/m)
    dx_coarse_mm    : coarse voxel size (mm)
    verbose         : print progress
    save_path       : where to write torso_grid.npz (geometry/conductivity
                       metadata only -- the factorisation itself isn't
                       serialisable, see module docstring)
    use_sbm         : if True, solve with the Smoothed Boundary Method
                       instead of the sharp/staircased boundary -- builds
                       a smoothed domain function ψ from the FINE
                       segmentation (captures sub-coarse-cell boundary
                       position, unlike smoothing an already-binarized
                       coarse mask) and solves ∇·(ψσ∇φ)=ψb. See
                       hrdayapy.ecg.functions.build_torso_grid module
                       docstring ("Smoothed Boundary Method") for the
                       full derivation and a synthetic-sphere validation.
                       Default False reproduces the original sharp
                       boundary exactly.
    sbm_interface_width_mm : ψ transition width (only used if use_sbm).
                       ~1-2 coarse cells is the usual SBM choice.
    sbm_psi_cutoff  : ψ floor / system-inclusion threshold (only used if
                       use_sbm).

    Returns
    -------
    dict with keys: shape, dx_mm, origin_mm, sigma_vol, active_mask,
    tissue_mask, active_ids, active_id_map, surface_flat, surface_rows,
    surface_xyz, pin_row, solve_fn, A_scipy, use_sbm, psi_vol (SBM only)
    """
    return _build_torso_grid(
        torso_nrrd_path=torso_nrrd_path,
        sigma_map=sigma_map,
        dx_coarse_mm=dx_coarse_mm,
        out_path=save_path,
        verbose=verbose,
        use_sbm=use_sbm,
        sbm_interface_width_mm=sbm_interface_width_mm,
        sbm_psi_cutoff=sbm_psi_cutoff,
    )


def load_torso_grid(
    save_path,
    torso_nrrd_path,
    sigma_map: dict,
    *,
    dx_coarse_mm: float = 2.0,
    verbose: bool = True,
):
    """
    Load a previously saved torso_grid.npz and re-factorise the stiffness
    matrix (the sparse factor can't be serialised to disk). Needs the
    same torso_nrrd_path / sigma_map / dx_coarse_mm compute_torso_grid
    was called with -- see module docstring.
    """
    return _load_torso_grid(
        out_path=save_path,
        torso_nrrd_path=torso_nrrd_path,
        sigma_map=sigma_map,
        dx_coarse_mm=dx_coarse_mm,
        verbose=verbose,
    )
