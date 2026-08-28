"""
simulation/coupled.py
========================
Compute/load pair for the coupled Purkinje + myocardium monodomain
simulation (Step 7) -- the main Vm solve that everything downstream
(animation, ECG forward solve) is built on.

Two outputs come out of one run:
  1. The main coupled result (surface + Purkinje branch voltage traces,
     used for animation) -- save_path / load_coupled.
  2. Optional volumetric Vm snapshots on every myocardial node (needed
     later for the ECG forward solve) -- vm_save_path / load_vm_snapshots,
     only produced if you ask for it (vm_save_dt is not None).

Note: load_coupled's return shape mirrors the saved .npz (time,
comp_nodes, comp_edges, surf_verts, surf_faces, vert_to_surf_node,
surf_frames, bmap_*, branch_*) -- this is NOT the same dict shape that
compute_coupled returns directly in memory (run_coupled's own richer
"results" dict). If you're chaining straight from compute_coupled, use
its return value directly; load_coupled is for starting midway from a
saved file.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import run_coupled
from .._spacing_resolve import resolve_voxel_size


def compute_coupled(
    nodes, elements, act_times, S, Z, phi,
    *,
    anatomy=None,
    stim_protocol=None,
    ectopic_region=None,
    voxel_size=0.4,
    dx_p: float = 0.2,
    sigma_P: float = 0.45,
    sigma_M: float = 1.31e-4,
    Cm: float = 0.01,
    A_P: float = 133.33,
    A_M: float = 0.14,
    R_P: float = 0.015,
    dt: float = 0.1,
    T: float = 400.0,
    theta: float = 0.5,
    stim_len_mm: float = 2.0,
    c_pmj: float = 0.05,
    n_pmj: int = 1,
    cg_tol: float = 1e-5,
    cg_max_iter: int = 500,
    n_frames: int = 100,
    device=None,
    phi_endo_max: float = 0.35,
    phi_epi_min: float = 0.65,
    save_path=None,
    vm_save_dt=None,
    vm_save_path=None,
    vm_tmp_dir=None,
):
    """
    Run the coupled Purkinje + myocardium monodomain simulation.

    Parameters
    ----------
    nodes, elements, act_times : from purkinje.compute_network / load_network
    S       : (Nx,Ny,Nz) bool  -- full myocardium mask
    Z       : (Nx,Ny,Nz) bool  -- cut mask, for the animated surface
    phi     : (Nx,Ny,Nz) float -- transmural coordinate, sets ionic
              heterogeneity via phi_endo_max / phi_epi_min
    anatomy : dict, optional -- from coordinates.compute_geometry /
              load_geometry. Only needed (and only used) when
              voxel_size="from_geometry"; ignored otherwise.
    voxel_size : float | "from_geometry" -- mm per voxel. Pass a float
              to hand-pick a value as before, or "from_geometry" to pull
              it from anatomy["spacing_mm"] instead, so it can't drift
              out of sync with what S/Z were actually resampled to
              (requires anatomy=...).
    stim_protocol   : list[dict] or None -- stimulus sites/timing
    ectopic_region  : (Nx,Ny,Nz) bool or None -- from
                       coordinates.compute_stim_region / compute_roi
    (remaining physics/solver parameters -- see run_coupled's own
    docstring for full detail; defaults here match run_coupled's)
    save_path    : if given, the surface/branch traces are written here (.npz)
    vm_save_dt   : sampling interval in ms for full volumetric Vm
                   snapshots; None (default) disables this second output
    vm_save_path : where to write the volumetric Vm snapshots, required
                   if vm_save_dt is not None
    vm_tmp_dir   : directory for the on-disk scratch buffer used while
                   accumulating Vm snapshots during the run (avoids holding
                   the full volumetric history in RAM). Defaults to the
                   same directory as vm_save_path. The scratch file is
                   deleted automatically once the run finishes.

    Returns
    -------
    results : dict -- run_coupled's raw in-memory result (times,
              comp_nodes, comp_edges, surf_verts, surf_faces,
              vert_to_surf_node, frames_surf, branch_map,
              branch_frames_p, ...)
    """
    voxel_size = resolve_voxel_size(voxel_size, anatomy, param_name="voxel_size")
    return run_coupled(
        nodes=nodes, elements=elements, activation_times=act_times,
        S=S, Z=Z, phi=phi,
        voxel_size=voxel_size, dx_p=dx_p, sigma_P=sigma_P, sigma_M=sigma_M,
        Cm=Cm, A_P=A_P, A_M=A_M, R_P=R_P, dt=dt, T=T, theta=theta,
        stim_protocol=stim_protocol, ectopic_region=ectopic_region,
        stim_len_mm=stim_len_mm, c_pmj=c_pmj, n_pmj=n_pmj,
        cg_tol=cg_tol, cg_max_iter=cg_max_iter, n_frames=n_frames,
        device=device, out_npz=save_path,
        phi_endo_max=phi_endo_max, phi_epi_min=phi_epi_min,
        vm_save_dt=vm_save_dt, out_vm_npz=vm_save_path,
        vm_tmp_dir=vm_tmp_dir,
    )


def load_coupled(path):
    """
    Load a previously saved coupled-simulation .npz (surface + Purkinje
    branch traces). No recomputation. See module docstring for how this
    differs from compute_coupled's direct return value.
    """
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved coupled simulation at {path}")
    z = np.load(path)
    return {k: z[k] for k in z.files}
