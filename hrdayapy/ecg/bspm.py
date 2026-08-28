"""
ecg/bspm.py
=============
Compute/load pair for the body surface potential map (BSPM), Steps 11-12
of the ECG forward solve: per-frame impressed-current + Poisson solve
mapping myocardial Vm snapshots onto the torso surface.

compute_bspm needs:
  - vm_snapshots_path : from simulation.compute_coupled's vm_save_path
  - grid              : from ecg.compute_torso_grid / load_torso_grid
  - registration      : from ecg.compute_registration / load_registration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .functions import run_bspm_loop
from .._spacing_resolve import resolve_voxel_size

# Defaults below mirror the values HrdayaPy's legacy config.py derives for
# a myocardium modelled with CPL_SIGMA_M = 2.625e-4 (intracellular
# conductivity, monodomain-scaled) and a 0.4 mm fine voxel grid (the
# simulation.compute_coupled default voxel_size). Override sigma_i /
# h_fine_mm if your patient's simulation used different physics/voxel
# parameters.
DEFAULT_SIGMA_I = 0.2625      # S/m
DEFAULT_H_FINE_MM = 0.4       # mm


def compute_bspm(
    vm_snapshots_path,
    grid: dict,
    registration: dict,
    *,
    anatomy=None,
    sigma_i: float = DEFAULT_SIGMA_I,
    h_fine_mm=DEFAULT_H_FINE_MM,
    cg_tol_mV: float = 0.01,
    cg_max_iter: int = 500,
    cg_metric: str = "rms",
    t_window=None,
    verbose: bool = True,
    save_path=None,
    full_field_frames=None,
    full_field_save_path=None,
):
    """
    Compute body surface potential maps from Vm snapshots.

    Parameters
    ----------
    vm_snapshots_path : path to *_vm_snapshots.npz
    grid              : dict from ecg.compute_torso_grid / load_torso_grid
    registration      : dict from ecg.compute_registration / load_registration
    anatomy           : dict, optional -- from coordinates.compute_geometry /
                        load_geometry. Only needed (and only used) when
                        h_fine_mm="from_geometry"; ignored otherwise.
    sigma_i           : intracellular/myocardial conductivity (S/m) -- see
                        module-level note on defaults
    h_fine_mm         : float | "from_geometry" -- fine myocardial voxel
                        size (mm), must match the voxel_size the coupled
                        simulation was run with. Pass a float to hand-pick
                        a value as before, or "from_geometry" to pull it
                        from anatomy["spacing_mm"] instead, so it can't
                        drift out of sync with the coupled simulation's
                        own voxel_size (requires anatomy=..., and only
                        helps if compute_coupled was ALSO run with
                        voxel_size="from_geometry" off the same anatomy).
    cg_tol_mV         : absolute convergence target for the PCG solve, in
                        mV of body-surface potential, applied to whichever
                        statistic `cg_metric` selects of z = M_inv_diag *
                        r, the Jacobi-preconditioned residual (an
                        approximate per-node phi error; see
                        compute_bspm.py module docstring). Interpretable
                        and independent of each frame's source magnitude,
                        unlike a relative residual tolerance. Default
                        0.01 mV.
    cg_max_iter       : iteration cap
    cg_metric         : which statistic of the per-node error estimate
                        to test against cg_tol_mV --
                          'rms'  (default) bulk/typical error, cheapest
                          'p99'  99th-percentile, robust worst-case
                          'linf' true worst-case (every node guaranteed
                                 under cg_tol_mV) -- strictest, most
                                 iterations
                        See _pcg_gpu docstring in compute_bspm.py for
                        the full tradeoff.
    t_window          : optional (t_min_ms, t_max_ms) to restrict the
                        solve to a sub-window of frames (e.g. QRS/T only)
    save_path         : where to write bspm.npz
    full_field_frames : optional array of frame indices (post-t_window) at
                        which to ALSO capture the complete (not
                        surface-restricted) active-node phi field, for
                        whole-volume cross-level convergence comparison.
                        Keep this small (tens of frames) -- the full field
                        is 40-100x the surface-only signal in size. See
                        run_bspm_loop docstring for the exact sizing.
                        None (default): no full-field capture.
    full_field_save_path : where to write the full-field capture (a
                        separate, potentially much larger .npz from
                        save_path). Required if full_field_frames is given.

    Returns
    -------
    bspm_signal : (N_surf, n_frames) float32 [mV]
    """
    if save_path is None:
        raise ValueError("compute_bspm requires save_path (run_bspm_loop "
                          "writes results incrementally as it solves).")
    h_fine_mm = resolve_voxel_size(h_fine_mm, anatomy, param_name="h_fine_mm")
    return run_bspm_loop(
        vm_snapshots_path=Path(vm_snapshots_path),
        grid=grid,
        registration=registration,
        sigma_i=sigma_i,
        h_fine_mm=h_fine_mm,
        out_path=Path(save_path),
        cg_tol_mV=cg_tol_mV,
        cg_max_iter=cg_max_iter,
        cg_metric=cg_metric,
        t_window=t_window,
        verbose=verbose,
        full_field_frames=full_field_frames,
        full_field_out_path=(Path(full_field_save_path)
                              if full_field_save_path is not None else None),
    )


def load_bspm(path):
    """
    Load a previously saved bspm.npz. No recomputation.

    Returns
    -------
    dict with keys: surface_xyz, surface_flat, dx_mm, bspm_signal, time_ms,
    reg_R, reg_t, reg_scale, reg_dice, sigma_i, h_fine_mm
    """
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved BSPM at {path}")
    z = np.load(str(path))
    return {k: z[k] for k in z.files}
