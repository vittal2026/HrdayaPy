"""
inspect_pipeline/torso_registration.py
=========================================
Compute/load pair for registering the ground-truth dataset's torso mesh
onto your simulated BSPM's torso surface (rigid ICP) -- the step that
makes "pick a point and compare simulated vs recorded" possible.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import run_torso_registration


def compute_torso_registration(
    bspm_npz_path,
    gt_torso_mat_path,
    *,
    max_iterations: int = 60,
    tol_mm: float = 1e-4,
    target_subsample: int = 20_000,
    verbose: bool = True,
    save_path=None,
):
    """
    Register the dataset's torso mesh onto the simulated BSPM's torso
    surface via rigid ICP.

    Parameters
    ----------
    bspm_npz_path     : path to bspm.npz, from ecg.compute_bspm
    gt_torso_mat_path : path to the ground-truth dataset's torso mesh (.mat)
    max_iterations, tol_mm, target_subsample : ICP settings
    save_path         : if given, the result is written here as .npz

    Returns
    -------
    registration : dict with keys R (3x3), t (3,), pre_rmse, pre_mean,
                   rmse, mean_dist, max_dist, n_iter
    """
    result = run_torso_registration(
        bspm_npz_path=bspm_npz_path, gt_torso_mat_path=gt_torso_mat_path,
        max_iterations=max_iterations, tol_mm=tol_mm,
        target_subsample=target_subsample, verbose=verbose,
    )
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(save_path), **result)
    return result


def load_torso_registration(path):
    """Load a previously saved torso registration. No recomputation."""
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved torso registration at {path}")
    data = np.load(str(path))
    return {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
