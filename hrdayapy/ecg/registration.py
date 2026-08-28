"""
ecg/registration.py
=====================
Compute/load pair for heart-to-torso registration (Step 9).

Registers the simulation heart point cloud (coords_mm in vm_snapshots.npz,
from hrdayapy.simulation.compute_coupled's vm_save_path output) to the
myocardium region (heart_label) of the full torso NRRD segmentation, via
a brute-force search over the 24 cube rotations plus isotropic scale +
translation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .functions import run_registration


def compute_registration(
    vm_snapshots_path,
    torso_nrrd_path,
    *,
    heart_label: int = 2,
    dice_vox_mm: float = 3.0,
    verbose: bool = True,
    save_path=None,
):
    """
    Register the coupled-simulation heart point cloud to the torso NRRD.

    Parameters
    ----------
    vm_snapshots_path : path to *_vm_snapshots.npz (from
                         simulation.compute_coupled's vm_save_path)
    torso_nrrd_path    : path to torso.seg.nrrd (full torso segmentation)
    heart_label        : NRRD label index for myocardium (default 2)
    dice_vox_mm         : voxel size used during rotation-search Dice
                          evaluation (mm)
    verbose            : print progress
    save_path          : if given, the registration is written here (.npz)

    Returns
    -------
    dict with keys: R (3x3), t (3,), scale (float), dice (float)
    """
    result = run_registration(
        vm_snapshots_path=Path(vm_snapshots_path),
        torso_nrrd_path=Path(torso_nrrd_path),
        heart_label=heart_label,
        dice_vox_mm=dice_vox_mm,
        verbose=verbose,
    )
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(save_path),
            R=result["R"], t=result["t"],
            scale=result["scale"], dice=result["dice"],
        )
    return result


def load_registration(path):
    """Load a previously saved heart-to-torso registration. No recomputation."""
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved registration at {path}")
    data = np.load(str(path))
    return {
        "R": data["R"],
        "t": data["t"],
        "scale": float(data["scale"]),
        "dice": float(data["dice"]),
    }
