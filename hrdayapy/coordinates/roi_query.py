"""
coordinates/roi_query.py
===========================
Compute/load pair for a coordinate-space ROI defined by a query dict,
e.g. {"ventricle": "LV", "psi": (0.8, 0.9), "theta_deg": (60, 80)}.

This is the range-based sibling of stimulus_region.py's target+/-tolerance
selection -- same idea (pick out a subset of the myocardium by its
coordinates), different way of specifying the window. Use whichever
reads more naturally for what you're doing.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import build_roi_mask


def compute_roi(query, chi_labels, psi, theta, phi=None, *, save_path=None):
    """
    Select myocardial voxels matching a coordinate-space query.

    Parameters
    ----------
    query      : dict, any subset of "ventricle" ("LV"/"RV"/"both"),
                 "psi": (lo, hi), "theta_deg": (lo, hi), "theta_rad": (lo, hi),
                 "phi": (lo, hi)
    chi_labels : (Nx,Ny,Nz) int   -- 0=bg, 1=LV, 2=RV, from compute_chi
    psi        : (Nx,Ny,Nz) float -- from compute_psi
    theta      : (Nx,Ny,Nz) float -- from compute_theta
    phi        : (Nx,Ny,Nz) float, optional -- from compute_phi
    save_path  : if given, the boolean ROI is written here as .npy

    Returns
    -------
    roi : (Nx,Ny,Nz) bool
    """
    roi = build_roi_mask(query, chi_labels=chi_labels, psi=psi, theta=theta, phi=phi)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, roi)
    return roi


def load_roi(path):
    """Load a previously saved ROI mask. No recomputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved ROI at {path}")
    return np.load(path)
