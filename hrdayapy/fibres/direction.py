"""
fibres/direction.py
======================
Compute/load pair for the rule-based myocardial fibre direction field.
Needs phi (transmural) and psi (apicobasal) from hrdayapy.coordinates --
no additional inputs (e.g. DT-MRI) required.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import generate_fibre_direction


def compute_fibres(
    S,
    phi,
    psi,
    spacing_mm,
    *,
    alpha_endo_deg: float = 60.0,
    alpha_epi_deg: float = -60.0,
    bbox_pad: int = 4,
    verbose: bool = True,
    save_path=None,
):
    """
    Solve for the rule-based fibre direction field.

    Parameters
    ----------
    S                  : (Nx,Ny,Nz) bool -- myocardium mask
    phi                : (Nx,Ny,Nz) float -- transmural coordinate (compute_phi)
    psi                : (Nx,Ny,Nz) float -- apicobasal coordinate (compute_psi)
    spacing_mm         : (3,) float -- anatomy["spacing_mm"]
    alpha_endo_deg, alpha_epi_deg : helix angle (deg) at phi=0 / phi=1
    bbox_pad           : internal memory-saving crop padding (voxels);
                         default is safely larger than every stencil used
    verbose            : print progress + sanity-check diagnostics
    save_path          : if given, f is written here as .npy

    Always recomputes: this always runs the full solve and overwrites
    save_path, even if a file is already sitting there.
    (generate_fibre_direction itself has a "load if it already exists"
    shortcut for its own save_path, but this wrapper routes around it via
    a temp path so that flag -- True in the calling script -- reliably
    means "compute", full stop. Same convention as coordinates.compute_phi
    / compute_psi.)

    Returns
    -------
    f : (Nx,Ny,Nz,3) float32 -- unit fibre direction cosines, 0 outside S
    """
    tmp_path = None
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = save_path.with_name(save_path.stem + "_tmp_compute.npy")
        if tmp_path.exists():
            tmp_path.unlink()

    f = generate_fibre_direction(
        S=S, phi=phi, psi=psi, spacing_mm=spacing_mm,
        alpha_endo_deg=alpha_endo_deg, alpha_epi_deg=alpha_epi_deg,
        bbox_pad=bbox_pad, verbose=verbose, save_path=tmp_path,
    )

    if tmp_path is not None:
        tmp_path.replace(save_path)

    return f


def load_fibres(path):
    """Load a previously saved fibre direction field. No recomputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved fibre field at {path}")
    return np.load(path)
