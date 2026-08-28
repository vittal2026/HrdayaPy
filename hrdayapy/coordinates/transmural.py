"""
coordinates/transmural.py
===========================
Compute/load pair for the transmural coordinate phi (0 = endocardium,
1 = epicardium). Solves a Laplace equation between the two surfaces.

The same pair is used for both phi and phi_rv -- they're the same
computation with a different surface_label input (anatomy["surface_label"]
for phi, anatomy["surface_label_rv"] for the RV-specific version). Call
compute_phi twice with two different save paths if you need both.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import generate_transmural_coordinate


def compute_phi(
    S,
    surface_label,
    *,
    device="cuda",
    tol: float = 1e-6,
    maxiter: int = 10_000,
    save_path=None,
):
    """
    Solve for the transmural coordinate phi.

    Parameters
    ----------
    S             : (Nx,Ny,Nz) bool/uint8 -- myocardium mask
    surface_label : (Nx,Ny,Nz) int8 -- +1 epicardium / -1 endocardium / 0
                    interior. Use anatomy["surface_label"] for phi, or
                    anatomy["surface_label_rv"] for phi_rv.
    device        : "cuda" or "cpu"
    tol, maxiter  : conjugate-gradient solver tolerance / iteration cap
    save_path     : if given, phi is written here as .npy

    Always recomputes: this always runs the full solve and overwrites
    save_path, even if a file is already sitting there.
    (generate_transmural_coordinate itself has a "load if it already
    exists" shortcut for its own save_path, but this wrapper routes
    around it via a temp path so that flag -- True in the calling
    script -- reliably means "compute", full stop.)

    Returns
    -------
    phi : (Nx,Ny,Nz) float, in [0, 1] inside the myocardium, NaN elsewhere
    """
    tmp_path = None
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = save_path.with_name(save_path.stem + "_tmp_compute.npy")
        if tmp_path.exists():
            tmp_path.unlink()

    phi = generate_transmural_coordinate(
        S=S, surface_label=surface_label, device=device,
        tol=tol, maxiter=maxiter, save_path=tmp_path,
    )

    if tmp_path is not None:
        tmp_path.replace(save_path)

    return phi


def load_phi(path):
    """Load a previously saved phi (or phi_rv) field. No recomputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved phi at {path}")
    return np.load(path)
