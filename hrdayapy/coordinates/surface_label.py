"""
coordinates/surface_label.py
===============================
Compute/load pair for the epi/endo surface label volume
(+1 = epicardium, -1 = endocardium, 0 = interior).

This isn't a new computation -- it's already inside the anatomy dict
that compute_geometry returns (anatomy["surface_label"]). It gets its
own save/load pair here because it's an output people reuse on its own
(e.g. straight into compute_phi) without needing the rest of anatomy.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def compute_surface_label(anatomy, *, save_path=None, key="surface_label"):
    """
    Pull a surface label volume out of an anatomy dict (from
    compute_geometry) and optionally save it on its own.

    Parameters
    ----------
    anatomy   : dict -- from compute_geometry / load_geometry
    save_path : if given, the label volume is written here as .npy
    key       : "surface_label" (default) or "surface_label_rv" for the
                RV-specific version used by phi_rv

    Returns
    -------
    surface_label : (Nx,Ny,Nz) int8 -- +1 epi / -1 endo / 0 interior
    """
    surface_label = anatomy[key]
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, surface_label)
    return surface_label


def load_surface_label(path):
    """Load a previously saved surface_label volume. No recomputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved surface_label at {path}")
    return np.load(path)
