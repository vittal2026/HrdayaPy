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
from .functions._multires import coarsen_mask, upsample_field


def compute_phi(
    S,
    surface_label,
    *,
    device="cuda",
    tol: float = 1e-6,
    maxiter: int = 10_000,
    save_path=None,
    coarsen_factor: int = 1,
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
    coarsen_factor: int, default 1 (no coarsening). If > 1, solves the
                    Laplace equation on S/surface_label downsampled by
                    this integer factor per axis, then trilinearly
                    upsamples back to S's full resolution (exact epi=1.0 /
                    endo=0.0 Dirichlet values re-applied afterward at the
                    true fine-resolution surface_label). The expensive
                    part at fine resolution is assembling and solving the
                    sparse Laplacian (memory scales with the number of
                    myocardial voxels), not S's resolution per se -- phi
                    is smooth/low-frequency, so a coarse solve captures
                    its large-scale shape well. Use this when S's own
                    resolution is finer than the solve needs to be
                    accurate at. See functions/_multires.py for the
                    tradeoff this makes. Use 1 whenever a full-resolution
                    solve is actually affordable.

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

    if coarsen_factor > 1:
        S_bool = S.astype(bool)
        coarse_shape = tuple(-(-s // coarsen_factor) for s in S_bool.shape)
        print(f"  [transmural] coarsen_factor={coarsen_factor}: solving on "
              f"a ~{coarse_shape} grid, then upsampling to {S_bool.shape}")

        S_c    = coarsen_mask(S_bool, coarsen_factor)
        epi_c  = coarsen_mask(surface_label == 1, coarsen_factor) & S_c
        endo_c = coarsen_mask(surface_label == -1, coarsen_factor) & S_c & ~epi_c
        surface_label_c = np.zeros(S_c.shape, dtype=np.int8)
        surface_label_c[epi_c]  = 1
        surface_label_c[endo_c] = -1

        phi_c = generate_transmural_coordinate(
            S=S_c, surface_label=surface_label_c, device=device,
            tol=tol, maxiter=maxiter, save_path=None,
        )
        phi = upsample_field(
            phi_c, S_bool.shape, fine_mask=S_bool,
            dirichlet_fine=[
                (surface_label == -1, 0.0),
                (surface_label == 1, 1.0),
            ],
        )
        if tmp_path is not None:
            np.save(tmp_path, phi)
            tmp_path.replace(save_path)
        return phi

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
