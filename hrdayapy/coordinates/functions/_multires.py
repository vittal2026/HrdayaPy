"""
coordinates/functions/_multires.py
====================================
Shared coarse-solve + upsample helper for the apicobasal/transmural Laplace
solves (psi, phi).

Both fields solve an elliptic (Laplace) PDE, whose solutions are smooth and
low-frequency by construction -- the field varies gently across many
voxels, with no fine local structure to resolve. That means a coarse-grid
solve captures the same large-scale shape as a fine-grid solve; what's lost
is only sub-voxel accuracy in exactly where an isosurface of psi/phi falls,
not the field's overall shape. When downstream stages only need *a* field
of the correct output shape and macroscopic pattern -- not the fine-grid
solve's full accuracy -- solving small and upsampling is a large, safe
memory/time win over solving at full resolution.

This is a single-level approximation of geometric multigrid: solve once on
a coarsened grid, then interpolate straight back up. A full multigrid
V-cycle would follow that with fine-grid smoothing sweeps to recover
fine-scale accuracy -- deliberately skipped here, since that fine-scale
accuracy is exactly what this helper is trading away for speed/memory.

NOT a substitute for a real fine-resolution solve wherever psi/phi's exact
values matter (e.g. if used to place quantitatively precise fibre angles
or thin-transmural-band cutoffs) -- verify against a real solve at a
resolution you trust before relying on this for anything accuracy-critical.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def coarsen_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a boolean mask by an integer `factor` in every dimension,
    via max-pooling (True if ANY voxel in the block is True) rather than
    point-sampling.

    Point-sampling (e.g. mask[::factor, ::factor, ::factor], or
    scipy.ndimage.zoom(order=0)) can silently drop entire thin structures
    if none of their voxels happen to land exactly on a sample point.
    Max-pooling can't lose structure that way -- it only ever grows the
    mask slightly (by up to `factor` voxels at each true boundary). That
    dilation doesn't leak into the final answer: the coarse mask is only
    used to seed the smooth interior solve, and the caller re-masks the
    final upsampled field against the TRUE fine-resolution mask afterward.
    """
    mask = np.asarray(mask, dtype=bool)
    pad = tuple((-s) % factor for s in mask.shape)
    if any(pad):
        mask = np.pad(mask, [(0, p) for p in pad], mode="constant", constant_values=False)
    new_shape = tuple(s // factor for s in mask.shape)
    reshaped = mask.reshape(
        new_shape[0], factor, new_shape[1], factor, new_shape[2], factor
    )
    return reshaped.any(axis=(1, 3, 5))


def _fit_shape(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """scipy.ndimage.zoom's output shape can be off by a voxel or two from
    rounding when target_shape isn't an exact multiple of the coarse
    shape -- pad (edge-replicate) or crop so the result matches exactly."""
    out = arr
    pad_width = [(0, max(0, t - s)) for s, t in zip(out.shape, target_shape)]
    if any(p[1] for p in pad_width):
        out = np.pad(out, pad_width, mode="edge")
    return out[tuple(slice(0, t) for t in target_shape)]


def upsample_field(
    field_coarse: np.ndarray,
    fine_shape: tuple[int, int, int],
    fine_mask: np.ndarray,
    dirichlet_fine: list[tuple[np.ndarray, float]],
) -> np.ndarray:
    """
    Trilinearly upsample a coarse-grid scalar field to fine_shape, then
    re-impose exact boundary conditions and masking against the TRUE
    fine-resolution geometry.

    Parameters
    ----------
    field_coarse   : coarse-grid solution, NaN outside the coarse mask
    fine_shape     : target (Nx,Ny,Nz) -- the ACTUAL fine S's shape
    fine_mask      : (Nx,Ny,Nz) bool -- the TRUE fine-resolution myocardium
                     mask. The returned field is NaN everywhere outside it,
                     regardless of what the (deliberately more permissive)
                     coarse mask covered.
    dirichlet_fine : list of (fine_region_mask, value) pairs, e.g.
                     [(apex_mask & fine_mask, 1e-6),
                      (basal_mask & fine_mask, 1.0)]
                     -- applied AFTER interpolation, so boundary conditions
                     stay numerically exact even though the interior is
                     only an interpolated approximation of a coarse solve.

    Returns
    -------
    field : (Nx,Ny,Nz) float32, exactly fine_shape.
    """
    filled = np.nan_to_num(field_coarse, nan=0.0).astype(np.float32)
    zoom_factors = [f / c for f, c in zip(fine_shape, filled.shape)]
    field = zoom(filled, zoom_factors, order=1)  # trilinear
    field = _fit_shape(field, fine_shape).astype(np.float32)

    field[~fine_mask] = np.nan
    for region_mask, value in dirichlet_fine:
        field[region_mask] = value
    return field
