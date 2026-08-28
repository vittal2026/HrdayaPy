"""
cut_plotting.py
================
Plots a scalar field (e.g. phi) on the surface of a user-defined cut
mask Z -- typically the Slicer-exported subset loaded from
cfg.CUT_MASK_PATH (see run_step_1b_cut_mask) -- rather than computing
an automatic geometric half-plane cut from the full myocardium mask.

This used to derive its own cut on the fly (an SVD-based half-plane
through the basal slice), which is why it always "sliced in half" no
matter what CUT_MASK_PATH pointed to -- that cosmetic cut never read
from disk at all. Now the cut is whatever Z you pass in: the full
mask, a half, a wedge, a single short-axis slab -- whatever your
Slicer export defines.

Masking note
------------
Selection of which voxels to render is done on a separate boolean
scalar ("valid" = Z & ~isnan(field)), not on the field's own value.
Thresholding directly on field magnitude (as the original version did)
silently drops any voxel whose field value is exactly 0 -- harmless
for phi/psi (only a thin endo/apex surface), but it would erase an
entire chamber for a field like chi where 0 is a real, common value.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


def _field_to_mesh(Z: np.ndarray, field: np.ndarray,
                    smooth_iter: int = 50, pass_band: float = 0.1) -> pv.PolyData:
    valid = Z.astype(bool) & ~np.isnan(field)
    field_filled = np.where(valid, field, 0.0).astype(np.float32)

    grid = pv.ImageData()
    grid.dimensions = np.array(Z.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data["valid"] = valid.flatten(order="F").astype(np.float32)
    grid.cell_data["field"] = field_filled.flatten(order="F")

    mesh = grid.threshold(0.5, scalars="valid")
    if mesh.n_cells == 0:
        return mesh
    mesh = mesh.extract_surface().triangulate()

    if smooth_iter > 0 and mesh.n_points > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iter,
            pass_band=pass_band,
            boundary_smoothing=True,
            feature_smoothing=False,
            normalize_coordinates=True,
        )
    return mesh


def cut_plotting(
    Z: np.ndarray,
    field: np.ndarray,
    cmap: str = "turbo",
    clim: tuple[float, float] | None = None,
    smooth_surface_iter: int = 50,
    smooth_pass_band: float = 0.1,
):
    """
    Plot `field` on the surface of the user-defined cut mask `Z`.

    Parameters
    ----------
    Z     : (Nx,Ny,Nz) ndarray, bool or {0,1}
        The cut/subset mask to render -- e.g. Z from
        run_step_1b_cut_mask(cfg, ...), loaded from cfg.CUT_MASK_PATH.
        Must be the same shape as `field`.
    field : (Nx,Ny,Nz) ndarray, float
        The scalar field to colour the surface with (e.g. phi), NaN
        outside the myocardium.
    cmap  : pyvista/matplotlib colormap name.
    clim  : (min, max) colour limits; defaults to the field's own
        [nanmin, nanmax] over Z if not given.
    smooth_surface_iter, smooth_pass_band : Taubin smoothing controls.
    """
    if Z.shape != field.shape:
        raise ValueError(f"Z.shape {Z.shape} != field.shape {field.shape}")

    mesh = _field_to_mesh(Z, field, smooth_iter=smooth_surface_iter,
                           pass_band=smooth_pass_band)
    if mesh.n_cells == 0:
        raise RuntimeError(
            "Z contains no voxels with a defined field value -- nothing to plot. "
            "Check that Z overlaps the region where `field` is non-NaN."
        )

    if clim is None:
        vals = mesh["field"]
        clim = (float(np.nanmin(vals)), float(np.nanmax(vals)))

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="field", cmap=cmap, clim=clim)
    plotter.add_axes()
    plotter.show_bounds(grid="back")
    plotter.show()
