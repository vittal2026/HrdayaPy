"""
plot_stimulus_region.py
========================
Highlights a stimulus region (e.g. from select_coordinate_region) on
the surface of a myocardium mask, as a sanity check before using it to
seed a simulation. Works with either the full mask S or the cut mask Z
-- pass whichever one you want the region checked against. The
myocardium is shown as a uniform translucent grey shell so the
stimulus region reads unambiguously against it; the selected region is
overlaid as a contrasting opaque colour.

Public API
----------
    plot_stimulus_region(mask, region, ...)
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


def _mask_to_mesh(mask: np.ndarray, smooth_iter: int = 50,
                   pass_band: float = 0.1) -> pv.PolyData:
    grid = pv.ImageData()
    grid.dimensions = np.array(mask.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data["valid"] = mask.astype(bool).flatten(order="F").astype(np.float32)

    mesh = grid.threshold(0.5, scalars="valid")
    if mesh.n_cells == 0:
        return mesh
    mesh = mesh.extract_surface().triangulate()
    if smooth_iter > 0 and mesh.n_points > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iter, pass_band=pass_band,
            boundary_smoothing=True, feature_smoothing=False,
            normalize_coordinates=True,
        )
    return mesh


def plot_stimulus_region(
    mask: np.ndarray,
    region: np.ndarray,
    myo_color: str = "lightgray",
    myo_opacity: float = 0.25,
    region_color: str = "red",
    smooth_surface_iter: int = 50,
    smooth_pass_band: float = 0.1,
):
    """
    Plot the surface of `mask` as a uniform translucent grey shell, with
    `region` highlighted in `region_color`.

    Parameters
    ----------
    mask : (Nx,Ny,Nz) ndarray, bool or {0,1}
        Myocardium mask to render. Pass the full mask S (e.g. from
        cfg.PATH_S) to check the region against the whole heart, or the
        cut mask Z (e.g. from cfg.CUT_MASK_PATH) to check it against a
        Slicer-exported subset.
    region : (Nx,Ny,Nz) ndarray, bool
        The stimulus region (e.g. from select_coordinate_region) to
        highlight. Only the part of `region` that lies on the surface
        of `mask` (i.e. region & mask) is shown; voxels selected by
        `region` but outside `mask` are not visible -- if you're using
        the cut mask Z and voxels seem to be missing, widen the cut or
        check the region against the full mask S instead.
    myo_color, myo_opacity : colour/opacity of the myocardium shell.
        Plain and translucent by design, so the highlighted region is
        never competing with an anatomical field for attention.
    region_color : colour for the highlighted region (rendered opaque).
    smooth_surface_iter, smooth_pass_band : Taubin smoothing controls.
    """
    mask = mask.astype(bool)
    region = region.astype(bool) & mask

    base_mesh = _mask_to_mesh(mask, smooth_iter=smooth_surface_iter,
                               pass_band=smooth_pass_band)
    region_mesh = _mask_to_mesh(region, smooth_iter=smooth_surface_iter,
                                 pass_band=smooth_pass_band)

    n_region_vox = int(region.sum())
    print(f"  [stimulus region] {n_region_vox:,} voxels on the surface of mask")
    if n_region_vox == 0:
        print("  [stimulus region] WARNING: region is empty within mask -- "
              "nothing will be highlighted. Widen the target tolerances, "
              "or (if you passed the cut mask Z) check the region against "
              "the full mask S instead, in case it's simply outside the "
              "current cut.")

    plotter = pv.Plotter()
    if base_mesh.n_cells > 0:
        plotter.add_mesh(base_mesh, color=myo_color, opacity=myo_opacity)
    if region_mesh.n_cells > 0:
        plotter.add_mesh(region_mesh, color=region_color, opacity=1.0)

    plotter.add_axes()
    plotter.show_bounds(grid="back")
    plotter.show()
