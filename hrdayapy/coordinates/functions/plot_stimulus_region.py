]"""
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

from .mesh_viz_utils import mask_to_pv_surface


def plot_stimulus_region(
    mask: np.ndarray,
    region: np.ndarray,
    myo_color: str = "lightgray",
    myo_opacity: float = 0.25,
    region_color: str = "red",
    smooth_surface_iter: int = 50,
    smooth_pass_band: float = 0.1,
    voxel_size: float = 0.4,
    target_mm: float = 1.0,
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
    voxel_size, target_mm : passed to mesh_viz_utils.mask_to_pv_surface --
        target_mm (mm per marching-cubes step, default 1.0) keeps this
        plot's surface reconstruction cost roughly constant regardless of
        how fine voxel_size is, instead of it getting denser -- and
        slower to build and smooth -- purely because mask was built at a
        finer resolution. See mask_to_pv_surface's docstring for why this
        replaced the old per-voxel-cell ImageData()+threshold() approach.
    """
    mask = mask.astype(bool)
    region = region.astype(bool) & mask

    base_mesh = mask_to_pv_surface(mask, voxel_size=voxel_size, target_mm=target_mm,
                                    smooth_iter=smooth_surface_iter,
                                    pass_band=smooth_pass_band)
    region_mesh = mask_to_pv_surface(region, voxel_size=voxel_size, target_mm=target_mm,
                                      smooth_iter=smooth_surface_iter,
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
