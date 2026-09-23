"""
mesh_viz_utils.py
==================
Shared helper for turning a boolean voxel mask into a lightweight,
resolution-independent PyVista surface mesh for sanity-check plots.

Why this exists
----------------
Several plotting functions in this package (compute_basal_plane.py,
cut_plotting.py, find_apico_basal_axis.py, load_muscle_mask.py,
plot_stimulus_region.py, purkinje/visualise_purkinje.py) each build their
own pv.ImageData() over the *entire* voxel array, set scalar data on
every one of its cells, then .threshold() down to the True ones to get a
surface. That's O(total voxel count) -- roughly cubically in
1/voxel_size -- with no downsampling lever at all, unlike marching_cubes'
step_size. mask_to_pv_surface() replaces that whole block with
marching_cubes (via mesh_labelling.extract_surface_mesh), at a step size
derived from a physical target resolution rather than a voxel count, so
refining voxel_size no longer silently makes these plots slower.

This does not eliminate the base cost of scanning the input array --
marching_cubes still visits every voxel of S once, same as any of these
functions did before. It removes the *compounding* cost: a denser
output mesh (more Taubin-smoothing work, more triangles to render) that
served no purpose past what a sanity-check plot needs.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

from .mesh_labelling import extract_surface_mesh, mm_step_size


def mask_field_to_pv_surface(
    mask:  np.ndarray,
    field: np.ndarray,
    voxel_size:  float = 0.4,
    target_mm:   float = 1.0,
    smooth_iter: int   = 30,
    pass_band:   float = 0.1,
    mask_sigma:  float = 0.8,
    nan_exclude_radius_vox: float = 1.5,
):
    """
    Build the boundary surface of `mask` (voxel-index space, scaled to mm
    at the end via voxel_size), carrying `field` along as a per-vertex
    scalar. Cells inside `mask` with a NaN field value are kept in the
    mesh (so they render, just uncoloured) -- only voxels outside `mask`
    are dropped.

    This is the shared version of a function that used to be copy-pasted
    (with drift) across plot_activation_maps.py (both the anisotropic_
    simulation and simulation copies) and visualise_coordinates.py /
    _2.py / _4.py, none of which exposed any way to control the
    marching-cubes surface's physical resolution -- so, like the other
    functions this package's mm_step_size fixes, the output mesh (and
    Taubin smoothing cost on top of it) got denser purely because `mask`
    was built at a finer voxel_size, unrelated to whether that detail
    was wanted. See mm_step_size's docstring in mesh_labelling.py.

    Extracts a marching-cubes isosurface from a lightly Gaussian-blurred
    mask rather than a raw threshold -- marching cubes places vertices at
    sub-voxel, interpolated positions, so Taubin smoothing actually
    rounds the result off instead of leaving it stair-stepped (which a
    cell-thresholded voxel-cube surface does even after smoothing, since
    every face starts out axis-aligned).

    Field values are sampled at each new surface vertex via trilinear
    interpolation, not nearest-neighbour (nearest-neighbour produces
    spurious contour rings). Naively interpolating `field` directly
    would let real NaNs bleed into every vertex within one voxel of
    them, wiping out far more of the surface than intended -- instead
    NaNs are inpainted with their nearest valid value first (so there's
    nothing left to bleed), and interpolation runs smoothly on that
    filled array. A vertex is re-masked back to NaN only if it's more
    than `nan_exclude_radius_vox` voxels from the nearest *recorded*
    value -- not simply "my single nearest voxel happens to be unset",
    which is too strict whenever field isn't recorded on literally every
    voxel of mask (e.g. values scattered from a coarser simulation mesh
    onto this finer image grid).

    Returns None if mask has no True voxels, the blurred mask never
    crosses 0.5, the extracted mesh has 0 points, or field is all-NaN --
    same "return None, caller checks" convention the callers this
    replaces already used.
    """
    mask  = mask.astype(bool)
    field = field.astype(np.float32)

    Mp = np.pad(mask.astype(np.uint8), 2, mode="constant")
    Mp = gaussian_filter(Mp.astype(float), sigma=mask_sigma)
    if not np.any(Mp > 0.5):
        return None

    step_size = mm_step_size(voxel_size, target_mm)
    verts, faces, _, _ = marching_cubes(Mp, level=0.5, step_size=step_size)
    verts -= 2.0

    F = faces.shape[0]
    vtk_faces = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), vtk_faces)
    if mesh.n_points == 0:
        return None

    nan_mask = np.isnan(field)
    if nan_mask.any() and not nan_mask.all():
        dist_to_valid, nearest_idx = distance_transform_edt(
            nan_mask, return_indices=True)
        field_filled = field[tuple(nearest_idx)]
    elif nan_mask.all():
        return None
    else:
        field_filled = field
        dist_to_valid = np.zeros(field.shape, dtype=np.float32)

    sampled = map_coordinates(field_filled, verts.T, order=1, mode="nearest")
    if nan_mask.any():
        sampled_dist = map_coordinates(
            dist_to_valid.astype(np.float32), verts.T, order=1, mode="nearest")
        sampled[sampled_dist > nan_exclude_radius_vox] = np.nan
    mesh.point_data["field"] = sampled.astype(np.float32)

    if smooth_iter > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iter,
            pass_band=pass_band,
            boundary_smoothing=True,
            feature_smoothing=False,
            normalize_coordinates=True,
        )

    mesh.points *= voxel_size
    return mesh


def mask_to_pv_surface(
    mask: np.ndarray,
    voxel_size: float = 0.4,
    target_mm: float = 1.0,
    smooth_iter: int = 30,
    pass_band: float = 0.1,
) -> pv.PolyData:
    """
    Boolean (Nx,Ny,Nz) mask -> smoothed pv.PolyData surface, in voxel-
    index coordinates (same convention as extract_surface_mesh /
    manual_landmarks._to_pyvista elsewhere in this package -- scale by
    voxel_size yourself, or via mesh.points *= voxel_size, if you need
    physical units).

    Parameters
    ----------
    mask        : (Nx,Ny,Nz) bool or {0,1} -- the volume to surface
    voxel_size  : mm per voxel (isotropic) -- match whatever produced mask
    target_mm   : marching-cubes step size in mm rather than voxels (see
                  mesh_labelling.mm_step_size). 1.0 (default) is plenty
                  fine for a sanity-check plot; lower it only if you
                  need genuinely fine visual detail.
    smooth_iter, pass_band : Taubin smoothing controls, same meaning as
                  the pv.PolyData.smooth_taubin() calls this replaces.

    Returns
    -------
    pv.PolyData, or an empty one (mesh.n_points == 0) if mask has no
    True voxels -- check that before using the result, same convention
    the old threshold()-based code's mesh.n_cells == 0 checks used.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return pv.PolyData()

    step_size = mm_step_size(voxel_size, target_mm)
    verts, faces = extract_surface_mesh(mask, step_size=step_size)
    if faces.shape[0] == 0:
        return pv.PolyData()

    n_faces = faces.shape[0]
    vtk_faces = np.hstack(
        [np.full((n_faces, 1), 3, dtype=np.int64), faces]
    ).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), vtk_faces)

    if smooth_iter > 0 and mesh.n_points > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iter, pass_band=pass_band,
            normalize_coordinates=True,
        )
    return mesh
