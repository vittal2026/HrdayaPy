"""
visualise_coordinates.py
=========================
Unified PyVista visualisation of the four coordinate fields on the
surface of a user-defined cut mask Z, in a single 2x2 linked-camera
window:

    top-left      phi    transmural     0 = endocardium, 1 = epicardium
    top-right     psi    apicobasal     0 = apex,         1 = base
    bottom-left   chi    biventricular  0 = LV,           1 = RV
    bottom-right  theta  rotational     0 .. 2*pi, septum-anchored

`Z` is whatever subset of the myocardium you want rendered -- typically
the Slicer-exported mask loaded from cfg.CUT_MASK_PATH (see
run_step_1b_cut_mask). This module does not compute its own cut: it
renders exactly the voxels in `Z`. Earlier versions derived an
automatic SVD-based half-plane cut from S on every call, which is why
changing CUT_MASK_PATH in config.py had no visible effect -- that cut
never read the file at all.

The theta (rotational) subplot can be rendered on a different mask via
the optional `Z_theta` argument -- e.g. the full uncut myocardium mask
S, when you want to see the full 0..2*pi wrap-around rather than just
the cut-open view. phi/psi/chi always render on `Z`.

Surface construction
---------------------
Surfaces are built with the same marching-cubes-on-a-blurred-mask
approach as plot_activation_maps.py, rather than the old
pv.ImageData -> cell threshold -> extract_surface pipeline. A
cell-thresholded voxel-cube surface stays stair-stepped even after
Taubin smoothing, since every face starts out axis-aligned; marching
cubes places vertices at sub-voxel, interpolated positions, so the
same Taubin smoothing pass actually rounds it off.

Because surface geometry now comes purely from the mask (Z / Z_theta),
independent of the field's own values, field == 0 voxels are rendered
correctly without any special-casing -- this is what previously made
chi (0 = LV, the majority of the field) risky to threshold on directly.
Field values are sampled onto the mesh vertices by trilinear
interpolation of a NaN-inpainted copy of the field, and any vertex more
than `nan_exclude_radius_vox` voxels from the nearest real value is
re-masked to NaN (rendered in `nan_color`) so unrecorded regions read
as gray rather than silently taking on a neighbour's value.

Public API
----------
    visualise_coordinates(Z, phi=..., psi=..., chi=..., theta=..., Z_theta=...)
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes


def _mask_field_to_surface(
    mask: np.ndarray,
    field: np.ndarray | None,
    voxel_size: float = 1.0,
    smooth_iter: int = 50,
    pass_band: float = 0.1,
    mask_sigma: float = 0.8,
    nan_exclude_radius_vox: float = 1.5,
) -> pv.PolyData | None:
    """
    Build the boundary surface of `mask` (voxel-index space), carrying
    `field` along as a per-vertex scalar. Cells inside `mask` with a NaN
    field value are kept in the mesh (so they render, just uncoloured) --
    only voxels outside `mask` are dropped from the geometry.

    Mirrors plot_activation_maps._mask_field_to_surface: a marching-cubes
    isosurface of a lightly Gaussian-blurred mask, instead of a
    cell-thresholded voxel-cube surface, so Taubin smoothing actually
    rounds the surface off rather than just softening stair-steps.
    """
    mask = mask.astype(bool)

    Mp = np.pad(mask.astype(np.uint8), 2, mode="constant")
    Mp = gaussian_filter(Mp.astype(float), sigma=mask_sigma)
    if not np.any(Mp > 0.5):
        return None

    verts, faces, _, _ = marching_cubes(Mp, level=0.5)
    verts -= 2.0

    F = faces.shape[0]
    vtk_faces = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), vtk_faces)
    if mesh.n_points == 0:
        return None

    if field is not None:
        field = field.astype(np.float32)

        # Inpaint NaNs with their nearest valid value first, so trilinear
        # interpolation of the field has nothing to bleed off of, then
        # re-mask any vertex too far from real data back to NaN -- see
        # plot_activation_maps._mask_field_to_surface for the full
        # rationale (naive interpolation of the raw field would let NaNs
        # wipe out far more of the surface than intended).
        nan_mask = np.isnan(field)
        if nan_mask.all():
            field_filled = None
        elif nan_mask.any():
            dist_to_valid, nearest_idx = distance_transform_edt(
                nan_mask, return_indices=True)
            field_filled = field[tuple(nearest_idx)]
        else:
            field_filled = field
            dist_to_valid = np.zeros(field.shape, dtype=np.float32)

        if field_filled is not None:
            sampled = map_coordinates(field_filled, verts.T, order=1, mode="nearest")
            if nan_mask.any():
                sampled_dist = map_coordinates(
                    dist_to_valid.astype(np.float32), verts.T, order=1, mode="nearest")
                sampled[sampled_dist > nan_exclude_radius_vox] = np.nan
            mesh.point_data["field"] = sampled.astype(np.float32)

    if smooth_iter > 0 and mesh.n_points > 0:
        mesh = mesh.smooth_taubin(
            n_iter=smooth_iter,
            pass_band=pass_band,
            boundary_smoothing=True,
            feature_smoothing=False,
            normalize_coordinates=True,
        )

    mesh.points *= voxel_size
    return mesh


def visualise_coordinates(
    Z: np.ndarray,
    phi: np.ndarray | None = None,
    psi: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    smooth_surface_iter: int = 50,
    smooth_pass_band: float = 0.1,
    mask_sigma: float = 0.8,
    nan_exclude_radius_vox: float = 1.5,
    voxel_size: float = 1.0,
    nan_color: str = "gray",
    window_size: tuple[int, int] = (1600, 1200),
    Z_theta: np.ndarray | None = None,
    background_color: str = "white",
    screenshot_path: str | Path | None = None,
    off_screen: bool = False,
    interactive: bool = True,
):
    """
    Render phi / psi / chi / theta side by side in a single 2x2 PyVista
    window, with a linked camera (rotate / zoom in one subplot, all four
    follow).

    Parameters
    ----------
    Z : (Nx,Ny,Nz) ndarray, bool or {0,1}
        The cut/subset mask to render phi/psi/chi on -- e.g. Z from
        run_step_1b_cut_mask(cfg, ...), loaded from cfg.CUT_MASK_PATH.
        Must be the same shape as phi/psi/chi/theta.
    phi, psi, chi, theta : (Nx,Ny,Nz) float arrays, NaN outside the
        myocardium -- the four coordinate fields. Any left as None
        renders as a plain grey surface in that subplot.
    Z_theta : (Nx,Ny,Nz) ndarray, bool or {0,1}, optional
        Mask to render the theta (rotational) subplot on, instead of Z --
        e.g. the full uncut myocardium mask S. Defaults to Z when omitted,
        so existing calls are unaffected. Must match theta's shape.
    smooth_surface_iter, smooth_pass_band : Taubin smoothing controls.
    mask_sigma : Gaussian blur sigma (voxels) applied to the mask before
        marching cubes -- higher gives a rounder but less exact surface.
    nan_exclude_radius_vox : a surface vertex renders in `nan_color` only
        if it's more than this many voxels from the nearest recorded
        field value -- raise it if genuinely-valid regions look patchy,
        lower it if gray regions look too filled-in.
    voxel_size : physical size of one voxel, applied uniformly to scale
        the rendered mesh (index-space coordinates by default).
    nan_color : colour for mesh regions with no field value nearby.
    window_size : PyVista window size in pixels.
    """
    Z = Z.astype(bool)
    Z_theta = Z if Z_theta is None else Z_theta.astype(bool)

    # A shape-mismatched field no longer aborts the whole call -- it's
    # dropped back to None (so its subplot renders as a plain/"not
    # provided" surface instead of crashing) and a warning is printed.
    # This matters most for theta, which can legitimately be computed on
    # a different-shaped grid than Z when Z_theta isn't also supplied
    # (or was itself mistakenly left at a mismatched default).
    fields = {"phi": (phi, Z), "psi": (psi, Z), "chi": (chi, Z), "theta": (theta, Z_theta)}
    for name, (field, mask) in fields.items():
        if field is not None and field.shape != mask.shape:
            print(
                f"  [visualise_coordinates] WARNING: {name}.shape {field.shape} "
                f"!= mask.shape {mask.shape} -- ignoring {name} for this render "
                f"(its subplot will be blank)."
            )
            fields[name] = (None, mask)
    phi, _ = fields["phi"]
    psi, _ = fields["psi"]
    chi, _ = fields["chi"]
    theta, _ = fields["theta"]

    specs = [
        ("phi",   phi,   Z,       "turbo",    (0.0, 1.0),        "Transmural  \u03c6  (0=endo, 1=epi)"),
        ("psi",   psi,   Z,       "turbo",    (0.0, 1.0),        "Apicobasal  \u03c8  (0=apex, 1=base)"),
        ("chi",   chi,   Z,       "coolwarm", (0.0, 1.0),        "Biventricular  \u03c7  (0=LV, 1=RV)"),
        ("theta", theta, Z_theta, "gray",     (0.0, 2 * np.pi),  "Rotational  \u03b8  (0..2\u03c0, septum-anchored)"),
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    plotter = pv.Plotter(shape=(2, 2), window_size=window_size)

    for (row, col), (name, field, mask, cmap, clim, title) in zip(positions, specs):
        plotter.subplot(row, col)
        mesh = _mask_field_to_surface(
            mask, field, voxel_size=voxel_size,
            smooth_iter=smooth_surface_iter, pass_band=smooth_pass_band,
            mask_sigma=mask_sigma, nan_exclude_radius_vox=nan_exclude_radius_vox,
        )
        if mesh is None or mesh.n_points == 0:
            plotter.add_text(f"{title}\n(mask is empty here)", font_size=10)
            continue
        if field is None or "field" not in mesh.point_data:
            plotter.add_mesh(mesh, color="lightgray")
            plotter.add_text(f"{title}\n(not provided)", font_size=10)
        else:
            plotter.add_mesh(mesh, scalars="field", cmap=cmap, clim=clim,
                              nan_color=nan_color, smooth_shading=True,
                              scalar_bar_args={"title": name})
            plotter.add_text(title, font_size=10)
        plotter.add_axes()
        plotter.show_bounds(grid="back")

    plotter.link_views()
    plotter.show()
