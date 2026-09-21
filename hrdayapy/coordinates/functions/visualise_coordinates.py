"""
visualise_coordinates_transmural_apicobasal.py
================================================
Slimmed-down PyVista visualisation showing only the two "smooth" UVC
fields -- transmural (phi) and apicobasal (psi) -- side by side in a
single 1x2 linked-camera window:

    left    phi    transmural     0 = endocardium, 1 = epicardium
    right   psi    apicobasal     0 = apex,         1 = base

chi (biventricular) and theta (rotational) are not rendered -- this is
the deliberate difference from visualise_coordinates.py, not an
oversight. No bounding box is drawn either: only the surface itself,
so nothing in the render distracts from the two colour fields.

Drop-in compatibility
----------------------
`visualise_coordinates` here has the *same call signature* as the
original 2x2, four-field version (Z, phi, psi, chi, theta, Z_theta, and
all the smoothing/geometry kwargs), so it can be substituted in for
that module directly -- rename this file to visualise_coordinates.py
(or import it under that name) and existing call sites need no changes.

This file is self-contained: it does not import from any other
visualise_coordinates module, so it's safe to rename/replace with no
dangling cross-file dependency.

chi, theta, and Z_theta are still accepted (and shape-checked if given)
so call sites that already pass all four coordinate fields don't need
to be edited -- chi/theta/Z_theta are simply ignored for rendering.

Surface construction: marching cubes on a lightly Gaussian-blurred
mask, then NaN-aware trilinear field sampling and Taubin smoothing --
ported from plot_activation_maps.py.

Saving / headless use
----------------------
Mirrors the screenshot_path / off_screen / interactive convention used
by plot_activation_maps.py, so all of the pipeline's inspection figures
behave the same way from run_simulation.py:

    hp.coordinates.visualise_coordinates(
        Z, phi=phi, psi=psi,
        screenshot_path=f"{OUT}/{PATIENT_ID}_coordinates.png",
        off_screen=True, interactive=False,
    )

- screenshot_path : if given, the rendered window is saved to this path.
- off_screen      : if True, no window is opened at all (needed on a
                     machine without a display, e.g. over SSH/CI) --
                     rendering still happens, just into an off-screen
                     buffer, so screenshot_path still works.
- interactive      : if True (default), also pops up the window after
                     any screenshot is taken, exactly as before. Set
                     False for batch/headless runs so this doesn't block
                     waiting for someone to close a window that isn't
                     there.

Public API
----------
    visualise_coordinates(Z, phi=..., psi=..., chi=..., theta=..., Z_theta=...)
"""

from __future__ import annotations

from pathlib import Path

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

    Marching-cubes isosurface of a lightly Gaussian-blurred mask, instead
    of a cell-thresholded voxel-cube surface, so Taubin smoothing actually
    rounds the surface off rather than just softening stair-steps. See
    plot_activation_maps.py, which this is ported from, for the full
    rationale.
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
        # re-mask any vertex too far from real data back to NaN (naive
        # interpolation of the raw field would let NaNs wipe out far more
        # of the surface than intended).
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
    window_size: tuple[int, int] = (1600, 800),
    Z_theta: np.ndarray | None = None,
    background_color: str = "white",
    screenshot_path: str | Path | None = None,
    off_screen: bool = False,
    interactive: bool = True,
) -> pv.Plotter:
    """
    Render phi / psi side by side in a single 1x2 PyVista window, with a
    linked camera (rotate / zoom in one subplot, both follow). Surface
    only -- no bounding box.

    Signature-compatible with visualise_coordinates.visualise_coordinates
    so it can be substituted in directly: chi, theta, and Z_theta are
    accepted and shape-checked but not rendered.

    Parameters
    ----------
    Z : (Nx,Ny,Nz) ndarray, bool or {0,1}
        The cut/subset mask to render phi/psi on -- e.g. Z from
        run_step_1b_cut_mask(cfg, ...), loaded from cfg.CUT_MASK_PATH.
        Must be the same shape as phi/psi (and chi/theta, if given).
    phi, psi : (Nx,Ny,Nz) float arrays, NaN outside the myocardium --
        the two coordinate fields rendered here. Either left as None
        renders as a plain grey surface in that subplot.
    chi, theta, Z_theta : accepted for drop-in compatibility with
        visualise_coordinates.visualise_coordinates, shape-checked if
        provided, but otherwise ignored -- not rendered in this window.
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
    background_color : plotter background colour.
    screenshot_path : if given, save the rendered figure here (any
        format pv.Plotter.screenshot/.show(screenshot=...) supports,
        e.g. .png). Works whether or not `interactive` is True. When
        interactive (a window is shown), the screenshot is captured at
        the moment the window closes, using whatever camera angle you
        left it at -- rotate/zoom to the framing you want, then close
        with the 'q' key. Closing via the OS title-bar/X button instead
        destroys the render window before PyVista can grab it, and no
        screenshot is saved.
    off_screen : if True, render into an off-screen buffer instead of
        opening a window -- use on machines without a display. Combine
        with `screenshot_path` to save a figure headlessly, and leave
        `interactive=False` in that case (there's no window to show).
    interactive : if True (default), pop up the window after any
        screenshot is captured. Set False for unattended/batch runs.
    """
    Z = Z.astype(bool)
    # chi, theta, Z_theta are accepted for drop-in signature compatibility
    # but are never rendered here, so their shapes are deliberately *not*
    # checked against Z -- a caller can pass a chi/theta pair (and
    # Z_theta) from an entirely different grid/resolution than Z/phi/psi
    # without this function raising.
    for name, field in (("phi", phi), ("psi", psi)):
        if field is not None and field.shape != Z.shape:
            raise ValueError(f"{name}.shape {field.shape} != mask.shape {Z.shape}")

    specs = [
        ("phi", phi, Z, "turbo", (0.0, 1.0), "Transmural  \u03c6  (0=endo, 1=epi)"),
        ("psi", psi, Z, "turbo", (0.0, 1.0), "Apicobasal  \u03c8  (0=apex, 1=base)"),
    ]
    positions = [(0, 0), (0, 1)]

    plotter = pv.Plotter(shape=(1, 2), window_size=window_size, off_screen=off_screen)
    plotter.set_background(background_color)

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
        # Deliberately no plotter.show_bounds() -- surface only, no box.

    plotter.link_views()
    plotter.subplot(0, 0)
    plotter.camera_position = "iso"

    if screenshot_path:
        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path), auto_close=not interactive)
        print(f"  Screenshot saved -> {screenshot_path}")
    elif interactive:
        plotter.show()

    return plotter
