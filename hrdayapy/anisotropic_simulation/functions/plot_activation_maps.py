"""
plot_activation_maps.py
========================
PyVista visualisation of static activation (depolarisation), deactivation
(repolarisation), and APD (action-potential duration = deactivation -
activation) maps produced by activation_maps.py.

Side-by-side by design: activation, deactivation, and APD, sharing a
camera (rotate one, the others follow) so timing patterns are easy to
compare directly.

Colour limits default to a robust (percentile-based) range rather than the
raw [min, max] of each field. A handful of outlier nodes (e.g. a node that
barely repolarises by the end of the recording, or numerical chatter at a
mesh boundary) can otherwise stretch the colour scale so far that the rest
of the anatomy collapses into one or two colours. Pass `robust_clim=False`
(or an explicit `*_clim=(lo, hi)`) to opt back into plain min/max.

Mask options mirror the convention already used elsewhere in this package
(plot_stimulus_region.py, cut_plotting.py):
    mask_mode="full"  -> surface of the full myocardium mask S
                          (outer epicardial/endocardial shell only)
    mask_mode="cut"   -> surface of the Slicer-exported cut mask Z
                          (also exposes the interior cut face, so you can
                          see transmural timing, not just the outer shell)

Any myocardial voxel that IS part of the chosen mask but has no event at
the requested beat (never activated / never repolarised, or fewer beats
than requested) is rendered gray rather than dropped, so the anatomy stays
readable -- pass a different `nan_color` to change that.

Usage (standalone)
-------------------
    python plot_activation_maps.py --maps out/P001_activation_maps.npz \
        --full-mask out/P001_S.npy --cut-mask out/P001_cut_mask.npy \
        --mode cut --event 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes


# =============================================================================
# Surface construction (mirrors plot_stimulus_region._mask_to_mesh /
# visualise_purkinje._mask_to_mesh -- proven to work in this environment)
# =============================================================================

def _mask_field_to_surface(
    mask:  np.ndarray,
    field: np.ndarray,
    voxel_size:  float = 0.4,
    smooth_iter: int   = 30,
    pass_band:   float = 0.1,
    mask_sigma:  float = 0.8,
    nan_exclude_radius_vox: float = 1.5,
) -> pv.PolyData | None:
    """
    Build the boundary surface of `mask` (voxel-index space), carrying
    `field` along as a per-vertex scalar. Cells inside `mask` with a NaN
    field value are kept in the mesh (so they render, just uncoloured) --
    only voxels outside `mask` are dropped.

    Unlike the previous cell-threshold approach (pv.ImageData -> threshold
    -> extract_surface), this extracts a marching-cubes isosurface from a
    lightly Gaussian-blurred mask, matching the style used elsewhere in the
    pipeline. A cell-thresholded voxel-cube surface stays stair-stepped
    even after Taubin smoothing, since every face starts out axis-aligned;
    marching cubes places vertices at sub-voxel, interpolated positions,
    so the same Taubin smoothing pass actually rounds it off.
    """
    mask  = mask.astype(bool)
    field = field.astype(np.float32)

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

    # Sample the field at each new surface vertex via trilinear interpolation
    # (see visualise_coordinates.py's _field_to_mesh for why nearest-neighbour
    # produces spurious contour rings). Naively interpolating `field` directly
    # would let real NaNs bleed into every vertex sampled within one voxel of
    # them, wiping out far more of the surface than intended -- instead we
    # inpaint NaNs with their nearest valid value first (so there's nothing
    # left to bleed), and interpolate smoothly on that filled array.
    #
    # A vertex is re-masked back to NaN only if it's more than
    # `nan_exclude_radius_vox` voxels from the *nearest* recorded value --
    # not simply "my single nearest voxel happens to be unset". The latter
    # is too strict whenever `field` isn't recorded on literally every voxel
    # of `mask` (e.g. values scattered from a coarser simulation mesh onto
    # this finer image grid): most vertices' nearest voxel would land on an
    # empty in-between cell even while sitting right next to real data,
    # wiping out far more of the surface to gray than the data actually
    # warrants.
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


def _field_grid(mask_shape: tuple[int, int, int], vox_idx: np.ndarray,
                 values: np.ndarray) -> np.ndarray:
    """Scatter a (N_myo,) per-node array onto a (Nx,Ny,Nz) volume (NaN bg)."""
    grid = np.full(mask_shape, np.nan, dtype=np.float32)
    grid[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = values
    return grid


def _auto_clim(
    mask: np.ndarray,
    field: np.ndarray,
    robust: bool = True,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    """
    Colour limits for `field` over `mask`.

    robust=True (default): (lo, hi) are the given percentiles of the
        in-mask, non-NaN values. Values beyond that range still render --
        they're just clamped to the end-of-colormap colour -- so a handful
        of outlier nodes no longer wash out the scale for everyone else.
    robust=False: plain [min, max] of the in-mask values (old behaviour).
    """
    vals = field[mask.astype(bool)]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    if robust:
        lo, hi = np.percentile(vals, percentiles)
        lo, hi = float(lo), float(hi)
    else:
        lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        hi = lo + 1.0
    return (lo, hi)


# =============================================================================
# Main plotting function
# =============================================================================

def plot_activation_maps(
    maps,
    S: np.ndarray | None = None,
    Z: np.ndarray | None = None,
    *,
    mask_mode: str = "full",
    event: int = 1,
    voxel_size: float = 0.4,
    cmap: str = "turbo",
    apd_cmap: str = "viridis",
    activation_clim:   tuple[float, float] | None = None,
    deactivation_clim: tuple[float, float] | None = None,
    apd_clim:          tuple[float, float] | None = None,
    robust_clim: bool = True,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    show_apd: bool = True,
    nan_color: str = "lightgray",
    background_color: str = "white",
    smooth_surface_iter: int = 30,
    smooth_pass_band: float = 0.1,
    nan_exclude_radius_vox: float = 1.5,
    window_size: tuple[int, int] | None = None,
    show_scalar_bar: bool = True,
    link_cameras: bool = True,
    screenshot_path: str | Path | None = None,
    off_screen: bool = False,
    notebook: bool = False,
    interactive: bool = True,
) -> pv.Plotter:
    """
    Render activation (depolarisation), deactivation (repolarisation), and
    APD (action-potential duration) maps side by side for a single
    beat/event.

    Parameters
    ----------
    maps : ActivationMaps
        As returned by compute_activation_maps() / load_activation_maps().
    S, Z : (Nx,Ny,Nz) ndarray
        Full myocardium mask / Slicer-exported cut mask. Only the one
        selected by `mask_mode` needs to be provided. Shape must match the
        grid the simulation was run on (checked against maps.grid_shape
        if available).
    mask_mode : "full" | "cut"
        "full" -> outer shell of S only. "cut" -> shell of Z, which also
        exposes the interior cut face so transmural timing is visible.
    event : int
        1-based beat index -- 1 is the first beat/activation detected at
        each node, 2 the second, etc. Nodes with fewer beats than this
        render gray (see `nan_color`).
    activation_clim, deactivation_clim, apd_clim : (min, max), optional
        Colour limits in ms, one per panel. Default (None) to a robust
        percentile-based range over the chosen mask -- see `robust_clim`.
        Passing an explicit tuple always overrides `robust_clim` for that
        panel.
    robust_clim : if True (default) and a `*_clim` isn't given explicitly,
        colour limits are set from `clim_percentiles` of the in-mask
        values rather than the raw min/max, so a few outlier nodes don't
        wash out the colour scale for the rest of the anatomy. Set False
        to fall back to plain min/max.
    clim_percentiles : (low, high) percentiles used when robust_clim=True.
        Default (1, 99) clips the most extreme ~1% at each end.
    apd_cmap : colormap for the APD panel (kept distinct from `cmap`,
        which is shared by the activation/deactivation panels, since APD
        is a duration rather than a timestamp and a different palette
        makes that visually obvious).
    show_apd : if True (default), add a third panel with the APD map
        (deactivation - activation, i.e. maps.apd_map(event)). Set False
        to reproduce the original 2-panel activation/deactivation layout.
    nan_color : colour for mask voxels with no event at this beat.
    nan_exclude_radius_vox : a surface vertex renders gray only if it's
        more than this many voxels from the nearest recorded activation
        value -- raise it if genuinely-activated regions still look
        patchy/gray, lower it if gray regions look too filled-in.
    link_cameras : keep the two subplots' cameras synchronised.
    screenshot_path : if given, a screenshot is saved there.
    off_screen, notebook, interactive : passed through to pv.Plotter/.show().
    """
    if mask_mode not in ("full", "cut"):
        raise ValueError(f"mask_mode must be 'full' or 'cut', got {mask_mode!r}")

    mask = S if mask_mode == "full" else Z
    if mask is None:
        needed = "S" if mask_mode == "full" else "Z"
        raise ValueError(
            f"mask_mode={mask_mode!r} requires the {needed} mask to be "
            f"passed in, but it was None.")
    mask = np.asarray(mask)

    if maps.grid_shape is not None and tuple(int(x) for x in maps.grid_shape) != mask.shape:
        print(f"  Warning: mask.shape {mask.shape} != maps.grid_shape "
              f"{tuple(int(x) for x in maps.grid_shape)} -- results may be "
              f"mis-aligned if these come from different runs.")

    act_vals   = maps.activation_map(event)
    deact_vals = maps.deactivation_map(event)
    apd_vals   = deact_vals - act_vals  # NaN wherever either input is NaN

    n_act   = int(np.sum(~np.isnan(act_vals)))
    n_deact = int(np.sum(~np.isnan(deact_vals)))
    n_apd   = int(np.sum(~np.isnan(apd_vals)))
    print(f"  Beat {event}: {n_act:,}/{maps.N_myo:,} nodes activated, "
          f"{n_deact:,}/{maps.N_myo:,} nodes deactivated, "
          f"{n_apd:,}/{maps.N_myo:,} nodes with a full APD "
          f"(mask_mode={mask_mode!r}).")
    if n_act == 0:
        print(f"  Warning: no activation events found for beat {event} -- "
              f"maps.n_beats_activated = {maps.n_beats_activated}. "
              f"Try a smaller `event`, or check act_threshold.")
    if show_apd and n_apd < n_act:
        print(f"  Note: {n_act - n_apd:,} activated node(s) never "
              f"repolarised by this beat's window and render gray on the "
              f"APD panel (they still have an activation time).")
    n_apd_neg = int(np.sum(apd_vals[~np.isnan(apd_vals)] < 0))
    if n_apd_neg:
        print(f"  Warning: {n_apd_neg:,} node(s) have a negative APD -- "
              f"likely a second beat's activation paired against the "
              f"prior beat's deactivation. Check act/deact event indices "
              f"line up for this `event`.")

    act_grid   = _field_grid(mask.shape, maps.vox_idx, act_vals)
    deact_grid = _field_grid(mask.shape, maps.vox_idx, deact_vals)
    apd_grid   = _field_grid(mask.shape, maps.vox_idx, apd_vals)

    if activation_clim is None:
        activation_clim = _auto_clim(mask, act_grid, robust_clim, clim_percentiles)
    if deactivation_clim is None:
        deactivation_clim = _auto_clim(mask, deact_grid, robust_clim, clim_percentiles)
    if apd_clim is None:
        apd_clim = _auto_clim(mask, apd_grid, robust_clim, clim_percentiles)

    mesh_act = _mask_field_to_surface(
        mask, act_grid, voxel_size, smooth_surface_iter, smooth_pass_band,
        nan_exclude_radius_vox=nan_exclude_radius_vox)
    mesh_deact = _mask_field_to_surface(
        mask, deact_grid, voxel_size, smooth_surface_iter, smooth_pass_band,
        nan_exclude_radius_vox=nan_exclude_radius_vox)

    if mesh_act is None or mesh_deact is None:
        raise RuntimeError(
            "Mask produced an empty surface -- check that S/Z is non-empty "
            "and matches the simulation grid.")

    mesh_apd = None
    if show_apd:
        mesh_apd = _mask_field_to_surface(
            mask, apd_grid, voxel_size, smooth_surface_iter, smooth_pass_band,
            nan_exclude_radius_vox=nan_exclude_radius_vox)
        if mesh_apd is None:
            print("  Warning: APD surface came out empty (no node has both "
                  "an activation and a deactivation time) -- skipping the "
                  "APD panel.")
            show_apd = False

    n_cols = 3 if show_apd else 2
    if window_size is None:
        window_size = (2400, 800) if show_apd else (1600, 800)

    pl = pv.Plotter(shape=(1, n_cols), window_size=window_size,
                     off_screen=off_screen, notebook=notebook)
    pl.set_background(background_color)

    bar_common = dict(title_font_size=14, label_font_size=12, n_labels=5,
                       color="black" if background_color != "black" else "white")
    text_color = "black" if background_color != "black" else "white"

    pl.subplot(0, 0)
    pl.add_text(f"Activation (depolarisation) — beat {event}",
                font_size=12, color=text_color)
    pl.add_mesh(mesh_act, scalars="field", cmap=cmap, clim=activation_clim,
                nan_color=nan_color, smooth_shading=True,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={**bar_common, "title": "Activation (ms)"})
    pl.add_axes()

    pl.subplot(0, 1)
    pl.add_text(f"Deactivation (repolarisation) — beat {event}",
                font_size=12, color=text_color)
    pl.add_mesh(mesh_deact, scalars="field", cmap=cmap, clim=deactivation_clim,
                nan_color=nan_color, smooth_shading=True,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={**bar_common, "title": "Deactivation (ms)"})
    pl.add_axes()

    if show_apd:
        pl.subplot(0, 2)
        pl.add_text(f"APD (deactivation − activation) — beat {event}",
                    font_size=12, color=text_color)
        pl.add_mesh(mesh_apd, scalars="field", cmap=apd_cmap, clim=apd_clim,
                    nan_color=nan_color, smooth_shading=True,
                    show_scalar_bar=show_scalar_bar,
                    scalar_bar_args={**bar_common, "title": "APD (ms)"})
        pl.add_axes()

    if link_cameras:
        pl.link_views()
    pl.subplot(0, 0)
    pl.camera_position = "iso"

    if screenshot_path:
        pl.show(screenshot=str(screenshot_path), auto_close=not interactive)
        print(f"  Screenshot saved -> {screenshot_path}")
    elif interactive:
        pl.show()

    return pl


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot side-by-side activation/deactivation/APD maps")
    ap.add_argument("--maps", required=True,
                    help="Path to activation_maps.npz")
    ap.add_argument("--full-mask", default=None,
                    help="Path to the full myocardium mask (.npy), for mode=full")
    ap.add_argument("--cut-mask", default=None,
                    help="Path to the Slicer-exported cut mask (.npy), for mode=cut")
    ap.add_argument("--mode", choices=["full", "cut"], default="full")
    ap.add_argument("--event", type=int, default=1, help="1-based beat index")
    ap.add_argument("--voxel-size", type=float, default=0.4)
    ap.add_argument("--cmap", default="turbo",
                    help="Colormap for the activation/deactivation panels")
    ap.add_argument("--apd-cmap", default="viridis",
                    help="Colormap for the APD panel")
    ap.add_argument("--no-apd", action="store_true",
                    help="Skip the APD panel (2-panel activation/deactivation layout)")
    ap.add_argument("--no-robust-clim", action="store_true",
                    help="Use plain min/max colour limits instead of the default "
                         "outlier-robust percentile range")
    ap.add_argument("--clim-percentiles", type=float, nargs=2, default=(1.0, 99.0),
                    metavar=("LOW", "HIGH"),
                    help="Percentile range for robust colour limits (default: 1 99)")
    ap.add_argument("--out", default=None, help="Save a screenshot here instead of/as well as showing")
    args = ap.parse_args()

    try:
        from .activation_maps import load_activation_maps
    except ImportError:
        from activation_maps import load_activation_maps

    maps = load_activation_maps(args.maps)

    S = np.load(args.full_mask) if args.full_mask else None
    Z = np.load(args.cut_mask) if args.cut_mask else None

    plot_activation_maps(
        maps, S=S, Z=Z,
        mask_mode=args.mode, event=args.event,
        voxel_size=args.voxel_size, cmap=args.cmap, apd_cmap=args.apd_cmap,
        show_apd=not args.no_apd,
        robust_clim=not args.no_robust_clim,
        clim_percentiles=tuple(args.clim_percentiles),
        screenshot_path=args.out,
    )
