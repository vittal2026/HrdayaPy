"""
visualise_fibre_directions.py
================================
3D PyVista check for a fibre direction field: dense short line glyphs on
the epicardial surface, coloured over a solid myocardium surface.

Why this reuses purkinje's create_surface_mesh instead of writing fresh
marching-cubes / pv.ImageData code
--------------------------------------------------------------------------
purkinje/functions/visualise_purkinje.py's _mask_to_mesh (exposed there as
create_surface_mesh) transposes the mask with `np.transpose(mask, (1, 0, 2))`
before building the pv.ImageData, because PyVista's ImageData maps its own
X/Y/Z directly onto array axes 0/1/2 with no swap, while other conventions
in this pipeline (Purkinje node coordinates) use axis order (1, 0, 2). Its
docstring documents this as the fix for a real bug (tree nodes once
rendering off the surface because a separately-written extraction routine
disagreed with that convention). This is a deliberate, one-off cross-stage
import to reuse that tested extraction rather than risk a second,
independently-written routine silently disagreeing with it again -- the
only new logic this module owns is correctly UN-swapping that same
transform when sampling the volumetric fibre field f[i,j,k,:] onto the
resulting surface mesh points (done explicitly below, not assumed).

Public API
----------
    visualise_fibres(S, anatomy, f, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from hrdayapy.purkinje.functions.visualise_purkinje import create_surface_mesh


def visualise_fibres(
    S: np.ndarray,
    anatomy: dict,
    f: np.ndarray,
    *,
    phi: np.ndarray | None = None,
    phi_range: tuple[float, float] | None = None,
    glyph_stride: int = 2,
    glyph_length_mm: float = 2.5,
    surface_color: str = "tan",
    glyph_color: str = "blue",
    highlight_fraction: float = 0.0,
    highlight_color: str = "white",
    random_seed: int | None = None,
    background_color: str = "white",
    window_size: tuple[int, int] = (900, 900),
    screenshot_path: str | Path | None = None,
    interactive: bool = False,
):
    """
    Render a fibre direction field as line glyphs on the epicardial surface.

    Parameters
    ----------
    S                : (Nx,Ny,Nz) bool -- myocardium mask
    anatomy          : dict with "spacing_mm" (as returned by
                       coordinates.compute_geometry / load_geometry)
    f                : (Nx,Ny,Nz,3) float -- fibre direction cosines, as
                       returned by generate_fibre_direction / compute_fibres
    phi              : (Nx,Ny,Nz) float, optional -- transmural coordinate
                       (0 = endocardium, 1 = epicardium), as returned by
                       coordinates.compute_phi / load_phi. Required if
                       phi_range is given.
    phi_range        : (lo, hi), optional -- restrict glyph anchors to
                       surface points whose sampled phi falls in
                       [lo, hi], e.g. (0.95, 1.0) for a thin
                       near-epicardial band. The surface mesh itself
                       (extracted from S) is unaffected -- only which
                       points on it get a glyph. Ignored if phi is None.
    glyph_stride     : keep 1-in-N *filtered* surface points as glyph
                       anchors (density) -- applied after phi_range, so
                       stride density is relative to the restricted
                       region, not the whole surface
    glyph_length_mm  : visual length of each fibre line glyph
    surface_color    : PyVista colour name for the solid surface mesh
    glyph_color      : colour for the (1 - highlight_fraction) majority of
                       glyphs
    highlight_fraction : fraction (0-1) of the (post-stride) glyph anchors
                       to recolour to highlight_color instead of
                       glyph_color, chosen uniformly at random. 0.0 (the
                       default) recolours nothing -- every glyph is
                       glyph_color, same as before this parameter existed.
    highlight_color  : colour for the randomly-chosen highlight_fraction
                       of glyphs
    random_seed      : seed for the highlight_fraction draw, for a
                       reproducible split across repeated calls. None
                       (default) draws a fresh random split each call.
    background_color : PyVista colour name for the plot background
    window_size      : (width, height) in pixels
    screenshot_path  : if given, save a PNG here (created off-screen,
                       no window). If not given and interactive=False, no
                       output is produced -- pass one or the other.
    interactive      : if True, open a live rotatable PyVista window
                       instead of rendering off-screen.

    Returns
    -------
    None. Writes screenshot_path if given and/or opens a window if
    interactive=True.
    """
    try:
        import pyvista as pv
    except ImportError:
        sys.exit(
            "PyVista is not installed. Install it with:  pip install pyvista"
        )

    S = S.astype(bool)
    voxel_size = float(np.asarray(anatomy["spacing_mm"], dtype=float)[0])   # isotropic post-resample

    print("  [fibres] Extracting epicardial surface ...")
    surf = create_surface_mesh(S, voxel_size=voxel_size)
    if surf is None:
        sys.exit("Surface extraction failed -- see the warning printed above.")
    print(f"  [fibres] surface: {surf.n_points:,} pts, {surf.n_cells:,} faces")

    # See module docstring: create_surface_mesh transposes the mask with axes
    # (1, 0, 2) before extraction, so a surface point p_mm = (X, Y, Z)
    # corresponds to ORIGINAL array indices
    #     i = round(Y / voxel_size)   (original axis 0  <-  swapped axis 1)
    #     j = round(X / voxel_size)   (original axis 1  <-  swapped axis 0)
    #     k = round(Z / voxel_size)   (original axis 2, untouched)
    # and a fibre vector (fi, fj, fk) in the ORIGINAL array's axis order
    # must have its first two components swapped to point the right way in
    # this same (X, Y, Z) display frame: vector_display = (fj, fi, fk).
    print("  [fibres] Sampling fibre directions onto the surface ...")
    pts_vox = surf.points / voxel_size
    i_idx = np.clip(np.rint(pts_vox[:, 1]).astype(int), 0, S.shape[0] - 1)
    j_idx = np.clip(np.rint(pts_vox[:, 0]).astype(int), 0, S.shape[1] - 1)
    k_idx = np.clip(np.rint(pts_vox[:, 2]).astype(int), 0, S.shape[2] - 1)

    if not S[i_idx, j_idx, k_idx].all():
        print("  [fibres] (snapping a few off-mask samples to the nearest in-mask voxel)")
        edt_idx = distance_transform_edt(~S, return_distances=False, return_indices=True)
        raw = np.stack([i_idx, j_idx, k_idx], axis=0)
        off_mask = ~S[i_idx, j_idx, k_idx]
        raw[:, off_mask] = edt_idx[:, i_idx[off_mask], j_idx[off_mask], k_idx[off_mask]]
        i_idx, j_idx, k_idx = raw[0], raw[1], raw[2]

    f_sampled = f[i_idx, j_idx, k_idx, :]
    vec_display = f_sampled[:, [1, 0, 2]]
    surf["fibre_vec"] = vec_display

    # Optional phi-band restriction -- same (i_idx, j_idx, k_idx) sampling
    # as the fibre field above, so a surface point's phi is looked up at
    # the same voxel its fibre vector came from.
    anchor_mask = np.ones(surf.n_points, dtype=bool)
    if phi is not None and phi_range is not None:
        phi_sampled = phi[i_idx, j_idx, k_idx]
        lo, hi = phi_range
        anchor_mask = (phi_sampled >= lo) & (phi_sampled <= hi)
        print(f"  [fibres] phi in [{lo}, {hi}]: "
              f"{anchor_mask.sum():,} / {surf.n_points:,} surface points kept")
        if not anchor_mask.any():
            sys.exit(f"No surface points fall in phi_range={phi_range} -- "
                      f"nothing to glyph.")

    print("  [fibres] Building glyphs ...")
    anchor_pts = surf.points[anchor_mask][::glyph_stride]
    anchor_vec = vec_display[anchor_mask][::glyph_stride]
    n_anchors = anchor_pts.shape[0]

    # Random two-way colour split -- drawn per anchor point (i.e. per
    # glyph, post-stride), not per surface point, so glyph_stride's
    # density reduction happens first and the split fraction is exact
    # against what's actually drawn.
    if not 0.0 <= highlight_fraction <= 1.0:
        raise ValueError(f"highlight_fraction must be in [0, 1], got {highlight_fraction}")
    rng = np.random.default_rng(random_seed)
    is_highlight = rng.random(n_anchors) < highlight_fraction
    if highlight_fraction > 0.0:
        print(f"  [fibres] colour split: {is_highlight.sum():,} {highlight_color} / "
              f"{(~is_highlight).sum():,} {glyph_color}")

    def _build_glyphs(pts, vecs):
        src = pv.PolyData(pts)
        src["fibre_vec"] = vecs
        return src.glyph(
            orient="fibre_vec", scale=False, factor=glyph_length_mm,
            geom=pv.Line(pointa=(-0.5, 0, 0), pointb=(0.5, 0, 0)),
        )

    glyphs_main = _build_glyphs(anchor_pts[~is_highlight], anchor_vec[~is_highlight]) \
        if (~is_highlight).any() else None
    glyphs_highlight = _build_glyphs(anchor_pts[is_highlight], anchor_vec[is_highlight]) \
        if is_highlight.any() else None
    n_glyphs = (glyphs_main.n_cells if glyphs_main is not None else 0) \
        + (glyphs_highlight.n_cells if glyphs_highlight is not None else 0)
    print(f"  [fibres] {n_glyphs:,} line glyphs from {n_anchors:,} anchor points")

    pl = pv.Plotter(window_size=window_size, off_screen=not interactive)
    pl.set_background(background_color)
    pl.add_mesh(surf, color=surface_color, smooth_shading=True, opacity=1.0)
    if glyphs_main is not None:
        pl.add_mesh(glyphs_main, color=glyph_color, line_width=1.5, render_lines_as_tubes=False)
    if glyphs_highlight is not None:
        pl.add_mesh(glyphs_highlight, color=highlight_color, line_width=1.5, render_lines_as_tubes=False)
    pl.camera_position = "iso"

    if screenshot_path is not None:
        screenshot_path = Path(screenshot_path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        # pl.show(screenshot=...) forces a full render before capture --
        # calling pl.screenshot() directly (esp. off-screen) can grab a
        # stale/blank buffer.
        pl.show(screenshot=str(screenshot_path), auto_close=not interactive)
        print(f"  [fibres] Saved -> {screenshot_path}")
    elif interactive:
        pl.show(auto_close=False)
    else:
        print("  [fibres] Neither screenshot_path nor interactive=True given -- nothing rendered.")
