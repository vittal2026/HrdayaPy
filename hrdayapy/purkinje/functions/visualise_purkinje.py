"""
visualise_purkinje.py
=====================
PyVista visualisation for Purkinje networks.

Surface extraction now mirrors the proven _mask_to_mesh pattern used in
plot_stimulus_region.py (pv.ImageData + float32 cell_data + threshold +
extract_surface + triangulate + smooth_taubin), which is confirmed to work
in this environment.  The previous approaches (ImageData+uint8 or
skimage.marching_cubes) were silently failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from typing import Optional, Tuple


# ── Surface extraction ────────────────────────────────────────────────────────

def _mask_to_mesh(
    mask:        np.ndarray,
    voxel_size:  float = 0.4,
    smooth_iter: int   = 50,
    pass_band:   float = 0.1,
) -> Optional[pv.PolyData]:
    """
    Convert a binary voxel mask to a smoothed PyVista PolyData surface.

    Mirrors plot_stimulus_region._mask_to_mesh exactly (float32 cell_data,
    triangulate, smooth_taubin) — that function is confirmed working in this
    environment.  After extraction (voxel coords) the mesh is scaled by
    voxel_size so it lands in mm, matching nodes_mm = nodes * voxel_size.
    """
    try:
        # Root-convention alignment: the Purkinje tree nodes built
        # elsewhere in this pipeline (create_purkinje.py) use
        # (x, y, z) = (array axis 1, array axis 0, array axis 2) -- i.e.
        # axes 0 and 1 SWAPPED relative to plain numpy indexing --
        # because that's the order candidate voxel positions are built in
        # (np.column_stack([j, i, k]), see create_purkinje.py). pv.
        # ImageData's own cell ordering instead maps VTK-X/Y/Z directly
        # to array axis 0/1/2 with no swap. Without correcting for that
        # mismatch here, any surface built by this function sits rotated
        # (X and Y transposed) relative to the tree -- e.g. a marker
        # placed exactly at a tree node position would render off the
        # surface entirely, even though the coordinates are "correct" in
        # the tree's own frame. Transposing axes 0 and 1 up front makes
        # this function agree with that same convention.
        mask = np.transpose(mask, (1, 0, 2))

        grid = pv.ImageData()
        grid.dimensions = np.array(mask.shape) + 1
        grid.spacing    = (1, 1, 1)          # voxel coords; scale to mm below
        grid.origin     = (0, 0, 0)
        grid.cell_data["valid"] = (
            mask.astype(bool).flatten(order="F").astype(np.float32)
        )

        threshed = grid.threshold(0.5, scalars="valid")
        if threshed.n_cells == 0:
            print("  Warning: threshold produced no cells")
            return None

        mesh = threshed.extract_surface().triangulate()

        if mesh.n_points == 0:
            print("  Warning: surface has no points after triangulation")
            return None

        if smooth_iter > 0:
            mesh = mesh.smooth_taubin(
                n_iter=smooth_iter,
                pass_band=pass_band,
                boundary_smoothing=True,
                feature_smoothing=False,
                normalize_coordinates=True,
            )

        # Scale from voxel-index coords to mm
        mesh.points *= voxel_size

        return mesh

    except Exception as e:
        import traceback
        print(f"  Warning: surface extraction failed: {e}")
        traceback.print_exc()
        return None


def create_surface_mesh(
    voxel_mat:   np.ndarray,
    voxel_size:  float = 0.4,
    smooth_iter: int   = 50,
) -> Optional[pv.PolyData]:
    """Extract a smoothed surface from the full binary myocardium mask (S)."""
    return _mask_to_mesh(voxel_mat, voxel_size=voxel_size,
                         smooth_iter=smooth_iter)


def create_label_surface(
    surface_label: np.ndarray,
    label_value:   int,
    voxel_size:    float = 0.4,
    smooth_iter:   int   = 50,
    S:             Optional[np.ndarray] = None,
) -> Optional[pv.PolyData]:
    """
    Extract a single surface for one surface_label value.

    Epi (+1): use the full binary mask S if provided, because the outer
    boundary of S *is* the epicardial surface — no dilation needed and
    no double-surface artefact.

    Endo (-1): run marching cubes directly on the endo-labelled voxels
    without dilation.  The endo band is thick enough for marching cubes
    to find a clean single isosurface without any artificial thickening.

    Dilation was the cause of 4 surfaces (2 per label): dilating a thin
    shell creates a region with both inner and outer faces, and marching
    cubes faithfully extracts both.
    """
    if label_value == 1 and S is not None:
        # Outer boundary of the full myocardium mask = epicardium
        binary = S.astype(np.uint8)
    else:
        binary = (surface_label == label_value).astype(np.uint8)
    return _mask_to_mesh(binary, voxel_size=voxel_size, smooth_iter=smooth_iter)


# ── Main visualiser ───────────────────────────────────────────────────────────

def visualise_purkinje(
    nodes:            np.ndarray,
    elements:         np.ndarray,
    activation_times: np.ndarray,
    # Geometry
    voxel_mat:     Optional[np.ndarray] = None,
    surface_label: Optional[np.ndarray] = None,
    transmural:    Optional[np.ndarray] = None,   # kept for API compatibility
    voxel_size:    float = 0.4,
    resolution:    Optional[float] = None,        # legacy alias (cm)
    # Surface display
    show_surface:    bool  = True,
    epi_opacity:     float = 0.25,
    endo_opacity:    float = 0.15,
    surface_opacity: float = 0.25,
    epi_color:       str   = "lightgray",
    endo_color:      str   = "lightyellow",
    # Network display
    line_width:    float = 3.0,
    colormap:      str   = "jet",
    # Root marker
    show_root:     bool  = True,
    root_color:    str   = "blue",
    root_size:     float = 12.0,
    # Window
    window_size:      Tuple[int, int] = (1200, 900),
    background_color: str  = "white",
    camera_position:  Optional[str] = "iso",
    show_scalar_bar:  bool = False,
    screenshot_path:  Optional[str] = None,
    save_path:        Optional[str] = None,
    interactive:      bool = True,
    off_screen:       bool = False,
) -> pv.Plotter:
    """
    Visualise a Purkinje network with a translucent myocardial shell.

    Surface extraction uses the same pv.ImageData + float32 + smooth_taubin
    pattern as plot_stimulus_region._mask_to_mesh, which is confirmed to work.

    Saving / headless use
    ----------------------
    screenshot_path : if given, save the rendered figure here (e.g. .png).
        Works whether or not `interactive` is True. When interactive
        (a window is shown), the screenshot is captured at the moment the
        window closes, using whatever camera angle you left it at -- so
        rotate/zoom to the framing you want, then close with the 'q' key.
        Closing via the OS title-bar/X button instead destroys the render
        window before PyVista can grab it, and no screenshot is saved.
    off_screen : if True, render into an off-screen buffer instead of
        opening a window -- use on a machine with no display. Combine
        with screenshot_path and interactive=False for a fully headless
        save (e.g. from a batch run_simulation.py pass).
    interactive : if True (default), pop up the window after any
        screenshot is captured. Set False for unattended/batch runs, or
        automatically forced False when off_screen=True and a window
        cannot be shown anyway.
    """
    if off_screen and interactive:
        print("  Note: off_screen=True -- forcing interactive=False "
              "(no display to show a window on).")
        interactive = False
    if resolution is not None:
        voxel_size = resolution * 10.0

    print("Creating PyVista visualisation...")

    # ── Purkinje network mesh ─────────────────────────────────────────────────
    nodes_mm = nodes * voxel_size

    lines = []
    for elem in elements:
        lines.extend([2, int(elem[0]), int(elem[1])])

    pkn_mesh = pv.PolyData(nodes_mm, lines=lines)
    pkn_mesh["activation_time"] = activation_times

    if save_path:
        pkn_mesh.save(save_path)
        print(f"  Saved mesh -> {save_path}")

    # ── Plotter ───────────────────────────────────────────────────────────────
    pl = pv.Plotter(window_size=window_size, off_screen=off_screen)
    pl.set_background(background_color)

    # ── Surfaces ──────────────────────────────────────────────────────────────
    if show_surface:
        if surface_label is not None:
            print("  Extracting epicardial surface...")
            epi_surf = create_label_surface(
                surface_label, label_value=1, voxel_size=voxel_size, S=voxel_mat)
            if epi_surf is not None:
                print(f"    Epi: {epi_surf.n_points:,} pts, {epi_surf.n_cells:,} faces")
                pl.add_mesh(epi_surf, color=epi_color, opacity=epi_opacity,
                            smooth_shading=True, label="Epicardium")

            print("  Extracting endocardial surface...")
            endo_surf = create_label_surface(
                surface_label, label_value=-1, voxel_size=voxel_size)
            if endo_surf is not None:
                print(f"    Endo: {endo_surf.n_points:,} pts, {endo_surf.n_cells:,} faces")
                pl.add_mesh(endo_surf, color=endo_color, opacity=endo_opacity,
                            smooth_shading=True, label="Endocardium")

        elif voxel_mat is not None:
            print("  Extracting myocardium surface (S)...")
            surf = create_surface_mesh(voxel_mat, voxel_size=voxel_size)
            if surf is not None:
                print(f"    Surface: {surf.n_points:,} pts, {surf.n_cells:,} faces")
                pl.add_mesh(surf, color=epi_color, opacity=surface_opacity,
                            smooth_shading=True, label="Myocardium")
        else:
            print("  Warning: no voxel_mat or surface_label — surface skipped")

    # ── Purkinje network ──────────────────────────────────────────────────────
    print("  Adding Purkinje network...")
    scalar_bar_args = {
        "title":           "Activation (ms)",
        "title_font_size":  14,
        "label_font_size":  12,
        "n_labels":          5,
        "position_x":       0.85,
        "position_y":       0.05,
        "width":            0.1,
        "height":           0.6,
    }
    pl.add_mesh(
        pkn_mesh,
        scalars="activation_time",
        cmap=colormap,
        line_width=line_width,
        render_lines_as_tubes=True,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=scalar_bar_args if show_scalar_bar else None,
    )

    # ── Root marker ───────────────────────────────────────────────────────────
    if show_root:
        root_mesh = pv.PolyData(nodes_mm[0].reshape(1, 3))
        pl.add_mesh(root_mesh, color=root_color,
                    point_size=root_size,
                    render_points_as_spheres=True,
                    label="Root")

    # ── Camera / axes / legend ────────────────────────────────────────────────
    if camera_position:
        pl.camera_position = camera_position

    pl.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                line_width=2, color="black")

    if show_root or (show_surface and
                     (surface_label is not None or voxel_mat is not None)):
        pl.add_legend(size=(0.18, 0.12), loc="upper right")

    if screenshot_path:
        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
        # pl.show(screenshot=...) triggers a full render before capturing --
        # calling pl.screenshot() directly without a prior show()/render()
        # can grab a stale/blank buffer, especially off-screen.
        pl.show(screenshot=str(screenshot_path), auto_close=not interactive)
        print(f"  Screenshot saved -> {screenshot_path}")
    elif interactive:
        print("  Opening interactive window (close to continue)...")
        pl.show()

    return pl


