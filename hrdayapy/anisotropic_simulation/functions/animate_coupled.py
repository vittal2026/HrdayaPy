"""
animate_coupled.py
==================
Interactive PyVista animation of a coupled Purkinje + myocardium simulation.

Loads the self-contained NPZ written by run_coupled() and renders:
  - A triangulated myocardium surface (from the cut mask Z, marching cubes)
    coloured by transmembrane voltage.
  - The Purkinje network as a line mesh (clipped to nodes outside Z),
    coloured by the same voltage colourmap.

Both actors share a single scalar bar and colourmap.

Three ways to use this
-----------------------
1. Interactive real-time playback (original behaviour, unchanged):

    animate_coupled("coupled_sim.npz")

2. Headless "activation snapshots" -- a handful of PNG stills at chosen
   simulation times, e.g. to show the wavefront sweeping through the
   myocardium at a few representative moments. No display required, no
   real-time sleeping -- only the requested frames are ever rendered:

    animate_coupled(
        "coupled_sim.npz",
        off_screen=True,
        snapshot_times_ms=[10, 40, 70, 100, 150],
        snapshot_dir="out/P001_snapshots",
    )

   Produces one PNG per requested time, named
   "{snapshot_dir}/{snapshot_prefix}_t{time:07.2f}ms.png".

3. Headless movie export -- the full animation written to disk as an
   .mp4/.gif instead of played back live:

    animate_coupled(
        "coupled_sim.npz",
        off_screen=True,
        movie_path="out/P001_vm_animation.mp4",
        movie_fps=24,
    )

Usage (CLI)
-----------
    python animate_coupled.py path/to/coupled_sim.npz
    python animate_coupled.py path/to/coupled_sim.npz --off-screen \
        --snapshot-times 10,40,70,100,150 --snapshot-dir out/P001_snapshots
    python animate_coupled.py path/to/coupled_sim.npz --off-screen \
        --movie out/P001_vm_animation.mp4

Arguments (when called standalone)
------------------------------------
    --npz             path to coupled_sim.npz
    --vmin            colourbar minimum in mV  (default -85)
    --vmax            colourbar maximum in mV  (default  35)
    --cmap            matplotlib colourmap     (default jet)
    --speed           playback speed multiplier (default 0.1)
    --off-screen      render without opening a window (needs no display)
    --snapshot-times  comma-separated list of simulation times (ms) to
                      save as individual PNG stills
    --snapshot-dir    directory to save snapshot PNGs into (required if
                      --snapshot-times is given)
    --movie           path to save the full animation as a video/gif
                      (.mp4 or .gif) instead of playing it back live
    --movie-fps       frames per second for --movie (default 24)
"""

from __future__ import annotations

import os
import sys
import ctypes
import argparse
import time as time_module
from pathlib import Path

import numpy as np

# ── Force NVIDIA GPU before VTK imports (Windows) ────────────────────────────
try:
    ctypes.CDLL("nvapi64.dll")
except Exception:
    pass
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"

import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()


# =============================================================================
# Helpers
# =============================================================================

def _nearest_frame_indices(t_arr: np.ndarray, times_ms: list[float]) -> list[int]:
    """Map requested simulation times (ms) to the nearest saved frame index,
    de-duplicated but kept in the order the corresponding times occur."""
    idx = sorted({int(np.argmin(np.abs(t_arr - t))) for t in times_ms})
    return idx


def _frame_indices_ordered(t_arr: np.ndarray, times_ms: list[float]) -> list[int]:
    """Same mapping as `_nearest_frame_indices`, but keeps one entry per
    requested time, in the order given (not sorted/de-duplicated) -- used
    for panel grids where each requested time must land in its own,
    specific subplot slot."""
    return [int(np.argmin(np.abs(t_arr - t))) for t in times_ms]


def _load_coupled_data(npz_path: str, vmin: float) -> dict:
    """Load a coupled_sim.npz and assemble everything both `animate_coupled`
    and `plot_activation_snapshot_grid` need: the static myocardium surface
    geometry, the per-frame surface Vm, and the reconstructed per-frame
    Purkinje Vm. Shared here so the two entry points can't drift apart."""
    data = np.load(npz_path, allow_pickle=False)

    t_arr = data["time"]  # (n_frames,)
    n_frames = len(t_arr)

    comp_nodes = data["comp_nodes"]  # (N_comp, 3)  mm
    comp_edges = data["comp_edges"]  # (K', 2)       clipped to outside Z

    surf_verts = data["surf_verts"]  # (V, 3)  mm
    surf_faces = data["surf_faces"]  # (F, 3)
    vert_to_surf_node = data["vert_to_surf_node"]  # (V,)
    surf_frames = data["surf_frames"]  # (n_frames, N_surf)

    bmap_keys = sorted(
        [k for k in data.files if k.startswith("bmap_") and k != "bmap_myo"],
        key=lambda k: int(k.split("_")[1]))
    branch_keys = sorted(
        [k for k in data.files if k.startswith("branch_") and k != "branch_myo"],
        key=lambda k: int(k.split("_")[1]))
    branch_map = [data[k].tolist() for k in bmap_keys]
    N_comp = len(comp_nodes)

    V_pkn = np.full((n_frames, N_comp), float(vmin), dtype=np.float32)
    for b_idx, key in enumerate(branch_keys):
        node_ids = branch_map[b_idx]
        V_pkn[:, node_ids] = data[key]

    return dict(
        t_arr=t_arr, n_frames=n_frames,
        comp_nodes=comp_nodes, comp_edges=comp_edges,
        surf_verts=surf_verts, surf_faces=surf_faces,
        vert_to_surf_node=vert_to_surf_node, surf_frames=surf_frames,
        V_pkn=V_pkn,
    )


def _build_myo_mesh(surf_verts: np.ndarray, surf_faces: np.ndarray) -> pv.PolyData:
    """Static geometry only (no scalars) -- callers set Vm per copy/frame."""
    n_faces = len(surf_faces)
    faces_pv = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int_),
        surf_faces,
    ]).ravel()
    return pv.PolyData(surf_verts.astype(np.float64), faces_pv)


def _build_pkn_mesh(comp_nodes: np.ndarray, comp_edges: np.ndarray) -> pv.PolyData:
    lines_pv = np.hstack([
        np.full((len(comp_edges), 1), 2, dtype=np.int_),
        comp_edges,
    ]).ravel()
    mesh = pv.PolyData()
    mesh.points = comp_nodes.astype(np.float64)
    mesh.lines = lines_pv
    return mesh


# =============================================================================
# Main animation function
# =============================================================================

def animate_coupled(
    npz_path: str,
    vmin:     float = -85.0,
    vmax:     float =  35.0,
    cmap:     str   = "jet",
    speed:    float = 0.1,
    notebook: bool  = False,
    window_size: tuple[int, int] = (1200, 900),
    # ── Saving / headless controls ──────────────────────────────────────────
    off_screen:        bool = False,
    interactive:       bool = True,
    snapshot_times_ms: list[float] | None = None,
    snapshot_dir:      str | None = None,
    snapshot_prefix:   str | None = None,
    movie_path:        str | None = None,
    movie_fps:         int = 24,
) -> list[str] | None:
    """
    Animate coupled Purkinje + myocardium surface simulation.

    Parameters
    ----------
    npz_path : path to the NPZ written by run_coupled()
    vmin     : colourbar minimum (mV)
    vmax     : colourbar maximum (mV)
    cmap     : matplotlib colourmap name
    speed    : playback speed multiplier (1.0 = real cardiac time). Ignored
        in snapshot mode; ignored (frames written at `movie_fps` instead of
        wall-clock time) when `movie_path` is given.
    notebook : pass True when running inside a Jupyter notebook.
    off_screen : if True, render into an off-screen buffer instead of
        opening a window -- use on a machine with no display. Forces
        `interactive=False`. Required (or at least strongly recommended)
        for `snapshot_times_ms` / `movie_path` on a headless machine.
    interactive : if True (default) and neither `snapshot_times_ms` nor
        `movie_path` is given, plays the animation back live in a window,
        exactly as before. Ignored (no window is shown) whenever a
        snapshot batch or a movie is being written -- see those below.
    snapshot_times_ms : if given, skip real-time playback entirely and
        instead render + save exactly one PNG per requested simulation
        time (nearest available saved frame). Cheap and fast since only
        the requested frames are ever rendered. Requires `snapshot_dir`.
    snapshot_dir : output directory for the PNGs above (created if it
        doesn't exist).
    snapshot_prefix : filename prefix for snapshot PNGs (default: the
        NPZ's stem, e.g. "P001_coupled").
    movie_path : if given, write the *entire* animation to this path as a
        video (.mp4, needs imageio-ffmpeg) or .gif instead of playing it
        back interactively. Frames are written at `movie_fps`, independent
        of `speed`/wall-clock time.
    movie_fps : frames per second for `movie_path`.

    Returns
    -------
    list[str] of saved PNG paths, if `snapshot_times_ms` was given.
    None otherwise (interactive playback / movie export).
    """
    if off_screen and interactive:
        interactive = False

    snapshot_mode = snapshot_times_ms is not None
    movie_mode = movie_path is not None
    if snapshot_mode and snapshot_dir is None:
        raise ValueError("snapshot_dir is required when snapshot_times_ms is given.")
    if snapshot_mode and movie_mode:
        raise ValueError("Pass either snapshot_times_ms or movie_path, not both "
                          "-- run twice if you want both outputs.")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {npz_path} ...")
    d = _load_coupled_data(npz_path, vmin)
    t_arr, n_frames        = d["t_arr"], d["n_frames"]
    comp_nodes, comp_edges = d["comp_nodes"], d["comp_edges"]
    surf_verts, surf_faces = d["surf_verts"], d["surf_faces"]
    vert_to_surf_node      = d["vert_to_surf_node"]
    surf_frames            = d["surf_frames"]
    V_pkn                  = d["V_pkn"]
    N_comp                 = len(comp_nodes)

    dt_frame = float(t_arr[1] - t_arr[0]) if n_frames > 1 else 1.0
    sleep_s  = (dt_frame / 1000.0) / speed

    print(f"  Frames          : {n_frames}")
    print(f"  Purkinje nodes  : {N_comp}  ({len(comp_edges)} edges)")
    print(f"  Surface verts   : {len(surf_verts)}  faces: {len(surf_faces)}")
    print(f"  Surface nodes   : {surf_frames.shape[1]}")
    print(f"  Sim dt/frame    : {dt_frame:.3f} ms")
    if snapshot_mode:
        print(f"  Mode            : snapshot stills ({len(snapshot_times_ms)} requested)")
    elif movie_mode:
        print(f"  Mode            : movie export -> {movie_path} @ {movie_fps} fps")
    else:
        print(f"  Sleep/frame     : {sleep_s * 1000:.2f} ms  (speed={speed}x)")
        print(f"  Total anim time : {sleep_s * n_frames:.1f} s  "
              f"(sim {t_arr[-1]:.1f} ms)")
    print()

    # ── Build PyVista meshes ──────────────────────────────────────────────────

    myo_mesh       = _build_myo_mesh(surf_verts, surf_faces)
    myo_mesh["Vm"] = surf_frames[0][vert_to_surf_node]   # per-vertex voltage

    pkn_mesh       = _build_pkn_mesh(comp_nodes, comp_edges)
    pkn_mesh["Vm"] = V_pkn[0]

    # ── Plotter ───────────────────────────────────────────────────────────────
    pl = pv.Plotter(notebook=notebook, off_screen=off_screen,
                     window_size=window_size,
                     title="Cardiac EP — Coupled Simulation")

    scalar_bar_args = dict(
        title          = "Vm  (mV)",
        title_font_size= 14,
        label_font_size= 12,
        n_labels       = 5,
        position_x     = 0.88,
        position_y     = 0.05,
        width          = 0.08,
        height         = 0.80,
        color          = "white",
    )

    # Myocardium surface — opaque, shared colourbar
    pl.add_mesh(
        myo_mesh,
        scalars          = "Vm",
        cmap             = cmap,
        clim             = [vmin, vmax],
        show_scalar_bar  = True,
        scalar_bar_args  = scalar_bar_args,
        smooth_shading   = True,
        lighting         = True,
    )

    # Purkinje network — lines, no separate scalar bar (same range)
    pl.add_mesh(
        pkn_mesh,
        scalars         = "Vm",
        cmap            = cmap,
        clim            = [vmin, vmax],
        show_scalar_bar = False,
        line_width      = 2,
        render_lines_as_tubes = True,
    )

    pl.add_text("t = 0.00 ms", name="time_label", font_size=12, color="white")
    pl.set_background("black")
    pl.camera_position = "iso"

    def _update_frame(f: int) -> None:
        myo_mesh["Vm"] = surf_frames[f][vert_to_surf_node]
        pkn_mesh["Vm"] = V_pkn[f]
        pl.add_text(f"t = {t_arr[f]:.2f} ms", name="time_label",
                    font_size=12, color="white")

    # =========================================================================
    # Mode 1 -- snapshot stills: render only the requested frames, save PNGs,
    # no window, no sleeping.
    # =========================================================================
    if snapshot_mode:
        out_dir = Path(snapshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = snapshot_prefix or Path(npz_path).stem

        frame_idxs = _nearest_frame_indices(t_arr, snapshot_times_ms)
        saved_paths = []
        for f in frame_idxs:
            _update_frame(f)
            out_path = out_dir / f"{prefix}_t{t_arr[f]:07.2f}ms.png"
            pl.show(screenshot=str(out_path), auto_close=False)
            saved_paths.append(str(out_path))
            print(f"  Saved snapshot t={t_arr[f]:.2f} ms -> {out_path}")
        pl.close()
        print(f"\nSaved {len(saved_paths)} snapshot(s) -> {snapshot_dir}")
        return saved_paths

    # =========================================================================
    # Mode 2 -- movie export: write every frame to a video/gif at a fixed
    # frame rate, no wall-clock sleeping, no interactive window.
    # =========================================================================
    if movie_mode:
        Path(movie_path).parent.mkdir(parents=True, exist_ok=True)
        pl.show(auto_close=False, interactive_update=True)
        if str(movie_path).lower().endswith(".gif"):
            pl.open_gif(str(movie_path), fps=movie_fps)
        else:
            pl.open_movie(str(movie_path), framerate=movie_fps)

        for f in range(n_frames):
            _update_frame(f)
            pl.render()
            pl.write_frame()

        pl.close()
        print(f"\nSaved movie ({n_frames} frames @ {movie_fps} fps) -> {movie_path}")
        return None

    # =========================================================================
    # Mode 3 -- original interactive real-time playback, unchanged.
    # =========================================================================
    pl.show(auto_close=False, interactive_update=True)

    for f in range(n_frames):

        # Window-closed guard
        try:
            rw = pl.ren_win
            if rw is None or not rw.GetInteractor():
                break
        except Exception:
            break

        _update_frame(f)

        try:
            pl.render()
            pl.update()
        except Exception:
            break

        time_module.sleep(sleep_s)

    # ── Clean shutdown ────────────────────────────────────────────────────────
    try:
        pl.close()
    except Exception:
        pass

    return None


# =============================================================================
# Composite snapshot grid (e.g. 2x4 activation-sequence figure)
# =============================================================================

def plot_activation_snapshot_grid(
    npz_path: str,
    times_ms: list[float],
    grid_shape: tuple[int, int] | None = None,
    vmin:  float = -85.0,
    vmax:  float =  35.0,
    cmap:  str   = "jet",
    show_purkinje: bool = True,
    panel_size: tuple[int, int] = (350, 350),
    background_color: str = "white",
    camera_position: str | tuple = "iso",
    link_views: bool = True,
    show_time_labels: bool = True,
    title: str | None = None,
    screenshot_path: str | Path | None = None,
    off_screen: bool = False,
    interactive: bool = True,
) -> pv.Plotter:
    """
    One static figure, one panel per requested time -- e.g. an 8-panel
    2x4 activation-sequence plot showing the wavefront sweeping across the
    myocardium at 8 chosen moments, all sharing one colour scale.

    This is a different thing from `animate_coupled(..., snapshot_times_ms=...,
    snapshot_dir=...)`, which saves one PNG *per* time. Here every requested
    time is a subplot inside a single saved image.

    Parameters
    ----------
    npz_path : path to the NPZ written by run_coupled()
    times_ms : simulation times (ms) to render, one per panel, in reading
        order (left-to-right, top-to-bottom). Each is snapped to the
        nearest saved frame.
    grid_shape : (rows, cols). Default: as close to square as possible,
        preferring more columns than rows (e.g. 8 times -> (2, 4)).
    vmin, vmax, cmap : shared colour scale across every panel -- this is
        the point of a snapshot grid, so don't pass robust/per-panel
        limits here the way plot_activation_maps.py does per-field.
    show_purkinje : overlay the Purkinje network (same colour scale) on
        each panel.
    panel_size : pixels per panel; the window is panel_size scaled by
        grid_shape.
    camera_position : applied identically to every panel so the anatomy
        lines up across the grid. "iso" or an explicit PyVista camera
        position tuple.
    link_views : keep all panel cameras synchronised if you interact with
        the figure (cosmetic once saved, useful if interactive=True).
    show_time_labels : print "t = ... ms" in the corner of each panel.
    title : optional figure-level title, placed above the top-left panel.
    screenshot_path, off_screen, interactive : same convention as the rest
        of the pipeline (see plot_activation_maps.py / visualise_purkinje.py).
        Default here is off_screen=False, interactive=True -- a window
        opens with all panels laid out (cameras linked, so rotating one
        rotates all of them), you rotate/zoom to the view you want, then
        press 'q' (not the OS close button -- see visualise_purkinje.py's
        docstring note) to close it, which is exactly when the screenshot
        of that final view is captured. Pass off_screen=True,
        interactive=False instead once you're happy with the framing and
        want to regenerate the same figure unattended in a batch run.

    Returns
    -------
    pv.Plotter
    """
    print(f"Loading {npz_path} ...")
    d = _load_coupled_data(npz_path, vmin)
    t_arr, comp_nodes, comp_edges = d["t_arr"], d["comp_nodes"], d["comp_edges"]
    surf_verts, surf_faces = d["surf_verts"], d["surf_faces"]
    vert_to_surf_node, surf_frames = d["vert_to_surf_node"], d["surf_frames"]
    V_pkn = d["V_pkn"]

    n_panels = len(times_ms)
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n_panels)))
        rows = int(np.ceil(n_panels / cols))
        # prefer wide over tall (e.g. 8 -> 2x4, not 3x3)
        if rows > cols:
            rows, cols = cols, rows
        grid_shape = (rows, cols)
    rows, cols = grid_shape
    if rows * cols < n_panels:
        raise ValueError(f"grid_shape {grid_shape} has {rows * cols} slots, "
                          f"but {n_panels} times were requested.")

    frame_idxs = _frame_indices_ordered(t_arr, times_ms)
    print(f"  Panels: {n_panels}  grid: {rows}x{cols}  "
          f"times -> frames: {list(zip(times_ms, frame_idxs))}")

    window_size = (panel_size[0] * cols, panel_size[1] * rows)
    pl = pv.Plotter(shape=(rows, cols), window_size=window_size, off_screen=off_screen)
    pl.set_background(background_color)

    text_color = "black" if background_color != "black" else "white"

    for i, f in enumerate(frame_idxs):
        row, col = divmod(i, cols)
        pl.subplot(row, col)

        myo_mesh = _build_myo_mesh(surf_verts, surf_faces)
        myo_mesh["Vm"] = surf_frames[f][vert_to_surf_node]
        # Shared colour bar: only draw one, anchored on the last panel so
        # it doesn't get visually associated with any single timepoint.
        is_last_panel = (i == n_panels - 1)
        pl.add_mesh(
            myo_mesh, scalars="Vm", cmap=cmap, clim=[vmin, vmax],
            smooth_shading=True, show_scalar_bar=is_last_panel,
            scalar_bar_args=dict(title="Vm (mV)", color=text_color,
                                  n_labels=5, title_font_size=14,
                                  label_font_size=12),
        )

        if show_purkinje:
            pkn_mesh = _build_pkn_mesh(comp_nodes, comp_edges)
            pkn_mesh["Vm"] = V_pkn[f]
            pl.add_mesh(pkn_mesh, scalars="Vm", cmap=cmap, clim=[vmin, vmax],
                        show_scalar_bar=False, line_width=2,
                        render_lines_as_tubes=True)

        if show_time_labels:
            pl.add_text(f"t = {t_arr[f]:.0f} ms", font_size=10, color=text_color)

        pl.camera_position = camera_position

    if title:
        pl.subplot(0, 0)
        pl.add_text(title, position="upper_edge", font_size=14, color=text_color)

    if link_views:
        pl.link_views()

    if screenshot_path:
        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
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
        description="Animate coupled Purkinje + myocardium surface simulation")
    ap.add_argument("npz",    nargs="?", default=None,
                    help="Path to coupled_sim.npz")
    ap.add_argument("--npz",  dest="npz_opt", default=None,
                    help="Alternative: --npz path/to/coupled_sim.npz")
    ap.add_argument("--vmin", type=float, default=-85.0)
    ap.add_argument("--vmax", type=float, default=  35.0)
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--speed",type=float, default=0.1)
    ap.add_argument("--off-screen", action="store_true",
                    help="Render without opening a window (headless-safe)")
    ap.add_argument("--snapshot-times", default=None,
                    help="Comma-separated simulation times (ms) to save as "
                         "individual PNG stills, e.g. 10,40,70,100,150")
    ap.add_argument("--snapshot-dir", default=None,
                    help="Output directory for --snapshot-times PNGs")
    ap.add_argument("--movie", dest="movie_path", default=None,
                    help="Save the full animation as a video/gif here "
                         "(.mp4 or .gif) instead of an interactive window")
    ap.add_argument("--movie-fps", type=int, default=24)
    args = ap.parse_args()

    npz_path = args.npz or args.npz_opt
    if npz_path is None:
        ap.print_help()
        sys.exit(1)

    snapshot_times_ms = None
    if args.snapshot_times:
        snapshot_times_ms = [float(x) for x in args.snapshot_times.split(",")]

    animate_coupled(
        npz_path          = npz_path,
        vmin              = args.vmin,
        vmax              = args.vmax,
        cmap              = args.cmap,
        speed             = args.speed,
        off_screen        = args.off_screen,
        snapshot_times_ms = snapshot_times_ms,
        snapshot_dir      = args.snapshot_dir,
        movie_path        = args.movie_path,
        movie_fps         = args.movie_fps,
    )
