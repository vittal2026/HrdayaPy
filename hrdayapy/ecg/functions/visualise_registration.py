"""
ecg/functions/visualise_registration.py
=========================================
Side-by-side PyVista visualisation to sanity-check the heart-to-torso
registration.

Left panel  — ground truth anatomy from the full torso NRRD:
              torso surface (grey, translucent) + heart surface (red).

Right panel — registered simulation data:
              torso surface (grey, translucent) + snapshot point cloud
              coloured by Vm(t), looping continuously.

Usage
-----
Called standalone, or wherever the ecg stage's registration step wants a
visual check:

    python ecg/functions/visualise_registration.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import time as time_module
from pathlib import Path

import numpy as np

try:
    ctypes.CDLL("nvapi64.dll")
except Exception:
    pass
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"


# ---------------------------------------------------------------------------
# Helpers — surface building
# ---------------------------------------------------------------------------

def _mask_to_pv_mesh(binary_vol: np.ndarray,
                     spacing:    np.ndarray,
                     origin:     np.ndarray,
                     step:       int = 2,
                     target_mm:  float | None = 1.0):
    """Marching-cubes surface → PyVista PolyData in physical mm.

    step, target_mm : target_mm (default 1.0mm) overrides step with a
        physical-mm step size, using spacing.mean() as the isotropic
        voxel size (spacing here can be anisotropic; this is an
        approximation for sizing the mesh only, not a correctness
        concern -- see mesh_labelling.mm_step_size for the general
        version of this idea). Set target_mm=None to use the literal
        step value instead. Without this, step (a voxel count) makes
        the extracted mesh get denser purely because binary_vol was
        built at a finer spacing, regardless of whether that detail is
        wanted for this registration check.
    """
    from skimage.measure import marching_cubes
    import pyvista as pv

    if target_mm is not None:
        step = max(1, round(target_mm / float(np.mean(spacing))))
    verts, faces, _, _ = marching_cubes(
        binary_vol.astype(np.float32),
        level=0.5, step_size=step, allow_degenerate=False,
    )
    verts_mm = verts * spacing[np.newaxis, :] + origin[np.newaxis, :]
    n        = len(faces)
    pv_faces = np.hstack([np.full((n, 1), 3, np.int32), faces]).ravel()
    return pv.PolyData(verts_mm.astype(np.float32), pv_faces)


def _load_nrrd(nrrd_path: Path):
    import nrrd
    volume, header = nrrd.read(str(nrrd_path))
    volume = np.asarray(volume)
    if "space directions" in header:
        sd      = np.asarray(header["space directions"], dtype=np.float64)
        spacing = np.array([np.linalg.norm(sd[i]) for i in range(3)])
        origin  = np.asarray(header.get("space origin", [0., 0., 0.]), np.float64)
    elif "spacings" in header:
        spacing = np.asarray(header["spacings"], np.float64)
        origin  = np.zeros(3, np.float64)
    else:
        spacing = np.ones(3, np.float64)
        origin  = np.zeros(3, np.float64)
    return volume, spacing, origin


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualise_registration(
    registration_npz:  Path,
    vm_snapshots_path: Path,
    torso_nrrd_path:   Path,
    heart_label:       int   = 2,
    mc_step_torso:     int   = 2,
    mc_step_heart:     int   = 1,
    mc_target_mm:      float | None = 1.0,
    vmin:              float = -85.0,
    vmax:              float =  35.0,
    cmap:              str   = "jet",
    frame_delay_ms:    float = 100.0,
) -> None:
    """
    Open a side-by-side PyVista window:

    Left  — NRRD anatomy (torso + heart label-2 surface).
    Right — Registered snapshot point cloud animated by Vm(t).

    Parameters
    ----------
    registration_npz  : path to heart_registration.npz
    vm_snapshots_path : path to vm_snapshots.npz
    torso_nrrd_path   : path to torso.seg.nrrd
    heart_label       : NRRD label index for myocardium (default 2)
    mc_step_torso     : marching-cubes step for torso surface
    mc_step_heart     : marching-cubes step for NRRD heart surface
    vmin / vmax       : Vm colourbar range [mV]
    cmap              : matplotlib colourmap
    frame_delay_ms    : ms to sleep between frames
    """
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    # ── 1. Load registration params ───────────────────────────────────────
    print("[viz] Loading registration params ...")
    reg   = np.load(str(registration_npz))
    R     = reg["R"].astype(np.float64)
    t     = reg["t"].astype(np.float64)
    scale = float(reg["scale"])
    dice  = float(reg["dice"])
    print(f"  scale={scale:.4f}  dice={dice:.4f}")
    print(f"  R =\n{np.round(R, 4)}")
    print(f"  t = {np.round(t, 2)} mm")

    # ── 2. Load Vm snapshots ──────────────────────────────────────────────
    print("\n[viz] Loading vm_snapshots ...")
    snap      = np.load(str(vm_snapshots_path))
    coords_mm = snap["coords_mm"].astype(np.float64)   # (N, 3)
    Vm_frames = snap["Vm"].astype(np.float32)           # (n_frames, N)
    time_ms   = snap["time_vm"].astype(np.float64)      # (n_frames,)
    n_frames  = len(time_ms)
    print(f"  Nodes={len(coords_mm):,}  frames={n_frames}")

    # Apply registration
    registered = (R @ (coords_mm * scale).T).T + t     # (N, 3)

    # ── 3. Load NRRD and build surfaces ───────────────────────────────────
    print("\n[viz] Building NRRD surfaces ...")
    volume, spacing, origin = _load_nrrd(torso_nrrd_path)

    torso_mesh = _mask_to_pv_mesh(volume >= 1,           spacing, origin, step=mc_step_torso, target_mm=mc_target_mm)
    heart_mesh = _mask_to_pv_mesh(volume == heart_label, spacing, origin, step=mc_step_heart, target_mm=mc_target_mm)
    print(f"  Torso: {torso_mesh.n_points:,} verts")
    print(f"  Heart: {heart_mesh.n_points:,} verts")

    # ── 4. Build animated point cloud ─────────────────────────────────────
    cloud        = pv.PolyData(registered.astype(np.float32))
    cloud["Vm"]  = Vm_frames[0]

    # Shared scalar bar args
    scalar_bar_args = dict(
        title           = "Vm  (mV)",
        title_font_size = 13,
        label_font_size = 11,
        n_labels        = 5,
        position_x      = 0.91,
        position_y      = 0.05,
        width           = 0.06,
        height          = 0.80,
        color           = "white",
    )

    # ── 5. Build plotter ──────────────────────────────────────────────────
    print("\n[viz] Opening PyVista window ...")
    pl = pv.Plotter(
        shape=(1, 2),
        window_size=(1600, 800),
        title="Registration Check — Left: NRRD anatomy | Right: Registered Vm",
    )

    # ── LEFT panel — NRRD anatomy ─────────────────────────────────────────
    pl.subplot(0, 0)
    pl.add_text("NRRD anatomy\n(ground truth)",
                font_size=10, color="white", position="upper_left")
    pl.add_mesh(torso_mesh, color="#aaaaaa", opacity=0.12,
                label="Torso surface")
    pl.add_mesh(heart_mesh, color="#ff6666", opacity=0.55,
                label="Heart (label 2)")
    pl.add_legend(bcolor=None, border=False, size=(0.30, 0.10))
    pl.set_background("black")
    pl.add_axes()

    # ── RIGHT panel — registered snapshot animated ────────────────────────
    pl.subplot(0, 1)
    pl.add_text(f"Registered snapshot  (Dice={dice:.3f})\nt = {time_ms[0]:.1f} ms",
                name="time_label", font_size=10, color="white",
                position="upper_left")

    # Torso surface — same as left for spatial reference
    pl.add_mesh(torso_mesh.copy(), color="#aaaaaa", opacity=0.12)

    # Animated point cloud
    pl.add_points(
        cloud,
        scalars         = "Vm",
        cmap            = cmap,
        clim            = [vmin, vmax],
        point_size      = 3,
        render_points_as_spheres = False,
        show_scalar_bar = True,
        scalar_bar_args = scalar_bar_args,
    )
    pl.set_background("black")
    pl.add_axes()

    # Link cameras so both panels rotate together
    pl.link_views()

    # ── 6. Animation loop ─────────────────────────────────────────────────
    sleep_s = frame_delay_ms / 1000.0
    pl.show(auto_close=False, interactive_update=True)

    frame = 0
    while True:
        # Guard against window being closed
        try:
            if pl.ren_win is None or not pl.ren_win.GetInteractor():
                break
        except Exception:
            break

        # Update point cloud scalars
        cloud["Vm"] = Vm_frames[frame]

        pl.subplot(0, 1)
        pl.add_text(
            f"Registered snapshot  (Dice={dice:.3f})\nt = {time_ms[frame]:.1f} ms",
            name="time_label", font_size=10, color="white",
            position="upper_left",
        )

        try:
            pl.render()
            pl.update()
        except Exception:
            break

        time_module.sleep(sleep_s)
        frame = (frame + 1) % n_frames   # continuous loop

    try:
        pl.close()
    except Exception:
        pass
