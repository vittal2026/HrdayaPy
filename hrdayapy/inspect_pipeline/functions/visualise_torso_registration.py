"""
inspect_pipeline/functions/visualise_torso_registration.py
=============================================================
Side-by-side PyVista sanity check for Step 13's torso registration.

Left panel  — dataset torso mesh RAW (unregistered) overlaid on the
              simulated BSPM torso surface cloud.
Right panel — dataset torso mesh AFTER registration overlaid on the same
              simulated cloud.

If the two already line up well on the left (expected, since both are
derived from the same anatomy), the right panel should look almost
identical but slightly tighter.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

try:
    ctypes.CDLL("nvapi64.dll")
except Exception:
    pass
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"


def _to_pv_mesh(nodes: np.ndarray, faces: np.ndarray):
    import pyvista as pv
    n_tris = faces.shape[0]
    pv_faces = np.hstack([np.full((n_tris, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(nodes.astype(np.float32), pv_faces)


def visualise_torso_registration(
    registration_npz:  Path,
    bspm_npz_path:      Path,
    gt_torso_mat_path:  Path,
    point_size:         float = 3.0,
    window_size:        tuple[int, int] = (1600, 800),
) -> None:
    """Open a two-panel PyVista window: registration before vs after."""
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    from .load_ground_truth import load_gt_torso_mesh
    from .register_torso import apply_torso_registration

    # ── Load ────────────────────────────────────────────────────────────
    print("[viz] Loading torso registration ...")
    reg = np.load(str(registration_npz))
    R, t = reg["R"].astype(np.float64), reg["t"].astype(np.float64)
    rmse, pre_rmse = float(reg["rmse"]), float(reg["pre_rmse"])

    print("[viz] Loading dataset torso mesh ...")
    nodes, faces, _leadlinks = load_gt_torso_mesh(gt_torso_mat_path)
    gt_mesh_raw = _to_pv_mesh(nodes, faces)
    gt_mesh_reg = _to_pv_mesh(apply_torso_registration(nodes, R, t), faces)

    print("[viz] Loading simulated BSPM torso surface ...")
    bspm = np.load(str(bspm_npz_path))
    sim_cloud = pv.PolyData(bspm["surface_xyz"].astype(np.float32))

    # ── Plotter ─────────────────────────────────────────────────────────
    pv.set_plot_theme("dark")
    pl = pv.Plotter(
        shape=(1, 2),
        window_size=window_size,
        title="Torso registration check — Left: before | Right: after",
    )

    pl.subplot(0, 0)
    pl.add_text(f"Before registration\nRMSE = {pre_rmse:.2f} mm",
                font_size=11, color="white", position="upper_left")
    pl.add_mesh(gt_mesh_raw, color="peachpuff", opacity=0.55,
                label="Dataset torso (raw)")
    pl.add_points(sim_cloud, color="#3fa7ff", point_size=point_size,
                  render_points_as_spheres=True, label="Simulated surface")
    pl.add_legend(bcolor=None, border=False, size=(0.32, 0.10))
    pl.set_background("black")
    pl.add_axes()

    pl.subplot(0, 1)
    pl.add_text(f"After registration\nRMSE = {rmse:.2f} mm",
                font_size=11, color="white", position="upper_left")
    pl.add_mesh(gt_mesh_reg, color="lightgreen", opacity=0.55,
                label="Dataset torso (registered)")
    pl.add_points(sim_cloud.copy(), color="#3fa7ff", point_size=point_size,
                  render_points_as_spheres=True, label="Simulated surface")
    pl.add_legend(bcolor=None, border=False, size=(0.32, 0.10))
    pl.set_background("black")
    pl.add_axes()

    pl.link_views()
    pl.camera_position = "iso"
    pl.show()

    try:
        pl.close()
    except Exception:
        pass
