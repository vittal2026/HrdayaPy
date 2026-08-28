"""
inspect_pipeline/functions/pick_and_compare.py
=================================================
Step 14 — pick a point on the torso, compare simulated vs recorded traces.

Displays the dataset's torso mesh AFTER registration (Step 13) — i.e.
already sitting in the simulated BSPM's frame — together with the
simulated surface point cloud for visual reference. Right-click anywhere
on the torso mesh to pick a point; close the window and the voltage trace
at that location is plotted side by side:

    Left  — simulated φ_e(t), nearest node in bspm.npz's surface_xyz
    Right — recorded φ_e(t), nearest electrode in the dataset's ts.potvals

Both electrode lookup and simulated-node lookup use the SAME registered
point, so no inverse transform is needed anywhere in this function.
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

import matplotlib.pyplot as plt


def _to_pv_mesh(nodes: np.ndarray, faces: np.ndarray):
    import pyvista as pv
    n_tris = faces.shape[0]
    pv_faces = np.hstack([np.full((n_tris, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(nodes.astype(np.float32), pv_faces)


def pick_and_compare_traces(
    registration_npz:  Path,
    bspm_npz_path:      Path,
    gt_torso_mat_path:  Path,
    gt_ts_mat_path:     Path,
    gt_fs:              float | None = None,
    point_size:         float = 4.0,
) -> dict | None:
    """
    Open an interactive PyVista window; after it's closed, plot the
    simulated and recorded voltage traces at the picked location.

    Returns
    -------
    dict with keys: xyz, idx_sim, idx_gt, dist_sim_mm, dist_gt_mm,
    trace_sim, time_sim, trace_gt, time_gt   — or None if nothing was picked.
    """
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    from .load_ground_truth import load_gt_torso_mesh, load_gt_bspm
    from .register_torso import apply_torso_registration

    # ── Load ────────────────────────────────────────────────────────────
    print("[pick_and_compare] Loading torso registration ...")
    reg = np.load(str(registration_npz))
    R, t = reg["R"].astype(np.float64), reg["t"].astype(np.float64)

    print("[pick_and_compare] Loading dataset torso mesh + recording ...")
    nodes, faces, leadlinks = load_gt_torso_mesh(gt_torso_mat_path)
    potvals = load_gt_bspm(gt_ts_mat_path)          # (128, n_samples_gt)
    n_electrodes = leadlinks.shape[0]
    if potvals.shape[0] != n_electrodes:
        print(f"  WARNING: {n_electrodes} leadlinks but potvals has "
              f"{potvals.shape[0]} channels -- ordering assumption may not hold.")

    nodes_reg = apply_torso_registration(nodes, R, t)     # dataset -> sim frame
    electrode_points = nodes_reg[leadlinks]                # (128, 3), sim frame
    gt_mesh = _to_pv_mesh(nodes_reg, faces)

    print("[pick_and_compare] Loading simulated BSPM ...")
    bspm_data   = np.load(str(bspm_npz_path))
    surface_xyz = bspm_data["surface_xyz"].astype(np.float64)   # sim frame
    bspm_signal = bspm_data["bspm_signal"].astype(np.float32)   # (N_surf, n_frames)
    time_sim    = bspm_data["time_ms"].astype(np.float64)

    # ── Interactive pick ────────────────────────────────────────────────
    picked: dict = {"xyz": None}

    def on_pick(point) -> None:
        picked["xyz"] = np.asarray(point, dtype=np.float64)
        d_gt = np.linalg.norm(electrode_points - point, axis=1)
        idx_gt = int(np.argmin(d_gt))
        print(f"  Picked {np.round(point, 1)} mm -> "
              f"nearest electrode #{idx_gt} ({d_gt[idx_gt]:.1f} mm away)")

    pv.set_plot_theme("dark")
    pl = pv.Plotter(title="Torso — right-click a point, then close to compare traces")
    pl.add_mesh(gt_mesh, color="peachpuff", opacity=0.9, smooth_shading=True,
                show_edges=False, label="Dataset torso (registered)")
    pl.add_points(electrode_points, color="blue", point_size=6,
                  render_points_as_spheres=True, label="Recording electrodes")
    pl.add_points(surface_xyz, color="#3fa7ff", opacity=0.25, point_size=2,
                  render_points_as_spheres=True, label="Simulated surface")
    pl.enable_point_picking(
        callback       = on_pick,
        picker         = "point",
        left_clicking  = False,
        show_message   = ("Right-click a point on the torso to select it.\n"
                           "Close the window when done to compare traces."),
        color          = "magenta",
        point_size     = 12,
    )
    pl.add_legend(bcolor=None, border=False)
    pl.add_axes()
    pl.add_title("Right-click to pick, close window to compare traces")
    pl.show()

    try:
        pl.close()
    except Exception:
        pass

    if picked["xyz"] is None:
        print("[pick_and_compare] No point was picked. Nothing to plot.")
        return None

    xyz = picked["xyz"]

    # ── Nearest simulated surface node ─────────────────────────────────
    d_sim   = np.linalg.norm(surface_xyz - xyz, axis=1)
    idx_sim = int(np.argmin(d_sim))
    trace_sim = bspm_signal[idx_sim, :]

    # ── Nearest recording electrode ────────────────────────────────────
    d_gt   = np.linalg.norm(electrode_points - xyz, axis=1)
    idx_gt = int(np.argmin(d_gt))
    trace_gt = potvals[idx_gt, :]

    n_samples_gt = trace_gt.shape[0]
    if gt_fs is not None:
        time_gt = np.arange(n_samples_gt) / gt_fs
        xlabel_gt = "Time (s)"
    else:
        time_gt = np.arange(n_samples_gt)
        xlabel_gt = "Sample"

    # ── Plot side by side ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(time_sim, trace_sim, color="#e05252", linewidth=1.3)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel(r"$\phi_e$  (mV)")
    axes[0].set_title(f"Simulated (HrdayaPy)\nnode {idx_sim}, {d_sim[idx_sim]:.1f} mm from pick")
    axes[0].grid(True, alpha=0.3)
    axes[0].margins(x=0)

    axes[1].plot(time_gt, trace_gt, color="black", linewidth=0.9)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel(xlabel_gt)
    axes[1].set_ylabel("amplitude (mV, assumed)")
    axes[1].set_title(f"Ground truth (patient {Path(gt_ts_mat_path).stem.split('-')[0]})\n"
                       f"electrode #{idx_gt}, {d_gt[idx_gt]:.1f} mm from pick")
    axes[1].grid(True, color="lightgray", linewidth=0.5)
    axes[1].margins(x=0)

    fig.suptitle(f"Picked point {np.round(xyz, 1)} mm  "
                 f"(note: simulation and recording time axes are independent)")
    fig.tight_layout()
    plt.show()

    return dict(
        xyz=xyz, idx_sim=idx_sim, idx_gt=idx_gt,
        dist_sim_mm=float(d_sim[idx_sim]), dist_gt_mm=float(d_gt[idx_gt]),
        trace_sim=trace_sim, time_sim=time_sim,
        trace_gt=trace_gt, time_gt=time_gt,
    )
