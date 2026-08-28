"""
inspect_pipeline/functions/pick_and_compare_unipolar_raw.py
==============================================================
Step 14c(raw) -- pick ONE electrode on the (registered) torso, compare
simulated vs recorded FULL RAW TRACE at that electrode (no beat detection,
no beat-averaging).

Non-averaging sibling of pick_and_compare_unipolar.py (Step 14c): same
electrode-pick UI and electrode-only CAR referencing, but instead of
detecting beats and plotting a representative average beat, this just
plots each dataset's whole CAR'd trace over its own native time axis
(ms for the simulation, samples/ms for the recording) end to end:

  - both datasets are common-average-referenced (CAR) independently;
    the simulated dataset's reference is computed from ONLY the surface
    nodes nearest the recording electrodes (not all surface nodes), so
    the two CARs are computed over matching electrode sets -- same
    referencing convention as the averaged-beat version, so the two are
    directly comparable side by side,
  - no beat detection, no windowing, no averaging: what you see is
    exactly the picked channel's whole trace.

Electrode picking is shared with pick_and_compare_unipolar.py /
pick_and_compare_bipolar.py via electrode_pick.py -- click ONE electrode
(re-clicking replaces the pick) then CLOSE the window; only then is
anything plotted.
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
from scipy.spatial import cKDTree


def _to_pv_mesh(nodes: np.ndarray, faces: np.ndarray):
    import pyvista as pv
    n_tris = faces.shape[0]
    pv_faces = np.hstack([np.full((n_tris, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(nodes.astype(np.float32), pv_faces)


def pick_and_compare_unipolar_raw_traces(
    registration_npz:  Path,
    bspm_npz_path:      Path,
    gt_torso_mat_path:  Path,
    gt_ts_mat_path:     Path,
    gt_fs:              float | None = None,
    snap_warn_mm:       float = 15.0,
) -> dict | None:
    """
    Open an interactive PyVista window on the (registered) torso; click ONE
    electrode, then close the window to plot the simulated vs. recorded
    FULL raw (CAR'd, non-beat-averaged) trace at that electrode.

    Parameters
    ----------
    gt_fs : recording sample rate in Hz. If given, the recorded panel's
        x-axis is also shown in ms (purely cosmetic). None keeps it in
        raw samples.

    Returns
    -------
    dict with keys: idx, electrode_xyz, time_sim_ms, trace_sim,
    time_gt, trace_gt -- or None if nothing was picked.
    """
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    from .load_ground_truth import load_gt_torso_mesh, load_gt_bspm
    from .register_torso import apply_torso_registration
    from .electrode_pick import pick_electrodes
    from .beat_utils import common_average_reference, common_average_reference_using_channels

    # ── Load ────────────────────────────────────────────────────────────
    print("[pick_and_compare_unipolar_raw] Loading torso registration ...")
    reg = np.load(str(registration_npz))
    R, t = reg["R"].astype(np.float64), reg["t"].astype(np.float64)

    print("[pick_and_compare_unipolar_raw] Loading dataset torso mesh + recording ...")
    nodes, faces, leadlinks = load_gt_torso_mesh(gt_torso_mat_path)
    gt_potvals = load_gt_bspm(gt_ts_mat_path).astype(np.float64)   # (128, n_samples_gt)
    n_electrodes = leadlinks.shape[0]
    if gt_potvals.shape[0] != n_electrodes:
        print(f"  WARNING: {n_electrodes} leadlinks but potvals has "
              f"{gt_potvals.shape[0]} channels -- ordering assumption may not hold.")

    nodes_reg = apply_torso_registration(nodes, R, t)
    electrode_points = nodes_reg[leadlinks]                     # (128, 3), sim frame
    gt_mesh = _to_pv_mesh(nodes_reg, faces)

    print("[pick_and_compare_unipolar_raw] Loading simulated BSPM ...")
    bspm_data   = np.load(str(bspm_npz_path))
    surface_xyz = bspm_data["surface_xyz"].astype(np.float64)
    bspm_signal = bspm_data["bspm_signal"].astype(np.float64)   # (N_surf, n_frames)
    time_sim    = bspm_data["time_ms"].astype(np.float64)

    print(f"\n[pick_and_compare_unipolar_raw] Nearest surface node per electrode "
          f"({surface_xyz.shape[0]:,} candidates, {n_electrodes} electrodes)")
    tree = cKDTree(surface_xyz)
    snap_dist, nearest_surf_idx = tree.query(electrode_points, k=1)
    print(f"  snap distance: min={snap_dist.min():.2f} mm  "
          f"mean={snap_dist.mean():.2f} mm  max={snap_dist.max():.2f} mm")
    if snap_dist.max() > snap_warn_mm:
        print(f"    [warn] at least one electrode is >{snap_warn_mm:.0f} mm "
              f"from the nearest simulated surface node.")

    # ── CAR both datasets, independently, ONCE (doesn't depend on the pick) ─
    # Same convention as pick_and_compare_unipolar.py: the simulated CAR
    # uses only the electrode-nearest surface nodes, so both CARs are taken
    # over the same (electrode) channel set.
    print(f"\n[pick_and_compare_unipolar_raw] Common-average-referencing both "
          f"datasets (simulated CAR uses only the {len(nearest_surf_idx)} "
          f"electrode-nearest surface nodes)")
    bspm_car, _ = common_average_reference_using_channels(bspm_signal, nearest_surf_idx)
    gt_car, _   = common_average_reference(gt_potvals)

    # ── Pick ONE electrode ───────────────────────────────────────────────
    picked = pick_electrodes(
        gt_mesh, electrode_points, max_picks=1, snap_warn_mm=snap_warn_mm,
        window_title="Torso -- pick ONE electrode, 'c' to clear",
    )
    if not picked:
        print("\n[pick_and_compare_unipolar_raw] No electrode picked -- nothing to plot.")
        return None
    idx = picked[0]
    electrode_xyz = electrode_points[idx]
    print(f"\n[pick_and_compare_unipolar_raw] Electrode #{idx}  "
          f"{np.round(electrode_xyz, 1)} mm -- plotting full trace...")

    trace_sim = bspm_car[nearest_surf_idx[idx], :]
    trace_gt  = gt_car[idx, :]

    n_samples_gt = trace_gt.shape[0]
    if gt_fs is not None:
        time_gt = np.arange(n_samples_gt) / gt_fs * 1000.0
        xlabel_gt = "Time (ms)"
    else:
        time_gt = np.arange(n_samples_gt)
        xlabel_gt = "Sample"

    # ── Plot: whole trace, side by side ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(time_sim, trace_sim, color="#e05252", linewidth=1.1)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel(r"$\phi_e$ (mV, CAR)")
    axes[0].set_title(f"Simulated (HrdayaPy), CAR\nelectrode #{idx}")
    axes[0].grid(True, alpha=0.3)
    axes[0].margins(x=0)

    axes[1].plot(time_gt, trace_gt, color="black", linewidth=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel(xlabel_gt)
    axes[1].set_ylabel(r"$\phi_e$ (mV, CAR)")
    axes[1].set_title(f"Recorded (patient {Path(gt_ts_mat_path).stem.split('-')[0]}), CAR\n"
                       f"electrode #{idx}")
    axes[1].grid(True, color="lightgray", linewidth=0.5)
    axes[1].margins(x=0)

    fig.suptitle(f"Electrode #{idx}  {np.round(electrode_xyz, 1)} mm  --  full trace, no beat averaging  "
                 f"(note: simulation and recording time axes are independent)")
    fig.tight_layout()
    try:
        fig.canvas.manager.set_window_title("Full trace -- unipolar")
    except Exception:
        pass
    plt.show()

    return dict(
        idx=idx, electrode_xyz=electrode_xyz,
        time_sim_ms=time_sim, trace_sim=trace_sim,
        time_gt=time_gt, trace_gt=trace_gt,
    )
