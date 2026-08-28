"""
inspect_pipeline/functions/pick_and_compare_bipolar.py
=========================================================
Step 14d -- pick TWO electrodes (A, B) on the (registered) torso, compare
simulated vs recorded REPRESENTATIVE AVERAGE BEAT of the A-B difference.

Bipolar sibling of pick_and_compare_unipolar.py (Step 14c): same
registered-torso display, electrode set, and beat-averaging pipeline
(beat_utils.py, ported from the experimental
bipolar_representative_beat.py), applied to the A-B difference trace
instead of a single electrode's trace:

  - both datasets are common-average-referenced (CAR) independently --
    note this has NO EFFECT on the final A-B difference itself (CAR
    subtracts the same per-sample, cross-channel mean from every
    channel, so it cancels in A-B); it's applied for consistency and
    because the fiducial detection below needs it,
  - beat fiducials are found from the RMS, ACROSS ALL CHANNELS, of the
    CAR'd derivative (robust regardless of which two electrodes end up
    picked, since it doesn't depend on A or B individually),
  - each dataset is windowed in its own native units (ms for the
    simulation, samples for the recording) as a fraction of its own
    median beat-to-beat interval, so no shared/assumed sample rate is
    needed between the two,
  - averaging is NaN-padded: a beat that's incomplete at one edge of the
    window still contributes everywhere it has data; it's never
    discarded outright.

Electrode picking is shared with pick_and_compare_unipolar.py via
electrode_pick.py -- click TWO electrodes (A then B; a third click drops
A and keeps the newest two) then CLOSE the window; only then is anything
plotted.
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


def pick_and_compare_bipolar_beat_traces(
    registration_npz:  Path,
    bspm_npz_path:      Path,
    gt_torso_mat_path:  Path,
    gt_ts_mat_path:     Path,
    gt_fs:              float | None = None,
    pre_frac:           float = 0.30,
    post_frac:          float = 0.60,
    prominence_frac:    float = 0.35,
    min_rr_sim:         int = 250,
    min_rr_gt:          int = 200,
    snap_warn_mm:       float = 15.0,
) -> dict | None:
    """
    Open an interactive PyVista window on the (registered) torso; click TWO
    electrodes (A, B), then close the window to plot the simulated vs.
    recorded representative average beat of the A-B bipolar trace.

    Parameters are the same as pick_and_compare_unipolar_beat_traces.

    Returns
    -------
    dict with keys: idx_a, idx_b, xyz_a, xyz_b, t_rel_sim_ms, avg_sim,
    count_sim, beats_sim, t_rel_gt, avg_gt, count_gt, beats_gt, fid_sim,
    fid_gt -- or None if fewer than 2 electrodes were picked.
    """
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    from .load_ground_truth import load_gt_torso_mesh, load_gt_bspm
    from .register_torso import apply_torso_registration
    from .electrode_pick import pick_electrodes
    from .beat_utils import (
        common_average_reference,
        detect_fiducials_multichannel,
        beat_window_from_rr,
        extract_average_beat,
        plot_average_beat_panel,
    )

    # ── Load ────────────────────────────────────────────────────────────
    print("[pick_and_compare_bipolar] Loading torso registration ...")
    reg = np.load(str(registration_npz))
    R, t = reg["R"].astype(np.float64), reg["t"].astype(np.float64)

    print("[pick_and_compare_bipolar] Loading dataset torso mesh + recording ...")
    nodes, faces, leadlinks = load_gt_torso_mesh(gt_torso_mat_path)
    gt_potvals = load_gt_bspm(gt_ts_mat_path).astype(np.float64)   # (128, n_samples_gt)
    n_electrodes = leadlinks.shape[0]
    if gt_potvals.shape[0] != n_electrodes:
        print(f"  WARNING: {n_electrodes} leadlinks but potvals has "
              f"{gt_potvals.shape[0]} channels -- ordering assumption may not hold.")

    nodes_reg = apply_torso_registration(nodes, R, t)
    electrode_points = nodes_reg[leadlinks]                     # (128, 3), sim frame
    gt_mesh = _to_pv_mesh(nodes_reg, faces)

    print("[pick_and_compare_bipolar] Loading simulated BSPM ...")
    bspm_data   = np.load(str(bspm_npz_path))
    surface_xyz = bspm_data["surface_xyz"].astype(np.float64)
    bspm_signal = bspm_data["bspm_signal"].astype(np.float64)   # (N_surf, n_frames)
    time_sim    = bspm_data["time_ms"].astype(np.float64)
    dt_sim      = float(np.median(np.diff(time_sim)))

    print(f"\n[pick_and_compare_bipolar] Nearest surface node per electrode "
          f"({surface_xyz.shape[0]:,} candidates, {n_electrodes} electrodes)")
    tree = cKDTree(surface_xyz)
    snap_dist, nearest_surf_idx = tree.query(electrode_points, k=1)
    print(f"  snap distance: min={snap_dist.min():.2f} mm  "
          f"mean={snap_dist.mean():.2f} mm  max={snap_dist.max():.2f} mm")
    if snap_dist.max() > snap_warn_mm:
        print(f"    [warn] at least one electrode is >{snap_warn_mm:.0f} mm "
              f"from the nearest simulated surface node.")

    # ── CAR both datasets, independently, ONCE (doesn't depend on the pick) ─
    # No electrode-only reference needed here (unlike the unipolar sibling):
    # CAR cancels in the A-B difference regardless of which channels the
    # reference mean is taken over.
    print("\n[pick_and_compare_bipolar] Common-average-referencing both datasets")
    bspm_car, _ = common_average_reference(bspm_signal)
    gt_car, _   = common_average_reference(gt_potvals)

    # ── Beat fiducials + window sizes, ONCE (also independent of the pick) ──
    print("\n[pick_and_compare_bipolar] Detecting beats (RMS-of-derivative, "
          "across all channels)")
    fid_sim, _ = detect_fiducials_multichannel(
        bspm_car, min_distance=min_rr_sim, prominence_frac=prominence_frac, label="simulated",
    )
    fid_gt, _ = detect_fiducials_multichannel(
        gt_car, min_distance=min_rr_gt, prominence_frac=prominence_frac, label="recorded",
    )
    if len(fid_sim) == 0 or len(fid_gt) == 0:
        raise RuntimeError(
            "No beats detected in one of the two datasets -- loosen "
            "prominence_frac or check the CAR'd RMS-of-derivative signal."
        )
    print(f"  simulated: {len(fid_sim)} beat(s) at frames {fid_sim} "
          f"(t = {np.round(time_sim[fid_sim], 1)} ms)")
    print(f"  recorded:  {len(fid_gt)} beat(s) at samples {fid_gt}")

    pre_sim, post_sim, _ = beat_window_from_rr(
        fid_sim, pre_frac, post_frac, fallback_length=bspm_car.shape[1], label="simulated",
    )
    pre_gt, post_gt, _ = beat_window_from_rr(
        fid_gt, pre_frac, post_frac, fallback_length=gt_car.shape[1], label="recorded",
    )

    # ── Pick TWO electrodes (A, B) ──────────────────────────────────────
    picked = pick_electrodes(
        gt_mesh, electrode_points, max_picks=2, snap_warn_mm=snap_warn_mm,
        window_title="Torso -- pick TWO electrodes (A, B), 'c' to clear",
    )
    if len(picked) < 2:
        print(f"\n[pick_and_compare_bipolar] Only {len(picked)} electrode(s) "
              f"picked -- need 2 for a bipolar (A-B) trace. Nothing to plot.")
        return None
    idx_a, idx_b = picked[0], picked[1]
    xyz_a, xyz_b = electrode_points[idx_a], electrode_points[idx_b]
    print(f"\n[pick_and_compare_bipolar] A = electrode #{idx_a}  {np.round(xyz_a, 1)} mm")
    print(f"B = electrode #{idx_b}  {np.round(xyz_b, 1)} mm")
    print("Computing average beat(s) of the A-B bipolar trace...")

    surf_a = nearest_surf_idx[idx_a]
    surf_b = nearest_surf_idx[idx_b]
    trace_sim_car = bspm_car[surf_a, :] - bspm_car[surf_b, :]   # CAR cancels here, kept for consistency
    trace_gt_car  = gt_car[idx_a, :] - gt_car[idx_b, :]

    avg_sim, count_sim, beats_sim = extract_average_beat(trace_sim_car, fid_sim, pre_sim, post_sim)
    avg_gt, count_gt, beats_gt   = extract_average_beat(trace_gt_car, fid_gt, pre_gt, post_gt)

    t_rel_sim = np.arange(-pre_sim, post_sim) * dt_sim   # ms, 0 = fiducial
    t_rel_gt  = np.arange(-pre_gt, post_gt).astype(np.float64)   # samples, 0 = fiducial
    gt_xlabel = "Samples relative to fiducial"
    if gt_fs is not None:
        t_rel_gt = t_rel_gt / gt_fs * 1000.0
        gt_xlabel = "Time relative to fiducial (ms)"

    # ── Plot: average beat (top) + beat-count coverage (bottom) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col",
                              gridspec_kw={"height_ratios": [3, 1]})

    plot_average_beat_panel(
        axes[0, 0], axes[1, 0], t_rel_sim, avg_sim, count_sim, beats_sim,
        color="#e05252", label="average beat",
        title=f"Simulated average beat (CAR), A-B\nA=#{idx_a} B=#{idx_b}, {len(fid_sim)} beat(s) detected",
    )
    axes[0, 0].set_ylabel(r"$\phi_e$ (mV, CAR)")
    axes[1, 0].set_xlabel("Time relative to fiducial (ms)")

    plot_average_beat_panel(
        axes[0, 1], axes[1, 1], t_rel_gt, avg_gt, count_gt, beats_gt,
        color="black", label="average beat",
        title=f"Recorded average beat (CAR), A-B\nA=#{idx_a} B=#{idx_b}, {len(fid_gt)} beat(s) detected",
    )
    axes[1, 1].set_xlabel(gt_xlabel)

    fig.suptitle(f"A = electrode #{idx_a}    B = electrode #{idx_b}    (bipolar = A - B)  "
                 f"(note: simulation and recording axes are independent, native units)")
    fig.tight_layout()
    try:
        fig.canvas.manager.set_window_title("Average beat -- bipolar A-B")
    except Exception:
        pass
    plt.show()

    return dict(
        idx_a=idx_a, idx_b=idx_b, xyz_a=xyz_a, xyz_b=xyz_b,
        t_rel_sim_ms=t_rel_sim, avg_sim=avg_sim, count_sim=count_sim, beats_sim=beats_sim,
        t_rel_gt=t_rel_gt, avg_gt=avg_gt, count_gt=count_gt, beats_gt=beats_gt,
        fid_sim=fid_sim, fid_gt=fid_gt,
    )
