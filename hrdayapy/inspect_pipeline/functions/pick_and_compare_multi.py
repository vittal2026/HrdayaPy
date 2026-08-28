"""
inspect_pipeline/functions/pick_and_compare_multi.py
=======================================================
Step 14b -- pick up to N_POINTS points on the torso, compare simulated vs
recorded averaged-beat morphology at all of them in one N-panel plot.

This is the multi-point sibling of pick_and_compare.py (Step 14): same
interactive right-click picker, same registered-torso display, but instead
of plotting one raw voltage trace per side, each picked point gets:

    1. CAR (common-average reference) on both datasets independently,
    2. multi-channel beat detection (shared across all picked points --
       computed once from the full channel set, not per point),
    3. a NaN-padded averaged beat at that point,
    4. global (not per-channel) z-score normalization by the standard
       deviation of the *averaged* beat, pooled across the selected
       points and time -- NOT pooled across raw individual beats, and
       NOT per-channel. See average_beat_compare.py's docstring for why
       the averaged beat (rather than raw beats) is the right input to a
       normalization scalar: it's noise-suppressed, so the scalar
       reflects signal amplitude rather than beat-to-beat noise level.
       Per-point (rather than per-channel) pooling is deliberate too --
       it preserves the relative amplitude relationship between the
       selected points, which per-channel normalization would destroy.
    5. resampling onto a common fraction-of-beat-window grid before
       overlay/correlation, so no assumption is made about the (currently
       unverified -- see bspm_viewer.py's SAMPLE_RATE_HZ placeholder)
       ground-truth recording's true sample rate.

Note on the normalization step: it changes what the overlay plot looks
like, but NOT the reported Pearson r values -- r is invariant to any
independent affine rescaling of either signal. The normalization exists
purely to make the visual comparison honest; don't read anything into it
when interpreting the r numbers.
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
from scipy.signal import find_peaks

# --- Beat-detection / windowing defaults (same as average_beat_compare.py) -
PRE_FRAC            = 0.30
POST_FRAC           = 0.60
PROMINENCE_FRAC      = 0.35
MIN_RR_SAMPLES_SIM   = 20
MIN_RR_SAMPLES_GT    = 200
N_COMMON_GRID        = 300


def _to_pv_mesh(nodes: np.ndarray, faces: np.ndarray):
    import pyvista as pv
    n_tris = faces.shape[0]
    pv_faces = np.hstack([np.full((n_tris, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(nodes.astype(np.float32), pv_faces)


def _detect_fiducials(rms_signal, min_distance, prominence_frac=PROMINENCE_FRAC):
    dynamic_range = np.nanmax(rms_signal) - np.nanmin(rms_signal)
    prominence = prominence_frac * dynamic_range
    peaks, _ = find_peaks(rms_signal, distance=min_distance, prominence=prominence)
    return peaks


def _extract_average_beat(signal, fiducials, pre, post):
    """Single-channel NaN-padded average beat (same logic as
    average_beat_compare.py's extract_average_beat)."""
    n = pre + post
    beats = np.full((len(fiducials), n), np.nan)
    for i, f in enumerate(fiducials):
        start, end = f - pre, f + post
        seg_start, seg_end = max(start, 0), min(end, len(signal))
        if seg_start >= seg_end:
            continue
        out_start = seg_start - start
        out_end = out_start + (seg_end - seg_start)
        beats[i, out_start:out_end] = signal[seg_start:seg_end]
    return np.nanmean(beats, axis=0)


def _resample_to_common_grid(rows, n_common=N_COMMON_GRID):
    """rows : (n_rows, n_native), possibly NaN at the edges. Interpolates
    each row's fraction-of-window axis [0,1] onto a common grid."""
    n_rows, n_native = rows.shape
    x_native = np.linspace(0.0, 1.0, n_native)
    x_common = np.linspace(0.0, 1.0, n_common)
    out = np.full((n_rows, n_common), np.nan)
    for r in range(n_rows):
        row = rows[r]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        out[r] = np.interp(x_common, x_native[valid], row[valid])
    return out, x_common


def pick_and_compare_multi_traces(
    registration_npz:  Path,
    bspm_npz_path:      Path,
    gt_torso_mat_path:  Path,
    gt_ts_mat_path:     Path,
    n_points:           int = 6,
    point_size:         float = 4.0,
) -> dict | None:
    """
    Open an interactive PyVista window; right-click up to `n_points`
    locations on the (registered) torso. Close the window at any time --
    whatever was picked (1 up to n_points) is compared. For each picked
    point, plots the normalized simulated vs. recorded averaged beat and
    reports a Pearson correlation coefficient.

    Returns
    -------
    dict with keys: xyz (n_picked,3), idx_sim, idx_gt (n_picked,),
    dist_sim_mm, dist_gt_mm (n_picked,), r_values (n_picked,),
    avg_sim (n_picked, N_COMMON_GRID), avg_gt (n_picked, N_COMMON_GRID)
    -- or None if nothing was picked.
    """
    import pyvista as pv
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    from .load_ground_truth import load_gt_torso_mesh, load_gt_bspm
    from .register_torso import apply_torso_registration

    # ── Load ────────────────────────────────────────────────────────────
    print("[pick_and_compare_multi] Loading torso registration ...")
    reg = np.load(str(registration_npz))
    R, t = reg["R"].astype(np.float64), reg["t"].astype(np.float64)

    print("[pick_and_compare_multi] Loading dataset torso mesh + recording ...")
    nodes, faces, leadlinks = load_gt_torso_mesh(gt_torso_mat_path)
    potvals = load_gt_bspm(gt_ts_mat_path).astype(np.float64)   # (128, n_samples_gt)
    n_electrodes = leadlinks.shape[0]
    if potvals.shape[0] != n_electrodes:
        print(f"  WARNING: {n_electrodes} leadlinks but potvals has "
              f"{potvals.shape[0]} channels -- ordering assumption may not hold.")

    nodes_reg = apply_torso_registration(nodes, R, t)
    electrode_points = nodes_reg[leadlinks]                     # (128, 3), sim frame
    gt_mesh = _to_pv_mesh(nodes_reg, faces)

    print("[pick_and_compare_multi] Loading simulated BSPM ...")
    bspm_data   = np.load(str(bspm_npz_path))
    surface_xyz = bspm_data["surface_xyz"].astype(np.float64)
    bspm_signal = bspm_data["bspm_signal"].astype(np.float64)   # (N_surf, n_frames)
    time_sim    = bspm_data["time_ms"].astype(np.float64)

    # ── Interactive multi-pick ──────────────────────────────────────────
    picked_xyz: list[np.ndarray] = []

    def on_pick(point) -> None:
        if len(picked_xyz) >= n_points:
            print(f"  Already have {n_points} points -- close the window to compare.")
            return
        xyz = np.asarray(point, dtype=np.float64)
        picked_xyz.append(xyz)
        d_gt = np.linalg.norm(electrode_points - xyz, axis=1)
        idx_gt = int(np.argmin(d_gt))
        print(f"  [{len(picked_xyz)}/{n_points}] Picked {np.round(xyz, 1)} mm -> "
              f"nearest electrode #{idx_gt} ({d_gt[idx_gt]:.1f} mm away)")

    pv.set_plot_theme("dark")
    pl = pv.Plotter(title=f"Torso -- right-click up to {n_points} points, "
                          f"then close to compare")
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
        show_message   = (f"Right-click up to {n_points} points on the torso.\n"
                           f"Close the window when done to compare traces."),
        color          = "magenta",
        point_size     = 12,
    )
    pl.add_legend(bcolor=None, border=False)
    pl.add_axes()
    pl.add_title(f"Right-click up to {n_points} points, close window to compare")
    pl.show()

    try:
        pl.close()
    except Exception:
        pass

    if len(picked_xyz) == 0:
        print("[pick_and_compare_multi] No points were picked. Nothing to plot.")
        return None

    n_picked = len(picked_xyz)
    if n_picked < n_points:
        print(f"[pick_and_compare_multi] Only {n_picked}/{n_points} points picked "
              f"-- proceeding with what was picked.")

    xyz_arr = np.stack(picked_xyz, axis=0)   # (n_picked, 3)

    # ── Nearest sim node / GT electrode per picked point ────────────────
    idx_sim_list, idx_gt_list, dist_sim_list, dist_gt_list = [], [], [], []
    for xyz in picked_xyz:
        d_sim = np.linalg.norm(surface_xyz - xyz, axis=1)
        i_sim = int(np.argmin(d_sim))
        d_gt  = np.linalg.norm(electrode_points - xyz, axis=1)
        i_gt  = int(np.argmin(d_gt))
        idx_sim_list.append(i_sim); dist_sim_list.append(float(d_sim[i_sim]))
        idx_gt_list.append(i_gt);   dist_gt_list.append(float(d_gt[i_gt]))

    # ── CAR both datasets independently (own arbitrary reference removed) ─
    bspm_car = bspm_signal - bspm_signal.mean(axis=0, keepdims=True)
    gt_car   = potvals    - potvals.mean(axis=0, keepdims=True)

    # ── Beat detection: multi-channel RMS-of-derivative, computed ONCE
    #    across the FULL channel set (not just the picked points) --
    #    same logic/rationale as average_beat_compare.py ──────────────────
    rms_dsim = np.sqrt(np.mean(np.diff(bspm_car, axis=1) ** 2, axis=0))
    rms_dgt  = np.sqrt(np.mean(np.diff(gt_car,  axis=1) ** 2, axis=0))
    fid_sim = _detect_fiducials(rms_dsim, min_distance=MIN_RR_SAMPLES_SIM) + 1
    fid_gt  = _detect_fiducials(rms_dgt,  min_distance=MIN_RR_SAMPLES_GT) + 1

    if len(fid_sim) == 0 or len(fid_gt) == 0:
        raise RuntimeError(
            "No beats detected in one of the two datasets -- loosen "
            "PROMINENCE_FRAC or check the CAR'd RMS signal."
        )

    rr_sim = np.median(np.diff(fid_sim)) if len(fid_sim) >= 2 else bspm_car.shape[1] / 2.0
    rr_gt  = np.median(np.diff(fid_gt))  if len(fid_gt)  >= 2 else gt_car.shape[1]   / 2.0
    pre_sim, post_sim = int(round(PRE_FRAC * rr_sim)), int(round(POST_FRAC * rr_sim))
    pre_gt,  post_gt  = int(round(PRE_FRAC * rr_gt)),  int(round(POST_FRAC * rr_gt))

    # ── Averaged beat per picked point ──────────────────────────────────
    avg_sim_native = np.stack([
        _extract_average_beat(bspm_car[i_sim], fid_sim, pre_sim, post_sim)
        for i_sim in idx_sim_list
    ], axis=0)   # (n_picked, pre_sim+post_sim)
    avg_gt_native = np.stack([
        _extract_average_beat(gt_car[i_gt], fid_gt, pre_gt, post_gt)
        for i_gt in idx_gt_list
    ], axis=0)   # (n_picked, pre_gt+post_gt)

    # ── Global (per-dataset, pooled across the n_picked points) z-score
    #    normalization, computed from the AVERAGED beat -- for plotting
    #    only, does not affect the r values below ─────────────────────────
    sim_scale = np.nanstd(avg_sim_native)
    gt_scale  = np.nanstd(avg_gt_native)
    avg_sim_norm = avg_sim_native / sim_scale if sim_scale > 0 else avg_sim_native
    avg_gt_norm  = avg_gt_native  / gt_scale  if gt_scale  > 0 else avg_gt_native

    # ── Resample onto a common fraction-of-window grid (sample-rate-free) ─
    sim_common, x_common = _resample_to_common_grid(avg_sim_norm)
    gt_common,  _         = _resample_to_common_grid(avg_gt_norm)

    # ── Pearson r per point ──────────────────────────────────────────────
    r_values = []
    for k in range(n_picked):
        a, b = sim_common[k], gt_common[k]
        valid = ~(np.isnan(a) | np.isnan(b))
        r_values.append(float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() >= 2 else np.nan)

    print("\n[pick_and_compare_multi] Per-point morphology correlation "
          "(representative averaged beat):")
    for k in range(n_picked):
        print(f"  Point {k+1}: sim node {idx_sim_list[k]:6d} / GT electrode "
              f"#{idx_gt_list[k]:3d}   r = {r_values[k]:.3f}")

    # ── Plot: one panel per picked point ────────────────────────────────
    n_cols = 3 if n_picked > 3 else n_picked
    n_rows = int(np.ceil(n_picked / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    for k in range(n_picked):
        ax = axes[k // n_cols][k % n_cols]
        ax.plot(x_common, sim_common[k], color="#8a1c1c", linewidth=1.8, label="simulated")
        ax.plot(x_common, gt_common[k],  color="black",   linewidth=1.4, label="recorded", alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_title(f"Point {k+1}: sim {idx_sim_list[k]} / GT #{idx_gt_list[k]}\n"
                     f"r = {r_values[k]:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=7)
    # hide unused axes
    for k in range(n_picked, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle(f"{n_picked}-point averaged-beat comparison "
                 f"(CAR + global z-score normalization; common fraction-of-window axis)")
    fig.tight_layout()
    plt.show()

    return dict(
        xyz=xyz_arr,
        idx_sim=np.array(idx_sim_list), idx_gt=np.array(idx_gt_list),
        dist_sim_mm=np.array(dist_sim_list), dist_gt_mm=np.array(dist_gt_list),
        r_values=np.array(r_values),
        avg_sim=sim_common, avg_gt=gt_common, x_common=x_common,
    )
