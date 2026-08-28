"""
EASI_leads.py
===========================
Interactively pick four points on the SIMULATED torso surface -- E, S, A,
I, in that order -- then plot the three bipolar EASI leads:

    ES = phi(S) - phi(E)
    AS = phi(A) - phi(S)
    AI = phi(A) - phi(I)

Only loads {PATIENT_ID}_bspm.npz (no ground truth, no registration --
keep this next to run_simulation.py and run it any time after RUN_BSPM).

CHANGES from previous version (math/logic unchanged throughout):
  - The four picks (E, S, A, I) are now saved to
    {PATIENT_ID}_easi_picks.json right after picking, and can optionally
    be reloaded on a later run (USE_SAVED_PICKS) instead of re-picking by
    eye every time metrics need to be recomputed or the figure regenerated.
  - The derived bipolar EASI lead signals are now saved to
    {PATIENT_ID}_easi_leads.npz. This is the file ecg_metrics.py reads, and
    is the "source data" the manuscript figure caption should point to.
  - Saved picks are now ALWAYS re-snapped to the CURRENTLY LOADED surface
    by physical (mm) coordinate, never trusted as a raw node index --
    each level's torso grid has its own, differently-sized/ordered
    surface point cloud (see build_torso_grid.py), so an index saved
    against one resolution's surface_xyz is not meaningful, and can even
    be out-of-range, against another's. Only the physical xyz location is
    portable across resolutions; the index is level-specific and is
    re-derived by nearest-neighbour snap every time it's needed. This is
    what makes the new bipolar convergence check below possible: the SAME
    picked electrode location can be snapped onto each level's own
    surface independently.
  - NEW: a bipolar-lead convergence check (RUN_BIPOLAR_CONVERGENCE_CHECK).
    Reuses the E/S/A/I picks made on the primary BSPM (whatever
    resolution BSPM_NPZ points at) as fixed physical (mm) electrode
    locations, re-snaps them onto every {PATIENT_ID}_bspm_conv_lvl*.npz
    produced by run_experiment7_dgx.py's convergence sweep, and compares
    the resulting ES/AS/AI traces across resolutions -- a cheap, targeted,
    clinically-relevant complement to the full-surface/full-volume
    convergence work in run_experiment7_dgx.py, focused on exactly the
    three EASI bipolar signals a clinician-facing comparison actually
    depends on. See RUN_BIPOLAR_CONVERGENCE_CHECK below.

Unlike plot_nodal_lines.py, there is no recording-electrode grid to snap
to here: every one of the ~10^5 points in the simulated surface point
cloud is a valid pick, and each click snaps to whichever one of THOSE is
nearest the cursor.

Controls
--------
  right-click  : hover/pick a candidate point (shown as a yellow marker) --
                 same click convention the rest of the pipeline uses for
                 point picking on this surface
  F            : finalise the current candidate as the next point
                 (E -> S -> A -> I, in that order)
  U            : undo the last finalised point, in case of a mis-click
  Q / close    : once all 4 are finalised, close the window to bring up
                 the bipolar lead plot
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

os.environ.setdefault("VTK_SILENCE_GET_VOID_POINTER_WARNINGS", "1")

# =============================================================================
# Inputs -- edit to match run_simulation.py for this patient
# =============================================================================

PATIENT_ID = "P001"

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "outputs" / PATIENT_ID
OUT        = str(OUT_DIR)

BSPM_NPZ = f"{OUT}/{PATIENT_ID}_bspm.npz"
PICKS_JSON = f"{OUT}/{PATIENT_ID}_easi_picks.json"
LEADS_NPZ = f"{OUT}/{PATIENT_ID}_easi_leads.npz"

# If True and PICKS_JSON exists, reuse the saved E/S/A/I picks instead of
# opening the interactive picker again (e.g. when you just want to redo the
# figure or feed ecg_metrics.py after already picking once). The saved picks
# are re-snapped onto BSPM_NPZ's own surface_xyz by physical coordinate (see
# snap_xyz_to_surface) -- so this is safe even if BSPM_NPZ now points at a
# different resolution than the one the picks were originally made on,
# though accuracy of the snap degrades the coarser/further that resolution
# gap is (a warning prints if any snap distance looks large for the
# current grid's dx).
USE_SAVED_PICKS = True

# --- Bipolar-lead convergence check -----------------------------------------
# After the usual primary-BSPM pick + plot workflow (unchanged above), also
# re-snap those same E/S/A/I physical locations onto every convergence-sweep
# level found on disk and compare the resulting ES/AS/AI traces across
# resolutions. This is a MUCH cheaper, targeted complement to the full-
# surface/full-volume convergence metrics in run_experiment7_dgx.py -- just
# 3 scalar time series per level, no spatial interpolation needed for the
# comparison itself (only for the picking/snapping step, which is a single
# cheap per-level KD-tree nearest-neighbour query).
#
# Requires run_experiment7_dgx.py's Part B (RUN_CONVERGENCE_SWEEP) to have
# already produced {PATIENT_ID}_bspm_conv_lvl*.npz files in OUT_DIR -- if
# none are found, this step is skipped with a one-line notice, not an error.
RUN_BIPOLAR_CONVERGENCE_CHECK = True
CONV_BSPM_GLOB = f"{PATIENT_ID}_bspm_conv_lvl*.npz"
# Warn (not fail) if a pick's nearest available node, at some level, is
# farther than this many multiples of that level's dx away -- a large snap
# distance means the physical location you picked doesn't have a nearby
# surface representative at that resolution (e.g. it landed in a concavity
# or near a boundary feature that's smoothed away at coarser dx), which
# would silently compare two not-quite-the-same anatomical points and
# should be visible in the output rather than hidden.
SNAP_WARN_DX_MULTIPLE = 1.5
BIPOLAR_CONV_RESULTS_NPZ = f"{OUT}/{PATIENT_ID}_bipolar_convergence_results.npz"
BIPOLAR_CONV_PLOT_PNG = f"{OUT}/{PATIENT_ID}_bipolar_convergence.png"

# How many of the finest available levels to overlay in the convergence
# PLOT (the metrics table/order-fit below still use ALL discovered levels --
# this only thins out what's drawn). E.g. 2 = only the two finest levels.
CONVERGENCE_PLOT_N_LEVELS = 2

# =============================================================================
# Settings
# =============================================================================

LABELS = ["E", "S", "A", "I"]
LABEL_COLORS = {"E": "red", "S": "dodgerblue", "A": "limegreen", "I": "orange"}

# =============================================================================
# Lead definitions -- one entry per derived ECG trace.
#
# Each LeadDef bundles everything about ONE plotted trace: what to call it
# internally (`name`, used as the dict/npz key everywhere downstream --
# ecg_metrics.py, {PATIENT_ID}_easi_leads.npz, the convergence-check
# tables), how it's actually computed from the picked E/S/A/I signals
# (`compute`), what to print it as (`label`, the clinical lead name --
# may differ from `name`), and how to draw it on the primary plot
# (`color`, `linewidth`). Fields are independent of one another: redefine
# `compute` to change the formula without touching styling or naming;
# restyle one lead without touching how any lead is computed; rename
# `label` without touching the `name` key that other files key off of.
#
# To add, remove, restyle, or recompute a lead, edit LEAD_DEFS below --
# nothing else in the script needs to change, since every consumer
# (picking-independent computation, the primary plot, the convergence
# check/plot/table) iterates this list rather than hardcoding leads.
# =============================================================================

from dataclasses import dataclass
from typing import Callable


@dataclass
class LeadDef:
    name: str                                   # stable key: dict/npz keys, table columns
    label: str                                  # axis/legend text shown to the user
    compute: Callable[[dict[str, np.ndarray]], np.ndarray]   # phi (label -> signal) -> trace
    color: str = "black"                        # primary-plot trace color
    linewidth: float = 1.2                      # primary-plot trace line width


LEAD_DEFS: list[LeadDef] = [
    LeadDef(name="SE", label="ES", compute=lambda phi: phi["S"] - phi["E"]),
    LeadDef(name="AS", label="AS", compute=lambda phi: phi["A"] - phi["S"]),
    LeadDef(name="AI", label="AI", compute=lambda phi: phi["A"] - phi["I"]),
]

LEAD_BY_NAME: dict[str, LeadDef] = {ld.name: ld for ld in LEAD_DEFS}


def compute_leads(phi: dict[str, np.ndarray],
                   lead_defs: list[LeadDef] = LEAD_DEFS) -> dict[str, np.ndarray]:
    """Apply every LeadDef's `compute` to the picked-point signals `phi`
    (label -> signal), keyed by each LeadDef's `name`. The single place
    that turns the lead definitions above into actual traces -- used by
    both the primary workflow and the per-level convergence check, so
    the two stay in sync automatically whenever LEAD_DEFS changes."""
    return {ld.name: ld.compute(phi) for ld in lead_defs}


def lead_label(name: str, lead_defs: list[LeadDef] = LEAD_DEFS) -> str:
    """Display label for a lead `name`, falling back to the name itself
    if it isn't one of LEAD_DEFS (e.g. a legacy saved results file)."""
    by_name = {ld.name: ld.label for ld in lead_defs}
    return by_name.get(name, name)


POINT_SIZE_MARKER = 16
PICK_TOLERANCE = 0.015   # fraction of render-window diagonal

SAVE_PLOT_PNG = True

# Figure size (width, height) in inches for the saved bipolar-leads plot.
# Lower the width (or raise the height) to make the figure less wide.
BIPOLAR_PLOT_FIGSIZE = (4, 7)

# Title styling for both the primary-plot title and the convergence-plot
# title -- kept as its own small, independently-tweakable block since the
# two titles sit side by side (one per column) and need to fit within a
# narrower width than a single full-figure title would get.
TITLE_FONTSIZE = 10
TITLE_FONTWEIGHT = "bold"
TITLE_FONTFAMILY = "serif"
TITLE_WRAP_CHARS = 34   # wrap width in characters, tuned to BIPOLAR_PLOT_FIGSIZE's column width


def style_title(ax, text: str, wrap_chars: int = TITLE_WRAP_CHARS) -> None:
    """Set an axes title with the shared font styling above, wrapping
    long text onto multiple lines first so it stays inside its column
    instead of overflowing into the neighbouring one (the failure mode
    with a single long unwrapped title at a large sans-serif size)."""
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width=wrap_chars))
    ax.set_title(wrapped, fontsize=TITLE_FONTSIZE, fontweight=TITLE_FONTWEIGHT,
                 fontfamily=TITLE_FONTFAMILY)



# =============================================================================
# Interactive picking
# =============================================================================

def pick_four_points(surface_xyz: np.ndarray) -> dict[str, int]:
    """Opens a PyVista window over the simulated surface point cloud.
    Returns {label: index_into_surface_xyz} once all 4 labels are
    finalised and the window is closed."""
    import pyvista as pv

    tree = cKDTree(surface_xyz)

    state = {
        "stage": 0,                 # index into LABELS: which point we're selecting
        "candidate_idx": None,      # last hovered/clicked point, not yet finalised
        "finalized": {},            # label -> index into surface_xyz
    }

    pv.set_plot_theme("dark")
    pl = pv.Plotter()

    # same rendering convention the rest of the pipeline uses for this
    # exact point cloud (see inspect_pipeline/functions/pick_and_compare.py)
    # Build a solid surface from the point cloud (picking still snaps to the
    # original surface_xyz points via the KDTree below, so this is purely
    # for the opaque visual -- it doesn't change which nodes can be picked).
    cloud = pv.PolyData(surface_xyz.astype(np.float32))
    surf_mesh = cloud.reconstruct_surface()
    pl.add_mesh(surf_mesh, color="#e8b593", opacity=1.0, smooth_shading=True,
                label="Simulated surface")

    def prompt_text() -> str:
        if state["stage"] >= len(LABELS):
            return "All 4 points selected!\nClose this window to view the bipolar leads."
        label = LABELS[state["stage"]]
        done = ", ".join(f"{k}\u2713" for k in state["finalized"]) or "none yet"
        return (f"Click near a point for '{label}', then press F to confirm.\n"
                f"(U = undo last)   Selected so far: {done}")

    def refresh_prompt():
        pl.add_text(prompt_text(), position="upper_left", font_size=12,
                    color="white", name="prompt")

    def on_pick(point):
        if state["stage"] >= len(LABELS):
            return
        _, idx = tree.query(point, k=1)
        state["candidate_idx"] = int(idx)
        snapped = surface_xyz[idx]
        pl.add_point_labels(
            snapped.reshape(1, 3), ["candidate"], name="candidate_marker",
            point_color="yellow", text_color="yellow", point_size=POINT_SIZE_MARKER,
            render_points_as_spheres=True, shape=None, always_visible=True,
        )
        refresh_prompt()

    def finalize():
        if state["stage"] >= len(LABELS) or state["candidate_idx"] is None:
            return
        label = LABELS[state["stage"]]
        idx = state["candidate_idx"]
        state["finalized"][label] = idx
        state["candidate_idx"] = None
        state["stage"] += 1

        pt = surface_xyz[idx].reshape(1, 3)
        pl.add_point_labels(
            pt, [label], name=f"finalized_{label}",
            point_color=LABEL_COLORS[label], text_color=LABEL_COLORS[label],
            point_size=POINT_SIZE_MARKER, render_points_as_spheres=True,
            shape=None, always_visible=True, font_size=20, bold=True,
        )
        pl.remove_actor("candidate_marker")
        print(f"  [{label}] finalised -> surface node {idx} at "
              f"({pt[0,0]:.1f}, {pt[0,1]:.1f}, {pt[0,2]:.1f}) mm")
        refresh_prompt()

    def undo():
        if state["stage"] == 0:
            return
        state["stage"] -= 1
        label = LABELS[state["stage"]]
        removed_idx = state["finalized"].pop(label, None)
        pl.remove_actor(f"finalized_{label}")
        state["candidate_idx"] = None
        print(f"  [{label}] selection undone (was surface node {removed_idx})")
        refresh_prompt()

    pl.enable_point_picking(callback=on_pick, picker="point",
                             left_clicking=False, show_message=False,
                             show_point=False, tolerance=PICK_TOLERANCE)
    pl.add_key_event("f", finalize)
    pl.add_key_event("F", finalize)
    pl.add_key_event("u", undo)
    pl.add_key_event("U", undo)

    refresh_prompt()
    pl.add_axes()
    pl.show()

    if len(state["finalized"]) < len(LABELS):
        missing = [l for l in LABELS if l not in state["finalized"]]
        raise RuntimeError(
            f"Window closed before all 4 points were finalised "
            f"(missing: {missing}). Re-run and press F after each click."
        )
    return state["finalized"]


def save_picks(picks: dict[str, int], surface_xyz: np.ndarray,
               save_path: str) -> None:
    """Persist the manually-picked E/S/A/I node indices and their
    coordinates, so this exact pick is reproducible without re-clicking,
    and so the figure caption can cite where E/S/A/I actually landed."""
    record = {
        label: {
            "index": int(idx),
            "xyz_mm": surface_xyz[idx].round(2).tolist(),
        }
        for label, idx in picks.items()
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved EASI picks -> {save_path}")


def load_picks(save_path: str) -> dict[str, int]:
    """Reload previously-saved E/S/A/I picks (index only; coordinates are
    for reference/logging and are re-read from the bspm npz at use time).

    NOTE: the returned indices are only valid against the EXACT surface_xyz
    they were originally picked on. Do not use them directly against a
    different resolution's BSPM -- use load_picks_xyz + snap_xyz_to_surface
    instead (main() below does this automatically). Kept for backward
    compatibility / any external script that only needs the original index.
    """
    with open(save_path) as f:
        record = json.load(f)
    return {label: entry["index"] for label, entry in record.items()}


def load_picks_xyz(save_path: str) -> dict[str, np.ndarray]:
    """Reload previously-saved E/S/A/I picks as physical (mm) coordinates
    -- the ONLY part of a saved pick that's portable across resolutions.
    Each torso grid level has its own surface point cloud (different
    node count and ordering, see build_torso_grid.py's per-level
    _find_surface_ids), so a saved node INDEX is meaningless -- even
    potentially out of range -- against any surface_xyz other than the
    exact one it was picked from. The physical location is what's
    reusable; re-derive the appropriate index for whichever surface
    you're currently working with via snap_xyz_to_surface.
    """
    with open(save_path) as f:
        record = json.load(f)
    return {label: np.asarray(entry["xyz_mm"], dtype=np.float64)
            for label, entry in record.items()}


def snap_xyz_to_surface(
    picks_xyz: dict[str, np.ndarray], surface_xyz: np.ndarray,
    dx_mm: float | None = None, context: str = "",
    warn_dx_multiple: float = SNAP_WARN_DX_MULTIPLE,
) -> tuple[dict[str, int], dict[str, float]]:
    """Nearest-neighbour snap each physical (mm) pick onto THIS surface's
    own point cloud, returning {label: index_into_surface_xyz} plus the
    per-label snap distance (mm) for transparency/QA.

    This is the operation that makes a pick portable across torso
    resolutions: the physical location stays fixed, and the index it
    resolves to is re-derived fresh for whichever surface_xyz is passed
    in. Prints a warning (does not raise) if a snap distance exceeds
    warn_dx_multiple * dx_mm, since that means the picked location
    doesn't have a nearby representative at this resolution -- e.g. a
    concavity or fine anatomical feature that coarser voxelization
    smoothed away -- and comparing across levels at that point may not
    be comparing quite the same anatomy. dx_mm=None (unknown) skips the
    distance check but still snaps and reports raw distances.
    """
    tree = cKDTree(surface_xyz)
    idx_out: dict[str, int] = {}
    dist_out: dict[str, float] = {}
    label_ctx = f" [{context}]" if context else ""
    for label, xyz in picks_xyz.items():
        dist, idx = tree.query(xyz, k=1)
        idx_out[label] = int(idx)
        dist_out[label] = float(dist)
        flag = ""
        if dx_mm is not None and dist > warn_dx_multiple * dx_mm:
            flag = (f"  [warn] snap distance {dist:.2f} mm exceeds "
                    f"{warn_dx_multiple}x dx ({dx_mm:.2f} mm) -- this "
                    f"pick may not have a comparable representative here")
        print(f"    snap{label_ctx} [{label}]: -> node {idx}, "
              f"{dist:.2f} mm from picked location{flag}")
    return idx_out, dist_out


# =============================================================================
# Bipolar-lead convergence check (targeted, cheap complement to the
# full-surface/full-volume convergence work in run_experiment7_dgx.py)
# =============================================================================

def discover_convergence_levels(out_dir: str, patient_id: str,
                                 glob_pattern: str) -> list[tuple[int, str]]:
    """Find every {patient_id}_bspm_conv_lvl{N}.npz in out_dir, parsed and
    sorted by level index N. Returns [(level, path), ...] sorted ascending
    by level (matching run_experiment7_dgx.py's convention: level 0 =
    coarsest, higher = finer)."""
    pattern = re.compile(rf"{re.escape(patient_id)}_bspm_conv_lvl(\d+)\.npz$")
    found = []
    for p in Path(out_dir).glob(glob_pattern):
        m = pattern.search(p.name)
        if m:
            found.append((int(m.group(1)), str(p)))
    found.sort(key=lambda t: t[0])
    return found


def compute_bipolar_leads_for_level(
    bspm_path: str, picks_xyz: dict[str, np.ndarray], level: int,
) -> dict:
    """Load one convergence-sweep level's BSPM, snap the fixed physical
    picks onto its own surface, and compute the same three bipolar leads
    as the primary workflow (ES, AS, AI). Returns a dict with time_ms,
    leads, dx_mm, snap_indices, snap_distances -- everything needed for
    the cross-level comparison and for a QA audit trail of exactly which
    node each level actually used."""
    data = np.load(bspm_path)
    surface_xyz = data["surface_xyz"].astype(np.float64)
    bspm_signal = data["bspm_signal"].astype(np.float64)
    time_ms = data["time_ms"].astype(np.float64)
    dx_mm = float(data["dx_mm"]) if "dx_mm" in data.files else None

    print(f"  Level {level} (dx={dx_mm if dx_mm is not None else '?'} mm): "
          f"{surface_xyz.shape[0]:,} surface nodes, {bspm_signal.shape[1]:,} frames")
    idx, dist = snap_xyz_to_surface(picks_xyz, surface_xyz, dx_mm=dx_mm,
                                     context=f"level {level}")

    phi = {label: bspm_signal[i, :] for label, i in idx.items()}
    leads = compute_leads(phi)

    return dict(level=level, dx_mm=dx_mm, time_ms=time_ms, leads=leads,
                snap_indices=idx, snap_distances=dist)


def _fit_convergence_order_1d(dx_values: list[float], err_values: list[float]) -> float:
    """log-log slope of err vs dx -- same convention as
    run_experiment7_dgx.py's fit_convergence_order (~1 expected for a
    first-order-accurate discretization; this script is standalone so
    the fit is duplicated here rather than imported)."""
    log_dx = np.log(np.asarray(dx_values))
    log_err = np.log(np.asarray(err_values))
    p, _ = np.polyfit(log_dx, log_err, 1)
    return float(p)


def bipolar_cross_level_metrics(level_results: list[dict]) -> dict:
    """Consecutive-level-pair RMS/L-inf disagreement for each of the three
    bipolar leads, plus a fitted convergence order per lead -- same
    methodological convention (consecutive pairs, log-log order fit) as
    the full-surface/full-volume checks, applied here to just 3 scalar
    time series per level instead of a spatial field, so no interpolation
    is needed for the comparison itself (only the snap step above needed
    one nearest-neighbour query).

    Assumes all levels share the same time_ms (same source Vm snapshots,
    no t_window) -- raises if that assumption doesn't hold rather than
    silently comparing misaligned frames.
    """
    lead_names = list(level_results[0]["leads"].keys())
    t0 = level_results[0]["time_ms"]
    for r in level_results[1:]:
        if len(r["time_ms"]) != len(t0) or not np.allclose(r["time_ms"], t0):
            raise RuntimeError(
                f"time_ms mismatch between level {level_results[0]['level']} "
                f"({len(t0)} frames) and level {r['level']} "
                f"({len(r['time_ms'])} frames) -- these must be IDENTICAL "
                f"for a valid direct comparison (same source Vm snapshots, "
                f"no t_window). If a t_window WAS used for some levels, "
                f"interpolate onto a common time grid before comparing -- "
                f"not done automatically here to avoid silently masking a "
                f"real mismatch."
            )

    pair_results = []
    for i in range(len(level_results) - 1):
        r_coarse, r_fine = level_results[i], level_results[i + 1]
        row = dict(level_coarse=r_coarse["level"], level=r_fine["level"],
                   dx_mm_coarse=r_coarse["dx_mm"], dx_mm=r_fine["dx_mm"])
        for name in lead_names:
            diff = r_coarse["leads"][name] - r_fine["leads"][name]
            row[f"{name}_rms"] = float(np.sqrt(np.mean(diff**2)))
            row[f"{name}_linf"] = float(np.max(np.abs(diff)))
            row[f"{name}_linf_frame"] = int(np.argmax(np.abs(diff)))
        pair_results.append(row)

    orders = {}
    dx_have_all = all(r["dx_mm"] is not None for r in level_results)
    if dx_have_all and len(level_results) >= 3:
        dx_fine_per_pair = [r["dx_mm"] for r in pair_results]
        for name in lead_names:
            linf_per_pair = [r[f"{name}_linf"] for r in pair_results]
            try:
                orders[name] = _fit_convergence_order_1d(dx_fine_per_pair, linf_per_pair)
            except Exception:
                orders[name] = float("nan")

    return dict(lead_names=lead_names, pair_results=pair_results, orders=orders)


def plot_bipolar_convergence(level_results: list[dict], save_path: str | None = None,
                              axes=None):
    """Overlay the plotted levels' ES/AS/AI traces -- one row per lead.

    Styling: the COARSE level is drawn as a solid orange line, and the
    FINE level is drawn as a dotted green line on top of it, so the two
    traces (and any divergence between them) are immediately readable.
    This assumes exactly two levels are passed in (the normal case, via
    CONVERGENCE_PLOT_N_LEVELS = 2); if more than two are passed, the
    extras fall back to light grey so nothing is silently dropped,
    though the coarse/fine styling only really tells its story with two.

    If `axes` is given (one matplotlib Axes per lead, e.g. a column of
    subplots on a figure a caller already built), draws into those
    instead of creating its own figure/window -- used so this plot can
    sit side by side with the main bipolar-leads plot on one shared
    figure. If `axes` is None, creates its own standalone figure (and
    saves it to `save_path` if given).
    """
    import matplotlib.pyplot as plt

    lead_names = list(level_results[0]["leads"].keys())
    n_levels = len(level_results)

    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(lead_names), 1, figsize=BIPOLAR_PLOT_FIGSIZE,
                                  sharex=True, sharey=True)
        if len(lead_names) == 1:
            axes = [axes]

    # coarse -> fine styling: solid orange for coarse, dotted green for fine,
    # drawn ON TOP of the coarse line; any extra middle levels (n_levels > 2)
    # drawn underneath in light grey so nothing is silently dropped
    styles = []
    for i in range(n_levels):
        if i == n_levels - 1:
            styles.append(dict(color="green", linestyle=":", linewidth=1.8, zorder=3))
        elif i == 0:
            styles.append(dict(color="orange", linestyle="-", linewidth=1.4, zorder=2))
        else:
            styles.append(dict(color="0.6", linestyle="-", linewidth=1.0, zorder=1))

    for ax, name in zip(axes, lead_names):
        for i, r in enumerate(level_results):
            dx_label = f"{r['dx_mm']:.2f} mm" if r["dx_mm"] is not None else "?"
            role = "fine" if i == n_levels - 1 else ("coarse" if i == 0 else "mid")
            ax.plot(r["time_ms"], r["leads"][name],
                     label=f"lvl {r['level']} (dx={dx_label}, {role})",
                     alpha=0.9, **styles[i])
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_ylabel(f"{lead_label(name)}\n(mV)")
        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=8, ncol=min(n_levels, 4), loc="upper right")
    axes[-1].set_xlabel("Time (ms)")
    style_title(axes[0], "Coarse vs. fine convergence (same fixed E/S/A/I "
                          "electrode locations, re-snapped per level)")

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved plot -> {save_path}")

    return fig


def run_bipolar_convergence_check(picks_xyz: dict[str, np.ndarray],
                                   axes=None) -> dict | None:
    """Full targeted convergence-check workflow: discover available
    levels, snap the fixed physical picks onto each, compute leads,
    compare consecutive pairs, plot, save results. Returns the results
    dict, or None if fewer than 2 convergence-sweep levels were found
    (nothing to compare).

    If `axes` is given, the coarse/fine overlay is drawn into those
    (see plot_bipolar_convergence) instead of its own standalone
    figure/PNG -- used to put it side by side with the main bipolar
    leads plot on one shared figure."""
    levels = discover_convergence_levels(OUT, PATIENT_ID, CONV_BSPM_GLOB)
    if len(levels) < 2:
        print(f"\n[bipolar convergence check] found {len(levels)} level(s) matching "
              f"'{CONV_BSPM_GLOB}' in {OUT} -- need at least 2 to compare. "
              f"Skipping (run run_experiment7_dgx.py's RUN_CONVERGENCE_SWEEP first "
              f"if you want this check).")
        return None

    print(f"\n=== Bipolar-lead convergence check ({len(levels)} levels found) ===")
    print(f"  Re-snapping the same 4 physical picks (E/S/A/I) onto each level's "
          f"own surface -- an index saved at one resolution is NOT reused "
          f"directly at another (see module docstring).")

    level_results = []
    for level, path in levels:
        level_results.append(compute_bipolar_leads_for_level(path, picks_xyz, level))

    metrics = bipolar_cross_level_metrics(level_results)

    print(f"\n  {'pair':>10} {'dx_coarse':>10} {'dx_fine':>9}", end="")
    for name in metrics["lead_names"]:
        print(f" {lead_label(name)+'_RMS':>10} {lead_label(name)+'_Linf':>11}", end="")
    print()
    for row in metrics["pair_results"]:
        pair_str = f"({row['level_coarse']},{row['level']})"
        dxc = f"{row['dx_mm_coarse']:.3f}" if row['dx_mm_coarse'] is not None else "?"
        dxf = f"{row['dx_mm']:.3f}" if row['dx_mm'] is not None else "?"
        print(f"  {pair_str:>10} {dxc:>10} {dxf:>9}", end="")
        for name in metrics["lead_names"]:
            print(f" {row[name+'_rms']:>10.4f} {row[name+'_linf']:>11.4f}", end="")
        print()

    if metrics["orders"]:
        print(f"\n  Fitted convergence order (log-log slope of L-inf disagreement vs "
              f"fine-level dx, across all {len(metrics['pair_results'])} consecutive "
              f"pairs):")
        for name, p in metrics["orders"].items():
            print(f"    {lead_label(name)}: p={p:.2f}")
    else:
        print(f"\n  (Convergence order not fitted -- need >=3 levels with known dx_mm; "
              f"got {len(level_results)} level(s).)")

    plot_levels = level_results[-CONVERGENCE_PLOT_N_LEVELS:]
    save_path = None if axes is not None else BIPOLAR_CONV_PLOT_PNG
    plot_bipolar_convergence(plot_levels, save_path=save_path, axes=axes)

    save_kwargs = dict(
        levels=np.array([r["level"] for r in level_results], dtype=np.int32),
        dx_mm=np.array([r["dx_mm"] if r["dx_mm"] is not None else np.nan
                        for r in level_results], dtype=np.float64),
        time_ms=level_results[0]["time_ms"],
        pick_labels=np.array(LABELS),
        pick_xyz_mm=np.array([picks_xyz[l] for l in LABELS]),
    )
    for r in level_results:
        for name, trace in r["leads"].items():
            save_kwargs[f"lvl{r['level']}_{name}"] = trace
    for name in metrics["lead_names"]:
        if name in metrics["orders"]:
            save_kwargs[f"order_{name}"] = metrics["orders"][name]
    np.savez(BIPOLAR_CONV_RESULTS_NPZ, **save_kwargs)
    print(f"\n  Saved -> {BIPOLAR_CONV_RESULTS_NPZ}")

    return metrics


def main():
    if not Path(BSPM_NPZ).exists():
        raise FileNotFoundError(
            f"{BSPM_NPZ} not found -- run run_simulation.py with RUN_BSPM = True first."
        )

    print("Loading simulated BSPM ...")
    data = np.load(BSPM_NPZ)
    surface_xyz = data["surface_xyz"].astype(np.float64)     # (N_surf, 3)
    bspm_signal = data["bspm_signal"].astype(np.float64)     # (N_surf, n_frames), mV
    time_ms     = data["time_ms"].astype(np.float64)         # (n_frames,)
    print(f"  {surface_xyz.shape[0]:,} surface points, {bspm_signal.shape[1]:,} frames")

    primary_dx_mm = float(data["dx_mm"]) if "dx_mm" in data.files else None

    if USE_SAVED_PICKS and Path(PICKS_JSON).exists():
        print(f"\nReusing saved EASI picks -> {PICKS_JSON}")
        print(f"  Re-snapping saved physical locations onto THIS surface "
              f"(not reusing the saved index directly -- see module docstring):")
        picks_xyz = load_picks_xyz(PICKS_JSON)
        picks, _ = snap_xyz_to_surface(picks_xyz, surface_xyz, dx_mm=primary_dx_mm,
                                        context="primary")
    else:
        print("\nOpening picking window ...")
        print("  right-click a point, press F to confirm, in order E -> S -> A -> I")
        print("  (U to undo the last confirmed point)\n")
        picks = pick_four_points(surface_xyz)
        save_picks(picks, surface_xyz, PICKS_JSON)

    # Physical (mm) coordinates of the final picks on THIS surface -- the
    # portable representation reused below for the convergence check (and
    # for re-snapping onto any other resolution in future runs).
    picks_xyz = {label: surface_xyz[idx] for label, idx in picks.items()}

    print("\nFinal picks:")
    for label in LABELS:
        idx = picks[label]
        print(f"  {label}: surface node {idx} at {tuple(np.round(surface_xyz[idx], 1))} mm")

    phi = {label: bspm_signal[idx, :] for label, idx in picks.items()}

    leads = compute_leads(phi)

    # =========================================================================
    # Plot -- one shared figure, two columns side by side:
    #   left  = the main simulated bipolar EASI leads (solid black)
    #   right = coarse-vs-fine convergence overlay (solid orange = coarse,
    #           dotted green = fine), if convergence-sweep levels are found
    # instead of two separate windows/PNGs.
    # =========================================================================
    import matplotlib.pyplot as plt

    # decide up front whether a convergence column is even possible, so we
    # know whether to lay out 1 or 2 columns before creating the figure
    do_conv = RUN_BIPOLAR_CONVERGENCE_CHECK
    if do_conv:
        conv_levels_found = discover_convergence_levels(OUT, PATIENT_ID, CONV_BSPM_GLOB)
        do_conv = len(conv_levels_found) >= 2

    n_rows = len(leads)
    n_cols = 2 if do_conv else 1
    fig_w, fig_h = BIPOLAR_PLOT_FIGSIZE
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w * n_cols, fig_h),
                              sharex=True, sharey="col")
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    left_axes = axes[:, 0]
    for ax, ld in zip(left_axes, LEAD_DEFS):
        ax.plot(time_ms, leads[ld.name], color=ld.color, linewidth=ld.linewidth)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel(f"{ld.label}\n(mV)")
        ax.grid(alpha=0.25)
    left_axes[-1].set_xlabel("Time (ms)")
    style_title(left_axes[0], f"Simulated bipolar leads -- E:{picks['E']}  "
                               f"S:{picks['S']}  A:{picks['A']}  I:{picks['I']}")

    # =========================================================================
    # Bipolar-lead convergence check (targeted, cheap complement to the
    # full-surface/full-volume convergence work in run_experiment7_dgx.py)
    # -- reuses these same physical E/S/A/I locations, re-snapped onto
    # every available convergence-sweep level's own surface. Drawn into the
    # right column of the SAME figure above rather than its own window/PNG.
    # =========================================================================
    conv_metrics = None
    if do_conv:
        conv_metrics = run_bipolar_convergence_check(picks_xyz, axes=axes[:, 1])
    elif RUN_BIPOLAR_CONVERGENCE_CHECK:
        # requested but nothing to compare -- still runs (prints its own
        # one-line notice) so the caller sees why the right column is absent
        conv_metrics = run_bipolar_convergence_check(picks_xyz)

    fig.tight_layout()

    if SAVE_PLOT_PNG:
        png_path = f"{OUT}/{PATIENT_ID}_bipolar_leads.png"
        fig.savefig(png_path, dpi=150)
        print(f"\nSaved plot -> {png_path}")

    # =========================================================================
    # Save derived bipolar EASI lead signals so that ecg_metrics.py -- and
    # figure regeneration -- don't need to re-run the picker or reload/
    # recompute from the raw BSPM each time.
    # =========================================================================
    np.savez(
        LEADS_NPZ,
        time_ms=time_ms,
        pick_labels=np.array(LABELS),
        pick_indices=np.array([picks[label] for label in LABELS]),
        pick_xyz_mm=np.array([picks_xyz[label] for label in LABELS]),
        **{f"bipolar_{name}": trace for name, trace in leads.items()},
    )
    print(f"Saved derived leads -> {LEADS_NPZ}")

    plt.show()

    return picks, leads, conv_metrics


if __name__ == "__main__":
    main()
