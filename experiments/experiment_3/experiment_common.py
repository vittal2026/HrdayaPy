#!/usr/bin/env python3
"""
experiments/experiment_common.py
===================================================================
SINGLE shared solver + CV-computation + PMJ-capture core for
Experiments 2, 3, and 4 of the manuscript (grid-resolution
convergence, conductivity scaling, and PMJ capture threshold /
propagation delay respectively).

This file replaces the three previously-separate, copy-pasted
experiment_N/experiment_common.py files. It lives in experiments/,
one level above experiment_2/, experiment_3/, experiment_4/, and is
imported by each run_experimentN.py via a small sys.path adjustment
(see the top of each run_experimentN.py) -- there is now exactly ONE
copy of every geometry/estimator/solver-wrapper function, and exactly
ONE copy of every shared physical/discretisation constant (imported
from base_values.py, also in this directory), removing the drift
between experiments that motivated this refactor:
  - REF_DX_P / REF_DX_M / REF_DT previously disagreed with the grid
    Experiment 2 actually reported as its "level 0" (0.3152 mm vs the
    stale 0.5 mm default) -- now defined once, in base_values.py.
  - SIGMA_P previously risked drifting from the (deleted) production
    config.py without anyone noticing -- now defined once, in
    base_values.py, with print_base_values() available so every run
    echoes the live values to its console output / log.

ANISOTROPIC PORT (this revision)
-----------------------------------------------------------------
The myocardium is now solved with hrdayapy's ANISOTROPIC solver
(hrdayapy.anisotropic_simulation.compute_anisotropic_coupled), not the
isotropic hrdayapy.simulation.compute_coupled these experiments
originally used -- see base_values.py's module docstring for the full
rationale (SIGMA_M is retired; SIGMA_L/SIGMA_T/FIBRE_DIRECTION are
taken directly from niederer_values.py, fibre uniform along Z, the
slab's long axis). Concretely, relative to the pre-anisotropic version
of this file:

  - build_myocardium() now also returns a per-voxel fibre_dir array
    (uniform along FIBRE_DIRECTION), fourth return value.
  - run_one() / run_directional() now call compute_anisotropic_coupled
    with sigma_l/sigma_t + fibre_dir instead of compute_coupled with a
    single sigma_M, and load_vm_snapshots/compute_activation_maps are
    imported from hrdayapy.anisotropic_simulation.functions (matching
    niederer_common.run_niederer's pattern) rather than
    hrdayapy.simulation.functions.
  - Both wrappers now undo the anisotropic solver's own internal
    Purkinje-tree axis convention on the returned myocardial
    coordinates (a swap of array axes 0 and 1) immediately after
    computing activation maps -- see base_values.py's module docstring
    and niederer_common.run_niederer() for the same correction applied
    there. Everything downstream of that point (CV fits, PMJ-coupled-
    node lookup, capture/delay analysis) already assumes the raw,
    un-swapped array-axis frame that P_ROOT/P_GRAZE/P_PMJ are defined
    in, so this correction MUST happen inside run_one/run_directional,
    not downstream.
  - Only run_directional() (Experiment 4's solve wrapper) has actually
    been exercised against the anisotropic solver as part of this
    port -- run_experiment2.py/run_experiment3.py (not touched here)
    still call run_one() positionally with a single conductivity
    argument; that call site will need updating to pass sigma_l/
    sigma_t before Experiments 2/3 are themselves re-run on the
    anisotropic solver, and Experiment 3's sweep design (currently "a
    single sigma_M value") needs its own rethink for two independent
    conductivities -- both deliberately OUT OF SCOPE for this revision,
    which targets Experiment 4 only.

Experiments 2, 3, and 4 all run on the SAME coupled Purkinje-fibre +
myocardial-slab geometry (build_myocardium / build_purkinje below) and
extract conduction velocity from activation times via the SAME
restricted-region linear fit (estimate_myocardial_cv /
estimate_purkinje_cv). They differ only in which parameter is swept:

  - Experiment 2 sweeps discretisation (dt, dx_p, dx_m) at fixed
    conductivity (run_experiment2.py).
  - Experiment 3 sweeps conductivity at fixed discretisation
    (run_experiment3.py) -- sweep design TBD post-anisotropy, see above.
  - Experiment 4 sweeps PMJ coupling conductance (c_pmj) and coupled-
    node count (n_pmj), in both the orthodromic and antidromic
    directions, at fixed discretisation and fixed (sigma_l, sigma_t)
    (run_experiment4.py).

This module owns everything that is identical across all three:

  1. Geometry construction
       build_myocardium, build_purkinje, physical_to_voxel_idx
  2. Shared physical/discretisation parameters
       -- imported from base_values.py, not redefined here.
  3. The single-run solve wrappers
       run_one()          -- fixed c_pmj/n_pmj (Experiments 2, 3)
       run_directional()  -- call-time c_pmj/n_pmj/direction (Exp. 4)
     Both build geometry, call
     hrdayapy.anisotropic_simulation.compute_anisotropic_coupled, and
     extract per-node activation times in both domains.
  4. Purkinje trace assembly / threshold-crossing helpers
       purkinje_ordered_trace, first_crossing_times
  5. CV estimation restricted to the "Eikonal-like" interior region
       estimate_myocardial_cv, estimate_purkinje_cv
  6. PMJ-specific capture/delay analysis (Experiment 4 only; unused by
     but harmless to import for Experiments 2 and 3)
       build_ectopic_region, find_pmj_coupled_myo_nodes,
       pmj_capture_delay, DIRECTIONS, PROPAGATION_CHECK_RADIUS_MM,
       PURKINJE_PROPAGATION_CHECK_MM

run_experiment2.py, run_experiment3.py, and run_experiment4.py are thin
sweep drivers on top of this module: none should reimplement any of
the above.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hrdayapy.anisotropic_simulation import compute_anisotropic_coupled
from hrdayapy.anisotropic_simulation.functions import (
    load_vm_snapshots, compute_activation_maps,
)
# Same interpolated-threshold-crossing routine the myocardial activation
# maps use, applied here to the Purkinje branch traces too, so both
# domains get activation times from an identical algorithm. This helper is
# a generic Vm/t threshold-crossing interpolator, not specific to the
# isotropic solver, so it is still sourced from hrdayapy.simulation
# (there is no anisotropic-specific copy).
from hrdayapy.simulation.functions.activation_maps import _interp_crossings

# =============================================================================
# Shared parameters -- SINGLE SOURCE OF TRUTH, imported from base_values.py.
# Do NOT redefine any of these locally in this file or in any
# run_experimentN.py; if a value needs to change, change it in
# base_values.py so all three experiments pick it up identically.
#
# SIGMA_M is gone (retired -- see base_values.py's module docstring);
# SIGMA_L / SIGMA_T / FIBRE_DIRECTION are the anisotropic myocardial
# conductivity pair + fibre field that replace it.
# =============================================================================
from base_values import (
    SIGMA_P, SIGMA_L, SIGMA_T, FIBRE_DIRECTION, CM, A_M, R_P,
    C_PMJ, N_PMJ, ACT_THRESHOLD,
    REF_DT, REF_DX_P, REF_DX_M,
    STIM_LEN_MM, STIM_AMP, STIM_DUR,
    P_ROOT, P_GRAZE, P_PMJ, MYO_PAD_VOX,
    print_base_values,
)

THETA       = 0.5        # Crank-Nicolson theta
CG_TOL      = 1e-8
CG_MAX_ITER = 500

# Vm snapshots only need to be dense enough to interpolate the -40 mV
# upstroke crossing accurately (see _interp_crossings) -- they do NOT need
# to be saved at the solver's own dt. Saving at dt means snapshot memory
# scales as (T/dt) x N_myo, which blows up at fine resolution levels
# (33.8 GiB at Experiment 2's level 2, dt=0.0125 ms, 630,000 myocardial
# nodes) even though the solve itself completes fine. Capping the save
# cadence at a fixed value keeps snapshot memory roughly constant across
# levels instead of compounding the refinement in both dt and dx_m
# simultaneously. Applied in both run_one() and run_directional() below --
# do not call compute_coupled with vm_save_dt=dt directly anywhere else.
VM_SAVE_DT = 0.05   # ms

# =============================================================================
# Fixed geometry (not swept by any of the three experiments)
# =============================================================================
SLAB_X_MM, SLAB_Y_MM, SLAB_Z_MM = 2.5, 2.5, 50.0   # myocardial slab, long axis z

# Experiment 4: far-field myocardial ectopic stimulus used for the
# ANTIDROMIC (myocardium -> Purkinje) capture/delay protocol. Placed near
# the slab's distal (z=SLAB_Z_MM) face, i.e. ~46 mm from the PMJ at z=0 --
# deliberately "considerably far away" so the retrograde wavefront has
# travelled the full length of the same interior region used for CV
# fitting (see MYO_ENTRY_BUFFER_MM / MYO_EXIT_BUFFER_MM below) and is a
# well-developed planar front by the time it reaches the PMJ, rather than
# a near-field stimulus artefact -- and comparable in magnitude to the
# orthodromic protocol's own root-to-PMJ distance (~50.6 mm), so the two
# directions are not confounded by one having a much shorter "run-up"
# than the other. Not used by Experiments 2/3.
ANTI_STIM_OFFSET_FROM_FAR_MM = 4.0   # set-in distance from the far (z=SLAB_Z_MM) face
ANTI_STIM_RADIUS_MM          = 2.0   # matches STIM_LEN_MM, for a fair comparison
ANTI_STIM_TARGET_MM = np.array(
    [1.25, 1.25, SLAB_Z_MM - ANTI_STIM_OFFSET_FROM_FAR_MM])   # (1.25, 1.25, 46) mm

# =============================================================================
# Output location -- each experiment writes to its own outputs/ directory,
# resolved relative to THIS file's parent's parent (i.e.
# experiments/experiment_N/outputs), not a single shared directory.
# run_experimentN.py passes its own out_dir explicitly to run_one() /
# run_directional(); OUT_DIR here is only a fallback default.
# =============================================================================
OUT_DIR = Path(__file__).resolve().parent / "experiment_2" / "outputs"


# =============================================================================
# Geometry construction
# =============================================================================
def build_myocardium(dx_m: float):
    """
    Full cuboidal slab mask at voxel size dx_m, with a 1-voxel zero-padding
    border (see MYO_PAD_VOX). S = Z (no cutting: this is a synthetic
    benchmark slab, not an anatomical geometry) and phi=0 everywhere
    (homogeneous endocardial TTP06 throughout -- a single cell type,
    matching standard CV-benchmark practice).

    Also returns fibre_dir, a per-voxel (Nx, Ny, Nz, 3) fibre-direction
    field, uniform along base_values.FIBRE_DIRECTION (Z, the slab's long
    axis, SLAB_Z_MM) wherever S==1 -- required by the anisotropic solver
    (hrdayapy.anisotropic_simulation.compute_anisotropic_coupled), which
    takes SIGMA_L/SIGMA_T + this per-voxel fibre field instead of a single
    isotropic SIGMA_M. Same pattern as
    niederer_common.build_niederer_myocardium's fibre_dir construction.

    The slab's own (x=0..SLAB_X_MM, y=0..SLAB_Y_MM, z=0..SLAB_Z_MM) physical
    frame starts at voxel index MYO_PAD_VOX in the padded array -- see
    physical_to_voxel_idx() for the corresponding coordinate transform used
    when placing the Purkinje fibre.
    """
    Nx = max(2, int(round(SLAB_X_MM / dx_m))) + 2 * MYO_PAD_VOX
    Ny = max(2, int(round(SLAB_Y_MM / dx_m))) + 2 * MYO_PAD_VOX
    Nz = max(2, int(round(SLAB_Z_MM / dx_m))) + 2 * MYO_PAD_VOX
    S = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    S[MYO_PAD_VOX:-MYO_PAD_VOX, MYO_PAD_VOX:-MYO_PAD_VOX, MYO_PAD_VOX:-MYO_PAD_VOX] = 1
    Z = S.copy()
    phi = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    fibre_dir = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    fibre_dir[S == 1] = FIBRE_DIRECTION   # uniform, along array axis 2 (= long/Z axis)

    print(f"  Myocardium grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} voxels "
          f"(incl. {MYO_PAD_VOX}-voxel padding) @ dx_m={dx_m} mm, fibre || Z")
    return S, Z, phi, fibre_dir


def physical_to_voxel_idx(points_mm: np.ndarray, dx_m: float) -> np.ndarray:
    """Physical mm coords (in the slab's own x/y/z=0 frame) -> voxel-index
    units in the padded mask built by build_myocardium(), i.e. what
    compute_coupled's `nodes` argument expects (it multiplies by voxel_size
    internally)."""
    return points_mm / dx_m + MYO_PAD_VOX


def build_purkinje(dx_m: float):
    """
    Purkinje fibre as a 2-segment polyline (root -> graze -> PMJ).

    `nodes` must be supplied in the SAME voxel-index units as the
    myocardium grid: hrdayapy's compute_coupled() internally computes
    `nodes_mm = nodes * voxel_size` using the myocardium's own
    `voxel_size` (= dx_m here). So control-point physical mm positions
    are divided by dx_m to land at the correct physical location
    regardless of dx_m.

    `elements` is the polyline's edge list; `activation_times` is a
    dummy monotonically-increasing per-node array only used internally
    to identify the root (the node with the minimum value) -- it plays
    no role in the actual simulated dynamics.
    """
    points_mm = np.stack([P_ROOT, P_GRAZE, P_PMJ], axis=0)   # (3, 3)
    nodes = physical_to_voxel_idx(points_mm, dx_m).astype(np.float64)
    elements = np.array([[0, 1], [1, 2]], dtype=np.int64)
    activation_times = np.array([0.0, 1.0, 2.0], dtype=np.float64)  # dummy, node 0 = root

    seg_lengths = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    print(f"  Purkinje fibre: {seg_lengths.sum():.2f} mm total "
          f"({seg_lengths[0]:.2f} + {seg_lengths[1]:.2f} mm segments)")
    return nodes, elements, activation_times


# =============================================================================
# Purkinje trace assembly / activation-time extraction
# =============================================================================
def purkinje_ordered_trace(results):
    """
    Concatenate the 2 branches' compartment-node voltage traces and
    coordinates into a single root->PMJ ordered sequence (branch 0 ends
    where branch 1 begins -- the shared "graze" node -- so branch 1's
    first entry is dropped to avoid double-counting it).

    Returns
    -------
    coords_mm   : (N_p_ordered, 3) float32, root -> PMJ order
    arc_length  : (N_p_ordered,)   float32, cumulative distance from root [mm]
    Vm          : (n_frames, N_p_ordered) float32
    t           : (n_frames,) float32
    """
    comp_nodes = results["comp_nodes"]          # (N_comp, 3), all compartment nodes
    branch_map = results["branch_map"]          # list of comp-node-id lists, per branch
    branch_frames_p = results["branch_frames_p"]  # list of (n_frames, len(bmap)) arrays
    t = np.asarray(results["times"], dtype=np.float32)

    bmap0, bmap1 = branch_map[0], branch_map[1]
    assert bmap0[-1] == bmap1[0], "expected branch 0 and 1 to share the graze node"
    ordered_ids = list(bmap0) + list(bmap1[1:])

    Vm0 = np.asarray(branch_frames_p[0], dtype=np.float32)   # (n_frames, len(bmap0))
    Vm1 = np.asarray(branch_frames_p[1], dtype=np.float32)   # (n_frames, len(bmap1))
    Vm = np.concatenate([Vm0, Vm1[:, 1:]], axis=1)            # (n_frames, N_p_ordered)

    coords_mm = comp_nodes[ordered_ids].astype(np.float32)
    seg = np.linalg.norm(np.diff(coords_mm, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)

    return coords_mm, arc_length, Vm, t


def first_crossing_times(Vm: np.ndarray, t: np.ndarray, threshold: float) -> np.ndarray:
    """
    First upward threshold crossing per node/column, via the same
    linear-interpolation scheme as hrdayapy's myocardial activation maps
    (activation_maps._interp_crossings). NaN where the node never
    crosses `threshold`.
    """
    edge_mask = (Vm[:-1] < threshold) & (Vm[1:] >= threshold)
    idx_node, t_cross = _interp_crossings(Vm, t, edge_mask, threshold)
    out = np.full(Vm.shape[1], np.nan, dtype=np.float32)
    for node, tc in zip(idx_node, t_cross):
        if np.isnan(out[node]) or tc < out[node]:
            out[node] = tc
    return out


# =============================================================================
# Single-run solve wrapper -- fixed c_pmj/n_pmj (Experiments 2 and 3)
# =============================================================================
def run_one(sigma_l: float, sigma_t: float, sigma_P: float, dt: float, dx_p: float,
            dx_m: float, T: float, tag: str, out_dir: Path,
            delete_intermediate_files: bool = True) -> dict:
    """
    Build the coupled Purkinje-fibre + myocardial-slab geometry at the
    requested (dx_p, dx_m), solve with the requested (sigma_l, sigma_t,
    sigma_P, dt, T) on the ANISOTROPIC solver, and return per-node
    activation-time arrays for both domains. PMJ coupling is fixed at the
    base_values.py defaults (C_PMJ, N_PMJ).

    NOTE on the anisotropic port: this signature has changed (sigma_M ->
    sigma_l, sigma_t) as part of porting Experiment 4 to the anisotropic
    solver (see this module's docstring). run_experiment2.py and
    run_experiment3.py's call sites have NOT been updated as part of this
    change -- they will need to pass sigma_l/sigma_t explicitly before
    Experiments 2/3 themselves run on the anisotropic solver.

    out_dir must be passed explicitly by the caller (each experiment's own
    outputs/ directory) -- there is no shared default across experiments.

    delete_intermediate_files (default True): the three full-field npz
    files this writes (coupled_{tag}.npz, vm_snapshots_{tag}.npz,
    activation_maps_{tag}.npz) are pure scratch -- nothing downstream of
    this function (run_experimentN.py's save_results()) ever reads them
    again; only the scalar/per-node arrays returned here are used. Left
    alone, a full sweep (many tags, each holding O(T/vm_save_dt x N_myo)
    Vm data) accumulates disk usage roughly linearly in sweep size with
    no benefit. Set False only if you specifically want to keep the raw
    Vm fields around afterwards (e.g. for a one-off animation/debug pass
    on a single tag) -- do not disable this globally for a full sweep.
    """
    S, Z, phi, fibre_dir = build_myocardium(dx_m)
    nodes, elements, act_times = build_purkinje(dx_m)

    n_steps = int(round(T / dt))
    stim_protocol = [dict(name="purkinje_root", target="root",
                          amp=STIM_AMP, dur=STIM_DUR, onset=0.0)]

    vm_save_dt = max(dt, VM_SAVE_DT)  # never save Vm snapshots more finely than needed

    coupled_path = out_dir / f"coupled_{tag}.npz"
    vm_path = out_dir / f"vm_snapshots_{tag}.npz"
    act_map_path = out_dir / f"activation_maps_{tag}.npz"

    t0 = time.time()
    results = compute_anisotropic_coupled(
        nodes=nodes, elements=elements, act_times=act_times,
        S=S, Z=Z, phi=phi, fibre_dir=fibre_dir,
        stim_protocol=stim_protocol, ectopic_region=None,
        voxel_size=dx_m, dx_p=dx_p,
        sigma_P=sigma_P, sigma_l=sigma_l, sigma_t=sigma_t,
        Cm=CM, A_M=A_M, R_P=R_P,
        dt=dt, T=T, theta=THETA,
        stim_len_mm=STIM_LEN_MM, c_pmj=C_PMJ, n_pmj=N_PMJ,
        cg_tol=CG_TOL, cg_max_iter=CG_MAX_ITER,
        n_frames=n_steps,
        save_path=coupled_path,
        vm_save_dt=vm_save_dt,
        vm_save_path=vm_path,
    )
    print(f"    solve wall-clock: {time.time() - t0:.1f} s")

    vm_snap = load_vm_snapshots(vm_path)
    myo_maps = compute_activation_maps(
        vm_snap, S, act_threshold=ACT_THRESHOLD, deact_threshold=ACT_THRESHOLD,
        save_path=act_map_path,
    )
    myo_coords = myo_maps.coords_mm
    myo_coords = myo_coords[:, [1, 0, 2]]  # undo anisotropic solver's Purkinje-tree
                                            # axis convention -- see base_values.py's
                                            # module docstring and niederer_common's
                                            # run_niederer() for the same correction
    if myo_maps.activation_times.shape[1] == 0:
        print("    WARNING: no myocardial node activated within T.")
        myo_act = np.full(myo_maps.N_myo, np.nan, dtype=np.float32)
    else:
        myo_act = myo_maps.activation_times[:, 0]

    p_coords, p_arc_length, p_Vm, p_t = purkinje_ordered_trace(results)
    p_act = first_crossing_times(p_Vm, p_t, ACT_THRESHOLD)

    # Free the large in-memory Vm arrays now that activation times have
    # been extracted from them -- don't rely on Python's GC to get to
    # this before the next sweep iteration's arrays are allocated.
    del results, vm_snap, myo_maps, p_Vm

    if delete_intermediate_files:
        for p in (coupled_path, vm_path, act_map_path):
            p.unlink(missing_ok=True)

    return dict(myo_coords=myo_coords, myo_act=myo_act,
                p_arc_length=p_arc_length, p_act=p_act)


# =============================================================================
# CV estimation, restricted to the "Eikonal-like" interior region
# -----------------------------------------------------------------------
# A single global "distance-from-stimulus vs activation-time" fit is not
# appropriate here because several parts of the geometry do NOT obey a
# simple planar (Eikonal-like) activation law and would bias a naive fit:
#
#   Myocardial domain
#     - Lateral (x/y) walls: no-flux boundary conditions locally distort
#       the wavefront curvature near the slab's side faces.
#     - The PMJ-entry (z=0) face and its neighbourhood: the wavefront
#       enters from a near-point-like source (the PMJ, coupled to only
#       N_PMJ=4 myocardial nodes) and is strongly curved/spherical close
#       to it, not planar.
#     - The far (z=SLAB_Z) face: approaching a no-flux boundary, the
#       wavefront can locally decelerate/accelerate before it arrives.
#   Purkinje domain
#     - Near the root: the stimulus itself is applied there, so the
#       earliest activation times reflect stimulus artefacts, not steady
#       free propagation.
#     - Near the PMJ (fibre's distal end): the electrotonic load imposed
#       by the coupled myocardium (which the free-running fibre does not
#       otherwise see) locally perturbs propagation velocity.
#
# Both estimators therefore restrict the linear activation_time-vs-distance
# fit, in each domain, to an interior region defined by explicit exclusion
# buffers, fit CV = 1 / slope there, and report the fit R^2 and number of
# points used so a poor/biased fit is visible rather than silently accepted.
# =============================================================================
MYO_LATERAL_BUFFER_MM = 2.0   # exclude voxels within this distance of the x/y slab walls
MYO_ENTRY_BUFFER_MM   = 8.0   # exclude voxels within this distance of the PMJ-entry (z=0) face
MYO_EXIT_BUFFER_MM    = 8.0   # exclude voxels within this distance of the far (z=SLAB_Z) face
PMJ_RADIUS_EXCLUDE_MM = 6.0   # additionally exclude voxels within this 3D radius of the PMJ itself

PURKINJE_ROOT_BUFFER_MM = 5.0  # exclude fibre arc-length within this distance of the root (stimulus site)
PURKINJE_PMJ_BUFFER_MM  = 5.0  # exclude fibre arc-length within this distance of the PMJ (distal end)

MIN_FIT_POINTS_MYO = 20
MIN_FIT_POINTS_PKJ = 5
R2_WARN_THRESHOLD = 0.99


def estimate_myocardial_cv(coords_mm: np.ndarray, act_ms: np.ndarray, dx_m: float) -> dict:
    """Linear fit of activation_time vs. z (the propagation axis), restricted
    to voxels away from the lateral walls, the PMJ-entry face, the far face,
    and the PMJ itself. CV = 1 / slope [mm/ms]."""
    offset = MYO_PAD_VOX * dx_m
    x0, x1 = offset, offset + SLAB_X_MM
    y0, y1 = offset, offset + SLAB_Y_MM
    z0, z1 = offset, offset + SLAB_Z_MM
    pmj_mm = P_PMJ + offset  # PMJ location in the same (padded-mm) frame as coords_mm

    x, y, z = coords_mm[:, 0], coords_mm[:, 1], coords_mm[:, 2]
    finite = np.isfinite(act_ms)
    lateral_ok = ((x >= x0 + MYO_LATERAL_BUFFER_MM) & (x <= x1 - MYO_LATERAL_BUFFER_MM) &
                  (y >= y0 + MYO_LATERAL_BUFFER_MM) & (y <= y1 - MYO_LATERAL_BUFFER_MM))
    z_ok = (z >= z0 + MYO_ENTRY_BUFFER_MM) & (z <= z1 - MYO_EXIT_BUFFER_MM)
    dist_pmj = np.linalg.norm(coords_mm - pmj_mm[None, :], axis=1)
    pmj_ok = dist_pmj >= PMJ_RADIUS_EXCLUDE_MM

    mask = finite & lateral_ok & z_ok & pmj_ok
    n_used = int(mask.sum())
    if n_used < MIN_FIT_POINTS_MYO:
        return dict(cv_mm_per_ms=np.nan, cv_cm_per_s=np.nan, n_points=n_used, r2=np.nan, slope=np.nan)

    zc, tc = z[mask], act_ms[mask]
    slope, intercept = np.polyfit(zc, tc, 1)   # t = slope*z + intercept
    pred = slope * zc + intercept
    ss_res = np.sum((tc - pred) ** 2)
    ss_tot = np.sum((tc - tc.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    cv = 1.0 / slope
    return dict(cv_mm_per_ms=float(cv), cv_cm_per_s=float(cv * 100.0),
                n_points=n_used, r2=float(r2), slope=float(slope))


def estimate_purkinje_cv(arc_length_mm: np.ndarray, act_ms: np.ndarray) -> dict:
    """Linear fit of activation_time vs. arc length along the fibre,
    restricted to the interior segment away from the root (stimulus site)
    and the PMJ (distal, electrotonically loaded end). CV = 1 / slope."""
    total_len = float(arc_length_mm[-1])
    finite = np.isfinite(act_ms)
    interior = ((arc_length_mm >= PURKINJE_ROOT_BUFFER_MM) &
                (arc_length_mm <= total_len - PURKINJE_PMJ_BUFFER_MM))
    mask = finite & interior
    n_used = int(mask.sum())
    if n_used < MIN_FIT_POINTS_PKJ:
        return dict(cv_mm_per_ms=np.nan, cv_cm_per_s=np.nan, n_points=n_used, r2=np.nan, slope=np.nan)

    s, t = arc_length_mm[mask], act_ms[mask]
    slope, intercept = np.polyfit(s, t, 1)
    pred = slope * s + intercept
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    cv = 1.0 / slope
    return dict(cv_mm_per_ms=float(cv), cv_cm_per_s=float(cv * 100.0),
                n_points=n_used, r2=float(r2), slope=float(slope))


def print_cv_table(title: str, results: list, param_col: str, param_key: str,
                    param_fmt: str = ">16.4g") -> None:
    """Shared tabular printer for all sweeps -- results is a list of dicts
    each containing at least `param_key`, cv_cm_per_s, n_points, r2."""
    print(f"\n{title}")
    print(f"{param_col:>16} {'CV (cm/s)':>12} {'n_pts':>8} {'R^2':>8}")
    for r in results:
        flag = "  <-- low R^2" if (np.isfinite(r["r2"]) and r["r2"] < R2_WARN_THRESHOLD) else ""
        print(f"{r[param_key]:{param_fmt}} {r['cv_cm_per_s']:>12.2f} "
              f"{r['n_points']:>8d} {r['r2']:>8.4f}{flag}")


# =============================================================================
# Experiment 4 (Section 2.11.4): PMJ capture threshold and propagation delay,
# orthodromic and antidromic, as a function of (c_pmj, n_pmj). Unused by
# Experiments 2 and 3, kept here rather than in a separate file so there is
# still only ONE experiment_common.py to keep in sync.
#
# run_directional() below is a drop-in generalisation of run_one() above
# that additionally accepts (c_pmj, n_pmj) as call-time parameters (run_one
# fixes them at the base_values.py defaults C_PMJ/N_PMJ) and a `direction`
# switch between:
#
#   "orthodromic" : the physiological direction, identical stimulus to
#                   run_one() (Purkinje root stimulus; wave crosses the PMJ
#                   from Purkinje -> myocardium).
#   "antidromic"  : a single far-field myocardial ectopic stimulus (see
#                   ANTI_STIM_TARGET_MM / ANTI_STIM_RADIUS_MM above), driven
#                   via compute_coupled's built-in target="ectopic"
#                   mechanism; wave crosses the PMJ from myocardium ->
#                   Purkinje.
#
# Capture and delay are reported per-junction rather than via a global
# timing metric (Section 2.11.4 rationale): this synthetic benchmark has
# exactly one PMJ, so "capture fraction" here is either a fraction over the
# n_pmj coupled myocardial nodes (orthodromic) or a binary 0/1 over the
# single Purkinje terminal (antidromic) -- see pmj_capture_delay().
# =============================================================================
DIRECTIONS = ("orthodromic", "antidromic")

# Distance/arc-length beyond which activation counts as genuine propagation
# away from the junction, rather than local depolarisation of the
# immediately-coupled patch that fails to spread further (liminal-length /
# source-sink mismatch -- see pmj_capture_delay). Chosen independently of
# the CV-fit exclusion buffers above (those define where near-source
# curvature ends for a *velocity* fit; these define where "did it
# propagate at all" is unambiguous) but of comparable magnitude.
PROPAGATION_CHECK_RADIUS_MM = 6.0    # mm from the PMJ, myocardial domain
PURKINJE_PROPAGATION_CHECK_MM = 5.0  # mm of arc length from the PMJ, Purkinje domain


def build_ectopic_region(S_shape: tuple, dx_m: float,
                          target_mm: np.ndarray | None = None,
                          radius_mm: float | None = None) -> np.ndarray:
    """
    Boolean voxel mask, same shape as the padded myocardium grid returned
    by build_myocardium(dx_m), True within `radius_mm` mm (Euclidean, in
    the slab's own physical frame) of `target_mm`. Defaults to the
    antidromic far-field stimulus site (ANTI_STIM_TARGET_MM /
    ANTI_STIM_RADIUS_MM). Passed directly as compute_coupled's
    `ectopic_region` argument for a `target="ectopic"` stimulus entry --
    the same mechanism config.py's own ectopic-focus protocol used to use
    (Section 2.7), just built here from a physical-mm target point instead
    of a UVC-space one, since this synthetic benchmark has no UVC fields.
    """
    if target_mm is None:
        target_mm = ANTI_STIM_TARGET_MM
    if radius_mm is None:
        radius_mm = ANTI_STIM_RADIUS_MM
    ii, jj, kk = np.indices(S_shape)
    # physical mm coords of each voxel centre, slab-local frame -- inverse
    # of physical_to_voxel_idx()
    x = (ii - MYO_PAD_VOX) * dx_m
    y = (jj - MYO_PAD_VOX) * dx_m
    z = (kk - MYO_PAD_VOX) * dx_m
    d2 = (x - target_mm[0]) ** 2 + (y - target_mm[1]) ** 2 + (z - target_mm[2]) ** 2
    return (d2 <= radius_mm ** 2)


def find_pmj_coupled_myo_nodes(myo_coords_mm: np.ndarray, dx_m: float, n_pmj: int) -> np.ndarray:
    """
    Identify the n_pmj myocardial nodes actually coupled to the PMJ, using
    the SAME nearest-neighbour-by-Euclidean-distance rule that
    hrdayapy.simulation.functions.purkinje_myocardium_pipeline.map_pmj uses
    internally to build the real PMJ patch -- so, for a given n_pmj, this
    reconstructs exactly the same patch the solver actually coupled,
    without needing map_pmj's internal pmj_comp_ids/pmj_patches (which are
    not part of compute_coupled's returned results dict).
    """
    offset = MYO_PAD_VOX * dx_m
    pmj_mm = P_PMJ + offset   # PMJ location in the same padded-mm frame as myo_coords_mm
    dist = np.linalg.norm(myo_coords_mm - pmj_mm[None, :], axis=1)
    k = min(n_pmj, len(myo_coords_mm))
    return np.argsort(dist)[:k]


def pmj_capture_delay(myo_coords: np.ndarray, myo_act: np.ndarray, p_act: np.ndarray,
                       p_arc_length: np.ndarray, dx_m: float, n_pmj: int,
                       direction: str) -> dict:
    """
    Local, single-junction capture/delay summary for Experiment 4.

    Reports TWO distinct notions of "capture", because they are not the
    same thing and conflating them is misleading (see below):

    junction_capture_fraction : orthodromic -- fraction of the n_pmj
                        coupled myocardial nodes that activate within T.
                        antidromic -- 0.0/1.0, whether the single Purkinje
                        terminal node activates within T.
                        This is a LOCAL, near-source quantity: because
                        current is injected directly into these few nodes,
                        it is close to 1.0 whenever c_pmj > 0, essentially
                        regardless of whether the excitation goes on to
                        propagate anywhere. On its own it is a poor proxy
                        for "did the PMJ actually work".
    propagating_capture : whether excitation actually reaches BEYOND the
                        immediate junction -- orthodromic: any myocardial
                        node farther than PROPAGATION_CHECK_RADIUS_MM from
                        the PMJ activates within T (i.e. a genuine
                        wavefront was launched into the bulk, not just a
                        local, sub-liminal depolarisation of the coupled
                        patch that decays back down without spreading --
                        the classic liminal-length / source-sink-mismatch
                        failure mode: a source patch too small relative to
                        its diffusively-coupled sink can locally exceed
                        threshold and still fail to propagate). antidromic:
                        Purkinje-side activation reaches farther than
                        PURKINJE_PROPAGATION_CHECK_MM of arc length from
                        the PMJ (i.e. genuinely invades the tree, rather
                        than only depolarising the terminal compartment).
    n_myo_activated     : total count of myocardial nodes that crossed
                        threshold within T (diagnostic; from myo_act
                        directly, independent of n_pmj/direction).
    delay_ms            : receiving-side activation time minus
                        driving-side activation time, computed from the
                        immediate junction only (so it is defined whenever
                        junction_capture_fraction > 0), but only MEANINGFUL
                        as a propagation delay when propagating_capture is
                        also True -- when it is False, delay_ms reflects
                        the timing of a local, non-propagating depolarisation
                        rather than a true PMJ transit time, and should be
                        interpreted (or reported/plotted) accordingly.

    In this geometry, myocardial diffusion has no path to the Purkinje
    fibre other than through the explicit PMJ coupling term (the two
    domains are otherwise disjoint -- see build_purkinje), so any
    Purkinje-side activation recorded here under the antidromic protocol is
    necessarily a genuine transjunctional capture event, not a diffusion
    leakage artefact -- propagating_capture in the antidromic case is
    therefore about retrograde invasion of the TREE, not about whether the
    junction itself fired.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    p_terminal_act = float(p_act[-1]) if np.isfinite(p_act[-1]) else np.nan
    coupled_idx = find_pmj_coupled_myo_nodes(myo_coords, dx_m, n_pmj)
    myo_coupled_act = myo_act[coupled_idx]
    myo_finite = np.isfinite(myo_coupled_act)
    n_myo_activated = int(np.isfinite(myo_act).sum())

    if direction == "orthodromic":
        junction_capture_fraction = float(myo_finite.mean()) if len(myo_coupled_act) else np.nan
        delay_ms = (float(np.mean(myo_coupled_act[myo_finite]) - p_terminal_act)
                    if myo_finite.any() and np.isfinite(p_terminal_act) else np.nan)

        offset = MYO_PAD_VOX * dx_m
        pmj_mm = P_PMJ + offset
        dist_from_pmj = np.linalg.norm(myo_coords - pmj_mm[None, :], axis=1)
        far_field = np.isfinite(myo_act) & (dist_from_pmj >= PROPAGATION_CHECK_RADIUS_MM)
        propagating_capture = bool(far_field.any())
    else:  # antidromic
        junction_capture_fraction = float(np.isfinite(p_terminal_act))
        delay_ms = (float(p_terminal_act - np.mean(myo_coupled_act[myo_finite]))
                    if myo_finite.any() and np.isfinite(p_terminal_act) else np.nan)

        total_len = float(p_arc_length[-1])
        retrograde_reach = ((p_arc_length <= total_len - PURKINJE_PROPAGATION_CHECK_MM) &
                             np.isfinite(p_act))
        propagating_capture = bool(retrograde_reach.any())

    return dict(pmj_coupled_myo_idx=coupled_idx,
                pmj_junction_capture_fraction=junction_capture_fraction,
                pmj_propagating_capture=propagating_capture,
                pmj_n_myo_activated=n_myo_activated,
                pmj_delay_ms=delay_ms)


def run_directional(sigma_l: float, sigma_t: float, sigma_P: float, dt: float,
                     dx_p: float, dx_m: float, T: float, tag: str, c_pmj: float,
                     n_pmj: int, direction: str, out_dir: Path,
                     delete_intermediate_files: bool = True) -> dict:
    """
    Drop-in generalisation of run_one() for Experiment 4: same geometry and
    activation-time extraction, but (c_pmj, n_pmj) are call-time parameters
    rather than fixed at the base_values.py defaults, and `direction`
    selects the orthodromic (Purkinje-root) or antidromic (far-field
    myocardial ectopic) stimulus protocol -- see the module docstring above.

    Runs on the ANISOTROPIC solver (sigma_l/sigma_t + a per-voxel fibre
    field uniform along Z, the slab's long axis, matching base_values.py's
    FIBRE_DIRECTION) -- see this module's docstring for what changed
    relative to the pre-anisotropic version. Experiment 4 fixes
    (sigma_l, sigma_t) at base_values.SIGMA_L/SIGMA_T and sweeps only
    c_pmj/n_pmj/direction, per the manuscript's Section 2.11.4 scope.

    Returns everything run_one() returns, plus the PMJ capture/delay
    summary from pmj_capture_delay().

    delete_intermediate_files (default True): see run_one()'s docstring --
    same rationale applies here, and matters MORE for Experiment 4 since
    its coarse+fine sweep calls this dozens of times per run_experiment4.py
    invocation. run_cell() (run_experiment4.py) only keeps the scalar
    summary dict this function returns; the three full-field npz files
    written per tag are never read again afterwards.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    S, Z, phi, fibre_dir = build_myocardium(dx_m)
    nodes, elements, act_times = build_purkinje(dx_m)

    ectopic_region = None
    if direction == "orthodromic":
        stim_protocol = [dict(name="purkinje_root", target="root",
                              amp=STIM_AMP, dur=STIM_DUR, onset=0.0)]
    else:
        ectopic_region = build_ectopic_region(S.shape, dx_m)
        if not ectopic_region.any():
            raise RuntimeError(
                "antidromic ectopic_region is empty at this dx_m -- widen "
                "ANTI_STIM_RADIUS_MM or check ANTI_STIM_TARGET_MM against SLAB_Z_MM.")
        stim_protocol = [dict(name="myo_far_field", target="ectopic",
                              amp=STIM_AMP, dur=STIM_DUR, onset=0.0)]

    n_steps = int(round(T / dt))
    vm_save_dt = max(dt, VM_SAVE_DT)  # never save Vm snapshots more finely than needed

    coupled_path = out_dir / f"coupled_{tag}.npz"
    vm_path = out_dir / f"vm_snapshots_{tag}.npz"
    act_map_path = out_dir / f"activation_maps_{tag}.npz"

    t0 = time.time()
    results = compute_anisotropic_coupled(
        nodes=nodes, elements=elements, act_times=act_times,
        S=S, Z=Z, phi=phi, fibre_dir=fibre_dir,
        stim_protocol=stim_protocol, ectopic_region=ectopic_region,
        voxel_size=dx_m, dx_p=dx_p,
        sigma_P=sigma_P, sigma_l=sigma_l, sigma_t=sigma_t,
        Cm=CM, A_M=A_M, R_P=R_P,
        dt=dt, T=T, theta=THETA,
        stim_len_mm=STIM_LEN_MM, c_pmj=c_pmj, n_pmj=n_pmj,
        cg_tol=CG_TOL, cg_max_iter=CG_MAX_ITER,
        n_frames=n_steps,
        save_path=coupled_path,
        vm_save_dt=vm_save_dt,
        vm_save_path=vm_path,
    )
    print(f"    [{direction}] solve wall-clock: {time.time() - t0:.1f} s")

    vm_snap = load_vm_snapshots(vm_path)
    myo_maps = compute_activation_maps(
        vm_snap, S, act_threshold=ACT_THRESHOLD, deact_threshold=ACT_THRESHOLD,
        save_path=act_map_path,
    )
    myo_coords = myo_maps.coords_mm
    myo_coords = myo_coords[:, [1, 0, 2]]  # undo anisotropic solver's Purkinje-tree
                                            # axis convention -- see base_values.py's
                                            # module docstring and niederer_common's
                                            # run_niederer() for the same correction
    if myo_maps.activation_times.shape[1] == 0:
        print("    WARNING: no myocardial node activated within T.")
        myo_act = np.full(myo_maps.N_myo, np.nan, dtype=np.float32)
    else:
        myo_act = myo_maps.activation_times[:, 0]

    p_coords, p_arc_length, p_Vm, p_t = purkinje_ordered_trace(results)
    p_act = first_crossing_times(p_Vm, p_t, ACT_THRESHOLD)

    pmj_result = pmj_capture_delay(myo_coords, myo_act, p_act, p_arc_length, dx_m, n_pmj, direction)

    # Free the large in-memory Vm arrays now that activation times have
    # been extracted from them -- don't rely on Python's GC to get to
    # this before the next sweep cell's arrays are allocated.
    del results, vm_snap, myo_maps, p_Vm

    if delete_intermediate_files:
        for p in (coupled_path, vm_path, act_map_path):
            p.unlink(missing_ok=True)

    return dict(myo_coords=myo_coords, myo_act=myo_act,
                p_arc_length=p_arc_length, p_act=p_act,
                direction=direction, c_pmj=c_pmj, n_pmj=n_pmj,
                **pmj_result)
