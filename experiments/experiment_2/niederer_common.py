"""
niederer_common.py
===============================================================================
Geometry, stimulus, and activation-time extraction for reproducing the
Niederer et al. (2011) N-version benchmark on the ANISOTROPIC solver. This
is a SEPARATE module from experiment_common.py, not an extension of it --
different geometry (bare slab, no Purkinje-driven stimulus), different
solver (hrdayapy.anisotropic_simulation, not hrdayapy.simulation), and a
different validation goal (absolute accuracy against niederer_values.py's
REFERENCE_ACTIVATION_TIMES_MS, not internal grid-refinement
self-consistency) -- see niederer_values.py's own module docstring for the
full rationale.

Reused, unmodified, from experiment_common.py's established patterns
(same conventions, deliberately not re-derived from scratch):
  - physical-mm <-> padded-voxel-index conversion (physical_to_voxel_idx)
  - the "cube/ball stimulus region as a boolean mask, built from the
    padded array's own MYO_PAD_VOX/dx_m" pattern (build_ectopic_region)
  - the load_vm_snapshots -> compute_activation_maps activation-time
    extraction pipeline (run_one)
Not reused: the Purkinje-fibre-driven root stimulus, the linear CV fit
(estimate_myocardial_cv) -- neither applies here.
===============================================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hrdayapy.anisotropic_simulation import compute_anisotropic_coupled
from hrdayapy.anisotropic_simulation.functions import load_vm_snapshots, compute_activation_maps

from niederer_values import (
    LX_MM, LY_MM, LZ_MM, MYO_PAD_VOX, FIBRE_DIRECTION,
    STIM_CUBE_SIDE_MM, STIM_AMP, STIM_DUR,
    SIGMA_L, SIGMA_T, CM, A_M, ACT_THRESHOLD,
    corner_points_mm,
)

THETA       = 0.5
CG_TOL      = 1e-8
CG_MAX_ITER = 500
VM_SAVE_DT  = 0.05   # ms -- same rationale as experiment_common.py: dense
                     # enough for accurate threshold-crossing interpolation
                     # without snapshot memory scaling as T/dt x N_myo.

# Inert Purkinje stub -- placed well outside the slab (near the P8 corner,
# elevated further beyond it) so it plays no geometric role either;
# c_pmj=0.0 in run_niederer() below is what actually guarantees
# inertness (zero coupling weight regardless of placement -- see
# _build_global_laplacian_gpu's w_pmj = c_pmj / n_patch_nodes), this
# placement is belt-and-suspenders on top of that, not a substitute for it.
STUB_A_MM = np.array([LX_MM + 5.0, LY_MM + 5.0, LZ_MM + 5.0])
STUB_B_MM = np.array([LX_MM + 5.0, LY_MM + 5.0, LZ_MM + 10.0])


# =============================================================================
# Geometry
# =============================================================================
def build_niederer_myocardium(dx_m: float):
    """
    Bare 20x7x3mm slab (niederer_values.LX_MM/LY_MM/LZ_MM), 1-voxel
    zero-padded, fibre uniform along the long (X) axis. S=Z (no cutting --
    synthetic benchmark, not anatomical), phi=0 everywhere (homogeneous
    single cell type, matching standard benchmark practice, same as
    experiment_common.py's build_myocardium).
    """
    Nx = max(2, int(round(LX_MM / dx_m))) + 2 * MYO_PAD_VOX
    Ny = max(2, int(round(LY_MM / dx_m))) + 2 * MYO_PAD_VOX
    Nz = max(2, int(round(LZ_MM / dx_m))) + 2 * MYO_PAD_VOX
    S = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    S[MYO_PAD_VOX:-MYO_PAD_VOX, MYO_PAD_VOX:-MYO_PAD_VOX, MYO_PAD_VOX:-MYO_PAD_VOX] = 1
    Z = S.copy()
    phi = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    fibre_dir = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    fibre_dir[S == 1] = FIBRE_DIRECTION   # uniform, along array axis 0 (= long/X axis)

    print(f"  Myocardium grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} voxels "
          f"(incl. {MYO_PAD_VOX}-voxel padding) @ dx_m={dx_m} mm, fibre || X")
    return S, Z, phi, fibre_dir


def physical_to_voxel_idx(points_mm: np.ndarray, dx_m: float) -> np.ndarray:
    """Same convention as experiment_common.py's function of the same
    name: slab-local physical mm -> voxel-index units in the padded mask,
    what compute_anisotropic_coupled's `nodes` argument expects."""
    return points_mm / dx_m + MYO_PAD_VOX


def build_purkinje_stub(dx_m: float):
    """
    Minimal 2-node, 1-edge polyline, entirely outside the slab (see
    STUB_A_MM/STUB_B_MM above). Exists only because
    purkinje_myocardium_anisotropic_solver requires non-empty
    nodes/elements/activation_times positionally -- there is no
    Purkinje-free call path (verified against the solver source; the
    established pattern for a myocardium-only stimulus in this codebase
    is an inert-but-present Purkinje domain, e.g.
    experiment_common.py's antidromic protocol, not an actual empty tree).
    Electrically inert via run_niederer()'s c_pmj=0.0, not via placement
    alone.
    """
    points_mm = np.stack([STUB_A_MM, STUB_B_MM], axis=0)
    nodes = physical_to_voxel_idx(points_mm, dx_m).astype(np.float64)
    elements = np.array([[0, 1]], dtype=np.int64)
    activation_times = np.array([0.0, 1.0], dtype=np.float64)   # dummy, node 0 = root
    return nodes, elements, activation_times


def build_corner_cube_stimulus(S_shape: tuple, dx_m: float) -> np.ndarray:
    """
    Boolean voxel mask, True within the STIM_CUBE_SIDE_MM cube whose
    corner sits at P1=(0,0,0) in the slab's own physical frame (matching
    "[t]he stimulus was applied within the cube marked S" at the P1
    corner, per the benchmark's own figure). AXIS-ALIGNED CUBE, not a
    Euclidean ball -- deliberately different from experiment_common.py's
    build_ectopic_region (which builds a ball, appropriate for its own
    far-field point-stimulus use case, not appropriate here).
    """
    ii, jj, kk = np.indices(S_shape)
    x = (ii - MYO_PAD_VOX) * dx_m
    y = (jj - MYO_PAD_VOX) * dx_m
    z = (kk - MYO_PAD_VOX) * dx_m
    return (x >= 0) & (x <= STIM_CUBE_SIDE_MM) & \
           (y >= 0) & (y <= STIM_CUBE_SIDE_MM) & \
           (z >= 0) & (z <= STIM_CUBE_SIDE_MM)


# =============================================================================
# Single-run solve wrapper
# =============================================================================
def run_niederer(dx_m: float, dt: float, T: float, tag: str, out_dir: Path) -> dict:
    """
    Build the bare-slab + inert-Purkinje-stub geometry, apply the corner-
    cube ectopic stimulus, solve with the anisotropic solver at
    niederer_values.py's SIGMA_L/SIGMA_T, and return per-node myocardial
    activation times + coordinates (Purkinje domain is not used downstream
    at all -- it's present only because the solver requires it).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    S, Z, phi, fibre_dir = build_niederer_myocardium(dx_m)
    nodes, elements, act_times = build_purkinje_stub(dx_m)
    ectopic_region = build_corner_cube_stimulus(S.shape, dx_m)
    if not ectopic_region.any():
        raise RuntimeError(
            f"corner-cube stimulus region is empty at dx_m={dx_m} -- "
            f"STIM_CUBE_SIDE_MM ({STIM_CUBE_SIDE_MM} mm) is smaller than one voxel.")
    print(f"  Corner-cube stimulus: {ectopic_region.sum():,} voxels "
          f"({STIM_CUBE_SIDE_MM} mm cube @ P1)")

    stim_protocol = [dict(name="corner_stimulus", target="ectopic",
                          amp=STIM_AMP, dur=STIM_DUR, onset=0.0)]

    n_steps = int(round(T / dt))
    vm_save_dt = max(dt, VM_SAVE_DT)

    t0 = time.time()
    results = compute_anisotropic_coupled(
        nodes=nodes, elements=elements, act_times=act_times,
        S=S, Z=Z, phi=phi, fibre_dir=fibre_dir,
        stim_protocol=stim_protocol, ectopic_region=ectopic_region,
        voxel_size=dx_m, dx_p=dx_m,   # dx_p irrelevant (stub only), matches dx_m for simplicity
        sigma_P=0.0, sigma_l=SIGMA_L, sigma_t=SIGMA_T,
        Cm=CM, A_M=A_M, R_P=0.01,   # A_P/R_P irrelevant (stub only)
        dt=dt, T=T, theta=THETA,
        stim_len_mm=STIM_CUBE_SIDE_MM, c_pmj=0.0, n_pmj=1,   # c_pmj=0.0 -- see module docstring
        cg_tol=CG_TOL, cg_max_iter=CG_MAX_ITER,
        n_frames=n_steps,
        save_path=out_dir / f"coupled_{tag}.npz",
        vm_save_dt=vm_save_dt,
        vm_save_path=out_dir / f"vm_snapshots_{tag}.npz",
    )
    print(f"    solve wall-clock: {time.time() - t0:.1f} s")

    vm_snap = load_vm_snapshots(out_dir / f"vm_snapshots_{tag}.npz")
    myo_maps = compute_activation_maps(
        vm_snap, S, act_threshold=ACT_THRESHOLD, deact_threshold=ACT_THRESHOLD,
        save_path=out_dir / f"activation_maps_{tag}.npz",
    )
    myo_coords = myo_maps.coords_mm
    myo_coords = myo_coords[:, [1, 0, 2]] # undo solver's Purkinje-tree axis
                                        # convention -- this benchmark's own
                                        # geometry (S/fibre_dir/corner_points_mm)
                                        # is in raw array-axis order throughout
    if myo_maps.activation_times.shape[1] == 0:
        print("    WARNING: no myocardial node activated within T.")
        myo_act = np.full(myo_maps.N_myo, np.nan, dtype=np.float32)
    else:
        myo_act = myo_maps.activation_times[:, 0]

    return dict(myo_coords=myo_coords, myo_act=myo_act, S=S, dx_m=dx_m)


# =============================================================================
# Point / diagonal activation-time extraction
# =============================================================================
def extract_point_activation_times(myo_coords_mm: np.ndarray, myo_act: np.ndarray,
                                    dx_m: float) -> dict:
    """
    Nearest-myocardial-node activation time at each of P1-P9 (nearest-
    neighbour, not exact match -- dx_m won't generally divide LX_MM/LY_MM/
    LZ_MM exactly, so a corner/centre point rarely lands exactly on a
    voxel centre). Returns {point_name: dict(t_ms, nearest_dist_mm)} --
    nearest_dist_mm lets you sanity-check the lookup actually found a
    voxel close to the intended point, not a stand-in from elsewhere in
    the domain.
    """
    offset = MYO_PAD_VOX * dx_m
    out = {}
    for name, p_mm in corner_points_mm().items():
        target = p_mm + offset
        dist = np.linalg.norm(myo_coords_mm - target[None, :], axis=1)
        idx = np.argmin(dist)
        out[name] = dict(t_ms=float(myo_act[idx]), nearest_dist_mm=float(dist[idx]))
    return out


def extract_diagonal_curve(myo_coords_mm: np.ndarray, myo_act: np.ndarray, dx_m: float,
                            n_samples: int = 200) -> dict:
    """
    Activation time along the standard P1->P8 diagonal line, sampled at
    n_samples evenly-spaced points along the line, each via nearest-
    myocardial-node lookup (same approach as extract_point_activation_times,
    applied along a line instead of at 9 fixed points).

    Returns dict(arc_length_mm, t_ms) -- arc_length_mm measured along the
    diagonal from P1 (0 mm) to P8 (full diagonal length).
    """
    offset = MYO_PAD_VOX * dx_m
    pts = corner_points_mm()
    p1, p8 = pts["P1"] + offset, pts["P8"] + offset
    diag_len = np.linalg.norm(p8 - p1)

    s = np.linspace(0.0, 1.0, n_samples)
    line_pts = p1[None, :] + s[:, None] * (p8 - p1)[None, :]

    t_ms = np.empty(n_samples, dtype=np.float32)
    for i, target in enumerate(line_pts):
        dist = np.linalg.norm(myo_coords_mm - target[None, :], axis=1)
        t_ms[i] = myo_act[np.argmin(dist)]

    return dict(arc_length_mm=(s * diag_len).astype(np.float32), t_ms=t_ms)
