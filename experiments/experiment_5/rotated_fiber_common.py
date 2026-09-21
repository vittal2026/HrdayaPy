from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hrdayapy.anisotropic_simulation import compute_anisotropic_coupled
from hrdayapy.anisotropic_simulation.functions import load_vm_snapshots, compute_activation_maps

from rotated_fiber_values import (
    LX_MM, LY_MM, LZ_MM, MYO_PAD_VOX,
    FIBRE_CONFIGS,
    STIM_FACE_THICKNESS_MM, STIM_AMP, STIM_DUR,
    SIGMA_L, SIGMA_T, CM, A_M, ACT_THRESHOLD,
    MARGIN_STIM_MM, MARGIN_FAR_MM, MARGIN_WALL_Y_MM, MARGIN_WALL_Z_MM,
    eikonal_normal_speed,
)

THETA       = 0.5
CG_TOL      = 1e-8
CG_MAX_ITER = 5000
VM_SAVE_DT  = 0.05


STUB_A_MM = np.array([LX_MM + 5.0, LY_MM + 5.0, LZ_MM + 5.0])
STUB_B_MM = np.array([LX_MM + 5.0, LY_MM + 5.0, LZ_MM + 10.0])


def build_rotated_fiber_myocardium(dx_m: float, fibre_direction: np.ndarray):
    Nx = max(2, int(round(LX_MM / dx_m))) + 2 * MYO_PAD_VOX
    Ny = max(2, int(round(LY_MM / dx_m))) + 2 * MYO_PAD_VOX
    Nz = max(2, int(round(LZ_MM / dx_m))) + 2 * MYO_PAD_VOX
    S = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    S[MYO_PAD_VOX:-MYO_PAD_VOX, MYO_PAD_VOX:-MYO_PAD_VOX, MYO_PAD_VOX:-MYO_PAD_VOX] = 1
    Z = S.copy()
    phi = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    fibre_dir = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    fibre_dir[S == 1] = fibre_direction

    print(f"  Myocardium grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} voxels "
          f"(incl. {MYO_PAD_VOX}-voxel padding) @ dx_m={dx_m} mm, "
          f"fibre = ({fibre_direction[0]:+.4f}, {fibre_direction[1]:+.4f}, {fibre_direction[2]:+.4f})")
    return S, Z, phi, fibre_dir


def physical_to_voxel_idx(points_mm: np.ndarray, dx_m: float) -> np.ndarray:
    return points_mm / dx_m + MYO_PAD_VOX


def build_purkinje_stub(dx_m: float):
    points_mm = np.stack([STUB_A_MM, STUB_B_MM], axis=0)
    nodes = physical_to_voxel_idx(points_mm, dx_m).astype(np.float64)
    elements = np.array([[0, 1]], dtype=np.int64)
    activation_times = np.array([0.0, 1.0], dtype=np.float64)
    return nodes, elements, activation_times


def build_face_stimulus(S_shape: tuple, dx_m: float) -> np.ndarray:
    ii, jj, kk = np.indices(S_shape)
    x = (ii - MYO_PAD_VOX) * dx_m
    return (x >= 0) & (x <= STIM_FACE_THICKNESS_MM)


def run_rotated_fiber(dx_m: float, dt: float, T: float, config_name: str,
                       out_dir: Path) -> dict:
    if config_name not in FIBRE_CONFIGS:
        raise ValueError(f"Unknown fibre configuration {config_name!r}; "
                          f"expected one of {list(FIBRE_CONFIGS)}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fibre_direction = FIBRE_CONFIGS[config_name]["direction"]
    tag = f"{config_name}_dx{dx_m}"

    S, Z, phi, fibre_dir = build_rotated_fiber_myocardium(dx_m, fibre_direction)
    nodes, elements, act_times = build_purkinje_stub(dx_m)
    ectopic_region = build_face_stimulus(S.shape, dx_m)
    if not ectopic_region.any():
        raise RuntimeError(
            f"face stimulus region is empty at dx_m={dx_m} -- "
            f"STIM_FACE_THICKNESS_MM ({STIM_FACE_THICKNESS_MM} mm) is smaller than one voxel.")
    print(f"  Full-face stimulus [{config_name}]: {ectopic_region.sum():,} voxels "
          f"(x in [0,{STIM_FACE_THICKNESS_MM}] mm, full y-z extent)")

    stim_protocol = [dict(name="face_stimulus", target="ectopic",
                          amp=STIM_AMP, dur=STIM_DUR, onset=0.0)]

    n_steps = int(round(T / dt))
    vm_save_dt = max(dt, VM_SAVE_DT)

    t0 = time.time()
    results = compute_anisotropic_coupled(
        nodes=nodes, elements=elements, act_times=act_times,
        S=S, Z=Z, phi=phi, fibre_dir=fibre_dir,
        stim_protocol=stim_protocol, ectopic_region=ectopic_region,
        voxel_size=dx_m, dx_p=dx_m,
        sigma_P=0.0, sigma_l=SIGMA_L, sigma_t=SIGMA_T,
        Cm=CM, A_M=A_M, R_P=0.01,
        dt=dt, T=T, theta=THETA,
        stim_len_mm=STIM_FACE_THICKNESS_MM, c_pmj=0.0, n_pmj=1,
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
    myo_coords = myo_coords[:, [1, 0, 2]]


    if myo_maps.activation_times.shape[1] == 0:
        print("    WARNING: no myocardial node activated within T.")
        myo_act = np.full(myo_maps.N_myo, np.nan, dtype=np.float32)
    else:
        myo_act = myo_maps.activation_times[:, 0]


    offset = MYO_PAD_VOX * dx_m
    myo_coords_mm = myo_coords - offset

    return dict(myo_coords_mm=myo_coords_mm, myo_act=myo_act, S=S, dx_m=dx_m,
                config_name=config_name, theta_deg=FIBRE_CONFIGS[config_name]["theta_deg"])


def restricted_region_mask(myo_coords_mm: np.ndarray) -> np.ndarray:
    x, y, z = myo_coords_mm[:, 0], myo_coords_mm[:, 1], myo_coords_mm[:, 2]
    return (
        (x >= MARGIN_STIM_MM) & (x <= LX_MM - MARGIN_FAR_MM) &
        (y >= MARGIN_WALL_Y_MM) & (y <= LY_MM - MARGIN_WALL_Y_MM) &
        (z >= MARGIN_WALL_Z_MM) & (z <= LZ_MM - MARGIN_WALL_Z_MM)
    )


def fit_cv_along_x(myo_coords_mm: np.ndarray, myo_act: np.ndarray,
                    min_nodes: int = 10) -> dict:
    mask = restricted_region_mask(myo_coords_mm) & np.isfinite(myo_act)
    n = int(mask.sum())
    if n < min_nodes:
        raise RuntimeError(
            f"Only {n} myocardial nodes fall inside the restricted CV-fit "
            f"region (need >= {min_nodes}) -- widen LX_MM/LY_MM/LZ_MM, "
            f"shrink the margins, or use a finer dx_m before trusting a fit.")

    x = myo_coords_mm[mask, 0].astype(np.float64)
    t = myo_act[mask].astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, t, rcond=None)

    if slope <= 0:
        raise RuntimeError(
            f"Non-positive activation-time slope ({slope:.4g} ms/mm) fit in "
            f"the restricted region -- activation time is not increasing "
            f"with x there. Check that the stimulus actually captured and "
            f"that the front has not already reflected off the far wall "
            f"before T; a corrupted fit region will silently produce a "
            f"meaningless (or negative) velocity otherwise.")

    v = 1.0 / slope
    pred = A @ np.array([slope, intercept])
    ss_res = float(np.sum((t - pred) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return dict(v_mm_per_ms=float(v), slope_ms_per_mm=float(slope),
                intercept_ms=float(intercept), r2=float(r2), n_nodes=n)


def compute_local_wavefront_normals(myo_coords_mm: np.ndarray, myo_act: np.ndarray,
                                     dx_m: float) -> dict:
    idx = np.rint(myo_coords_mm / dx_m).astype(np.int64)
    idx -= idx.min(axis=0)
    nx, ny, nz = (idx.max(axis=0) + 1).tolist()
    if idx.shape[0] != nx * ny * nz:
        raise RuntimeError(
            f"myo_coords_mm does not fill a dense {nx}x{ny}x{nz} box "
            f"({idx.shape[0]} nodes, expected {nx*ny*nz}) -- "
            f"compute_local_wavefront_normals assumes the solid-box "
            f"geometry build_rotated_fiber_myocardium produces; if the "
            f"myocardial mask has since gained holes/irregular shape this "
            f"grid-reconstruction approach needs a scattered-data (e.g. "
            f"KD-tree local-plane-fit) replacement instead.")

    T = np.full((nx, ny, nz), np.nan, dtype=np.float64)
    T[idx[:, 0], idx[:, 1], idx[:, 2]] = myo_act

    grad = np.full((nx, ny, nz, 3), np.nan, dtype=np.float64)
    for axis, n_ax in enumerate((nx, ny, nz)):
        if n_ax < 2:
            continue
        fwd = np.roll(T, -1, axis=axis)
        bwd = np.roll(T, 1, axis=axis)
        sl_fwd = [slice(None)] * 3; sl_fwd[axis] = -1
        sl_bwd = [slice(None)] * 3; sl_bwd[axis] = 0
        fwd[tuple(sl_fwd)] = np.nan
        bwd[tuple(sl_bwd)] = np.nan

        centred_ok = np.isfinite(fwd) & np.isfinite(bwd)
        g = np.where(centred_ok, (fwd - bwd) / (2.0 * dx_m), np.nan)

        fwd_only = np.isfinite(fwd) & np.isfinite(T) & ~centred_ok
        g = np.where(fwd_only, (fwd - T) / dx_m, g)
        bwd_only = np.isfinite(bwd) & np.isfinite(T) & ~centred_ok & ~fwd_only
        g = np.where(bwd_only, (T - bwd) / dx_m, g)

        grad[..., axis] = g

    grad_flat = grad[idx[:, 0], idx[:, 1], idx[:, 2], :]
    grad_mag = np.linalg.norm(grad_flat, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_local = grad_flat / grad_mag[:, None]
        cn_measured = 1.0 / grad_mag

    return dict(n_local=n_local, cn_measured_mm_per_ms=cn_measured,
                grad_t_ms_per_mm=grad_flat)


def validate_eikonal_pointwise(myo_coords_mm: np.ndarray, myo_act: np.ndarray,
                                dx_m: float, fibre_direction: np.ndarray,
                                v_l: float, v_t: float,
                                rel_error_tolerance: float,
                                min_nodes: int = 10) -> dict:
    mask = restricted_region_mask(myo_coords_mm) & np.isfinite(myo_act)
    normals = compute_local_wavefront_normals(myo_coords_mm, myo_act, dx_m)

    n_local = normals["n_local"][mask]
    cn_measured = normals["cn_measured_mm_per_ms"][mask]
    finite = np.isfinite(cn_measured) & np.all(np.isfinite(n_local), axis=1)
    n_finite = int(finite.sum())
    if n_finite < min_nodes:
        raise RuntimeError(
            f"Only {n_finite} restricted-region nodes have a resolvable "
            f"local gradient (need >= {min_nodes}) -- widen the geometry, "
            f"use a finer dx_m, or check that the front has actually "
            f"crossed the restricted x-range by T.")

    n_local = n_local[finite]
    cn_measured = cn_measured[finite]
    cn_predicted = eikonal_normal_speed(n_local, fibre_direction, v_l, v_t)
    rel_err = np.abs(cn_measured - cn_predicted) / cn_predicted

    rms_rel_error = float(np.sqrt(np.mean(rel_err**2)))
    passed = rms_rel_error < rel_error_tolerance

    stats = dict(
        n_nodes=n_finite,
        mean_rel_error=float(np.mean(rel_err)),
        median_rel_error=float(np.median(rel_err)),
        rms_rel_error=rms_rel_error,
        max_rel_error=float(np.max(rel_err)),
        rel_error_tolerance=rel_error_tolerance,
        passed=bool(passed),
        cn_measured_mean_mm_per_ms=float(np.mean(cn_measured)),
        cn_predicted_mean_mm_per_ms=float(np.mean(cn_predicted)),
    )

    print("=" * 70)
    print("validate_eikonal_pointwise(): local normal-speed vs. eikonal prediction")
    print("-" * 70)
    print(f"  n nodes (restricted region, finite gradient) = {n_finite}")
    print(f"  relative error: mean={stats['mean_rel_error']*100:.3f}%  "
          f"median={stats['median_rel_error']*100:.3f}%  "
          f"rms={stats['rms_rel_error']*100:.3f}%  max={stats['max_rel_error']*100:.3f}%")
    print(f"  rms rel. error vs. tolerance = {rms_rel_error*100:.3f}% "
          f"(require < {rel_error_tolerance*100:.1f}%)")
    print(f"  [{'PASS' if passed else 'FAIL'}]")
    print("=" * 70)

    return stats


def check_planarity(myo_coords_mm: np.ndarray, myo_act: np.ndarray, dx_m: float,
                     x_probe_frac: float = 0.5, x_tol_mm: float | None = None,
                     flat_rel_sd_threshold: float = 0.02) -> dict:
    if x_tol_mm is None:
        x_tol_mm = max(2.0 * dx_m, 0.2)

    x_lo, x_hi = MARGIN_STIM_MM, LX_MM - MARGIN_FAR_MM
    if x_hi <= x_lo:
        raise RuntimeError(
            f"MARGIN_STIM_MM ({MARGIN_STIM_MM}) and MARGIN_FAR_MM ({MARGIN_FAR_MM}) "
            f"leave no valid x-range within LX_MM={LX_MM} to probe.")
    x_probe = x_lo + x_probe_frac * (x_hi - x_lo)

    x, y, z = myo_coords_mm[:, 0], myo_coords_mm[:, 1], myo_coords_mm[:, 2]
    finite = np.isfinite(myo_act)
    slab = (np.abs(x - x_probe) <= x_tol_mm) & finite
    n_slab = int(slab.sum())
    if n_slab < 5:
        raise RuntimeError(
            f"Only {n_slab} finite-activation nodes found within {x_tol_mm:.3g} mm "
            f"of x={x_probe:.3g} mm -- widen x_tol_mm, use a finer dx_m, or check "
            f"that the front has actually reached this x by T.")

    x_s, y_s, z_s, t_s = x[slab], y[slab], z[slab], myo_act[slab]


    if x_s.max() > x_s.min():
        A_local = np.stack([x_s, np.ones_like(x_s)], axis=1)
        (slope_local, intercept_local), _, _, _ = np.linalg.lstsq(A_local, t_s, rcond=None)
        t_s = t_s - (slope_local * x_s + intercept_local) + t_s.mean()

    interior = (
        (z_s >= MARGIN_WALL_Z_MM) & (z_s <= LZ_MM - MARGIN_WALL_Z_MM) &
        (y_s >= MARGIN_WALL_Y_MM) & (y_s <= LY_MM - MARGIN_WALL_Y_MM)
    )
    near_wall = ~interior

    def _stats(t: np.ndarray) -> dict:
        if t.size == 0:
            return dict(n=0, mean_ms=float("nan"), sd_ms=float("nan"), range_ms=float("nan"))
        return dict(
            n=int(t.size),
            mean_ms=float(t.mean()),
            sd_ms=float(t.std(ddof=1)) if t.size > 1 else 0.0,
            range_ms=float(t.max() - t.min()),
        )

    interior_stats = _stats(t_s[interior])
    near_wall_stats = _stats(t_s[near_wall])

    print("=" * 70)
    print(f"check_planarity(): activation-time spread across the y-z cross-section "
          f"at x~={x_probe:.3g} mm (+/-{x_tol_mm:.3g} mm)")
    print("-" * 70)
    print(f"  interior band  (|wall dist| > margins): n={interior_stats['n']:4d}  "
          f"sd={interior_stats['sd_ms']:.4g} ms  range={interior_stats['range_ms']:.4g} ms")
    print(f"  near-wall band (excluded by margins):   n={near_wall_stats['n']:4d}  "
          f"sd={near_wall_stats['sd_ms']:.4g} ms  range={near_wall_stats['range_ms']:.4g} ms")

    rel_spread = float("nan")
    flat = False
    if interior_stats["n"] > 1 and interior_stats["mean_ms"] > 0:
        rel_spread = interior_stats["sd_ms"] / interior_stats["mean_ms"]
        flat = rel_spread < flat_rel_sd_threshold
        verdict = "OK -- interior band looks flat" if flat else\
            "WARNING -- interior band is NOT flat; widen MARGIN_WALL_Z_MM / LZ_MM before trusting the CV fit"
        print(f"  interior relative spread (sd/mean) = {rel_spread*100:.3f}%  [{verdict}]")
    else:
        print("  interior relative spread: not enough interior nodes to judge -- "
              "widen LZ_MM/LY_MM or shrink MARGIN_WALL_Z_MM/MARGIN_WALL_Y_MM.")
    print("=" * 70)

    return dict(
        x_probe_mm=float(x_probe), x_tol_mm=float(x_tol_mm),
        interior=interior_stats, near_wall=near_wall_stats,
        interior_rel_spread=rel_spread, interior_flat=flat,
    )
