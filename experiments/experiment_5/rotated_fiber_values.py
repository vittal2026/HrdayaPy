from __future__ import annotations
import numpy as np

from niederer_values import (
    LX_MM, LY_MM, LZ_MM, MYO_PAD_VOX,
    SIGMA_L, SIGMA_T, CM, A_M, ACT_THRESHOLD,
    STIM_AMP, STIM_DUR,
)


FIBRE_CONFIGS: dict[str, dict] = {
    "longitudinal": dict(direction=np.array([1.0, 0.0, 0.0]), theta_deg=0.0),
    "transverse":   dict(direction=np.array([0.0, 0.0, 1.0]), theta_deg=90.0),
    "rotated45":    dict(direction=np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), theta_deg=45.0),
}
CONFIG_ORDER = ["longitudinal", "transverse", "rotated45"]


STIM_FACE_THICKNESS_MM = 1.0


MARGIN_STIM_MM   = 3.0
MARGIN_FAR_MM    = 3.0
MARGIN_WALL_Y_MM = 0.5
MARGIN_WALL_Z_MM = 0.6


DX_M  = 0.1
DT_MS = 0.5


T = 150.0


REL_ERROR_TOLERANCE = 0.05


def predicted_v45(v_l: float, v_t: float) -> float:
    return float(np.sqrt((v_l**2 + v_t**2) / 2.0))


def predicted_v_theta(v_l: float, v_t: float, theta_deg: float) -> float:
    theta = np.deg2rad(theta_deg)
    return float(np.sqrt(v_l**2 * np.cos(theta)**2 + v_t**2 * np.sin(theta)**2))


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / norm


def eikonal_normal_speed(n: np.ndarray, f: np.ndarray, v_l: float, v_t: float) -> np.ndarray:
    n = _unit(np.asarray(n, dtype=np.float64))
    f = _unit(np.asarray(f, dtype=np.float64))
    n_dot_f = np.sum(n * f, axis=-1)
    return np.sqrt(v_l**2 * n_dot_f**2 + v_t**2 * (1.0 - n_dot_f**2))


def eikonal_velocity_vector(n: np.ndarray, f: np.ndarray, v_l: float, v_t: float) -> np.ndarray:
    n = _unit(np.asarray(n, dtype=np.float64))
    f = _unit(np.asarray(f, dtype=np.float64))
    n_dot_f = np.sum(n * f, axis=-1)
    cn = eikonal_normal_speed(n, f, v_l, v_t)
    n_perp = n - n_dot_f[..., None] * f
    v_vec = (v_l**2 * n_dot_f[..., None] * f + v_t**2 * n_perp) / cn[..., None]
    return v_vec


def print_rotated_fiber_values() -> None:
    print("=" * 70)
    print("rotated_fiber_values.py -- Experiment 5 parameters in effect")
    print("-" * 70)
    print(f"  geometry      = {LX_MM} x {LY_MM} x {LZ_MM} mm  (Niederer N-version geometry, reused as-is)")
    print(f"  SIGMA_L       = {SIGMA_L:.6g} mS/mm  (from niederer_values.py)")
    print(f"  SIGMA_T       = {SIGMA_T:.6g} mS/mm  (ratio {SIGMA_L/SIGMA_T:.3g}:1)")
    print(f"  fibre configs = {CONFIG_ORDER}")
    print(f"  stim face     = x in [0, {STIM_FACE_THICKNESS_MM}] mm, full y-z extent")
    print(f"  CV-fit margins (mm): stim={MARGIN_STIM_MM}, far={MARGIN_FAR_MM}, "
          f"wall_y={MARGIN_WALL_Y_MM}, wall_z={MARGIN_WALL_Z_MM}  [verify via check_planarity()]")
    print(f"  DX_M = {DX_M} mm, DT_MS = {DT_MS} ms  (single resolution -- no convergence sweep)")
    print(f"  T             = {T:.6g} ms")
    print(f"  REL_ERROR_TOLERANCE = {REL_ERROR_TOLERANCE*100:.1f}%")
    print("  Analytical target: v(theta)^2 = vL^2 cos^2(theta) + vT^2 sin^2(theta)")
    print("  (corrected 'velocity ellipse' form -- NOT the reciprocal 'slowness ellipse')")
    print("=" * 70)


if __name__ == "__main__":
    print_rotated_fiber_values()
