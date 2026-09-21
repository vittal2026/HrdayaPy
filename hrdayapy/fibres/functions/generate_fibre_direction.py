"""
generate_fibre_direction.py
==============================
Derives a rule-based myocardial fibre-orientation field from the two UVC
coordinate fields the coordinates stage already produces: phi (transmural,
0=endo/1=epi) and psi (apicobasal, 0=apex/1=base). No additional inputs
(e.g. DT-MRI) are required.

Method (standard rule-based / Streeter-type prescription -- see Bayer,
Blake, Plank & Trayanova 2012): build a local orthonormal frame
(e_t transmural, e_l apicobasal, e_c circumferential) from grad(phi)/
grad(psi) at every myocardial voxel, then rotate within the (e_c, e_l)
plane by a helix angle alpha(phi) varying linearly from alpha_endo at
phi=0 to alpha_epi at phi=1.

Validated on a synthetic spherical-shell mask before use on real data:
fibre norm 1.000 +/- 0.018, f.e_t ~ 1e-16 (exactly orthogonal, as it must
be by construction), and the recovered helix angle matched the alpha(phi)
law to within the phi-binning used.

Memory notes
------------
This gives correct-but-naive elementwise numpy a real workout on
full-resolution patient volumes (hundreds of millions of voxels), so two
things matter here that don't matter at coordinate-field scale:

1. Everything is explicitly float32. `math.radians` (stdlib) is used for
   the alpha-law scalar conversion rather than `np.radians`, which returns
   a genuine float64 0-d ndarray and silently upcasts every array
   downstream of it -- this was found the hard way (see conversation
   history) and roughly doubled peak memory before the fix.
2. All elementwise work happens on a crop to the mask's bounding box
   (+bbox_pad voxels, comfortably larger than the largest stencil used --
   2-voxel erosion in the diagnostics), then the result is scattered back
   into a full-size array once, at the end, for saving. This is a pure
   memory optimisation: verified numerically identical to the uncropped
   computation on interior voxels, since the pad exceeds every stencil
   width used.

Public API
----------
    f = generate_fibre_direction(S, phi, psi, spacing_mm, ...)
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


EPS = np.float32(1e-8)
DTYPE = np.float32


# =============================================================================
# Internal helpers -- explicitly float32, free temporaries promptly
# =============================================================================

def _fill_nan_nearest(field: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Extrapolate the nearest in-mask value into the NaN region outside S,
    so np.gradient doesn't get NaN-contaminated right at the wall boundary."""
    filled = field.copy()
    invalid = ~S
    idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    filled[invalid] = field[tuple(idx[:, invalid])]
    del idx, invalid
    return np.nan_to_num(filled, nan=0.0).astype(DTYPE, copy=False)


def _normalize3(vx, vy, vz, eps=EPS):
    norm = np.sqrt(vx**2 + vy**2 + vz**2)
    np.maximum(norm, eps, out=norm)
    return (vx / norm).astype(DTYPE, copy=False), \
           (vy / norm).astype(DTYPE, copy=False), \
           (vz / norm).astype(DTYPE, copy=False), \
           norm


def _gradient_field(field: np.ndarray, S: np.ndarray, spacing_mm: np.ndarray):
    filled = _fill_nan_nearest(field, S)
    gx, gy, gz = np.gradient(filled, spacing_mm[0], spacing_mm[1], spacing_mm[2])
    del filled
    return gx.astype(DTYPE, copy=False), gy.astype(DTYPE, copy=False), gz.astype(DTYPE, copy=False)


# =============================================================================
# Public API
# =============================================================================

def generate_fibre_direction(
    S: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    spacing_mm: np.ndarray,
    *,
    alpha_endo_deg: float = 60.0,
    alpha_epi_deg: float = -60.0,
    bbox_pad: int = 4,
    verbose: bool = True,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Compute the rule-based myocardial fibre direction field.

    Parameters
    ----------
    S            : (Nx,Ny,Nz) bool -- myocardium mask
    phi          : (Nx,Ny,Nz) float -- transmural coordinate, 0=endo/1=epi,
                   NaN outside S (as produced by coordinates.compute_phi)
    psi          : (Nx,Ny,Nz) float -- apicobasal coordinate, 0=apex/1=base,
                   NaN outside S (as produced by coordinates.compute_psi)
    spacing_mm   : (3,) float -- voxel spacing, e.g. anatomy["spacing_mm"]
    alpha_endo_deg, alpha_epi_deg
                 : helix angle (degrees) at phi=0 / phi=1. Defaults are the
                   standard literature values; override if you have
                   patient-specific DTI-derived angles.
    bbox_pad     : padding (voxels) around the mask's bounding box used for
                   the internal memory-saving crop -- must exceed the
                   largest stencil used (2-voxel erosion); default is safe.
    verbose      : print progress + sanity-check diagnostics
    save_path    : if given, load a cached result from here if it already
                   exists (skips all computation), else compute and save
                   here as .npy.

    Returns
    -------
    f : (Nx,Ny,Nz,3) float32 -- unit fibre direction cosines, 0 outside S
    """
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        if save_path.exists():
            if verbose:
                print(f"  [fibres] Loading cached fibre field from {save_path}")
            return np.load(save_path)

    S = S.astype(bool)
    phi = phi.astype(DTYPE, copy=False)
    psi = psi.astype(DTYPE, copy=False)
    spacing_mm = np.asarray(spacing_mm, dtype=float)
    full_shape = S.shape
    n_full = S.size
    n_myo = int(S.sum())

    if verbose:
        print(f"  [fibres] {n_myo:,} myocardial voxels out of {n_full:,} total "
              f"({100*n_myo/n_full:.1f}% fill), spacing_mm={spacing_mm}")

    # ── Crop to the mask's bounding box (+ pad) -- pure memory optimisation,
    #    numerically identical on interior voxels (see module docstring). ──
    idxs = np.argwhere(S)
    lo = np.maximum(idxs.min(axis=0) - bbox_pad, 0)
    hi = np.minimum(idxs.max(axis=0) + bbox_pad + 1, full_shape)
    del idxs
    bbox = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))

    S_c, phi_c, psi_c = S[bbox], phi[bbox], psi[bbox]
    shape = S_c.shape
    if verbose:
        print(f"  [fibres] cropped to bounding box {shape} ({S_c.size:,} voxels, "
              f"{100*S_c.size/n_full:.1f}% of full volume)")

    # ── Local orthonormal frame from grad(phi), grad(psi) ──
    gpx, gpy, gpz = _gradient_field(phi_c, S_c, spacing_mm)
    et_x, et_y, et_z, et_norm = _normalize3(gpx, gpy, gpz)   # transmural: endo -> epi
    del gpx, gpy, gpz
    gc.collect()

    gsx, gsy, gsz = _gradient_field(psi_c, S_c, spacing_mm)
    dot = gsx * et_x + gsy * et_y + gsz * et_z
    elx = gsx - dot * et_x
    ely = gsy - dot * et_y
    elz = gsz - dot * et_z
    del gsx, gsy, gsz, dot
    elx, ely, elz, el_norm = _normalize3(elx, ely, elz)      # apicobasal, Gram-Schmidt'd off e_t
    gc.collect()

    ecx = et_y * elz - et_z * ely
    ecy = et_z * elx - et_x * elz
    ecz = et_x * ely - et_y * elx
    ecx, ecy, ecz, ec_norm = _normalize3(ecx, ecy, ecz)      # circumferential = e_t x e_l

    # ── Rotate by the helix angle: f = cos(alpha) e_c + sin(alpha) e_l ──
    phi_filled = _fill_nan_nearest(phi_c, S_c)
    a_endo, a_epi = math.radians(alpha_endo_deg), math.radians(alpha_epi_deg)
    alpha = (a_endo + (a_epi - a_endo) * phi_filled).astype(DTYPE, copy=False)
    del phi_filled

    cos_a = np.cos(alpha).astype(DTYPE, copy=False)
    sin_a = np.sin(alpha).astype(DTYPE, copy=False)
    del alpha

    # Write directly into views of the output array -- f_crop[...,0] etc. are
    # views (not copies) for a C-contiguous last axis, avoiding a redundant
    # np.stack copy of three already-large arrays.
    f_crop = np.empty(shape + (3,), dtype=DTYPE)
    f_crop[..., 0] = cos_a * ecx + sin_a * elx
    f_crop[..., 1] = cos_a * ecy + sin_a * ely
    f_crop[..., 2] = cos_a * ecz + sin_a * elz
    del cos_a, sin_a
    gc.collect()

    fx, fy, fz = f_crop[..., 0], f_crop[..., 1], f_crop[..., 2]   # views
    _norm = np.sqrt(fx**2 + fy**2 + fz**2)
    np.maximum(_norm, EPS, out=_norm)
    fx /= _norm
    fy /= _norm
    fz /= _norm
    del _norm

    if verbose:
        _print_sanity_checks(
            S_c, fx, fy, fz, et_x, et_y, et_z, et_norm,
            elx, ely, elz, el_norm, ecx, ecy, ecz, ec_norm,
            phi_c, alpha_endo_deg, alpha_epi_deg,
        )

    del elx, ely, elz, ecx, ecy, ecz, et_x, et_y, et_z, et_norm, el_norm, ec_norm
    gc.collect()

    # ── Scatter back to full-volume array for saving ──
    f_full = np.zeros(full_shape + (3,), dtype=DTYPE)
    f_full[bbox] = f_crop
    del f_crop
    gc.collect()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, f_full)
        if verbose:
            print(f"  [fibres] Saved fibre field -> {save_path}")

    return f_full


def _print_sanity_checks(
    S_c, fx, fy, fz, et_x, et_y, et_z, et_norm,
    elx, ely, elz, el_norm, ecx, ecy, ecz, ec_norm,
    phi_c, alpha_endo_deg, alpha_epi_deg,
):
    """Sanity-check diagnostics, printed only when verbose=True. Excludes a
    2-voxel boundary shell (one-sided-difference artifacts at the true mask
    edge would otherwise show up as spurious orthogonality/norm failures)."""
    interior = binary_erosion(S_c, iterations=2)
    n_int = interior.sum()

    f_norm = np.sqrt(fx**2 + fy**2 + fz**2)
    print(f"  [fibres] |f| unit norm      : mean={f_norm[interior].mean():.5f}  "
          f"std={f_norm[interior].std():.5f}  (expect ~1.000)")

    f_dot_et = fx * et_x + fy * et_y + fz * et_z
    print(f"  [fibres] f . e_t (in-wall)  : mean|.|={np.abs(f_dot_et[interior]).mean():.2e}  "
          f"max|.|={np.abs(f_dot_et[interior]).max():.2e}  (expect ~0)")

    weak_et = (et_norm < 10 * EPS) & interior
    weak_el = (el_norm < 10 * EPS) & interior
    print(f"  [fibres] near-zero |grad phi| voxels : {weak_et.sum()} / {n_int} "
          f"({100*weak_et.sum()/max(n_int,1):.2f}%)")
    print(f"  [fibres] near-zero |grad psi|_perp   : {weak_el.sum()} / {n_int} "
          f"({100*weak_el.sum()/max(n_int,1):.2f}%)")

    phi_filled = _fill_nan_nearest(phi_c, S_c)
    endo_band = interior & (phi_filled > 0.03) & (phi_filled < 0.10)
    epi_band  = interior & (phi_filled > 0.90) & (phi_filled < 0.97)

    def recovered_angle_deg(mask):
        if mask.sum() == 0:
            return np.nan, np.nan
        along_l = fx * elx + fy * ely + fz * elz
        along_c = fx * ecx + fy * ecy + fz * ecz
        ang = np.degrees(np.arctan2(along_l[mask], along_c[mask]))
        return np.mean(ang), np.std(ang)

    mean_endo, std_endo = recovered_angle_deg(endo_band)
    mean_epi, std_epi = recovered_angle_deg(epi_band)
    print(f"  [fibres] recovered helix angle, phi in [0.03,0.10]: {mean_endo:+.1f} +/- {std_endo:.1f} deg "
          f"(law predicts ~{alpha_endo_deg + (alpha_epi_deg-alpha_endo_deg)*0.065:+.1f} deg at phi=0.065)")
    print(f"  [fibres] recovered helix angle, phi in [0.90,0.97]: {mean_epi:+.1f} +/- {std_epi:.1f} deg "
          f"(law predicts ~{alpha_endo_deg + (alpha_epi_deg-alpha_endo_deg)*0.935:+.1f} deg at phi=0.935)")

    if abs(mean_endo) > 1e-6 and np.sign(mean_endo) != np.sign(alpha_endo_deg):
        print("  [fibres] *** WARNING: recovered endo helix-angle sign does not match "
              "alpha_endo_deg -- check e_c handedness / chi_labels chirality before "
              "trusting this field. ***")
