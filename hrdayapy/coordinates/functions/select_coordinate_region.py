"""
select_coordinate_region.py
============================
Selects a region of the myocardium directly in (psi, phi, chi, theta)
coordinate space, by intersecting a target +/- tolerance window on
each of the four fields. This is the general building block for
defining ectopic stimulus sites -- or any other coordinate-addressed
region (scar, a fibrotic patch, a recording electrode footprint) --
anatomically, instead of by raw voxel index.

Each coordinate constraint is optional and independent: leave a target
as None to not constrain that coordinate at all. NaN coordinate values
(i.e. outside the myocardium) automatically fail every constraint, so
the result is always confined to tissue without an explicit mask
intersection.

Public API
----------
    region   = select_coordinate_region(psi=..., phi=..., chi=..., theta=..., ...)
    (i, j, k) = region_centroid_voxel(region)
"""

from __future__ import annotations

import numpy as np


def _circular_distance_deg(a_deg: np.ndarray, b_deg: float) -> np.ndarray:
    """Shortest distance between angles a and b on a circle, in degrees."""
    d = np.mod(a_deg - b_deg + 180.0, 360.0) - 180.0
    return np.abs(d)


def select_coordinate_region(
    psi: np.ndarray | None = None, psi_target: float | None = None, psi_tol: float = 0.05,
    phi: np.ndarray | None = None, phi_target: float | None = None, phi_tol: float = 0.05,
    chi: np.ndarray | None = None, chi_target: float | None = None, chi_tol: float = 0.5,
    theta: np.ndarray | None = None, theta_target_deg: float | None = None,
    theta_tol_deg: float = 10.0,
) -> np.ndarray:
    """
    Build a boolean voxel mask selecting the region of myocardium whose
    coordinates fall within the given target +/- tolerance windows.

    Parameters
    ----------
    psi, phi, chi, theta : (Nx,Ny,Nz) float arrays or None
        The coordinate fields (NaN outside the myocardium). Pass only
        the ones you want to constrain; the rest can be omitted.
    psi_target, phi_target : float in [0,1], or None (unconstrained)
        psi: 0 = apex, 1 = base.  phi: 0 = endocardium, 1 = epicardium.
    chi_target : float, 0.0 (LV) or 1.0 (RV), or None (unconstrained)
    theta_target_deg : float in [0,360), or None (unconstrained)
        Target rotational angle in DEGREES (theta itself is stored in
        radians in [0, 2*pi) -- degrees are just easier to specify by
        hand). Wraparound is handled correctly (e.g. target=355,
        tol=10 spans 345..360 and 0..5 as one continuous window).
    psi_tol, phi_tol : float, default 0.05
        Half-width of the [target-tol, target+tol] window.
    chi_tol : float, default 0.5
        Half-width of the chi window. Since chi is strictly binary
        (0.0 or 1.0), the default of 0.5 just means "whichever chamber
        is closer" -- there's normally no reason to change it.
    theta_tol_deg : float, default 10.0
        Half-width of the angular window in degrees.

    Returns
    -------
    region : (Nx,Ny,Nz) bool
        True where every supplied constraint is satisfied.

    Example
    -------
    A basal, epicardial, LV ectopic focus at theta = 180 degrees +/- 10:

        region = select_coordinate_region(
            psi=psi, psi_target=1.0, psi_tol=0.05,
            phi=phi, phi_target=1.0, phi_tol=0.05,
            chi=chi, chi_target=0.0,
            theta=theta, theta_target_deg=180.0, theta_tol_deg=10.0,
        )
    """
    fields = {"psi": psi, "phi": phi, "chi": chi, "theta": theta}
    provided = [f for f in fields.values() if f is not None]
    if not provided:
        raise ValueError("At least one coordinate field must be provided.")
    shape = provided[0].shape
    for name, f in fields.items():
        if f is not None and f.shape != shape:
            raise ValueError(f"{name}.shape {f.shape} != {shape}")

    region = np.ones(shape, dtype=bool)

    # NaN comparisons evaluate to False in numpy, so voxels outside the
    # myocardium (NaN coordinates) are excluded automatically by every
    # constraint below -- no explicit ~isnan() check is needed.
    if psi is not None and psi_target is not None:
        region &= np.abs(psi - psi_target) <= psi_tol
    if phi is not None and phi_target is not None:
        region &= np.abs(phi - phi_target) <= phi_tol
    if chi is not None and chi_target is not None:
        region &= np.abs(chi - chi_target) <= chi_tol
    if theta is not None and theta_target_deg is not None:
        theta_deg = np.degrees(theta)
        region &= _circular_distance_deg(theta_deg, theta_target_deg) <= theta_tol_deg

    return region


def region_centroid_voxel(region: np.ndarray) -> tuple[int, int, int]:
    """
    A single representative voxel for a region -- the actual selected
    voxel closest to the region's geometric centroid (not just the
    mean index, which can fall outside the region for a concave or
    crescent-shaped selection).

    Useful for seeding a point-like stimulus; for an area stimulus,
    use the `region` mask directly instead.
    """
    coords = np.argwhere(region)
    if coords.size == 0:
        raise RuntimeError(
            "Region is empty -- cannot compute a centroid. Widen the "
            "target tolerances or check that the coordinate fields were "
            "computed from the same mask."
        )
    centroid = coords.mean(axis=0)
    d2 = np.sum((coords - centroid) ** 2, axis=1)
    nearest = coords[np.argmin(d2)]
    return int(nearest[0]), int(nearest[1]), int(nearest[2])
