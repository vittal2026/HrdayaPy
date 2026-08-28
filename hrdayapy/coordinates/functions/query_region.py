"""
query_region.py
===============
Select and visualise an anatomical region of interest (ROI) by querying the
curvilinear coordinates (ψ, θ, φ) in voxel space.

All logic from the standalone query_region.py prototype has been refactored
into two functions with clean signatures so they can be called from the
pipeline or from a notebook without editing path constants:

    roi_mask = build_roi_mask(query, chi_labels, psi, theta, phi=None)
    visualise_roi(roi_mask, query, S, chi_labels, psi, mesh_step=2)

The QUERY dict format (any subset of keys):

    {
        "ventricle"  : "LV" | "RV" | "both"   (default "both")
        "psi"        : (lo, hi)                  e.g. (0.8, 0.9)
        "theta_deg"  : (lo, hi)                  e.g. (60, 80)   — degrees
        "theta_rad"  : (lo, hi)                  e.g. (1.05, 1.40) — radians
        "phi"        : (lo, hi)                  e.g. (0.3, 0.7)  — optional
    }
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()

from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


# =============================================================================
# Helpers
# =============================================================================

def _to_pyvista_surface(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    F = faces.shape[0]
    vtk_faces = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(verts.astype(np.float32), vtk_faces)


def _theta_range_mask(theta_vol: np.ndarray, lo_rad: float, hi_rad: float) -> np.ndarray:
    """
    Boolean volume True where theta ∈ [lo_rad, hi_rad).
    Handles wrap-around across the 0 / 2π seam correctly.
    """
    TWO_PI = 2 * np.pi
    lo = lo_rad % TWO_PI
    hi = hi_rad % TWO_PI

    with np.errstate(invalid="ignore"):
        if lo <= hi:
            return (theta_vol >= lo) & (theta_vol < hi)
        else:
            # wraps: e.g. 350°–20°
            return (theta_vol >= lo) | (theta_vol < hi)


# =============================================================================
# Core: build ROI mask
# =============================================================================

def build_roi_mask(
    query: dict,
    chi_labels: np.ndarray,
    psi: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply *query* to the coordinate volumes and return a boolean mask of the
    same shape.

    Parameters
    ----------
    query       : dict with any subset of "ventricle", "psi", "theta_deg",
                  "theta_rad", "phi" keys (see module docstring).
    chi_labels  : (Nx,Ny,Nz) int   0=bg, 1=LV, 2=RV
    psi         : (Nx,Ny,Nz) float apicobasal coordinate
    theta       : (Nx,Ny,Nz) float rotational coordinate [0, 2π), NaN outside
    phi         : (Nx,Ny,Nz) float transmural coordinate (optional)

    Returns
    -------
    roi : (Nx,Ny,Nz) bool
    """
    ventricle = query.get("ventricle", "both").upper()
    if ventricle == "LV":
        roi = chi_labels == 1
    elif ventricle == "RV":
        roi = chi_labels == 2
    else:
        roi = chi_labels > 0

    roi = roi.copy()
    print(f"  [query] Base ({ventricle}): {roi.sum():,} voxels")

    # ψ filter
    if "psi" in query:
        plo, phi_ = query["psi"]
        roi &= (psi >= plo) & (psi <= phi_)
        print(f"  [query] After ψ ∈ [{plo}, {phi_}]:  {roi.sum():,} voxels")

    # θ filter
    if "theta_deg" in query or "theta_rad" in query:
        if "theta_deg" in query:
            lo_deg, hi_deg = query["theta_deg"]
            lo_rad = np.deg2rad(lo_deg)
            hi_rad = np.deg2rad(hi_deg)
            print(f"  [query] θ filter: {lo_deg}°–{hi_deg}°  "
                  f"({lo_rad:.3f}–{hi_rad:.3f} rad)")
        else:
            lo_rad, hi_rad = query["theta_rad"]
            print(f"  [query] θ filter: {lo_rad:.3f}–{hi_rad:.3f} rad  "
                  f"({np.rad2deg(lo_rad):.1f}°–{np.rad2deg(hi_rad):.1f}°)")

        theta_mask = _theta_range_mask(theta, lo_rad, hi_rad)
        theta_mask &= ~np.isnan(theta)
        roi &= theta_mask
        print(f"  [query] After θ filter:              {roi.sum():,} voxels")

    # φ filter (optional)
    if "phi" in query and phi is not None:
        plo, phi_hi = query["phi"]
        roi &= (phi >= plo) & (phi <= phi_hi)
        print(f"  [query] After φ ∈ [{plo}, {phi_hi}]:  {roi.sum():,} voxels")

    return roi.astype(bool)


# =============================================================================
# Visualisation
# =============================================================================

def visualise_roi(
    roi_mask: np.ndarray,
    query: dict,
    S: np.ndarray,
    psi: np.ndarray,
    mesh_step: int = 2,
) -> None:
    """
    Two-panel PyVista window:
      Left  — transparent heart surface + red voxel cloud for the ROI
      Right — same ROI voxels coloured by ψ (confirms apicobasal extent)

    Parameters
    ----------
    roi_mask  : (Nx,Ny,Nz) bool   result of build_roi_mask
    query     : the same dict passed to build_roi_mask (for window title)
    S         : (Nx,Ny,Nz) bool   myocardium mask (for surface mesh)
    psi       : (Nx,Ny,Nz) float  apicobasal coordinate (for right panel)
    mesh_step : marching-cubes step size
    """
    print("\n  [query] Building surface mesh …")
    Sp = np.pad(S.astype(np.uint8), 2, mode="constant")
    Sp = gaussian_filter(Sp.astype(float), sigma=0.8)
    verts, faces, _, _ = marching_cubes(Sp, level=0.5, step_size=mesh_step)
    verts -= 2.0
    surf = _to_pyvista_surface(verts, faces)
    surf = surf.smooth_taubin(n_iter=30, pass_band=0.1, normalize_coordinates=True)

    roi_idx = np.argwhere(roi_mask).astype(np.float32)
    if len(roi_idx) == 0:
        print("  [query] WARNING: ROI is empty — nothing to display.")
        return

    psi_roi   = psi[roi_mask].astype(np.float32)
    roi_cloud = pv.PolyData(roi_idx)
    roi_cloud["psi"] = psi_roi

    # Build compact window title
    parts = []
    vent = query.get("ventricle", "both")
    parts.append(vent)
    if "psi" in query:
        parts.append(f"ψ∈[{query['psi'][0]:.2f},{query['psi'][1]:.2f}]")
    if "theta_deg" in query:
        parts.append(f"θ∈[{query['theta_deg'][0]}°,{query['theta_deg'][1]}°]")
    elif "theta_rad" in query:
        a, b = query["theta_rad"]
        parts.append(f"θ∈[{np.rad2deg(a):.0f}°,{np.rad2deg(b):.0f}°]")
    if "phi" in query:
        parts.append(f"φ∈[{query['phi'][0]:.2f},{query['phi'][1]:.2f}]")
    title = "  ·  ".join(parts)

    print(f"  [query] Opening PyVista window: {title}")
    print(f"  [query] ROI voxels: {len(roi_idx):,}")

    bg = "#111111"
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 780))
    pl.set_background(bg)

    pl.subplot(0, 0)
    pl.add_text(f"ROI (red)  —  {title}", font_size=10, color="white")
    pl.add_mesh(surf, color="#888888", opacity=0.18,
                show_edges=False, smooth_shading=True)
    pl.add_mesh(roi_cloud, color="#ee3333",
                point_size=5, render_points_as_spheres=True)

    pl.subplot(0, 1)
    pl.add_text(f"ROI coloured by ψ  —  {title}", font_size=10, color="white")
    pl.add_mesh(surf, color="#888888", opacity=0.18,
                show_edges=False, smooth_shading=True)
    pl.add_mesh(
        roi_cloud,
        scalars="psi",
        clim=[0.0, 1.0],
        cmap="plasma",
        point_size=5,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args=dict(
            title="ψ (apicobasal)",
            title_font_size=12,
            label_font_size=10,
            color="white",
            vertical=True,
            position_x=0.88,
            position_y=0.20,
            n_labels=5,
        ),
    )

    pl.link_views()
    pl.camera_position = "xz"
    pl.show()
    try:
        pl.close()
    except Exception:
        pass
