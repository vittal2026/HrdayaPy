"""
coordinates/__init__.py
=========================
Public API for the coordinates stage.

Every output of this stage has two ways in: compute it from the previous
stage's output, or load it back from a file you already saved. Nothing
here decides that for you -- call whichever you mean.

    S, anatomy       = compute_geometry(...)   / load_geometry(...)
    anatomy          = refine_geometry_with_psi(...)               (redo LV/RV split once psi exists)
    Z                = load_cut_mask(...)                          (Slicer export, load-only)
    surface_label    = compute_surface_label(...) / load_surface_label(...)
    chi, chi_labels  = compute_chi(...)        / load_chi(...)
    phi              = compute_phi(...)        / load_phi(...)     (also used for phi_rv)
    landmarks        = compute_landmarks(...)  / load_landmarks(...)
    psi              = compute_psi(...)        / load_psi(...)
    theta            = compute_theta(...)      / load_theta(...)
    stim_region      = compute_stim_region(...)/ load_stim_region(...)
    stim_region      = compute_stim_region_from_point(...) / load_stim_region_from_point(...)
    roi              = compute_roi(...)        / load_roi(...)

Every function takes its own paths as arguments -- nothing is read from
a shared config file, so the same script works for any patient.
"""

from .geometry import compute_geometry, load_geometry, load_cut_mask, refine_geometry_with_psi
from .surface_label import compute_surface_label, load_surface_label
from .biventricular import compute_chi, load_chi
from .transmural import compute_phi, load_phi
from .landmarks import compute_landmarks, load_landmarks
from .apicobasal import compute_psi, load_psi
from .rotational import compute_theta, load_theta
from .stimulus_region import (
    compute_stim_region, load_stim_region,
    compute_stim_region_from_point, load_stim_region_from_point,
    compute_stim_regions_from_points, load_stim_regions,
)
from .roi_query import compute_roi, load_roi

# Visualisation (no compute/load pair -- these just render, they don't
# produce a new saveable output)
from .functions import visualise_coordinates, plot_stimulus_region

__all__ = [
    "compute_geometry", "load_geometry", "load_cut_mask", "refine_geometry_with_psi",
    "compute_surface_label", "load_surface_label",
    "compute_chi", "load_chi",
    "compute_phi", "load_phi",
    "compute_landmarks", "load_landmarks",
    "compute_psi", "load_psi",
    "compute_theta", "load_theta",
    "compute_stim_region", "load_stim_region",
    "compute_stim_region_from_point", "load_stim_region_from_point",
    "compute_stim_regions_from_points", "load_stim_regions",
    "compute_roi", "load_roi",
    "visualise_coordinates", "plot_stimulus_region",
]
