"""
coordinates/functions/__init__.py
==================================
Public API for the coordinates sub-package.
"""

# Geometry / mask
from .load_muscle_mask              import load_muscle_mask
from .mesh_labelling                import label_ventricle_mesh, relabel_lv_rv_with_psi
from .manual_landmarks              import pick_and_save_landmarks
from .manual_stim_region            import pick_and_save_stim_region
from .find_apico_basal_axis         import find_apico_basal_axis
from .compute_basal_plane           import compute_basal_plane
from .compute_com                   import compute_com
from .split_lv_rv                   import split_lv_rv

# Coordinate fields
from .generate_transmural_coordinate    import generate_transmural_coordinate
from .generate_apicobasal_coordinate    import generate_apicobasal_coordinate
from .generate_biventricular_coordinate import generate_biventricular_coordinate
from .generate_rotational_coordinate    import (
    generate_rotational_coordinate,
    compute_septal_loop,
    compute_septal_loop_and_arcs,
    compute_theta,
    pick_septal_landmarks,
    pick_anterior_vertex,
)

# ROI querying
from .query_region import build_roi_mask, visualise_roi

# Visualisation / post-processing
from .cut_plotting              import cut_plotting
from .visualise_coordinates     import visualise_coordinates
from .select_coordinate_region  import select_coordinate_region, region_centroid_voxel
from .plot_stimulus_region      import plot_stimulus_region

__all__ = [
    # Geometry
    "load_muscle_mask",
    "label_ventricle_mesh",
    "relabel_lv_rv_with_psi",
    "pick_and_save_landmarks",
    "pick_and_save_stim_region",
    "find_apico_basal_axis",
    "compute_basal_plane",
    "compute_com",
    "split_lv_rv",
    # Coordinate fields
    "generate_transmural_coordinate",
    "generate_apicobasal_coordinate",
    "generate_biventricular_coordinate",
    "generate_rotational_coordinate",
    "compute_septal_loop",
    "compute_septal_loop_and_arcs",
    "compute_theta",
    "pick_septal_landmarks",
    "pick_anterior_vertex",
    # ROI
    "build_roi_mask",
    "visualise_roi",
    # Visualisation
    "cut_plotting",
    "visualise_coordinates",
    "select_coordinate_region",
    "region_centroid_voxel",
    "plot_stimulus_region",
]
