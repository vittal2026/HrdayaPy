"""
hrdayapy/_spacing_resolve.py
=============================
Shared helper for every downstream stage that needs a single isotropic
mm-per-voxel value (Purkinje monodomain, coupled simulation, ECG BSPM):
lets that one parameter be EITHER an explicit float (full freedom to
override) OR the string "from_geometry", which pulls the value straight
out of the anatomy dict produced by coordinates.compute_geometry /
load_geometry ("spacing_mm") -- so the number can't silently drift out of
sync with what the geometry was actually resampled to.

Deliberately has no dependency on the `coordinates` sub-package (just
expects a plain dict with a "spacing_mm" key), so it can be imported from
purkinje/simulation/ecg without any import-order or circularity concerns.
"""

from __future__ import annotations

import numpy as np

FROM_GEOMETRY = "from_geometry"


def resolve_voxel_size(voxel_size, anatomy=None, *, param_name: str = "voxel_size") -> float:
    """
    Resolve a voxel_size-like parameter to a concrete float (mm/voxel).

    Parameters
    ----------
    voxel_size : float | "from_geometry"
        - float            : used directly, as before -- full freedom to
                              hand-pick a value independent of geometry.
        - "from_geometry"   : pull the value from anatomy["spacing_mm"]
                              (produced by coordinates.compute_geometry /
                              load_geometry) instead of hand-typing it,
                              so this stage can't drift out of sync with
                              what the mask was actually resampled to.
    anatomy : dict, optional
        The anatomy dict from compute_geometry/load_geometry. Required
        (and only used) when voxel_size == "from_geometry".
    param_name : str
        Name to use in error messages (this helper is shared by several
        differently-named parameters: voxel_size, h_fine_mm, ...).

    Returns
    -------
    float -- resolved isotropic voxel size in mm.
    """
    if isinstance(voxel_size, str):
        if voxel_size.strip().lower() != FROM_GEOMETRY:
            raise ValueError(
                f"Unrecognised {param_name}={voxel_size!r}; the only "
                f"accepted string value is \"{FROM_GEOMETRY}\" (or pass a "
                "float directly)."
            )
        if anatomy is None or "spacing_mm" not in anatomy:
            raise ValueError(
                f"{param_name}=\"{FROM_GEOMETRY}\" requires the anatomy "
                "dict from coordinates.compute_geometry / load_geometry "
                "(pass it in as anatomy=...); it wasn't given, or doesn't "
                "contain a 'spacing_mm' entry (re-run compute_geometry "
                "with the updated load_muscle_mask to get one)."
            )
        spacing = np.asarray(anatomy["spacing_mm"], dtype=np.float64).ravel()
        if spacing.size > 1 and not np.allclose(spacing, spacing[0]):
            raise ValueError(
                f"anatomy['spacing_mm']={spacing} is anisotropic, but "
                f"{param_name} here must be a single isotropic value -- "
                f"pass a float explicitly instead of \"{FROM_GEOMETRY}\"."
            )
        return float(spacing[0])

    return float(voxel_size)
