"""
coordinates/functions/_nrrd_spacing.py
=======================================
Single shared place to pull a per-axis voxel spacing (mm) out of an NRRD
header. Same parsing rules already used ad hoc in
hrdayapy/ecg/functions/build_torso_grid.py and register_heart_to_torso.py --
factored out here so load_muscle_mask can use the identical logic instead
of assuming spacing.

Not part of the public package API (leading underscore) -- import it from
other `coordinates/functions/*.py` modules as needed, not from user code.
"""

from __future__ import annotations

import numpy as np


def spacing_from_nrrd_header(header: dict) -> np.ndarray:
    """
    Extract (sx, sy, sz) voxel spacing in mm from an NRRD header dict.

    Checks, in order:
      1. "space directions"  -- 3x3 matrix, spacing = row norms
      2. "spacings"          -- already a (3,) vector
      3. falls back to (1, 1, 1) mm if neither key is present

    Parameters
    ----------
    header : dict
        Header dict as returned by ``nrrd.read(path)``.

    Returns
    -------
    spacing : (3,) float64 ndarray
    """
    if "space directions" in header:
        sd = np.asarray(header["space directions"], dtype=np.float64)
        return np.array([np.linalg.norm(sd[i]) for i in range(3)])
    elif "spacings" in header:
        return np.asarray(header["spacings"], dtype=np.float64)
    else:
        return np.ones(3, dtype=np.float64)


def resolve_original_spacing(original_spacing, target_spacing, header: dict) -> np.ndarray:
    """
    Resolve the "original_spacing" argument shared by load_muscle_mask /
    load_cut_mask / compute_geometry into a concrete (3,) mm-per-voxel
    array, given three accepted forms:

    original_spacing = None
        No spacing information is assumed/available. The loaded grid is
        taken AS-IS and simply labelled with `target_spacing` -- i.e. no
        interpolation happens (zoom factor = 1). This is the old,
        pre-fix behaviour: whatever grid was on disk is assigned the
        requested spacing verbatim, so it's only correct if the file
        already happens to be sampled at target_spacing.

    original_spacing = "nrrd"
        Read the true spacing from the NRRD header itself (see
        `spacing_from_nrrd_header`), and use that as the ground truth
        physical spacing of the loaded array.

    original_spacing = float  (or a (3,) sequence of floats)
        The caller states the original spacing explicitly (isotropic if
        a single float, anisotropic if a 3-tuple), overriding whatever
        the header says.

    Parameters
    ----------
    original_spacing : None | "nrrd" | float | sequence of 3 floats
    target_spacing   : float | sequence of 3 floats
        Needed only for the `None` case, where it doubles as the
        assumed original spacing.
    header : dict
        NRRD header, used only when original_spacing == "nrrd".

    Returns
    -------
    spacing : (3,) float64 ndarray -- resolved original spacing, mm/voxel
    """
    if original_spacing is None:
        return np.broadcast_to(np.asarray(target_spacing, dtype=np.float64), (3,)).copy()

    if isinstance(original_spacing, str):
        if original_spacing.lower() == "nrrd":
            return spacing_from_nrrd_header(header)
        raise ValueError(
            f"Unrecognised original_spacing string {original_spacing!r}; "
            "the only accepted string value is \"nrrd\"."
        )

    return np.broadcast_to(np.asarray(original_spacing, dtype=np.float64), (3,)).copy()
