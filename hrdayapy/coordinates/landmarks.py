"""
coordinates/landmarks.py
==========================
Compute/load pair for the manually-picked landmarks (apex, basal region)
that psi and theta are built from.

compute_landmarks opens an interactive PyVista picker -- it needs a real
display. load_landmarks never opens a picker; it only reads a previous
result, so it's the one to use in headless/batch scripts.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import pick_and_save_landmarks


def compute_landmarks(
    S,
    save_path,
    *,
    mesh_step: int = 2,
    picking_mesh_step: int | None = None,
    apex_radius_mm: float = 6.0,
    basal_band_mm: float = 8.0,
):
    """
    Interactively pick the apex and basal region on the myocardium
    surface, then voxelize them.

    Opens a PyVista window: pick the apex (hover + P), then click a
    sequence of points around the base and close the window. Requires a
    display -- not for headless use.

    Parameters
    ----------
    S                 : (Nx,Ny,Nz) bool -- myocardium mask
    save_path         : landmarks are always written here as .npz
                         (required -- the picker result is not returned
                         without being saved)
    mesh_step         : marching-cubes step size for the mesh the final
                         apex/basal regions are voxelized on
    picking_mesh_step : marching-cubes step size for a separate, coarser
                         mesh used only for the interactive picking itself
                         (defaults to max(mesh_step, 4)). Keep this at its
                         default if you refine `S` (e.g. a finer
                         TARGET_SPACING_MM) and notice basal-region picking
                         getting slow -- that mesh no longer grows with
                         `S`'s resolution. Pass mesh_step here to restore
                         the old behaviour of picking directly on the
                         final-resolution mesh.
    apex_radius_mm, basal_band_mm :
                         region size around each pick, in mm

    Returns
    -------
    landmarks : dict with apex_voxels, basal_voxels (Nx,Ny,Nz) bool,
                plus the raw picked points
    """
    save_path = Path(save_path).with_suffix(".npz")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick into a temp file first. If picking fails or is cancelled partway
    # (a window closed early, fewer than 3 basal points clicked, etc.),
    # pick_and_save_landmarks raises before writing anything -- and because
    # we haven't touched save_path yet, whatever landmarks you already had
    # saved there are untouched. Only once picking fully succeeds do we
    # replace the real file.
    tmp_path = save_path.with_name(save_path.stem + "_tmp_pick.npz")
    if tmp_path.exists():
        tmp_path.unlink()

    landmarks = pick_and_save_landmarks(
        S, save_path=tmp_path, mesh_step=mesh_step,
        picking_mesh_step=picking_mesh_step,
        apex_radius_mm=apex_radius_mm,
        basal_band_mm=basal_band_mm,
    )

    tmp_path.replace(save_path)
    print(f"  [compute_landmarks] Saved -> {save_path}")
    return landmarks


def load_landmarks(path):
    """Load previously picked landmarks. Never opens a picker."""
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved landmarks at {path}")
    z = np.load(path)
    return {k: z[k] for k in z.files}
