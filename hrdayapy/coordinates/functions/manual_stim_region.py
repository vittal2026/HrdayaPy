"""
manual_stim_region.py
======================
Interactive point-and-grow alternative to stimulus_region.py's UVC
(psi/phi/chi/theta) target-window selection. Some ectopic foci are
easier to describe by "right about there" on the anatomy than by a
coordinate window -- this lets a person click a point on the myocardium
surface and grows a small solid region around it, producing a boolean
voxel mask in exactly the same (Nx,Ny,Nz) format select_coordinate_region
does, so it's a drop-in alternative wherever a stim_region mask is used
(compute_coupled's ectopic_region, etc).

Workflow
--------
    region = pick_and_save_stim_region(S, save_path=..., radius_mm=5.0)

opens one PyVista window (same hover + press P interaction as
pick_single_point in manual_landmarks.py -- left-click stays bound to
camera rotation): hover over the myocardium surface near where you want
the ectopic focus and press P; press P again elsewhere to move the pick;
close the window when you're happy with it.

The picked point sits on the marching-cubes surface, which passes
*between* voxel centers -- so it's snapped to the nearest actually-True
voxel of S before growing the region (via a KD-tree over S's True
voxels), rather than assumed to already be a valid myocardial voxel.
Growth itself is a solid Euclidean ball of radius `radius_mm` around
that seed voxel, intersected with S -- not a surface/geodesic band like
the landmark regions, since an ectopic focus is a volumetric myocardial
blob, not a Dirichlet band on the surface.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

from .mesh_labelling import extract_surface_mesh, mm_step_size
from .manual_landmarks import pick_single_point, _to_pyvista


def _snap_to_nearest_true_voxel(point, S: np.ndarray) -> tuple[int, int, int]:
    """
    Nearest voxel with S == True to an arbitrary (float) point in
    voxel-index space (e.g. a marching-cubes surface vertex, which sits
    between voxel centers rather than on one).

    Restricted to S's surface shell (S & ~eroded(S)), not every True
    voxel: a point picked on the marching-cubes surface can only ever be
    nearest to a surface voxel, never an interior one, so this changes
    nothing about the result. It does cut the KD-tree's point count from
    O(volume) -- which scales roughly cubically in 1/voxel_size -- down
    to O(surface area), roughly quadratic, which is the dominant cost of
    picking at a fine mesh resolution otherwise.
    """
    shell = S & ~binary_erosion(S)
    if not shell.any():
        shell = S   # degenerate case (S is 1 voxel thick or empty) --
                     # fall back to the old full-volume behaviour rather
                     # than raising on an edge case that isn't this
                     # function's to solve
    true_voxels = np.argwhere(shell)
    if true_voxels.size == 0:
        raise RuntimeError("S has no True voxels -- nothing to snap to.")
    tree = cKDTree(true_voxels)
    _, idx = tree.query(np.asarray(point, dtype=float))
    return tuple(int(v) for v in true_voxels[idx])


def _grow_ball_region(S: np.ndarray, seed_ijk: tuple[int, int, int],
                       radius_mm: float, voxel_size: float) -> np.ndarray:
    """
    Boolean (Nx,Ny,Nz) mask: True for voxels of S within `radius_mm`
    (Euclidean, physical units) of `seed_ijk`. Only touches a local
    bounding box around the seed, not the full volume, so this stays
    fast even for a large myocardium mask.
    """
    radius_vox = radius_mm / voxel_size
    pad = int(np.ceil(radius_vox)) + 1
    i0, j0, k0 = seed_ijk
    shape = S.shape

    lo = [max(0, i0 - pad), max(0, j0 - pad), max(0, k0 - pad)]
    hi = [min(shape[0], i0 + pad + 1), min(shape[1], j0 + pad + 1), min(shape[2], k0 + pad + 1)]

    ii, jj, kk = np.meshgrid(
        np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]),
        indexing="ij",
    )
    dist_mm = np.sqrt((ii - i0) ** 2 + (jj - j0) ** 2 + (kk - k0) ** 2) * voxel_size
    local_mask = (dist_mm <= radius_mm) & S[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    region = np.zeros(shape, dtype=bool)
    region[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = local_mask
    return region


def pick_and_save_stim_region(S: np.ndarray, save_path, *,
                               radius_mm: float = 5.0,
                               voxel_size: float = 0.4,
                               mesh_step: int = 2,
                               target_mm: float | None = 1.0,
                               verbose: bool = True) -> np.ndarray:
    """
    Full interactive pick -> grow -> save pipeline for a point-based
    ectopic stimulus region.

    Parameters
    ----------
    S           : (Nx,Ny,Nz) bool -- myocardium mask
    save_path   : region is written here as .npy (required -- the
                  picker result is not returned without being saved)
    radius_mm   : radius of the grown region around the picked point, in mm
    voxel_size  : mm per voxel (isotropic), for converting radius_mm to
                  voxel units -- match whatever compute_coupled will use
    mesh_step   : marching-cubes step size for the picking surface, in
                  voxels. Only takes effect when target_mm=None; see
                  target_mm below (the default) otherwise.
    target_mm   : marching-cubes step size for the picking surface, in mm
                  instead of voxels (see mm_step_size). Overrides
                  mesh_step. None disables this and uses the literal
                  mesh_step value instead -- set this if you need a
                  denser picking surface than 1 mm, or want the old
                  behaviour back.

    Returns
    -------
    region : (Nx,Ny,Nz) bool -- same format as compute_stim_region's output
    """
    def log(msg):
        if verbose:
            print(f"  [manual_stim_region] {msg}")

    save_path = Path(save_path).with_suffix(".npy")

    log("Extracting surface mesh ...")
    if target_mm is not None:
        mesh_step = mm_step_size(voxel_size, target_mm)
    verts, faces = extract_surface_mesh(S, step_size=mesh_step)
    mesh = _to_pyvista(verts, faces)

    # Smoothed copy for on-screen display only -- same point count, order,
    # and connectivity as `mesh` (Taubin smoothing moves vertices, it never
    # changes topology), so a vertex index picked on this mesh is still a
    # valid index into `verts` below. Matches manual_landmarks.py's
    # apex/basal picker, which does the same for the same reason (raw
    # marching-cubes output looks "bricky"/faceted otherwise).
    display_mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1, normalize_coordinates=True)

    log("Opening picker: ECTOPIC FOCUS (hover + press P once) ...")
    idx = pick_single_point(
        verts, faces, mesh=display_mesh, title="Pick the ECTOPIC FOCUS",
        instructions=(
            "Rotate freely with left-click-drag. Hover over the myocardium "
            "surface near where you want the ectopic focus and press P to "
            "pick (press P again elsewhere to move the pick). Close the "
            "window when you're happy with it."
        ),
    )
    if idx is None:
        raise RuntimeError("No point was picked.")
    picked_point = verts[idx]

    log(f"Picked surface point ({picked_point[0]:.2f}, {picked_point[1]:.2f}, "
        f"{picked_point[2]:.2f}); snapping to nearest myocardial voxel ...")
    seed_ijk = _snap_to_nearest_true_voxel(picked_point, S)
    log(f"Seed voxel: {seed_ijk}; growing a {radius_mm} mm ball "
        f"(voxel_size={voxel_size} mm) ...")

    region = _grow_ball_region(S, seed_ijk, radius_mm, voxel_size)
    log(f"Region size: {region.sum():,} voxels")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, region)
    log(f"Saved -> {save_path}")
    return region
