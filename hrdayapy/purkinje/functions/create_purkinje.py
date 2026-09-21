"""
Purkinje Network Generation Module
===================================
Enhanced implementation of the Constructive Optimization (CO) method for
generating physiological Purkinje networks for voxelized cardiac geometries.

Based on:
    Berg et al. "Enhanced optimization-based method for the generation of
    patient-specific models of Purkinje networks."
    Nature Scientific Reports (2023).
    https://doi.org/10.1038/s41598-023-38653-1

Backwards compatible with the original Ulysses et al. interface:
    nodes, elements, activation_times = create_purkinje(root, voxel_mat,
                                                        transmural, apicobasal)

New optional parameters (all have safe defaults so existing call-sites are
unchanged):
    root           : can now be passed as None to trigger interactive
                     picking (the argument is still required/positional,
                     just pass None instead of a coordinate). Opens a
                     PyVista picker on the myocardium surface (marching
                     cubes on voxel_mat) for you to pick the AV node/His
                     entry point by hand.
    pvj_locations  : (N,3) array of prescribed PVJ coordinates in voxel space
    pvj_lat        : (N,)  target Local Activation Times in ms at each PVJ
    lat_error_tol  : LAT tolerance for accepting an active PVJ connection (ms)
    n_a            : number of candidate segments evaluated per PVJ connection
    l_rate         : attempt PVJ connection every l_rate intermediate terminals
    min_degrees    : minimum bifurcation angle constraint (degrees)
    max_degrees    : maximum bifurcation angle constraint (degrees)
    min_seg_length : minimum segment length in voxels
    max_seg_length : maximum segment length in voxels
    cable_G        : cable internal conductivity  (mΩ/cm),  default = 7.9
    cable_Cf       : cable membrane capacitance   (µF/cm²), default = 3.4
    cable_tau      : cable time constant          (ms),      default = 0.1
    cable_d        : cable diameter               (µm),      default = 68.86

Utility helpers (also exported):
    compute_distance()   – segment-point distance (unchanged from original)
    local_opt()          – bifurcation optimiser  (unchanged from original)
    cable_conduction_velocity() – CV from Berg eq. (4)
    compute_lat_cable()  – LAT along a path via cable equation
    merge_purkinje_networks() – join His + LV + RV into one biventricular tree
    write_vtk_network()  – save result as ASCII VTK PolyData (MonoAlg3D ready)
    read_vtk_network()   – load a previously saved VTK network
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from typing import Tuple, Optional, List
import warnings
import heapq


# ---------------------------------------------------------------------------
# Constants (matching Berg et al. / Shocker defaults)
# ---------------------------------------------------------------------------

# Reference conduction velocity used in the merge-network step (µm/ms)
# = 1.9 m/s ≈ 2 m/s, consistent with graph.h REF_CV = 1900.0
_REF_CV_UM_MS = 1900.0


# ---------------------------------------------------------------------------
# Interactive AV node / root picking
# ---------------------------------------------------------------------------

def _pick_av_node_interactively(
    voxel_mat: np.ndarray,
    mesh_step: int = 1,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Interactive AV-node / root picker, used by create_purkinje() when
    ``root`` is not supplied.

    Extracts the myocardium surface from ``voxel_mat`` via marching
    cubes (skimage.measure.marching_cubes -- self-contained, no
    dependency on any other module of this pipeline) and opens a single
    PyVista window on it: hover over a point on the surface and press P
    to pick (press P again elsewhere to move the pick); rotate freely
    with left-click-drag in between. Close the window when you're happy
    with the pick. This mirrors the picking interaction used for the AV
    node in manual_landmarks.py's pick_single_point (plain
    enable_point_picking with a single-argument callback and an explicit
    nearest-point snap -- left-click is left alone for camera rotation,
    so it isn't bound to enable_surface_point_picking's use_picker=True
    2-argument callback nor to left_clicking=True).

    Returns
    -------
    root : tuple of float
        The picked point converted to the (x, y, z) voxel-space
        convention expected by create_purkinje's `root` argument --
        i.e. (column, row, depth), matching the order of the candidate
        position array `S` built later in create_purkinje from
        voxel_mat's (row, col, depth) axes via
        ``np.column_stack([coords_flat[1], coords_flat[0], coords_flat[2]])``.
    """
    from skimage.measure import marching_cubes
    import pyvista as pv

    if verbose:
        print("   Extracting myocardium surface (marching cubes)...")

    # Lightly smooth the binary mask before thresholding so marching cubes
    # doesn't just trace the raw voxel staircase. This only affects the
    # surface used for interactive picking here -- voxel_mat itself (and
    # everything downstream in create_purkinje) is untouched.
    vol = gaussian_filter(voxel_mat.astype(float), sigma=0.8)

    verts, faces, _, _ = marching_cubes(
        vol, level=0.5, step_size=mesh_step
    )
    n_faces  = faces.shape[0]
    vtk_faces = np.hstack(
        [np.full((n_faces, 1), 3, dtype=np.int64), faces]
    ).ravel()
    mesh = pv.PolyData(verts, vtk_faces)

    # Taubin smoothing further rounds off any remaining stairstep artifacts
    # without the shrinkage a plain Laplacian smooth would introduce, so the
    # picking surface stays close to the true myocardium boundary.
    mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1, normalize_coordinates=True)

    state = {"idx": None}
    plotter = pv.Plotter()
    instructions = (
        "Rotate freely with left-click-drag. Hover over a point on the "
        "mesh surface and press P to pick (press P again elsewhere to "
        "move the pick). Close the window when you're happy with it."
    )
    plotter.add_text(f"Pick the AV NODE\n{instructions}", font_size=12)
    plotter.add_mesh(mesh, color="lightcoral", show_edges=False, smooth_shading=True)

    def callback(point):
        idx = mesh.find_closest_point(point)
        state["idx"] = int(idx)
        coords = mesh.points[idx]
        print(f"  Picked vertex {idx} at ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")
        plotter.add_mesh(
            pv.Sphere(radius=mesh.length * 0.01, center=coords),
            color="yellow", name="picked_point",
        )
        plotter.add_text(
            f"vertex {idx}  ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})",
            name="pick_label", font_size=10, color="white", position="lower_left",
        )
        plotter.render()

    plotter.enable_point_picking(
        callback=callback,
        show_message=instructions,
        show_point=True,
        color="red",
        point_size=10,
        tolerance=0.025,
    )
    plotter.show()

    if state["idx"] is None:
        raise RuntimeError("No AV node point was picked.")

    # marching_cubes returns points in voxel array-index order (row, col, depth)
    picked = mesh.points[state["idx"]]
    root_point = (float(picked[1]), float(picked[0]), float(picked[2]))

    if verbose:
        print(f"   AV node picked at voxel coords (root convention): {root_point}")

    return root_point


# ---------------------------------------------------------------------------
# Candidate "allowed Purkinje region" helpers
# ---------------------------------------------------------------------------
def _nearest_lookup_factory(volume_bool: np.ndarray):
    """
    Cheap drop-in replacement for
        RegularGridInterpolator(axes, volume.astype(float), method='nearest',
                                 bounds_error=False, fill_value=0)
    for this module's specific use pattern: F(points) > 0.5 checks against
    a 0/1-valued volume, at integer-spaced grid axes (np.arange(shape[i])).

    RegularGridInterpolator's construction copies the WHOLE volume into
    itself as float64 and sets up generic N-D interpolation machinery it
    doesn't need here -- for a plain grid of integer axes, 'nearest'
    interpolation is exactly "round each coordinate to the nearest
    integer, then index", and out-of-[0, shape-1]-bounds points are
    exactly fill_value=0. This reproduces that behaviour with no data
    copy at construction and a single fancy-index per call instead of
    scipy's general interpolation codepath -- functionally identical
    output, much cheaper to build and to call.
    """
    shape  = volume_bool.shape
    vol_u8 = np.ascontiguousarray(volume_bool, dtype=np.uint8)

    def _F(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        # Bounds are checked against the CONTINUOUS coordinate range
        # [0, shape[d]-1], before rounding -- matching
        # RegularGridInterpolator's behaviour exactly. A point like 19.08
        # on a 20-point axis (valid indices 0..19) is outside the
        # continuous domain [0, 19] and gets fill_value=0, even though
        # rounding it would land on the valid index 19 -- checking
        # bounds on the rounded index instead (as an earlier version of
        # this function did) is NOT equivalent and was caught by a fuzz
        # test against RegularGridInterpolator before this was fixed.
        in_bounds = np.ones(len(points), dtype=bool)
        for d in range(3):
            in_bounds &= (points[:, d] >= 0) & (points[:, d] <= shape[d] - 1)
        out = np.zeros(len(points), dtype=np.float64)
        idx = np.rint(points[in_bounds]).astype(np.int64)
        out[in_bounds] = vol_u8[idx[:, 0], idx[:, 1], idx[:, 2]]
        return out

    return _F


# Shared by create_purkinje()'s own Step 1/2 (Purkinje layer / candidate
# terminal positions Si) and by pick_av_node_and_biventricular_roots()
# below, so both use exactly the same definition of "endocardial and
# within a maximum distance from the surface" -- i.e. a candidate voxel
# must have a low transmural coordinate (near the endocardium), an
# apicobasal coordinate below max_height, and -- when surface_depth_vox is
# given -- lie within that many voxels of a TRUE anatomical boundary.

def _near_surface_mask(
    voxel_mat: np.ndarray,
    surface_depth_vox: Optional[float],
    surface_reference_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Boolean mask of voxels within `surface_depth_vox` of a TRUE anatomical
    boundary, or None if surface_depth_vox is None (criterion disabled).

    Distance is measured against `surface_reference_mask` when given,
    NOT against `voxel_mat` itself. This matters whenever voxel_mat is a
    chamber-restricted sub-mask (e.g. S & (chi_labels == 1) for the LV):
    its own boundary includes not just the real endocardium/epicardium
    but also an ARTIFICIAL internal wall wherever the LV/RV split happens
    to cut through the myocardium -- most consequentially through the
    (thin) septum. distance_transform_edt(voxel_mat) can't tell a real
    surface from that internal cut, so nearly the *entire* septal
    half-thickness ends up "near a surface" simply for being close to the
    cut, not because it's actually close to the endocardium -- inflating
    candidate density in the septum relative to the free wall (where
    only a genuine thin endocardial shell qualifies) and biasing tree
    growth to disproportionately fill the septum. Pass the whole,
    unsplit myocardium mask as surface_reference_mask to measure against
    the real boundary instead and avoid that bias.

    This is the single most expensive step in candidate-region setup at
    fine voxel resolutions (distance_transform_edt is O(N) over the WHOLE
    reference array, not just the myocardium). Compute it once per
    create_purkinje() call and pass the result via
    precomputed_near_surface to _candidate_terminal_mask /
    _candidate_terminal_points below, rather than letting each call
    recompute it independently.
    """
    if surface_depth_vox is None:
        return None
    reference = voxel_mat if surface_reference_mask is None else surface_reference_mask
    dist_to_surface = distance_transform_edt(reference)
    return dist_to_surface <= surface_depth_vox


def _candidate_terminal_mask(
    voxel_mat: np.ndarray,
    transmural: np.ndarray,
    apicobasal: np.ndarray,
    depth: float,
    max_height: float,
    surface_depth_vox: Optional[float] = None,
    surface_reference_mask: Optional[np.ndarray] = None,
    precomputed_near_surface: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean mask of voxels allowed to host a Purkinje terminal/root.

    Pass precomputed_near_surface (from an earlier _near_surface_mask
    call with the same surface_depth_vox/surface_reference_mask) to skip
    recomputing the expensive distance transform here. When None
    (default -- unchanged behaviour for other callers), it's computed
    fresh from surface_depth_vox/surface_reference_mask as before.
    """
    valid_mask = voxel_mat.astype(bool) & (transmural < depth) & (apicobasal < max_height)
    near_surface = (
        precomputed_near_surface if precomputed_near_surface is not None
        else _near_surface_mask(voxel_mat, surface_depth_vox, surface_reference_mask)
    )
    if near_surface is not None:
        valid_mask = valid_mask & near_surface
    return valid_mask


def _candidate_terminal_points(
    voxel_mat: np.ndarray,
    transmural: np.ndarray,
    apicobasal: np.ndarray,
    depth: float,
    max_height: float,
    surface_depth_vox: Optional[float] = None,
    surface_reference_mask: Optional[np.ndarray] = None,
    precomputed_near_surface: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    (N,3) array of candidate voxel positions in (x,y,z) "root convention"
    -- i.e. the same column order as the S array built in create_purkinje's
    own Step 2, and the same order _pick_av_node_interactively returns.

    See _candidate_terminal_mask for precomputed_near_surface.
    """
    valid_mask = _candidate_terminal_mask(
        voxel_mat, transmural, apicobasal, depth, max_height,
        surface_depth_vox, surface_reference_mask, precomputed_near_surface,
    )
    lin_ind     = np.where(valid_mask.ravel())[0]
    coords_flat = np.unravel_index(lin_ind, voxel_mat.shape)
    return np.column_stack([coords_flat[1], coords_flat[0], coords_flat[2]])  # (x,y,z)


# ---------------------------------------------------------------------------
# Single-pick AV node -> automatic biventricular root finding
# ---------------------------------------------------------------------------

def pick_av_node_and_biventricular_roots(
    voxel_mat: np.ndarray,
    lv_mask: np.ndarray,
    rv_mask: np.ndarray,
    transmural: np.ndarray,
    apicobasal: np.ndarray,
    depth: float = 0.1,
    max_height: float = 0.9,
    surface_depth_vox: Optional[float] = None,
    mesh_step: int = 1,
    verbose: bool = True,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Pick a single AV-node point on the *whole* biventricular myocardium
    surface, then automatically find the two closest valid Purkinje roots
    to it -- one on the LV endocardium, one on the RV endocardium -- so
    you only ever click once instead of picking the LV and RV origins
    separately.

    A single connected tree still can't cross the septum (see the note in
    create_purkinje's caller in run_simulation.py), so this doesn't grow
    one tree from the pick -- it opens one picker on the combined surface,
    then hands back three points: the click itself (for the His stub) and
    the nearest valid LV/RV roots to grow the two chamber trees from with
    create_purkinje() / compute_network(), exactly as before.

    Parameters
    ----------
    voxel_mat : np.ndarray
        3-D binary mask of the *whole* myocardium (both ventricles) --
        this is what the interactive picker's surface is extracted from,
        so the person picking sees one unified heart surface, not a
        chamber cut in half.
    lv_mask, rv_mask : np.ndarray
        3-D binary masks restricting `voxel_mat` to the LV and RV
        chambers respectively (e.g. `S & (chi_labels == 1)` /
        `S & (chi_labels == 2)`).
    transmural : np.ndarray
        3-D transmural coordinate (0 = endocardium, 1 = epicardium).
    apicobasal : np.ndarray
        3-D apicobasal coordinate (0 = apex, 1 = base).
    depth : float
        Maximum transmural coordinate for an allowed root (default 0.1,
        matching create_purkinje's own `depth` default) -- i.e. "must be
        endocardial".
    max_height : float
        Maximum apicobasal coordinate for an allowed root (default 0.9,
        matching create_purkinje's own default).
    surface_depth_vox : float or None
        Maximum distance, in voxels, from a TRUE anatomical boundary --
        i.e. the "within a maximum distance from the surface" criterion.
        Measured against `voxel_mat` (the whole myocardium), not against
        `lv_mask`/`rv_mask` individually -- see create_purkinje's
        `surface_reference_mask` docstring for why that distinction
        matters (a chamber-restricted mask's own boundary includes an
        artificial internal cut through the septum, which is not a real
        surface). Pass the same value you plan to grow each chamber's
        tree with, so the roots this function finds are guaranteed to be
        valid candidates for that tree. Default None disables this
        criterion.
    mesh_step : int
        Marching-cubes step size for the picker surface (default 1).
    verbose : bool
        Print the pick and the two chosen roots (default True).

    Returns
    -------
    av_node : tuple of float
        The point you clicked, in voxel coordinates (root convention) --
        use this as the His-stub location when merging the two chamber
        trees with merge_purkinje_networks().
    lv_root : tuple of float
        Closest valid LV voxel to av_node -- pass as `root` to
        create_purkinje()/compute_network() for the LV tree.
    rv_root : tuple of float
        Closest valid RV voxel to av_node -- pass as `root` to
        create_purkinje()/compute_network() for the RV tree.
    """
    if verbose:
        print("Opening interactive picker on the whole biventricular "
              "myocardium surface -- pick the AV node once...")

    av_node     = _pick_av_node_interactively(voxel_mat, mesh_step=mesh_step, verbose=verbose)
    av_node_arr = np.array(av_node, dtype=float)

    # surface_reference_mask=voxel_mat (the whole, unsplit myocardium):
    # measures "near surface" against the real endocardium/epicardium,
    # not against lv_mask's/rv_mask's own boundary -- which would also
    # include the artificial LV/RV split cut through the septum. See
    # create_purkinje's surface_reference_mask docstring.
    lv_candidates = _candidate_terminal_points(
        lv_mask, transmural, apicobasal, depth, max_height, surface_depth_vox,
        surface_reference_mask=voxel_mat,
    )
    rv_candidates = _candidate_terminal_points(
        rv_mask, transmural, apicobasal, depth, max_height, surface_depth_vox,
        surface_reference_mask=voxel_mat,
    )

    if len(lv_candidates) == 0:
        raise ValueError(
            "No valid LV Purkinje candidate voxels found (endocardial + "
            "within the allowed surface distance). Relax depth/max_height/"
            "surface_depth_vox or check lv_mask."
        )
    if len(rv_candidates) == 0:
        raise ValueError(
            "No valid RV Purkinje candidate voxels found (endocardial + "
            "within the allowed surface distance). Relax depth/max_height/"
            "surface_depth_vox or check rv_mask."
        )

    lv_dists = np.linalg.norm(lv_candidates - av_node_arr, axis=1)
    rv_dists = np.linalg.norm(rv_candidates - av_node_arr, axis=1)
    lv_root  = lv_candidates[np.argmin(lv_dists)]
    rv_root  = rv_candidates[np.argmin(rv_dists)]

    if verbose:
        print(f"   AV node picked at        : {tuple(av_node_arr)}")
        print(f"   Nearest valid LV root at : {tuple(lv_root)} "
              f"(distance {lv_dists.min():.2f} voxels)")
        print(f"   Nearest valid RV root at : {tuple(rv_root)} "
              f"(distance {rv_dists.min():.2f} voxels)")

    return (
        tuple(float(c) for c in av_node_arr),
        tuple(float(c) for c in lv_root),
        tuple(float(c) for c in rv_root),
    )


# ---------------------------------------------------------------------------
# Public API — primary entry point
# ---------------------------------------------------------------------------

def create_purkinje(
    root: Optional[Tuple[float, float, float]],
    voxel_mat: np.ndarray,
    transmural: np.ndarray,
    apicobasal: np.ndarray,
    # ---- original parameters (defaults unchanged) ----
    n_term: int = 750,
    n_con_max: int = 20,
    theta_max: float = 63.0,
    cond_vel: float = 340.0,
    resolution: float = 0.25,
    depth: float = 0.1,
    max_height: float = 0.9,
    pkn_diff: int = 5,
    surface_depth_vox: Optional[float] = None,
    surface_reference_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
    show_progress: Optional[bool] = None,
    tqdm_position: Optional[int] = None,
    # ---- Berg et al. CO enhancements (all optional) ----
    pvj_locations: Optional[np.ndarray] = None,
    pvj_lat: Optional[np.ndarray] = None,
    lat_error_tol: float = 2.0,
    n_a: int = 120,
    l_rate: int = 25,
    min_degrees: float = 10.0,
    max_degrees: float = 90.0,
    min_seg_length: float = 0.0,
    max_seg_length: float = np.inf,
    cable_G: float = 7.9,
    cable_Cf: float = 3.4,
    cable_tau: float = 0.1,
    cable_d: float = 68.86,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a physiological Purkinje network for a voxelized cardiac geometry.

    Parameters
    ----------
    root : tuple of float or None
        His bundle / AV node entry point (h, k, l) in voxel units. Pass
        None to pick it interactively instead: a PyVista window opens on
        the myocardium surface extracted from voxel_mat -- hover + press
        P to pick a point, then close the window to continue -- and the
        picked point is used as root.
    voxel_mat : np.ndarray
        3-D binary mask — True/1 = tissue, False/0 = background.
    transmural : np.ndarray
        3-D transmural coordinate (0 = endocardium, 1 = epicardium).
    apicobasal : np.ndarray
        3-D apicobasal coordinate (0 = apex, 1 = base).
    n_term : int
        Target number of terminal branches (default 650).
    n_con_max : int
        Max candidate segments evaluated per intermediate branch (default 20).
    theta_max : float
        Maximum bifurcation angle in degrees (default 63°).
        Overridden by max_degrees when pvj_locations are supplied.
    cond_vel : float
        Conduction velocity in cm/s used for final activation time output
        (default 340 cm/s).
    resolution : float
        Voxel resolution in cm (default 0.25 cm).
    depth : float
        Maximum relative transmural depth for terminal nodes (default 0.1).
    max_height : float
        Maximum apicobasal coordinate for terminal nodes (default 0.9).
    pkn_diff : int
        Gaussian kernel sigma for Purkinje region smoothing (default 5).
    surface_depth_vox : float or None
        If given, restricts eligible Purkinje/terminal voxels to those
        within this many voxels of the *mask's own boundary* -- i.e.
        distance to the nearest non-tissue voxel (cavity or exterior),
        computed via a Euclidean distance transform over voxel_mat. This
        is independent of, and complementary to, the `depth` (transmural)
        criterion: `depth` excludes epicardial voxels (transmural far
        from 0), while `surface_depth_vox` excludes voxels that are deep
        inside thick myocardium despite having a low transmural
        coordinate there -- the classic failure mode being the septum,
        where transmural can read ~0 throughout much of its thickness
        because it's defined relative to the LV/RV endocardial surfaces,
        not to actual distance from any surface. Default None disables
        this criterion (fully backwards compatible).
    surface_reference_mask : np.ndarray or None
        The mask to measure surface_depth_vox's distance against, if
        different from voxel_mat. Matters specifically when voxel_mat is
        a chamber-restricted sub-mask of a larger myocardium (e.g. the
        LV-only `S & (chi_labels == 1)`, when growing the LV and RV
        trees separately): voxel_mat's own boundary then includes not
        just the real endocardium/epicardium but also an ARTIFICIAL
        internal wall wherever the LV/RV split happens to cut through
        the myocardium -- most consequentially through the (thin)
        septum, where nearly the *entire* half-thickness ends up "near a
        surface" simply for being close to that cut rather than to the
        real endocardium. That inflates candidate density in the septum
        relative to the free wall and biases the tree to fill the
        septum disproportionately. Pass the whole, unsplit myocardium
        mask here (e.g. plain `S`) to measure against the true boundary
        instead. Default None falls back to measuring against voxel_mat
        itself, matching the original (pre-split-mask) behaviour.
    verbose : bool
        Print progress information (default True).
    tqdm_position : int or None
        Fixes the growth/activation progress bars to a specific terminal
        row (tqdm's `position` argument) instead of the default single
        dynamic row. Set this when running multiple create_purkinje()
        calls concurrently (e.g. LV in one process, RV in another) so
        each one's bar stays on its own row instead of the two colliding
        -- see purkinje.network.compute_biventricular_networks, which
        sets this to 0/1 for its two worker processes. Default None
        (normal single-bar behaviour).
    show_progress : bool or None
        Independent override for just the two tqdm progress bars,
        separate from `verbose`'s per-stage text announcements ("[1/5]
        Defining Purkinje region...", etc.). Default None means "follow
        verbose" (the original behaviour: both on, or both off).
        Pass verbose=False, show_progress=True to get a quiet run with
        only the position-pinned progress bars -- what
        compute_biventricular_networks's parallel workers do, since
        interleaved *text* from two processes garbles the terminal but
        interleaved *position-pinned bars* (tqdm's own cross-process
        lock, see the same function) render cleanly.

    pvj_locations : np.ndarray, shape (N, 3), optional
        Active PVJ site coordinates in voxel space.  When provided the Berg
        et al. dual cost-function strategy is activated:
          • CFi  minimises total branch length for intermediate points
          • CFa  minimises |simulated_LAT − target_LAT| for PVJ points
    pvj_lat : np.ndarray, shape (N,), optional
        Target LAT in milliseconds for each PVJ in pvj_locations.
        Required when pvj_locations is given.
    lat_error_tol : float
        LAT error tolerance for accepting a PVJ connection (ms, default 2.0).
    n_a : int
        Number of candidate segments evaluated per active PVJ connection
        (default 120, Berg et al. Na parameter).
    l_rate : int
        Attempt PVJ connections every l_rate intermediate terminals
        (default 25, Berg et al. Lrate parameter).
    min_degrees : float
        Minimum bifurcation angle constraint in degrees (default 10°).
    max_degrees : float
        Maximum bifurcation angle constraint in degrees (default 90°).
    min_seg_length : float
        Minimum segment length in voxels (default 0, unconstrained).
    max_seg_length : float
        Maximum segment length in voxels (default inf, unconstrained).
    cable_G : float
        Internal conductivity for the cable equation (mΩ/cm, default 7.9).
    cable_Cf : float
        Membrane capacitance for the cable equation (µF/cm², default 3.4).
    cable_tau : float
        Time constant for the cable equation (ms, default 0.1).
    cable_d : float
        Cable diameter (µm, default 68.86 → CV ≈ 2 m/s).

    Returns
    -------
    nodes : np.ndarray, shape (n_nodes, 3)
        Node positions in voxel coordinates.
    elements : np.ndarray, shape (n_elements, 2)
        Element connectivity — each row is (proximal_node, distal_node).
    activation_times : np.ndarray, shape (n_nodes,)
        Activation time at each node in milliseconds.

    Notes
    -----
    When pvj_locations / pvj_lat are NOT supplied the function is fully
    backwards compatible with the original Ulysses et al. implementation.

    When pvj_locations / pvj_lat ARE supplied the Berg et al. CO method is
    used.  The cable equation conduction velocity (derived from cable_G,
    cable_Cf, cable_tau, cable_d) is used *during tree construction* to
    estimate LATs.  The final returned activation_times are still computed
    from path-length / cond_vel for consistency with the original interface.
    """

    use_co = (pvj_locations is not None)

    # Progress-bar visibility follows `verbose` unless explicitly overridden
    # (see show_progress's docstring above for why a caller would want to
    # decouple the two -- namely, running LV/RV growth concurrently).
    _show_progress = verbose if show_progress is None else show_progress

    # ------------------------------------------------------------------ #
    # Validate CO inputs                                                   #
    # ------------------------------------------------------------------ #
    if use_co:
        pvj_locations = np.asarray(pvj_locations, dtype=float)
        if pvj_lat is None:
            raise ValueError("pvj_lat must be provided together with pvj_locations.")
        pvj_lat = np.asarray(pvj_lat, dtype=float)
        if pvj_lat.shape[0] != pvj_locations.shape[0]:
            raise ValueError("pvj_locations and pvj_lat must have the same length.")

    # ------------------------------------------------------------------ #
    # Interactive AV node / root picking (only if root wasn't supplied)   #
    # ------------------------------------------------------------------ #
    if root is None:
        if verbose:
            print("No root/AV node supplied — opening interactive picker "
                  "on the myocardium surface...")
        root = _pick_av_node_interactively(voxel_mat, verbose=verbose)

    # Angle limits (radians)
    theta_max_rad   = np.deg2rad(theta_max)
    theta_min_rad   = np.deg2rad(min_degrees)
    theta_max_co_rad = np.deg2rad(max_degrees)

    # Effective angle used in local_opt
    angle_limit = theta_max_co_rad if use_co else theta_max_rad

    # Cable CV (voxel/ms) for LAT estimation during construction
    cv_cable_vox_ms = _cable_cv_voxel_ms(
        cable_G, cable_Cf, cable_tau, cable_d, resolution
    )

    if verbose:
        print("=" * 60)
        print("Purkinje Network Generation")
        if use_co:
            print("  Mode: Berg et al. Constructive Optimization (CO)")
            print(f"  Active PVJs: {len(pvj_locations)}")
            print(f"  LAT error tolerance: {lat_error_tol} ms")
            print(f"  Cable CV (voxel/ms): {cv_cable_vox_ms:.4f}")
        else:
            print("  Mode: Ulysses et al. (original)")
        print("=" * 60)
        print(f"Root position  : {root}")
        print(f"Target terminals: {n_term}")
        print(f"Angle range    : {min_degrees}° – {max_degrees if use_co else theta_max}°")
        print(f"Transmural depth: {depth}")
        print(f"Max apicobasal : {max_height}")
        print(f"Surface depth (vox): {surface_depth_vox if surface_depth_vox is not None else 'disabled'}")
        print("-" * 60)

    # ------------------------------------------------------------------ #
    # Step 1 — Define endocardial and Purkinje layers                     #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n[1/5] Defining Purkinje region...")

    pkn_layer = (transmural < depth) & voxel_mat

    # Computed once here and reused in Step 2 below via
    # precomputed_near_surface -- this distance_transform_edt is the most
    # expensive single operation in setup at fine voxel resolutions
    # (O(N) over the whole reference array), and Step 1/2 previously
    # recomputed it twice with identical inputs.
    near_surface = None
    if surface_depth_vox is not None:
        if verbose:
            ref_desc = "true myocardium (surface_reference_mask)" if surface_reference_mask is not None else "mask"
            print(f"   Computing distance to {ref_desc} surface (surface_depth_vox={surface_depth_vox})...")
        near_surface = _near_surface_mask(voxel_mat, surface_depth_vox, surface_reference_mask)
        pkn_layer    = pkn_layer & near_surface

    pkn_layer_diff = gaussian_filter(pkn_layer.astype(float), sigma=pkn_diff)
    threshold      = np.mean(pkn_layer_diff) * 0.5
    pkn_layer_diff = pkn_layer_diff > threshold

    if verbose:
        print(f"   Endocardial voxels   : {int(np.sum((transmural == 0) & voxel_mat))}")
        print(f"   Purkinje region voxels: {int(np.sum(pkn_layer_diff))}")

    # Nearest-neighbour lookup for fast in-region checks. See
    # _nearest_lookup_factory's docstring -- functionally identical to
    # RegularGridInterpolator(method='nearest', bounds_error=False,
    # fill_value=0) for this 0/1 grid, without its full-volume data copy.
    F = _nearest_lookup_factory(pkn_layer_diff)

    # ------------------------------------------------------------------ #
    # Step 2 — Identify candidate intermediate terminal positions (Si)    #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n[2/5] Identifying candidate terminal positions...")

    S = _candidate_terminal_points(
        voxel_mat, transmural, apicobasal, depth, max_height,
        surface_depth_vox, surface_reference_mask,
        precomputed_near_surface=near_surface,
    )
    n_pts = len(S)

    if verbose:
        print(f"   Found {n_pts} candidate positions (Si)")

    # Characteristic length scale (voxels)
    ld = 200  # kept identical to original

    if verbose:
        print(f"   Characteristic length ld = {ld} voxels")

    # ------------------------------------------------------------------ #
    # Step 3 — Pre-process active PVJs (Sa) — Berg et al. §Pre-processing #
    # ------------------------------------------------------------------ #
    # Sort PVJs by ascending target LAT so earlier activation sites are
    # attempted first, matching the paper's sorting strategy.
    if use_co:
        sort_idx      = np.argsort(pvj_lat)
        Sa            = pvj_locations[sort_idx].copy()   # active PVJ coords
        Sa_lat        = pvj_lat[sort_idx].copy()         # target LATs (ms)
        Sa_connected  = np.zeros(len(Sa), dtype=bool)    # connection flags
    else:
        Sa = Sa_lat = Sa_connected = None

    # ------------------------------------------------------------------ #
    # Step 4 — Initialise tree with root                                  #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n[3/5] Initialising tree from root...")

    root = np.array(root, dtype=float)

    # The initial root segment must respect max_seg_length too -- it used
    # to only be capped by `ld` (200 vox), completely bypassing the
    # min/max_seg_length constraints applied everywhere else in the tree.
    first_seg_thresh = min(ld, max_seg_length) if np.isfinite(max_seg_length) else ld

    rnd       = np.random.permutation(n_pts)
    i         = 0
    dist_node = S[rnd[i], :]

    while np.linalg.norm(root - dist_node) > first_seg_thresh and i < n_pts - 1:
        i += 1
        dist_node = S[rnd[i], :]

    first_seg_len = np.linalg.norm(root - dist_node)
    if first_seg_len > first_seg_thresh:
        raise ValueError(
            f"Could not find any candidate Purkinje voxel within "
            f"max_seg_length={max_seg_length} voxels of root {tuple(root)} "
            f"(closest candidate found was {first_seg_len:.2f} voxels away, "
            f"out of {n_pts} candidates searched). This almost always means "
            f"the root/AV-node point was not placed on (or near) the valid "
            f"endocardial Purkinje region -- re-pick a point directly on "
            f"the septal/endocardial surface, or relax max_seg_length."
        )

    S     = np.delete(S, rnd[i], axis=0)
    n_pts -= 1

    prox = root.reshape(1, 3)
    dist = dist_node.reshape(1, 3)

    if verbose:
        print(f"   First segment length: {first_seg_len:.2f} voxels")

    # ------------------------------------------------------------------ #
    # Step 5 — Main loop: grow the tree                                   #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n[4/5] Growing Purkinje tree...")
    if _show_progress:
        pbar = tqdm(total=n_term, desc="   Terminals added", unit="term",
                    position=tqdm_position, leave=True)
        pbar.update(1)

    k_term = 1
    i      = 0

    while k_term < n_term:

        # -------------------------------------------------------------- #
        # 5a — Attempt active PVJ connections every l_rate iterations     #
        #      (Berg et al. AttemptPVJConnection subroutine)              #
        # -------------------------------------------------------------- #
        if use_co and (k_term % l_rate == 0):
            prox, dist, Sa_connected = _attempt_pvj_connections(
                Sa, Sa_lat, Sa_connected,
                prox, dist,
                cv_cable_vox_ms, lat_error_tol, n_a,
                F, theta_min_rad, theta_max_co_rad,
                min_seg_length, max_seg_length,
                verbose=False,
            )

        # -------------------------------------------------------------- #
        # 5b — Adaptive distance threshold                                #
        # -------------------------------------------------------------- #
        d_thresh  = ld / np.sqrt(k_term)
        dist_flag = True
        counter   = 1

        while dist_flag:
            if counter >= 10:
                d_thresh *= 0.9
                counter   = 1

            if i >= n_pts:
                if verbose:
                    tqdm.write(
                        f"   Warning: ran out of candidates at {k_term} terminals"
                    )
                break

            rnd  = np.random.permutation(n_pts)
            term = S[rnd[i], :]
            i   += 1

            d_crit = compute_distance(term, prox, dist)

            if np.min(d_crit) > d_thresh:
                # Segment-length constraint (Berg et al. min/max_segment_length)
                seg_len_ok = _check_seg_length(
                    term, prox, dist, min_seg_length, max_seg_length
                )
                if seg_len_ok:
                    dist_flag = False

            counter += 1

        if i >= n_pts:
            break

        # -------------------------------------------------------------- #
        # 5c — Select candidate segments to try splitting                 #
        # -------------------------------------------------------------- #
        if prox.shape[0] <= n_con_max:
            n_con   = prox.shape[0]
            seg2con = np.arange(n_con)
        else:
            n_con   = n_con_max
            ind_sort = np.argsort(d_crit)
            seg2con  = ind_sort[:n_con_max]

        # -------------------------------------------------------------- #
        # 5d — Find best bifurcation point (CFi minimisation)            #
        # -------------------------------------------------------------- #
        cand_xbif = np.zeros((n_con, 3))
        min_len   = np.zeros(n_con)

        for j in range(len(seg2con)):
            cand_xbif[j, :], min_len[j] = local_opt(
                term,
                prox[seg2con[j], :],
                dist[seg2con[j], :],
                angle_limit,
                F,
                theta_min=theta_min_rad,
            )

        if np.all(np.isnan(cand_xbif)):
            continue

        ind_min  = np.nanargmin(min_len)
        x_bif    = cand_xbif[ind_min, :]
        seg2split = seg2con[ind_min]

        # -------------------------------------------------------------- #
        # 5e — Update tree structure                                      #
        # -------------------------------------------------------------- #
        tmp_prox = prox[seg2split, :].copy()
        tmp_dist = dist[seg2split, :].copy()

        prox = np.delete(prox, seg2split, axis=0)
        dist = np.delete(dist, seg2split, axis=0)

        prox = np.vstack([prox, tmp_prox, x_bif,   x_bif])
        dist = np.vstack([dist, x_bif,   tmp_dist, term ])

        k_term += 1
        S       = np.delete(S, rnd[i], axis=0)
        n_pts  -= 1
        i       = 0

        if _show_progress:
            pbar.update(1)

    if _show_progress:
        pbar.close()
    if verbose:
        print(f"   Final tree: {k_term} terminals, {len(prox)} segments")

    # ------------------------------------------------------------------ #
    # Step 5f — Post-processing: connect remaining PVJs                   #
    #           (Berg et al. PostProcessing subroutine)                   #
    # ------------------------------------------------------------------ #
    if use_co:
        if verbose:
            print("\n   [Post-processing] Connecting remaining active PVJs...")

        prox, dist, Sa_connected = _postprocess_pvjs(
            Sa, Sa_lat, Sa_connected,
            prox, dist,
            cv_cable_vox_ms, lat_error_tol, n_a,
            F, theta_min_rad, theta_max_co_rad,
            min_seg_length, max_seg_length,
            verbose=verbose,
        )

        n_unconnected = int(np.sum(~Sa_connected))
        if verbose:
            print(f"   PVJs connected: {int(np.sum(Sa_connected))} / {len(Sa)}")
            if n_unconnected:
                print(f"   Warning: {n_unconnected} PVJ(s) could not be connected.")

    # ------------------------------------------------------------------ #
    # Step 6 — Convert to node/element format                             #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n[5/5] Computing activation times...")

    nodes = np.vstack([prox, dist])
    n_el  = len(prox)

    elem = np.arange(2 * n_el).reshape(-1, 1)

    nodes, inverse_indices = np.unique(nodes, axis=0, return_inverse=True)
    elem  = inverse_indices.reshape(-1, 1)
    elem  = np.hstack([elem[:n_el], elem[n_el:]])

    # Ensure root is at index 0
    root_node = np.where(
        (nodes[:, 0] == root[0]) &
        (nodes[:, 1] == root[1]) &
        (nodes[:, 2] == root[2])
    )[0]

    if len(root_node) > 0:
        root_node = root_node[0]
        nodes[[0, root_node], :] = nodes[[root_node, 0], :]
        elem[elem == root_node] = -1
        elem[elem == 0]         = root_node
        elem[elem == -1]        = 0

    # BFS activation times
    n_nodes   = len(nodes)
    act_times = np.full(n_nodes, np.nan)
    act_times[0] = 0.0
    curr_nodes   = [0]

    remaining = n_nodes - 1
    if _show_progress:
        pbar = tqdm(total=remaining, desc="   Nodes activated", unit="node",
                    position=tqdm_position, leave=True)

    while np.any(np.isnan(act_times)):
        elem_in      = np.isin(elem[:, 0], curr_nodes)
        elem_indices = np.where(elem_in)[0]

        if len(elem_indices) == 0:
            break

        parent_nodes  = elem[elem_indices, 0]
        nodes_to_act  = elem[elem_indices, 1]
        path_len      = np.linalg.norm(
            nodes[nodes_to_act, :] - nodes[parent_nodes, :], axis=1
        )

        act_times[nodes_to_act] = act_times[parent_nodes] + path_len
        curr_nodes = nodes_to_act.tolist()

        if _show_progress:
            pbar.update(len(nodes_to_act))

    if _show_progress:
        pbar.close()

    # Convert path-length (voxels) → ms using original formula
    act_times = act_times * resolution / cond_vel * 100.0

    if verbose:
        print(f"   Activation range: {np.nanmin(act_times):.2f} – {np.nanmax(act_times):.2f} ms")
        print("\n" + "=" * 60)
        print("Purkinje network generation complete!")
        print("=" * 60)

    nodes = nodes[:, [0, 1, 2]]

    return nodes, elem, act_times


# ---------------------------------------------------------------------------
# Utility: compute_distance  (unchanged from original)
# ---------------------------------------------------------------------------

def compute_distance(
    term: np.ndarray,
    prox: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """
    Minimum distance from a terminal point to each existing segment.

    Parameters
    ----------
    term : (3,) array
    prox : (n_seg, 3) array  – proximal endpoints
    dist : (n_seg, 3) array  – distal endpoints

    Returns
    -------
    d_crit : (n_seg,) array
    """
    u = prox - dist
    v = term - dist
    w = term - prox

    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    norm_w = np.linalg.norm(w, axis=1)

    d      = np.sum(u * v, axis=1) / (norm_u ** 2 + 1e-10)
    crit_ind = (d <= 1) & (d >= 0)

    d_crit = np.full(prox.shape[0], np.nan)

    if np.any(crit_ind):
        cross_prod        = np.cross(v[crit_ind], w[crit_ind])
        d_crit[crit_ind]  = np.linalg.norm(cross_prod, axis=1) / norm_u[crit_ind]

    d_crit[~crit_ind] = np.minimum(norm_v[~crit_ind], norm_w[~crit_ind])

    return d_crit


# ---------------------------------------------------------------------------
# Utility: local_opt  (extended with theta_min constraint)
# ---------------------------------------------------------------------------

def local_opt(
    term: np.ndarray,
    prox: np.ndarray,
    dist: np.ndarray,
    theta_max: float,
    F: RegularGridInterpolator,
    theta_min: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Find optimal bifurcation point for a new intermediate terminal (CFi).

    Parameters
    ----------
    term      : (3,) terminal point to connect
    prox      : (3,) proximal node of segment to split
    dist      : (3,) distal node of segment to split
    theta_max : maximum bifurcation angle (radians)
    F         : valid-region interpolator
    theta_min : minimum bifurcation angle (radians, default 0)

    Returns
    -------
    x_bif   : (3,) optimal bifurcation point (NaN if no valid solution)
    min_len : minimum added path length (inf if no valid solution)
    """
    x1, y1, z1 = prox
    x2, y2, z2 = dist
    x3, y3, z3 = term

    n_e = 10
    eps = np.linspace(0, 1, n_e + 1)

    grid_eps, grid_eta = np.meshgrid(eps, eps)
    mask               = grid_eps + grid_eta <= 1
    grid_eps           = grid_eps[mask]
    grid_eta           = grid_eta[mask]

    grid_eps = grid_eps[1:-1]
    grid_eta = grid_eta[1:-1]

    psi1 = 1 - grid_eps - grid_eta
    psi2 = grid_eps
    psi3 = grid_eta

    grid_x = psi1 * x1 + psi2 * x2 + psi3 * x3
    grid_y = psi1 * y1 + psi2 * y2 + psi3 * y3
    grid_z = psi1 * z1 + psi2 * z2 + psi3 * z3

    # --- connectivity checks (all midpoints must lie in valid region) ---
    def _filter(gx, gy, gz, px, py, pz):
        pts   = np.column_stack([(gy + py) / 2, (gx + px) / 2, (gz + pz) / 2])
        valid = F(pts) > 0.5
        return gx[valid], gy[valid], gz[valid]

    points = np.column_stack([grid_y, grid_x, grid_z])
    valid  = F(points) > 0.5
    if not np.any(valid):
        return np.array([np.nan, np.nan, np.nan]), np.inf

    grid_x, grid_y, grid_z = grid_x[valid], grid_y[valid], grid_z[valid]

    for px, py, pz in [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]:
        grid_x, grid_y, grid_z = _filter(grid_x, grid_y, grid_z, px, py, pz)
        if len(grid_x) == 0:
            return np.array([np.nan, np.nan, np.nan]), np.inf

    # --- cost function (CFi): added total length ---
    v1 = np.column_stack([grid_x - x1, grid_y - y1, grid_z - z1])
    v2 = np.column_stack([grid_x - x2, grid_y - y2, grid_z - z2])
    v3 = np.column_stack([grid_x - x3, grid_y - y3, grid_z - z3])

    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    norm_v3 = np.linalg.norm(v3, axis=1)

    len_added = norm_v1 + norm_v2 + norm_v3 - np.linalg.norm(prox - dist)

    cos_angle   = np.sum(v2 * v3, axis=1) / (norm_v2 * norm_v3 + 1e-10)
    cos_angle   = np.clip(cos_angle, -1, 1)
    angle       = np.arccos(cos_angle)

    valid_angle = (angle < theta_max) & (angle > theta_min)

    if not np.any(valid_angle):
        return np.array([np.nan, np.nan, np.nan]), np.inf

    len_added = len_added[valid_angle]
    grid_x    = grid_x[valid_angle]
    grid_y    = grid_y[valid_angle]
    grid_z    = grid_z[valid_angle]

    min_idx = np.argmin(len_added)
    x_bif   = np.array([grid_x[min_idx], grid_y[min_idx], grid_z[min_idx]])

    return x_bif, len_added[min_idx]


# ---------------------------------------------------------------------------
# Cable equation helpers  (Berg et al. §Connection of an active PVJ branch)
# ---------------------------------------------------------------------------

def cable_conduction_velocity(
    G: float,
    Cf: float,
    tau: float,
    d: float,
) -> float:
    """
    Conduction velocity from Berg et al. eq. (4).

    CV = sqrt(G * d / (4 * Cf * tau))

    Parameters
    ----------
    G   : internal conductivity (mΩ/cm)
    Cf  : membrane capacitance  (µF/cm²)
    tau : time constant         (ms)
    d   : diameter              (µm)

    Returns
    -------
    CV in µm/ms  (≈ m/s numerically)
    """
    d_cm = d * 1e-4          # µm → cm
    G_si = G                 # mΩ/cm kept as-is; units cancel in sqrt
    return float(np.sqrt(G_si * d_cm / (4.0 * Cf * tau)))


def compute_lat_cable(
    path_length_voxels: float,
    resolution_cm: float,
    cv_voxel_ms: float,
) -> float:
    """
    Estimate LAT (ms) for a straight-line segment.

    Parameters
    ----------
    path_length_voxels : segment length in voxels
    resolution_cm      : voxel size in cm
    cv_voxel_ms        : conduction velocity in voxels/ms

    Returns
    -------
    LAT in ms
    """
    return path_length_voxels / (cv_voxel_ms + 1e-12)


# ---------------------------------------------------------------------------
# Merge utility  (mirrors merge_network() in graph.cpp)
# ---------------------------------------------------------------------------

def merge_purkinje_networks(
    his_nodes: np.ndarray,
    his_elem: np.ndarray,
    lv_nodes: np.ndarray,
    lv_elem: np.ndarray,
    rv_nodes: np.ndarray,
    rv_elem: np.ndarray,
    lv_root: np.ndarray,
    rv_root: np.ndarray,
    resolution: float = 0.25,
    cond_vel: float = 340.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge His-bundle, LV and RV Purkinje networks into one biventricular tree.

    Mirrors the logic of merge_networks() in scripts/merge-network/src/graph/
    graph.cpp:
      • Concatenates node lists with index offsets
      • Connects His node-1 to the closest node to lv_root and rv_root
      • Re-runs BFS to recompute activation times on the merged graph

    Parameters
    ----------
    his_nodes, his_elem : His-bundle nodes (n,3) and elements (m,2)
    lv_nodes,  lv_elem  : LV network nodes and elements
    rv_nodes,  rv_elem  : RV network nodes and elements
    lv_root             : (3,) LV root position in voxel coordinates
    rv_root             : (3,) RV root position in voxel coordinates
    resolution          : voxel size in cm (default 0.25)
    cond_vel            : conduction velocity in cm/s (default 340)
    verbose             : print summary (default True)

    Returns
    -------
    nodes, elements, activation_times  — same convention as create_purkinje()
    """
    n_his = len(his_nodes)
    n_lv  = len(lv_nodes)
    n_rv  = len(rv_nodes)

    # Offset element indices
    his_elem_off = his_elem.copy()
    lv_elem_off  = lv_elem.copy()  + n_his
    rv_elem_off  = rv_elem.copy()  + n_his + n_lv

    # Merged node array
    nodes = np.vstack([his_nodes, lv_nodes, rv_nodes])

    # Find closest node in merged graph to each ventricle root
    # (matching get_closest_point() which skips terminal nodes — here we
    # simply use the closest point overall, same effect for tree roots)
    lv_root_idx = _closest_node(nodes, lv_root)
    rv_root_idx = _closest_node(nodes, rv_root)

    # His bundle split node is index 1 (matches graph.cpp: list_nodes[1])
    his_split = 1

    # Two extra edges: His-split → LV-root and His-split → RV-root
    extra_edges = np.array([
        [his_split, lv_root_idx],
        [his_split, rv_root_idx],
    ])

    elements = np.vstack([his_elem_off, lv_elem_off, rv_elem_off, extra_edges])

    # Re-compute activation times via BFS from His root (node 0)
    n_nodes   = len(nodes)
    act_times = np.full(n_nodes, np.nan)
    act_times[0] = 0.0

    adj = [[] for _ in range(n_nodes)]
    for p, d in elements:
        length = float(np.linalg.norm(nodes[p] - nodes[d]))
        adj[p].append((d, length))

    queue     = [0]
    visited   = np.zeros(n_nodes, dtype=bool)
    visited[0] = True

    while queue:
        nxt = []
        for u in queue:
            for v, length in adj[u]:
                if not visited[v]:
                    visited[v]    = True
                    act_times[v]  = act_times[u] + length
                    nxt.append(v)
        queue = nxt

    # Convert voxel-path to ms
    act_times = act_times * resolution / cond_vel * 100.0

    if verbose:
        valid = ~np.isnan(act_times)
        print(f"Merged network: {n_nodes} nodes, {len(elements)} elements")
        print(f"Activation range: {np.nanmin(act_times):.2f} – "
              f"{np.nanmax(act_times):.2f} ms")

    return nodes, elements, act_times


# ---------------------------------------------------------------------------
# VTK I/O  (ASCII PolyData, MonoAlg3D compatible — matches write_network /
#            write_LAT in graph.cpp)
# ---------------------------------------------------------------------------

def write_vtk_network(
    filepath: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    activation_times: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> None:
    """
    Write a Purkinje network to an ASCII VTK PolyData file.

    Produces the same format as Shocker's write_LAT() / write_MonoAlg3D().

    Parameters
    ----------
    filepath         : output file path (e.g. 'outputs/network.vtk')
    nodes            : (n, 3) node positions
    elements         : (m, 2) element connectivity
    activation_times : (n,) LAT in ms — written as SCALARS LAT if provided
    sigma            : (n,) conductivity — written as SCALARS sigma if provided
                       (takes precedence over activation_times for MonoAlg3D)
    """
    n_nodes = len(nodes)
    n_elem  = len(elements)

    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 4.1\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {n_nodes} float\n")
        for x, y, z in nodes:
            f.write(f"{x:g} {y:g} {z:g}\n")

        f.write(f"LINES {n_elem} {n_elem * 3}\n")
        for p, d in elements:
            f.write(f"2 {int(p)} {int(d)}\n")

        scalar_data = sigma if sigma is not None else activation_times
        scalar_name = "sigma" if sigma is not None else "LAT"

        if scalar_data is not None:
            f.write(f"POINT_DATA {n_nodes}\n")
            f.write(f"SCALARS {scalar_name} float\n")
            f.write("LOOKUP_TABLE default\n")
            for v in scalar_data:
                f.write(f"{v:g}\n")


def read_vtk_network(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Read an ASCII VTK PolyData Purkinje network file (as written by Shocker).

    Returns
    -------
    nodes            : (n, 3) float array
    elements         : (m, 2) int array
    scalar_data      : (n,)  float array if present, else None
    """
    nodes    = []
    elements = []
    scalars  = []

    state = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('POINTS'):
                state = 'points'
                continue
            if line.startswith('LINES'):
                state = 'lines'
                continue
            if line.startswith('SCALARS'):
                state = 'scalars_header'
                continue
            if line.startswith('LOOKUP_TABLE'):
                state = 'scalars'
                continue
            if line.startswith('POINT_DATA') or line.startswith('DATASET') \
               or line.startswith('ASCII') or line.startswith('vtk'):
                state = None
                continue

            if state == 'points':
                vals = list(map(float, line.split()))
                if len(vals) == 3:
                    nodes.append(vals)
            elif state == 'lines':
                vals = list(map(int, line.split()))
                if len(vals) == 3 and vals[0] == 2:
                    elements.append([vals[1], vals[2]])
            elif state == 'scalars':
                try:
                    scalars.append(float(line))
                except ValueError:
                    pass

    nodes    = np.array(nodes,    dtype=float)
    elements = np.array(elements, dtype=int)
    scalars  = np.array(scalars,  dtype=float) if scalars else None

    return nodes, elements, scalars


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cable_cv_voxel_ms(
    G: float, Cf: float, tau: float, d: float, resolution_cm: float
) -> float:
    """CV from cable equation in voxels/ms."""
    cv_um_ms = cable_conduction_velocity(G, Cf, tau, d)  # µm/ms
    voxel_um = resolution_cm * 1e4                        # cm → µm
    return cv_um_ms / voxel_um                            # voxels/ms


def _lat_at_node(
    node_idx: int,
    prox: np.ndarray,
    dist: np.ndarray,
    root: np.ndarray,
    cv_vox_ms: float,
) -> float:
    """
    Estimate LAT at a given distal node by summing segment lengths from root.
    Uses a simple adjacency walk — cheap enough for small trees.
    """
    # Build adjacency from prox/dist arrays
    n = len(prox)
    adj: dict = {}
    for k in range(n):
        p_key = tuple(prox[k])
        d_key = tuple(dist[k])
        if p_key not in adj:
            adj[p_key] = []
        adj[p_key].append((d_key, float(np.linalg.norm(prox[k] - dist[k]))))

    # BFS from root
    start = tuple(root)
    target = tuple(dist[node_idx])
    visited = {start: 0.0}
    queue = [start]

    while queue:
        nxt = []
        for u in queue:
            for v, length in adj.get(u, []):
                if v not in visited:
                    visited[v] = visited[u] + length / cv_vox_ms
                    nxt.append(v)
        queue = nxt

    return visited.get(target, np.inf)


def _estimate_pvj_lat(
    pvj: np.ndarray,
    seg_prox: np.ndarray,
    seg_dist: np.ndarray,
    parent_lat_vox: float,
    cv_vox_ms: float,
) -> float:
    """
    Estimate LAT at pvj if connected to seg via a straight line.
    parent_lat_vox is the estimated LAT (ms) at the distal end of seg.
    """
    # Straight-line distance from distal end of seg to pvj
    d = float(np.linalg.norm(seg_dist - pvj))
    return parent_lat_vox + d / (cv_vox_ms + 1e-12)


def _compute_seg_lats(
    prox: np.ndarray,
    dist: np.ndarray,
    root: np.ndarray,
    cv_vox_ms: float,
) -> np.ndarray:
    """
    Compute estimated LAT (ms) at the distal end of every segment via BFS.
    """
    n_seg = len(prox)
    # Map node coords to index
    all_pts  = np.vstack([prox, dist])
    unique_pts, inv = np.unique(all_pts, axis=0, return_inverse=True)
    n_nodes  = len(unique_pts)

    prox_idx = inv[:n_seg]
    dist_idx = inv[n_seg:]

    # Find root index
    dists_to_root = np.linalg.norm(unique_pts - root, axis=1)
    root_idx      = int(np.argmin(dists_to_root))

    adj = [[] for _ in range(n_nodes)]
    for k in range(n_seg):
        p, d = int(prox_idx[k]), int(dist_idx[k])
        length = float(np.linalg.norm(unique_pts[p] - unique_pts[d]))
        adj[p].append((d, length))
        adj[d].append((p, length))  # undirected for robustness

    lat = np.full(n_nodes, np.inf)
    lat[root_idx] = 0.0
    heap = [(0.0, root_idx)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > lat[u]:
            continue
        for v, w in adj[u]:
            nd = d_u + w / (cv_vox_ms + 1e-12)
            if nd < lat[v]:
                lat[v] = nd
                heapq.heappush(heap, (nd, v))

    # Return LAT at distal end of each segment
    return lat[dist_idx]


def _attempt_pvj_connections(
    Sa, Sa_lat, Sa_connected,
    prox, dist,
    cv_vox_ms, lat_error_tol, n_a,
    F, theta_min, theta_max,
    min_seg_length, max_seg_length,
    verbose=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berg et al. AttemptPVJConnection — try to connect each unconnected PVJ.
    Uses CFa = |LAT(s) − T(PVJ)| as cost function.
    """
    root_approx = prox[0]  # root is first prox point

    # Pre-compute LAT at distal end of every current segment
    seg_lats = _compute_seg_lats(prox, dist, root_approx, cv_vox_ms)

    for pvj_idx in range(len(Sa)):
        if Sa_connected[pvj_idx]:
            continue

        pvj     = Sa[pvj_idx]
        t_pvj   = Sa_lat[pvj_idx]

        # Estimate LAT error for each segment (using straight-line distance)
        lat_errors = np.array([
            abs(_estimate_pvj_lat(pvj, prox[k], dist[k],
                                  seg_lats[k], cv_vox_ms) - t_pvj)
            for k in range(len(prox))
        ])

        # Evaluate the n_a best candidates properly
        candidate_segs = np.argsort(lat_errors)[:n_a]
        best_err = np.inf
        best_k   = None

        for k in candidate_segs:
            # Quick angle/region check via local_opt (CFa variant)
            x_bif, _ = local_opt(
                pvj, prox[k], dist[k], theta_max, F, theta_min=theta_min
            )
            if np.any(np.isnan(x_bif)):
                continue

            # Check seg-length constraints
            if not _check_seg_length(pvj, prox[[k]], dist[[k]],
                                     min_seg_length, max_seg_length):
                continue

            # Compute LAT via bifurcation point
            bif_lat = seg_lats[k] + (
                np.linalg.norm(dist[k] - x_bif) / (cv_vox_ms + 1e-12)
            )
            pvj_lat_est = bif_lat + (
                np.linalg.norm(x_bif - pvj) / (cv_vox_ms + 1e-12)
            )
            err = abs(pvj_lat_est - t_pvj)

            if err < best_err:
                best_err = err
                best_k   = k
                best_bif = x_bif

        if best_k is not None and best_err <= lat_error_tol:
            # Insert the bifurcation
            tmp_prox = prox[best_k].copy()
            tmp_dist = dist[best_k].copy()
            prox = np.delete(prox, best_k, axis=0)
            dist = np.delete(dist, best_k, axis=0)
            prox = np.vstack([prox, tmp_prox, best_bif, best_bif])
            dist = np.vstack([dist, best_bif, tmp_dist, pvj  ])
            Sa_connected[pvj_idx] = True

            # Recompute seg_lats after structural change
            seg_lats = _compute_seg_lats(prox, dist, root_approx, cv_vox_ms)

    return prox, dist, Sa_connected


def _postprocess_pvjs(
    Sa, Sa_lat, Sa_connected,
    prox, dist,
    cv_vox_ms, lat_error_tol, n_a,
    F, theta_min, theta_max,
    min_seg_length, max_seg_length,
    verbose=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berg et al. PostProcessing — connect remaining unconnected PVJs by:
      Phase II  : drop LAT tolerance, retry with geodesic connections
      Phase III : drop distance criterion, connect via closest segment
      Phase IV  : straight-line fallback to nearest segment
    """
    if verbose:
        print("   Phase II: retrying with relaxed LAT tolerance...")

    # Phase II — drop lat_error_tol completely
    prox, dist, Sa_connected = _attempt_pvj_connections(
        Sa, Sa_lat, Sa_connected,
        prox, dist,
        cv_vox_ms, np.inf, n_a,      # lat_error_tol=inf
        F, theta_min, theta_max,
        min_seg_length, max_seg_length,
        verbose=verbose,
    )

    if not np.any(~Sa_connected):
        return prox, dist, Sa_connected

    if verbose:
        print("   Phase III/IV: straight-line fallback for remaining PVJs...")

    # Phase III/IV — straight-line fallback for still-unconnected PVJs
    root_approx = prox[0]
    seg_lats    = _compute_seg_lats(prox, dist, root_approx, cv_vox_ms)

    for pvj_idx in range(len(Sa)):
        if Sa_connected[pvj_idx]:
            continue

        pvj   = Sa[pvj_idx]
        t_pvj = Sa_lat[pvj_idx]

        lat_errors = np.array([
            abs(_estimate_pvj_lat(pvj, prox[k], dist[k],
                                  seg_lats[k], cv_vox_ms) - t_pvj)
            for k in range(len(prox))
        ])

        best_k = int(np.argmin(lat_errors))

        # Straight-line bifurcation at midpoint of best segment
        x_bif    = (prox[best_k] + dist[best_k]) / 2.0
        tmp_prox = prox[best_k].copy()
        tmp_dist = dist[best_k].copy()
        prox = np.delete(prox, best_k, axis=0)
        dist = np.delete(dist, best_k, axis=0)
        prox = np.vstack([prox, tmp_prox, x_bif, x_bif])
        dist = np.vstack([dist, x_bif,   tmp_dist, pvj])
        Sa_connected[pvj_idx] = True

        seg_lats = _compute_seg_lats(prox, dist, root_approx, cv_vox_ms)

    return prox, dist, Sa_connected


def _check_seg_length(
    term: np.ndarray,
    prox: np.ndarray,
    dist: np.ndarray,
    min_len: float,
    max_len: float,
) -> bool:
    """
    Return True if any candidate segment-to-term connection satisfies
    the min/max length constraints from the .ini cost_function section.
    """
    if min_len <= 0.0 and np.isinf(max_len):
        return True
    lengths = np.linalg.norm(dist - term, axis=1)
    return bool(np.any((lengths >= min_len) & (lengths <= max_len)))


def _closest_node(nodes: np.ndarray, target: np.ndarray) -> int:
    """Return index of node in `nodes` closest to `target`."""
    return int(np.argmin(np.linalg.norm(nodes - target, axis=1)))


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Purkinje Network Generation Module (Berg et al. CO enhanced)")
    print("Import and call create_purkinje() — fully backwards compatible.")
    print()
    print("Basic usage (original interface):")
    print("  from create_purkinje import create_purkinje")
    print("  nodes, elem, times = create_purkinje(root, voxel_mat, phi, psi)")
    print()
    print("Enhanced usage (Berg et al. CO with active PVJs):")
    print("  nodes, elem, times = create_purkinje(")
    print("      root, voxel_mat, phi, psi,")
    print("      pvj_locations=pvj_coords,  # (N,3) voxel coords")
    print("      pvj_lat=pvj_times_ms,      # (N,)  target LATs in ms")
    print("      lat_error_tol=2.0,")
    print("      n_a=120,")
    print("      l_rate=25,")
    print("  )")
    print()
    print("Merge LV + RV + His into biventricular network:")
    print("  from create_purkinje import merge_purkinje_networks, write_vtk_network")
    print("  nodes, elem, times = merge_purkinje_networks(")
    print("      his_nodes, his_elem, lv_nodes, lv_elem, rv_nodes, rv_elem,")
    print("      lv_root, rv_root")
    print("  )")
    print("  write_vtk_network('bvn.vtk', nodes, elem, times)")
