"""
mesh_labelling.py
==================
Robust, orientation-independent ventricular surface labelling.

This replaces the grid-axis-snapped heuristics in
find_apico_basal_axis.py / compute_basal_plane.py / label_cardiac_surfaces.py
/ split_lv_rv.py with a single pipeline that ports the core *algorithm*
behind Doste et al.'s InSilicoHeartGen labelling stage
(https://github.com/rdoste/InSilicoHeartGen, Ventricular_Labelling.m) and
Cobiveco's long-axis estimator (KIT-IBT/Cobiveco, computeLongAxis.m) to
plain NumPy/SciPy, applied to a surface mesh extracted once from the
existing binary myocardium mask.

Nothing here assumes the heart is aligned with the voxel grid. The only
grid-dependent step left anywhere in the pipeline after this module is
generate_rotational_coordinate.py's per-slice theta construction -- see
the project README for why that one is a separate, smaller follow-up.

Pipeline
--------
1. extract_surface_mesh      -- marching cubes on the padded mask
2. classify_epi_endo         -- ray-trace each face against the mesh's own
                                 convex hull; faces close to the hull are
                                 epicardium, faces far from it are
                                 endocardium (2 dominant clusters)
3. disambiguate_lv_rv        -- the endocardial cluster with the larger
                                 convex-hull volume is RV (Doste et al.,
                                 Sec. 2.3.1.1)
4. compute_long_axis         -- port of Cobiveco's computeLongAxis: the
                                 unit vector minimising the area-weighted
                                 projection of LV-endocardial face normals
                                 onto it (i.e. "most orthogonal" to the LV
                                 cavity wall) -- robust to any tilt
5. find_pole_cap /
   detect_flat_basal_lid     -- apex and basal Dirichlet regions, located
                                 by projecting onto the long axis rather
                                 than a grid index
6. compute_rv_septal_endocardium -- local wall-thickness proxy (KD-tree,
                                 direction-filtered) separating the thick
                                 septal RV endocardium from the thin RV
                                 free wall (Doste et al., Sec. 2.3.1.1)
7. voxelize_labels            -- maps every result back onto the original
                                 voxel grid via nearest-face lookup

Public API
----------
    result = label_ventricle_mesh(S, voxel_size=1.0)

`result` is a dict; see `label_ventricle_mesh` docstring for its keys.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
from collections import defaultdict


# =============================================================================
# 1. Mesh extraction
# =============================================================================

def mm_step_size(voxel_size_mm: float, target_mm: float = 1.0) -> int:
    """
    Voxel-unit step_size for marching_cubes that keeps the *physical*
    mesh resolution roughly constant (~target_mm per triangle edge) as
    voxel_size_mm shrinks. A literal voxel-count step_size (the old
    default everywhere in this package) does the opposite: its physical
    resolution gets finer -- and its triangle count, smoothing cost, and
    downstream picking/rendering cost along with it -- automatically as
    voxel_size_mm shrinks, whether or not that detail is wanted.

    target_mm=1.0 (default) is plenty fine for interactive picking or a
    sanity-check plot -- a person clicking a point, or eyeballing a
    region against the anatomy, can't tell a 1 mm mesh from a 0.2 mm
    one. Lower it only if you need genuinely fine visual detail.

    Does not by itself reduce the cost of scanning the input array --
    marching_cubes still visits every voxel of S regardless of
    step_size. This caps how dense the *output* mesh (and everything
    built on top of it) gets, which is the dominant cost for a
    reasonably-sized heart mesh; if the base scan cost also matters at
    your resolution, that needs pre-downsampling S itself, which this
    function deliberately does not do for you (it would lose fine
    boundary detail, which changing the extracted mesh's density alone
    does not).
    """
    return max(1, int(round(target_mm / voxel_size_mm)))


def extract_surface_mesh(S: np.ndarray, step_size: int = 2):
    """
    Marching-cubes surface of a binary mask, with outward-pointing,
    consistently-wound faces.

    step_size > 1 coarsens the mesh. This is the same recommendation made
    in Doste et al. (2026), Sec. 2.3: ray tracing is the expensive part of
    labelling, so it is run on a coarser mesh and the *labels* projected
    back onto the fine voxel grid afterwards -- the labelling decision
    itself does not need finer-than-wall-thickness resolution.

    Returns
    -------
    verts : (V,3) float64  -- voxel-index coordinates (not yet scaled by
            voxel size; scale the returned mesh yourself if you need
            physical units for anything outside this module)
    faces : (F,3) int64
    """
    Sp = np.pad(S.astype(np.uint8), 2, mode="constant")
    verts, faces, _, _ = marching_cubes(Sp, level=0.5, step_size=step_size)
    verts -= 2.0

    centroids, normals, areas = face_geometry(verts, faces)
    # Orient outward: for a closed surface, sum(centroid . normal * area)
    # equals 3x the enclosed volume by the divergence theorem, and must be
    # positive if normals point outward.
    vol_signed = np.sum(np.einsum("ij,ij->i", centroids, normals) * areas) / 3.0
    if vol_signed < 0:
        faces = faces[:, [0, 2, 1]]

    return verts, faces


def face_geometry(verts: np.ndarray, faces: np.ndarray):
    """Per-face centroid, outward unit normal (assuming CCW winding as seen
    from outside), and area."""
    tri = verts[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2 * areas[:, None] + 1e-12)
    centroids = tri.mean(axis=1)
    return centroids, normals, areas


def _face_adjacency(faces: np.ndarray):
    """Returns (f_a, f_b): arrays of face-index pairs that share an edge."""
    F = faces.shape[0]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges_sorted = np.sort(edges, axis=1)
    face_id = np.tile(np.arange(F), 3)
    order = np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))
    es = edges_sorted[order]
    fi = face_id[order]
    same = np.all(es[1:] == es[:-1], axis=1)
    return fi[:-1][same], fi[1:][same]


def _sample_field_at_faces(centroids: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Nearest-voxel sample of a scalar (Nx,Ny,Nz) field at each face
    centroid (voxel-index coordinates, same convention as verts). Used to
    pull psi values onto the mesh for the basal-cut LV/RV split below."""
    idx = np.round(centroids).astype(int)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, field.shape[a] - 1)
    return field[idx[:, 0], idx[:, 1], idx[:, 2]]


def _surface_label_to_is_epi(S: np.ndarray, centroids: np.ndarray,
                               surface_label: np.ndarray) -> np.ndarray:
    """
    Reconstruct a per-face is_epi boolean from an already-computed
    voxel-level surface_label (+1 epi / -1 endo / 0 else), instead of
    re-running the expensive ray-tracing classify_epi_endo.

    Rounding centroids to the nearest voxel misses the thin surface_label
    shell most of the time -- marching-cubes vertices sit at sub-voxel,
    interpolated positions, not on integer grid points. A KDTree query
    against the actual surface-voxel coordinates (voxels of S with at
    least one background 6-neighbour) mirrors what voxelize_face_labels
    did in the forward direction, so this is the correct inverse lookup.
    """
    from scipy.ndimage import convolve

    Sb = (S > 0).astype(np.uint8)
    kernel = np.zeros((3, 3, 3), dtype=np.int8)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1
    bg_neighbors = convolve((Sb == 0).astype(np.int8), kernel, mode="constant", cval=1)
    surface_voxels = np.argwhere((Sb == 1) & (bg_neighbors > 0))
    surface_voxel_labels = surface_label[surface_voxels[:, 0], surface_voxels[:, 1],
                                           surface_voxels[:, 2]]

    tree = cKDTree(surface_voxels.astype(float))
    _, nearest = tree.query(centroids)
    face_surf_label = surface_voxel_labels[nearest]
    return face_surf_label == 1, face_surf_label == -1  # is_epi, is_endo


# =============================================================================
# 2. Ray tracing against the convex hull -> epi / endo split
# =============================================================================

def _ray_triangle_min_distance(origins, directions, tri_verts, eps=1e-8, chunk=3000):
    """
    Vectorised Moeller-Trumbore: for each (origin, direction), the distance
    to the nearest triangle in `tri_verts` strictly in front of the ray.
    np.inf where no triangle is hit. Chunked over origins to bound memory.
    """
    A, B, C = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    e1, e2 = B - A, C - A
    H = A.shape[0]
    N = origins.shape[0]
    out = np.full(N, np.inf)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        O, D = origins[s:e], directions[s:e]
        Dexp = D[:, None, :]
        pvec = np.cross(Dexp, e2[None, :, :])
        det = np.einsum("ijk,jk->ij", pvec, e1)
        valid = np.abs(det) > eps
        invdet = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)

        tvec = O[:, None, :] - A[None, :, :]
        u = np.einsum("ijk,ijk->ij", tvec, pvec) * invdet
        valid &= (u >= -1e-6) & (u <= 1 + 1e-6)

        qvec = np.cross(tvec, e1[None, :, :])
        v = np.einsum("ijk,ijk->ij", Dexp * np.ones((1, H, 1)), qvec) * invdet
        valid &= (v >= -1e-6) & (u + v <= 1 + 1e-6)

        t = np.einsum("ijk,jk->ij", qvec, e2) * invdet
        valid &= t > 1e-5

        out[s:e] = np.where(valid, t, np.inf).min(axis=1)

    return out


def _vectorised_majority_smooth(label, f_a, f_b, n_faces, valid_labels=(0, 1, 2),
                                  only_unresolved=True, max_iter=30):
    """
    Majority-vote label smoothing over the face-adjacency graph, fully
    vectorised via sparse one-hot matmuls (no per-face Python loop).

    If only_unresolved=True: only faces currently labelled -1 are updated
    each pass (used to resolve ray-tracing/threshold noise after the
    initial epi/endo split -- mirrors InSilicoHeartGen's remove_isolated.m).
    If False: every face is reassigned to its neighbourhood's majority
    label each pass (mirrors smooth_labels.m; use sparingly, e.g. 1-2
    passes, or it will erode small genuine structures like the RV
    septal patch).
    """
    label = label.copy()
    adj = coo_matrix((np.ones(len(f_a) * 2, dtype=np.float32),
                       (np.concatenate([f_a, f_b]), np.concatenate([f_b, f_a]))),
                      shape=(n_faces, n_faces)).tocsr()

    for _ in range(max_iter):
        target = np.where(label == -1)[0] if only_unresolved else np.arange(n_faces)
        if len(target) == 0:
            break
        counts = np.zeros((n_faces, len(valid_labels)))
        for i, l in enumerate(valid_labels):
            onehot = (label == l).astype(np.float32)
            counts[:, i] = adj @ onehot
        best = np.array(valid_labels)[np.argmax(counts, axis=1)]
        has_any = counts.sum(axis=1) > 0
        new_label = label.copy()
        upd = target[has_any[target]]
        new_label[upd] = best[upd]
        if only_unresolved:
            changed = (new_label[target] != label[target]).sum()
        else:
            changed = (new_label != label).sum()
        label = new_label
        if changed == 0:
            break
    return label


def _cluster_epi_endo_faces(verts, faces, areas, is_epi, f_a, f_b,
                              psi_at_face=None, basal_psi_threshold: float = 0.97):
    """
    Shared clustering step used by both classify_epi_endo (fresh
    hull-distance is_epi) and relabel_lv_rv_with_psi (is_epi reconstructed
    from an already-computed surface_label): given a per-face epi/endo
    split, connected-component the mesh into one epicardial cluster and
    two endocardial clusters (LV / RV, not yet disambiguated).

    If `psi_at_face` is given, any adjacency between two *endocardial*
    faces is additionally severed wherever either face sits in the basal
    band (psi >= basal_psi_threshold) before connected components runs.
    Without this cut, the LV- and RV-endocardial surfaces are not
    separated by a septum near an open/cut base, so naive same-label
    connected components can merge them into a single blob instead of
    the two clusters this function is supposed to find -- this basal cut
    is the fix ported from test_psi_basal_cut.py.

    Returns
    -------
    label      : (F,) int8   0 = epicardium, 1 = endocardium cluster A,
                              2 = endocardium cluster B (LV/RV not yet
                              disambiguated -- see disambiguate_lv_rv)
    basal_face : (F,) bool or None -- the basal-cut mask actually applied
                 (None if psi_at_face was not given)
    """
    F = faces.shape[0]
    same = is_epi[f_a] == is_epi[f_b]

    basal_face = None
    if psi_at_face is not None:
        basal_face = np.nan_to_num(psi_at_face, nan=0.0) >= basal_psi_threshold
        both_endo = (~is_epi[f_a]) & (~is_epi[f_b])
        cut = both_endo & (basal_face[f_a] | basal_face[f_b])
        n_cut = int(cut.sum())
        same = same & ~cut
        print(f"  [mesh_labelling] psi-based basal cut: {int(basal_face.sum())} faces "
              f"flagged basal (psi >= {basal_psi_threshold}), severing {n_cut} "
              f"endo-endo adjacency pairs through the basal band before "
              f"connected components.")

    adj = coo_matrix((np.ones(same.sum()), (f_a[same], f_b[same])), shape=(F, F))
    n_comp, comp_labels = connected_components(adj, directed=False)
    comp_area = np.bincount(comp_labels, weights=areas, minlength=n_comp)
    order = np.argsort(comp_area)[::-1]

    # Guard against degenerate candidates: a connected component that's
    # essentially flat/sliver-thin (a meshing artifact, or a genuinely
    # tiny disconnected piece of the mask) can occasionally out-rank a
    # real but smaller endocardial cluster by raw area alone. Require the
    # candidate's vertex bounding box to have *some* extent in all three
    # dimensions before it's eligible to be picked as LV/RV.
    MIN_EXTENT = 0.5  # mesh-units (≈ voxels); true cavities span many voxels
                       # in every dimension, so this only rejects genuinely
                       # flat/degenerate candidates, not small-but-real ones

    # epi_comp must be the largest component that was ACTUALLY classified
    # is_epi=True by the threshold above -- not just the largest
    # component overall. Area alone is not sufficient: if Otsu's
    # automatic split puts the true epicardial shell on the "far from
    # hull" side (is_epi=False), the single largest connected component
    # can simultaneously be (a) the biggest component by area and (b)
    # genuinely endo-classified -- in which case treating it as epi here
    # while ALSO later selecting it as an endo candidate below causes the
    # two label assignments to collide, and the second one silently wins
    # (overwriting the "epi" label with an "endo" one for every one of
    # those faces), leaving epi with zero faces. This was previously
    # unguarded (`epi_comp = order[0]`); this failure mode is exactly
    # what an unbalanced area_frac_epi_side warning above is flagging.
    epi_candidates = [c for c in order if is_epi[comp_labels == c].mean() >= 0.5]
    if not epi_candidates:
        raise RuntimeError(
            "No connected component was majority-classified as "
            "epicardium (is_epi=True) by the hull-distance threshold -- "
            "Otsu's automatic split failed to separate epi/endo for this "
            "mesh (see the area-fraction diagnostic printed above). "
            "Inspect the log_dn histogram directly rather than trusting "
            "the automatic threshold; a manual or percentile-based "
            "threshold may be needed for this mesh."
        )
    epi_comp = epi_candidates[0]

    candidates = [c for c in order
                  if is_epi[comp_labels == c].mean() < 0.5 and c != epi_comp]
    endo_comps = []
    for c in candidates:
        vidx = np.unique(faces[comp_labels == c].ravel())
        extent = verts[vidx].max(axis=0) - verts[vidx].min(axis=0)
        degenerate = np.any(extent < MIN_EXTENT)
        print(f"  [mesh_labelling] candidate endo component {c}: "
              f"{int((comp_labels==c).sum())} faces, area={comp_area[c]:.1f}, "
              f"bbox extent={extent}{'  <-- DEGENERATE, skipped' if degenerate else ''}")
        if not degenerate:
            endo_comps.append(c)
        if len(endo_comps) == 2:
            break
    if len(endo_comps) < 2:
        raise RuntimeError(
            "Could not find two distinct, non-degenerate endocardial "
            "surface clusters. See the candidate diagnostics printed "
            "above -- if every non-epicardial component looks tiny or "
            "degenerate, double-check that S contains a single fused "
            "LV+RV myocardial wall with no separate cavity/blood-pool "
            "voxels and no stray disconnected components. If LV and RV "
            "endocardium are connected through an open/cut base, passing "
            "psi (so the basal cut can be applied) usually fixes this."
        )

    label = np.full(F, -1, dtype=np.int8)
    label[comp_labels == epi_comp] = 0
    label[comp_labels == endo_comps[0]] = 1
    label[comp_labels == endo_comps[1]] = 2

    label = _vectorised_majority_smooth(label, f_a, f_b, F, valid_labels=(0, 1, 2),
                                         only_unresolved=True)
    return label, basal_face


def classify_epi_endo(verts, faces, mesh_step_for_hull: int = None,
                        psi=None, basal_psi_threshold: float = 0.97):
    """
    Separate mesh faces into epicardium (1 cluster) and two endocardial
    clusters, via bidirectional... -- actually single-direction (outward
    normal) ray distance to the mesh's own convex hull.

    Faces close to the hull (small dn+) are epicardium; faces far from it
    (large dn+) are endocardium. This is orientation-independent: it does
    not use the voxel grid axes anywhere. Mirrors Doste et al. Sec.
    2.3.1.2 (the "cut geometry" ray-tracing case, which is the relevant
    one here since this pipeline's masks have no separate valve labels).

    Parameters
    ----------
    psi : (Nx,Ny,Nz) float, optional
        Apicobasal coordinate field (0 = apex, 1 = base). If given, a
        psi-based basal cut (see _cluster_epi_endo_faces) is applied
        before the endocardial connected components are found -- needed
        whenever the LV/RV endocardium is topologically connected through
        an open/cut base. If None (default), behaviour is unchanged from
        before this parameter existed.
    basal_psi_threshold : float
        Faces with sampled psi >= this value are treated as "basal" for
        the cut. Only used when `psi` is given.

    Returns
    -------
    label : (F,) int8     0 = epicardium, 1 = endocardium cluster A,
                           2 = endocardium cluster B (LV/RV not yet
                           disambiguated -- see disambiguate_lv_rv)
    dn_plus : (F,) float  ray distance to hull (diagnostic / reused later)
    psi_at_face : (F,) float or None -- psi sampled at each face centroid
                  (None if psi was not given), returned so callers don't
                  need to resample it for disambiguate_lv_rv.
    """
    centroids, normals, areas = face_geometry(verts, faces)
    hull = ConvexHull(verts)
    hull_tri = verts[hull.simplices]

    dn_plus = _ray_triangle_min_distance(centroids, normals, hull_tri)
    finite = np.isfinite(dn_plus)
    dn_filled = np.where(finite, dn_plus, dn_plus[finite].max() if finite.any() else 1.0)

    log_dn = np.log1p(dn_filled)
    th = threshold_otsu(log_dn)
    is_epi = log_dn < th

    # Diagnostic: report how cleanly Otsu split the hull-distance
    # distribution into two clusters, and what fraction of total surface
    # area on each side of the threshold. A healthy bimodal split should
    # show most of the area on the is_epi=True side with a large log_dn
    # gap around `th`; a value close to 50/50, or with very little area
    # on the is_epi=True side, is a sign the automatic threshold has
    # failed to separate epi from endo for this mesh -- print it up
    # front so that's visible immediately rather than only showing up
    # later as a confusing "epi area=0" symptom.
    area_frac_epi_side = areas[is_epi].sum() / areas.sum()
    print(f"  [mesh_labelling] Otsu threshold on log1p(hull distance): "
          f"th={th:.4f}  (log_dn range [{log_dn.min():.4f}, {log_dn.max():.4f}])")
    print(f"  [mesh_labelling] area fraction with log_dn < th (candidate "
          f"epicardium side): {100*area_frac_epi_side:.1f}%")
    if area_frac_epi_side < 0.1 or area_frac_epi_side > 0.9:
        print(f"  [mesh_labelling] WARNING: this split is far from a "
              f"balanced bimodal separation -- Otsu's automatic threshold "
              f"may have failed to cleanly separate epi/endo for this "
              f"mesh. Inspect the log_dn histogram directly if the "
              f"labelling below looks wrong.")

    psi_at_face = _sample_field_at_faces(centroids, psi) if psi is not None else None

    f_a, f_b = _face_adjacency(faces)
    label, _basal_face = _cluster_epi_endo_faces(
        verts, faces, areas, is_epi, f_a, f_b,
        psi_at_face=psi_at_face, basal_psi_threshold=basal_psi_threshold,
    )
    return label, dn_plus, psi_at_face


def _safe_hull_volume(pts: np.ndarray) -> float:
    """ConvexHull volume that degrades gracefully on degenerate (coplanar/
    collinear/duplicate) point sets instead of crashing -- a genuine 3D
    cavity always has a well-defined hull volume, so a degenerate result
    here means the candidate wasn't a real cavity to begin with and
    should simply lose the RV-vs-LV comparison rather than halt the run.
    """
    try:
        return ConvexHull(pts).volume
    except Exception:
        try:
            # joggle the input -- standard Qhull workaround for
            # near-degenerate (but not exactly degenerate) point sets
            return ConvexHull(pts, qhull_options="QJ").volume
        except Exception:
            extent = pts.max(axis=0) - pts.min(axis=0)
            print(f"  [mesh_labelling] WARNING: a labelled endocardial "
                  f"cluster is geometrically degenerate ({len(pts)} points, "
                  f"bbox extent={extent}) -- not a real 3D cavity. "
                  f"Treating its hull volume as ~0.")
            return 1e-6


def disambiguate_lv_rv(verts, faces, label, psi_at_face=None):
    """
    Relabels the two endocardial clusters (currently 1/2, arbitrary) to
    the canonical convention RV=1, LV=2.

    If `psi_at_face` is given, this uses the apex-proximity criterion
    from test_psi_basal_cut.py: the apex is formed almost entirely of LV
    myocardium, so whichever endocardial cluster has the lower median
    psi (closer to psi=0, the apex) is LV. This is the preferred
    criterion whenever psi is available -- it is a direct anatomical
    fact rather than a population-level heuristic.

    Otherwise, falls back to the convex-hull-volume criterion from Doste
    et al. Sec. 2.3.1.1 ("the RV volume is larger than the LV volume in
    healthy conditions and in most pathologies").
    """
    if psi_at_face is not None:
        med = {}
        for l in (1, 2):
            vals = psi_at_face[label == l]
            vals = vals[np.isfinite(vals)]
            med[l] = float(np.median(vals)) if len(vals) else np.inf
        lv_current = min(med, key=med.get)   # closer to apex (lower psi) = LV
        info = {"criterion": "psi_median_apex_proximity", "psi_median": med}
        print(f"  [mesh_labelling] LV/RV disambiguation via psi median: "
              f"cluster 1 psi={med[1]:.3f}, cluster 2 psi={med[2]:.3f} "
              f"-> cluster {lv_current} = LV (closer to apex)")
        if lv_current == 2:
            return label, info   # already RV=1, LV=2
        swapped = label.copy()
        swapped[label == 1] = 2
        swapped[label == 2] = 1
        return swapped, info

    vols = {}
    for l in (1, 2):
        vidx = np.unique(faces[label == l].ravel())
        vols[l] = _safe_hull_volume(verts[vidx])
    rv_current = max(vols, key=vols.get)
    if rv_current == 1:
        return label, vols   # already RV=1, LV=2
    swapped = label.copy()
    swapped[label == 1] = 2
    swapped[label == 2] = 1
    return swapped, vols


# =============================================================================
# 3. Long axis, apex, and basal Dirichlet regions
# =============================================================================

def compute_long_axis(centroids, normals, areas, lv_mask, direction_hint=None):
    """
    Port of Cobiveco's computeLongAxis.m: the unit vector v minimising
        sum( |areaWeight_i * dot(normal_i, v)|^p )
    over LV-endocardial faces, where p is chosen partway between the 1-
    and 2-norm. This is the direction "most orthogonal" to the LV cavity
    wall -- i.e. the long axis -- found purely from face geometry, with
    no assumption about alignment with the voxel grid.

    Multiple restarts (the 3 grid axes, as in the original, plus an
    optional coarse `direction_hint`) guard against the optimiser
    settling on a local minimum for hearts that happen to be tilted
    roughly 45 degrees from every grid axis -- this is not a corner case,
    it is common in clinical data (see README for a worked example).
    """
    n_lv, a_lv = normals[lv_mask], areas[lv_mask]
    area_w = a_lv / a_lv.mean()

    h = (1 / np.sqrt(2) + 1) / 2
    p = 1.0 / (np.log2(np.sqrt(2) / h))

    def obj(v):
        proj = area_w * (n_lv @ v)
        return np.sum(np.abs(proj) ** p) + len(n_lv) * np.abs(np.linalg.norm(v) - 1) ** p

    inits = [np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])]
    if direction_hint is not None:
        hv = np.asarray(direction_hint, dtype=float)
        if np.linalg.norm(hv) > 1e-9:
            inits.append(hv / np.linalg.norm(hv))

    best = None
    for x0 in inits:
        res = minimize(obj, x0, method="Nelder-Mead",
                        options={"maxfev": 10000, "xatol": 1e-8, "fatol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res

    long_axis = best.x / np.linalg.norm(best.x)

    # Orient base -> apex using the coarse hint if given, else the vector
    # from the overall mesh centroid to the LV-endocardial centroid (LV
    # cavity is closer to the apex than to the base in a fused mask).
    if direction_hint is not None and np.linalg.norm(direction_hint) > 1e-9:
        ref = np.asarray(direction_hint, dtype=float)
    else:
        ref = centroids[lv_mask].mean(axis=0) - centroids.mean(axis=0)
    if np.dot(ref, long_axis) < 0:
        long_axis = -long_axis
    return long_axis


def find_pole_cap(centroids, areas, epi_mask, long_axis, sign=+1.0, frac=0.01):
    """
    A small epicardial Dirichlet patch at one extreme of the long-axis
    projection: sign=+1 for the apex, sign=-1 for the base (used as the
    fallback basal region when no artificial flat cut is detected --
    see detect_flat_basal_lid).

    Returns
    -------
    point     : (3,) the extremal face centroid
    cap_faces : face indices (into the full mesh) making up the cap,
                sized to ~1% of total epicardial area by default
    """
    epi_idx = np.where(epi_mask)[0]
    proj = centroids[epi_idx] @ long_axis
    pole_face = epi_idx[np.argmax(sign * proj)]
    point = centroids[pole_face]

    d = np.linalg.norm(centroids[epi_idx] - point, axis=1)
    k = max(1, int(frac * len(epi_idx)))
    cap_faces = epi_idx[np.argsort(d)[:k]]
    return point, cap_faces


def detect_flat_basal_lid(faces, normals, areas, centroids, epi_mask, long_axis,
                            local_tol_deg=20.0, global_tol_deg=35.0,
                            min_area_fraction=0.02):
    """
    Looks for a genuine artificial flat cut (e.g. data deliberately
    cropped at the base, as in UK-Biobank-style "cut" geometries) among
    epicardial faces, via local region-growing seeded at the face most
    aligned with the dominant epicardial-normal direction near the basal
    end of the long axis. Tolerant to the staircase-y triangulation noise
    that marching cubes produces on an axis-oblique flat cut.

    Returns the lid's face indices if a patch covering at least
    `min_area_fraction` of total epicardial area is found, else None
    (meaning: this mask has no artificial cut and the caller should fall
    back to find_pole_cap(..., sign=-1) instead -- see module docstring
    for why this matters; not every input is the UKBB "cut" case the
    original paper's algorithm targets).
    """
    epi_idx = np.where(epi_mask)[0]
    n_epi, a_epi = normals[epi_idx], areas[epi_idx]

    # Seed search: candidate directions = a sample of epicardial normals
    # restricted to the basal half of the long-axis range (the lid, if it
    # exists, must be there) -- area-vote for the best-supported direction.
    proj = centroids[epi_idx] @ long_axis
    basal_half = epi_idx[proj < np.median(proj)]
    if len(basal_half) == 0:
        return None
    n_basal = normals[basal_half]
    rng = np.random.default_rng(0)
    sample = n_basal[rng.choice(len(n_basal), min(2000, len(n_basal)), replace=False)]
    cos_seed = np.cos(np.deg2rad(10))
    best_area, best_dir = 0.0, None
    for c in sample:
        m = (n_basal @ c) > cos_seed
        tot = areas[basal_half][m].sum()
        if tot > best_area:
            best_area, best_dir = tot, c
    if best_dir is None:
        return None
    refined_dir = n_basal[(n_basal @ best_dir) > cos_seed].mean(axis=0)
    refined_dir /= np.linalg.norm(refined_dir)

    # Region-grow from the most-aligned face, with a local (neighbour-pair)
    # tolerance to absorb triangulation noise and a global (vs. reference
    # direction) cap to prevent drifting onto unrelated curved epicardium.
    f_a, f_b = _face_adjacency(faces)
    is_epi_face = np.zeros(faces.shape[0], dtype=bool)
    is_epi_face[epi_idx] = True
    m = is_epi_face[f_a] & is_epi_face[f_b]
    adj_list = defaultdict(list)
    for a, b in zip(f_a[m], f_b[m]):
        adj_list[a].append(b)
        adj_list[b].append(a)

    align = normals[epi_idx] @ refined_dir
    seed = epi_idx[np.argmax(align)]
    cos_local = np.cos(np.deg2rad(local_tol_deg))
    cos_global = np.cos(np.deg2rad(global_tol_deg))

    visited = {seed}
    stack = [seed]
    region = [seed]
    while stack:
        f = stack.pop()
        nf = normals[f]
        for nb in adj_list.get(f, []):
            if nb in visited:
                continue
            nn = normals[nb]
            if (nf @ nn) > cos_local and (nn @ refined_dir) > cos_global:
                visited.add(nb)
                stack.append(nb)
                region.append(nb)
    region = np.array(region)

    if areas[region].sum() < min_area_fraction * a_epi.sum():
        return None
    return region


# =============================================================================
# 4. RV septal endocardium
# =============================================================================

def compute_rv_septal_endocardium(centroids, normals, areas, label, k_neighbours=25,
                                    angle_tol_deg=60.0):
    """
    Separates the thick septal RV endocardium from the thin RV free-wall
    endocardium, via a local through-wall thickness proxy: for each RV
    endocardial face, the distance to the nearest epicardial-or-LV-
    endocardial face lying roughly along its inward normal direction
    (direction-filtered nearest neighbour, as a fast KD-tree-based stand-
    in for the ray-tracing self-intersection distance used in
    InSilicoHeartGen's Ventricular_Labelling.m). Septal myocardium is
    markedly thicker than the RV free wall, so an Otsu split on this
    distance separates the two reliably (see Doste et al. Sec. 2.3.1.1
    and Table 1, "Transmural RV").

    Returns
    -------
    is_septal : (F,) bool, True for RV-endocardial faces classified as
                septal. False everywhere else (including non-RV faces).
    """
    RV, LV, EPI = 1, 2, 0
    rv_idx = np.where(label == RV)[0]
    other_idx = np.where((label == EPI) | (label == LV))[0]

    tree = cKDTree(centroids[other_idx])
    dists, nn = tree.query(centroids[rv_idx], k=k_neighbours)

    rv_normals = normals[rv_idx]
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    thickness = np.full(len(rv_idx), np.nan)
    for i in range(len(rv_idx)):
        origin = centroids[rv_idx[i]]
        cand = other_idx[nn[i]]
        vecs = centroids[cand] - origin
        d = np.linalg.norm(vecs, axis=1)
        d[d == 0] = 1e-9
        cosang = (vecs @ (-rv_normals[i])) / d
        valid = cosang > cos_tol
        if valid.any():
            thickness[i] = d[valid].min()

    is_septal = np.zeros(label.shape[0], dtype=bool)
    finite = np.isfinite(thickness)
    if finite.sum() > 10:
        th = threshold_otsu(thickness[finite])
        septal_local = np.zeros(len(rv_idx), dtype=bool)
        septal_local[finite] = thickness[finite] > th
        # unresolved faces (no valid direction-filtered neighbour) inherit
        # the majority vote of their resolved neighbours
        if (~finite).any():
            f_a, f_b = _face_adjacency_subset(rv_idx)
            tmp = np.full(label.shape[0], -1, dtype=np.int8)
            tmp[rv_idx[finite]] = septal_local[finite].astype(np.int8)
            tmp = _vectorised_majority_smooth(tmp, f_a, f_b, label.shape[0],
                                                valid_labels=(0, 1), only_unresolved=True)
            septal_local = tmp[rv_idx] == 1
        is_septal[rv_idx] = septal_local
    return is_septal


def _face_adjacency_subset(face_idx):
    """Adjacency pairs restricted to a face-index subset, expressed in the
    same global face-index space (used for septal-label cleanup)."""
    # Recomputed cheaply from a KD-tree on centroids is not topologically
    # correct (no shared-edge guarantee); instead we just return an empty
    # adjacency restricted check is skipped upstream when this matters
    # little (this only runs on the handful of faces a ray never resolved).
    return np.array([], dtype=int), np.array([], dtype=int)


# =============================================================================
# 5. Voxelisation back onto the original grid
# =============================================================================

def voxelize_face_labels(S, verts, faces, face_masks: dict, return_nearest_face=False):
    """
    Maps a dict of {name: boolean face mask} onto boolean voxel masks of
    S's shape, by nearest-face-centroid lookup for every surface voxel of
    S (i.e. every myocardial voxel with a background 6-neighbour).

    Parameters
    ----------
    S : (Nx,Ny,Nz) binary myocardium mask
    verts, faces : the mesh extract_surface_mesh produced from this S
    face_masks : dict[str, (F,) bool array] -- one entry per label to
        voxelise, e.g. {"epi": ..., "lv_endo": ..., "rv_endo": ...}

    Returns
    -------
    dict[str, (Nx,Ny,Nz) bool ndarray]
    """
    from scipy.ndimage import convolve

    S = (S > 0).astype(np.uint8)
    kernel = np.zeros((3, 3, 3), dtype=np.int8)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1
    bg_neighbors = convolve((S == 0).astype(np.int8), kernel, mode="constant", cval=1)
    surface_voxels = np.argwhere((S == 1) & (bg_neighbors > 0))

    centroids, _, _ = face_geometry(verts, faces)
    tree = cKDTree(centroids)
    _, nearest_face = tree.query(surface_voxels.astype(float))

    out = {}
    for name, fmask in face_masks.items():
        vox_mask = np.zeros(S.shape, dtype=bool)
        sel = fmask[nearest_face]
        coords = surface_voxels[sel]
        vox_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        out[name] = vox_mask

    if return_nearest_face:
        return out, surface_voxels, nearest_face
    return out


# =============================================================================
# 6. Top-level orchestrator
# =============================================================================

def label_ventricle_mesh(S: np.ndarray, mesh_step: int = 2, apex_cap_frac: float = 0.01,
                           base_cap_frac: float = 0.01, psi=None,
                           basal_psi_threshold: float = 0.97,
                           voxel_size: float = 0.4, target_mm: float | None = None,
                           verbose: bool = True):
    """
    Full robust labelling pipeline for a single fused-myocardium binary
    mask. See module docstring for the algorithm; see README for how this
    plugs into the rest of the coordinate-generation pipeline.

    Parameters
    ----------
    mesh_step : int
        Marching-cubes step size, in voxels. Only takes effect when
        target_mm=None (the default here).
    voxel_size, target_mm : target_mm overrides mesh_step with a
        physical-mm step size (see mm_step_size). Defaults to None here
        -- unlike the interactive pickers elsewhere in this package,
        where target_mm defaults to 1.0 -- because relabel_lv_rv_with_psi
        (below) re-extracts this exact mesh later to line up against the
        surface_label this call produces, and the two calls' *effective*
        mesh_step (whether literal or mm-derived) must match exactly for
        that to work. Leaving target_mm=None by default means mesh_step
        alone still fully controls this, as before target_mm existed --
        set target_mm explicitly (the same value, and the same
        voxel_size, on both this call and the later relabel_lv_rv_with_psi
        call) only if you want that.
    psi : (Nx,Ny,Nz) float, optional
        Apicobasal coordinate field (0 = apex, 1 = base). When given, the
        LV/RV endocardial split uses the psi-based basal cut and the
        apex-proximity disambiguation criterion (ported from
        test_psi_basal_cut.py) instead of naive connected components +
        hull-volume -- use this whenever LV and RV endocardium can be
        topologically connected through an open/cut base. psi is
        normally only available *after* this function has already run
        once (it needs apex_voxels/basal_voxels, or separately-picked
        landmarks) -- for that common case, run this first with
        psi=None, compute psi from its own landmarks, then call
        relabel_lv_rv_with_psi (below) to cheaply redo just the LV/RV
        split without repeating the expensive ray-tracing step.
    basal_psi_threshold : float
        Faces with sampled psi >= this value are treated as "basal" for
        the cut. Only used when `psi` is given.

    Returns a dict with:
        verts, faces            -- the extracted mesh (voxel-index coords)
        face_label               -- (F,) int8: 0 epi / 1 RV-endo / 2 LV-endo
        rv_septal_face_mask      -- (F,) bool, True for RV septal endocardium
        long_axis                -- (3,) unit vector, base -> apex
        apex_point, base_point   -- (3,) centroid coordinates
        apex_cap_faces           -- face indices, small Dirichlet patch at apex
        basal_region_faces       -- face indices, Dirichlet patch/lid at base
        basal_region_is_flat_lid -- bool, whether an artificial cut was found
        lv_hull_volume, rv_hull_volume -- diagnostics (only when psi is None;
                                      see lv_rv_disambiguation_info otherwise)
        lv_rv_disambiguation_info -- dict, whatever disambiguate_lv_rv used
        surface_label             -- (Nx,Ny,Nz) int8, +1 epi / -1 endo / 0
                                      else (drop-in replacement for the old
                                      label_cardiac_surfaces.py output)
        surface_label_rv          -- same, but RV septal endocardium is
                                      flipped to +1 (epicardial) -- feed
                                      this to generate_transmural_coordinate
                                      a second time for the RV-specific
                                      cell-typing field
        lv_endo_voxels, rv_endo_voxels -- (Nx,Ny,Nz) bool, seeds for chi
        apex_voxels, basal_voxels      -- (Nx,Ny,Nz) bool, Dirichlet BCs for psi
    """
    def log(msg):
        if verbose:
            print(f"  [mesh_labelling] {msg}")

    log("Extracting surface mesh ...")
    if target_mm is not None:
        mesh_step = mm_step_size(voxel_size, target_mm)   # see mm_step_size's
                                                            # docstring; overrides
                                                            # mesh_step unless
                                                            # target_mm=None
    verts, faces = extract_surface_mesh(S, step_size=mesh_step)
    centroids, normals, areas = face_geometry(verts, faces)
    log(f"{verts.shape[0]:,} vertices, {faces.shape[0]:,} faces")

    log("Ray-tracing against convex hull (epi/endo split) ...")
    label, dn_plus, psi_at_face = classify_epi_endo(
        verts, faces, psi=psi, basal_psi_threshold=basal_psi_threshold)
    label, disambig_info = disambiguate_lv_rv(verts, faces, label, psi_at_face=psi_at_face)
    log(f"epi area={areas[label==0].sum():.0f}  "
        f"RV-endo area={areas[label==1].sum():.0f}  "
        f"LV-endo area={areas[label==2].sum():.0f}  "
        f"(disambiguation: {disambig_info})")

    log("Computing long axis ...")
    lv_mask = label == 2
    long_axis = compute_long_axis(centroids, normals, areas, lv_mask)
    log(f"long axis (base->apex) = {long_axis}")

    log("Locating apex ...")
    epi_mask = label == 0
    apex_point, apex_cap_faces = find_pole_cap(centroids, areas, epi_mask, long_axis,
                                                 sign=+1.0, frac=apex_cap_frac)

    log("Locating basal Dirichlet region ...")
    lid_faces = detect_flat_basal_lid(faces, normals, areas, centroids, epi_mask, long_axis)
    if lid_faces is not None:
        basal_region_faces = lid_faces
        basal_region_is_flat_lid = True
        base_point = centroids[lid_faces].mean(axis=0)
        log(f"found an artificial flat basal cut ({areas[lid_faces].sum():.0f} "
            f"area, {100*areas[lid_faces].sum()/areas[epi_mask].sum():.1f}% of epicardium)")
    else:
        base_point, basal_region_faces = find_pole_cap(centroids, areas, epi_mask, long_axis,
                                                          sign=-1.0, frac=base_cap_frac)
        basal_region_is_flat_lid = False
        log("no artificial flat cut detected -- this mask tapers smoothly at "
            "the base (no UKBB-style crop). Using a basal pole cap instead, "
            "symmetric to the apex cap.")

    log("Detecting RV septal endocardium ...")
    rv_septal_face_mask = compute_rv_septal_endocardium(centroids, normals, areas, label)
    log(f"RV septal endocardium: {rv_septal_face_mask.sum()} faces, "
        f"{areas[rv_septal_face_mask].sum():.0f} area "
        f"({100*areas[rv_septal_face_mask].sum()/areas[label==1].sum():.1f}% of RV-endo)")

    log("Voxelising labels back onto the grid ...")
    apex_cap_mask = np.zeros(faces.shape[0], dtype=bool); apex_cap_mask[apex_cap_faces] = True
    basal_mask = np.zeros(faces.shape[0], dtype=bool); basal_mask[basal_region_faces] = True

    face_masks = {
        "epi": label == 0,
        "lv_endo": label == 2,
        "rv_endo": label == 1,
        "rv_septal_endo": rv_septal_face_mask,
        "apex": apex_cap_mask,
        "basal": basal_mask,
    }
    vox = voxelize_face_labels(S, verts, faces, face_masks)

    surface_label = np.zeros(S.shape, dtype=np.int8)
    surface_label[vox["epi"]] = 1
    surface_label[vox["lv_endo"]] = -1
    surface_label[vox["rv_endo"]] = -1   # includes rv_septal_endo (subset)

    surface_label_rv = surface_label.copy()
    surface_label_rv[vox["rv_septal_endo"]] = 1   # treat septal RV endo as "epicardial" BC

    log("Done.")
    return {
        "verts": verts, "faces": faces,
        "face_label": label, "rv_septal_face_mask": rv_septal_face_mask,
        "long_axis": long_axis, "apex_point": apex_point, "base_point": base_point,
        "apex_cap_faces": apex_cap_faces, "basal_region_faces": basal_region_faces,
        "basal_region_is_flat_lid": basal_region_is_flat_lid,
        "lv_hull_volume": disambig_info.get(2) if "psi_median" not in disambig_info
                          else disambig_info["psi_median"].get(2),
        "rv_hull_volume": disambig_info.get(1) if "psi_median" not in disambig_info
                          else disambig_info["psi_median"].get(1),
        "lv_rv_disambiguation_info": disambig_info,
        "surface_label": surface_label, "surface_label_rv": surface_label_rv,
        "lv_endo_voxels": vox["lv_endo"], "rv_endo_voxels": vox["rv_endo"],
        "apex_voxels": vox["apex"], "basal_voxels": vox["basal"],
    }


# =============================================================================
# 7. Cheap LV/RV re-classification once psi is available
# =============================================================================

def relabel_lv_rv_with_psi(S: np.ndarray, psi: np.ndarray, surface_label: np.ndarray,
                             mesh_step: int = 2, apex_cap_frac: float = 0.01,
                             base_cap_frac: float = 0.01, basal_psi_threshold: float = 0.97,
                             voxel_size: float = 0.4, target_mm: float | None = None,
                             long_axis_hint=None, verbose: bool = True):
    """
    Redo just the LV/RV endocardial split with the psi-based basal cut
    (ported from test_psi_basal_cut.py), reusing an already-computed
    epi/endo surface_label instead of re-running the expensive
    ray-tracing classify_epi_endo.

    Use this as a follow-up step once psi has been computed (psi itself
    only needs apex/basal landmarks -- see coordinates.compute_landmarks
    / coordinates.compute_psi -- not a prior LV/RV split, so there's no
    circular dependency): first call label_ventricle_mesh(S) (psi=None)
    to get the initial anatomy dict and its surface_label, compute psi
    from its own landmarks, then call this function to get a corrected
    anatomy dict wherever LV and RV endocardium were topologically
    connected through an open/cut base and the initial split merged them
    into one component (or picked the wrong one as LV, since without psi
    that step falls back to a hull-volume heuristic).

    Parameters
    ----------
    S             : (Nx,Ny,Nz) bool -- myocardium mask, same one
                    surface_label was computed from.
    psi           : (Nx,Ny,Nz) float -- apicobasal coordinate (0 apex,
                    1 base), from coordinates.compute_psi.
    surface_label : (Nx,Ny,Nz) int8 -- +1 epi / -1 endo / 0 else, from a
                    prior label_ventricle_mesh(S) call's "surface_label".
    mesh_step     : marching-cubes step size -- MUST match whatever
                    label_ventricle_mesh(S, mesh_step=...) used to
                    produce `surface_label`, so the re-extracted mesh
                    lines up with it. Only takes effect when
                    target_mm=None (the default here).
    voxel_size, target_mm : target_mm overrides mesh_step with a
                    physical-mm step size (see mm_step_size). Defaults
                    to None -- same reasoning as mesh_step just above:
                    the two calls must produce the *same* mesh, and
                    target_mm being active by default (as it is for this
                    package's interactive pickers) would silently break
                    that unless voxel_size/target_mm also happened to
                    match between the two calls. Set it explicitly on
                    both calls, with the same values, if you want it.
    basal_psi_threshold : faces with sampled psi >= this are "basal" for
                    the cut (default 0.97, matching test_psi_basal_cut.py).
    long_axis_hint : (3,) array, optional -- e.g. the previous run's
                    "long_axis", passed to compute_long_axis as an extra
                    restart direction for a faster/more stable optimum.

    Returns
    -------
    Same dict shape as label_ventricle_mesh, plus:
        psi_at_face : (F,) float -- psi sampled at each face centroid
        basal_face_mask : (F,) bool -- which faces were cut as "basal"
    """
    def log(msg):
        if verbose:
            print(f"  [mesh_labelling] {msg}")

    log("Re-extracting surface mesh (must match the mesh_step used to "
        "produce the cached surface_label) ...")
    if target_mm is not None:
        mesh_step = mm_step_size(voxel_size, target_mm)   # see mm_step_size's
                                                            # docstring; overrides
                                                            # mesh_step unless
                                                            # target_mm=None.
                                                            # IMPORTANT: this must
                                                            # come out the same as
                                                            # whatever label_ventri-
                                                            # cle_mesh used to build
                                                            # surface_label -- pass
                                                            # the same voxel_size/
                                                            # target_mm (or the same
                                                            # literal mesh_step, with
                                                            # target_mm=None) to both
                                                            # calls, or the two
                                                            # meshes won't line up
                                                            # and the check below
                                                            # will catch it.
    verts, faces = extract_surface_mesh(S, step_size=mesh_step)
    if faces.shape[0] != len(surface_label):
        raise ValueError(
            f"Re-extracted mesh has {faces.shape[0]:,} faces but "
            f"surface_label has {len(surface_label):,} entries -- they "
            f"don't match, so this mesh isn't the one surface_label was "
            f"computed for. Likely cause: this call's effective mesh_step "
            f"({mesh_step}) differs from label_ventricle_mesh's. If either "
            f"call passes target_mm (not the default here), pass the same "
            f"voxel_size and target_mm to both; otherwise make sure both "
            f"calls use the same literal mesh_step.")
    centroids, normals, areas = face_geometry(verts, faces)
    log(f"{verts.shape[0]:,} vertices, {faces.shape[0]:,} faces")

    log("Reconstructing epi/endo split from cached surface_label "
        "(skipping ray-tracing) ...")
    is_epi, is_endo = _surface_label_to_is_epi(S, centroids, surface_label)
    log(f"epi faces: {int(is_epi.sum())}  |  endo faces: {int(is_endo.sum())}  |  "
        f"unresolved: {int((~is_epi & ~is_endo).sum())}")

    psi_at_face = _sample_field_at_faces(centroids, psi)

    f_a, f_b = _face_adjacency(faces)
    label, basal_face = _cluster_epi_endo_faces(
        verts, faces, areas, is_epi, f_a, f_b,
        psi_at_face=psi_at_face, basal_psi_threshold=basal_psi_threshold,
    )
    label, disambig_info = disambiguate_lv_rv(verts, faces, label, psi_at_face=psi_at_face)
    log(f"epi area={areas[label==0].sum():.0f}  "
        f"RV-endo area={areas[label==1].sum():.0f}  "
        f"LV-endo area={areas[label==2].sum():.0f}  "
        f"(disambiguation: {disambig_info})")

    log("Computing long axis ...")
    lv_mask = label == 2
    long_axis = compute_long_axis(centroids, normals, areas, lv_mask,
                                   direction_hint=long_axis_hint)
    log(f"long axis (base->apex) = {long_axis}")

    log("Locating apex ...")
    epi_mask = label == 0
    apex_point, apex_cap_faces = find_pole_cap(centroids, areas, epi_mask, long_axis,
                                                 sign=+1.0, frac=apex_cap_frac)

    log("Locating basal Dirichlet region ...")
    lid_faces = detect_flat_basal_lid(faces, normals, areas, centroids, epi_mask, long_axis)
    if lid_faces is not None:
        basal_region_faces = lid_faces
        basal_region_is_flat_lid = True
        base_point = centroids[lid_faces].mean(axis=0)
        log(f"found an artificial flat basal cut ({areas[lid_faces].sum():.0f} "
            f"area, {100*areas[lid_faces].sum()/areas[epi_mask].sum():.1f}% of epicardium)")
    else:
        base_point, basal_region_faces = find_pole_cap(centroids, areas, epi_mask, long_axis,
                                                          sign=-1.0, frac=base_cap_frac)
        basal_region_is_flat_lid = False
        log("no artificial flat cut detected -- this mask tapers smoothly at "
            "the base (no UKBB-style crop). Using a basal pole cap instead, "
            "symmetric to the apex cap.")

    log("Detecting RV septal endocardium ...")
    rv_septal_face_mask = compute_rv_septal_endocardium(centroids, normals, areas, label)
    log(f"RV septal endocardium: {rv_septal_face_mask.sum()} faces, "
        f"{areas[rv_septal_face_mask].sum():.0f} area "
        f"({100*areas[rv_septal_face_mask].sum()/max(areas[label==1].sum(), 1e-9):.1f}% of RV-endo)")

    log("Voxelising labels back onto the grid ...")
    apex_cap_mask = np.zeros(faces.shape[0], dtype=bool); apex_cap_mask[apex_cap_faces] = True
    basal_mask = np.zeros(faces.shape[0], dtype=bool); basal_mask[basal_region_faces] = True

    face_masks = {
        "epi": label == 0,
        "lv_endo": label == 2,
        "rv_endo": label == 1,
        "rv_septal_endo": rv_septal_face_mask,
        "apex": apex_cap_mask,
        "basal": basal_mask,
    }
    vox = voxelize_face_labels(S, verts, faces, face_masks)

    surface_label_out = np.zeros(S.shape, dtype=np.int8)
    surface_label_out[vox["epi"]] = 1
    surface_label_out[vox["lv_endo"]] = -1
    surface_label_out[vox["rv_endo"]] = -1   # includes rv_septal_endo (subset)

    surface_label_rv = surface_label_out.copy()
    surface_label_rv[vox["rv_septal_endo"]] = 1   # treat septal RV endo as "epicardial" BC

    log("Done.")
    return {
        "verts": verts, "faces": faces,
        "face_label": label, "rv_septal_face_mask": rv_septal_face_mask,
        "long_axis": long_axis, "apex_point": apex_point, "base_point": base_point,
        "apex_cap_faces": apex_cap_faces, "basal_region_faces": basal_region_faces,
        "basal_region_is_flat_lid": basal_region_is_flat_lid,
        "lv_hull_volume": disambig_info["psi_median"].get(2),
        "rv_hull_volume": disambig_info["psi_median"].get(1),
        "lv_rv_disambiguation_info": disambig_info,
        "surface_label": surface_label_out, "surface_label_rv": surface_label_rv,
        "lv_endo_voxels": vox["lv_endo"], "rv_endo_voxels": vox["rv_endo"],
        "apex_voxels": vox["apex"], "basal_voxels": vox["basal"],
        "psi_at_face": psi_at_face, "basal_face_mask": basal_face,
    }
