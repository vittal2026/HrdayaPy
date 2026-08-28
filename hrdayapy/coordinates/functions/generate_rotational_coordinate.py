"""
generate_rotational_coordinate.py
==================================
Compute the rotational (circumferential) coordinate θ ∈ [0, 2π) for every
myocardial voxel, using the anatomically-grounded algorithm developed in
identify_septal_ridge.py and compute_theta_pipeline.py.

Overview
--------
The algorithm is split into three clearly separated stages, each exposed as
a standalone callable function so any stage can be re-run independently:

  Stage 1 — compute_septal_loop(S, chi_labels)
      Skeletonise the LV–RV septal contact boundary into a single ordered
      3-D curve (the "seam loop") that runs from apex to base and back.

  Stage 2 — pick_septal_landmarks(surf_mesh, loop)  [interactive]
      Open a PyVista window: user picks APEX and AV NODE on the loop (hover
      + press P), then confirms which arc is the POSTERIOR ridge (press Y/C).
      Returns anterior_arc, posterior_arc as (N,3) arrays.

  Stage 2b — pick_anterior_vertex(surf_mesh, ant_arc, post_arc, loop)  [interactive]
      User picks the apical tip of the anterior ridge — the θ = 0 anchor
      point used to initialise the per-bin reference frame at the apex.

  Stage 3 — compute_theta(chi_labels, psi, long_axis, ant_arc, post_arc)
      Build θ independently for LV (chi=1) and RV (chi=2) and merge into
      one volume.  At each ψ bin:
        C(ψ)  = centroid of in-plane voxels (innermost 35% for LV; all for RV)
        e₁(ψ) = unit vector C(ψ) → anterior ridge at same ψ level  (θ = 0)
        e₂(ψ) = long_axis × e₁, sign fixed so posterior ridge → θ ≈ π
      The frame is smoothed along ψ, then θ = atan2(e₂·d, e₁·d) for each
      voxel displacement d = coord_perp − C.

Public API (used by coordinates/rotational.py's compute_theta)
----------------------------------------------------------------
    loop, ant_arc, post_arc = compute_septal_loop_and_arcs(
        S, chi_labels, long_axis, psi, mesh_step=2,
        ridge_save_path=None)              # loads cached .npz if it exists

    theta = compute_theta(
        chi_labels, psi, long_axis, ant_arc, post_arc,
        n_bins=100, lv_inner_frac=0.35, rv_inner_frac=0.0,
        smooth_sigma=2.0, psi_apex_threshold=0.05,
        save_path=None)

    theta = generate_rotational_coordinate(
        S, chi_labels, psi, long_axis,
        mesh_step=2, n_bins=100, smooth_sigma=2.0,
        ridge_save_path=None, anterior_vertex_save_path=None,
        save_path=None)                    # convenience wrapper for the full flow
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()

from collections import defaultdict
from pathlib import Path
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, gaussian_filter1d, binary_dilation
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize, ball


# =============================================================================
# Mesh / line helpers  (shared with visualisation steps)
# =============================================================================

def _to_pyvista_surface(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    F = faces.shape[0]
    vtk_faces = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(verts.astype(np.float32), vtk_faces)


def _to_pyvista_line(pts: np.ndarray, close: bool = False) -> pv.PolyData:
    pts = pts.astype(np.float32)
    if close:
        pts = np.vstack([pts, pts[:1]])
    N = len(pts) - 1
    if N < 1:
        return pv.PolyData()
    lines = np.hstack([
        np.full((N, 1), 2, dtype=np.int64),
        np.column_stack([np.arange(N), np.arange(1, N + 1)]),
    ]).ravel()
    mesh = pv.PolyData()
    mesh.points = pts
    mesh.lines = lines
    return mesh


def _build_surface_mesh(S: np.ndarray, mesh_step: int = 2):
    """Marching-cubes surface of S for visualisation.

    Purely cosmetic: the mask is lightly blurred before thresholding and the
    resulting mesh is Taubin-smoothed so the picking window doesn't show the
    raw voxel staircase. Landmarks picked in pick_septal_landmarks() snap to
    `loop` (a separately computed curve), not to this mesh, so smoothing it
    cannot shift the picked coordinates.
    """
    Sp = np.pad(S.astype(np.uint8), 2, mode="constant")
    Sp = gaussian_filter(Sp.astype(float), sigma=0.8)
    verts, faces, _, _ = marching_cubes(Sp, level=0.5, step_size=mesh_step)
    verts -= 2.0
    mesh = _to_pyvista_surface(verts, faces)
    mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1, normalize_coordinates=True)
    return mesh


# =============================================================================
# Stage 1 — Septal-loop extraction
# =============================================================================

def compute_septal_loop(
    S: np.ndarray,
    chi_labels: np.ndarray,
) -> np.ndarray:
    """
    Extract and order the LV–RV septal boundary loop on the outer surface
    of the myocardium.

    Algorithm
    ---------
    1. Seam voxels: myocardial voxels of one chi label that are 6-adjacent
       to the other label.
    2. Restrict to the outer surface of S (voxels with ≥1 background
       6-neighbour) → the 1-D boundary of the 2-D septal patch.
    3. Skeletonise to a 1-voxel-wide loop.
    4. Collapse knot clusters (degree > 2 vertices caused by voxel-grid
       staircase noise) to single representative vertices.
    5. Walk as an ordered chain with direction continuity.
    6. Gaussian-smooth (σ=4, periodic wrap) to reduce staircase artefacts.

    Parameters
    ----------
    S          : (Nx,Ny,Nz) bool   myocardium mask
    chi_labels : (Nx,Ny,Nz) int    0=background, 1=LV, 2=RV

    Returns
    -------
    loop : (N, 3) float   ordered voxel-index coordinates of the closed loop
    """
    kernel6 = np.zeros((3, 3, 3), dtype=np.int8)
    kernel6[1, 0, 1] = kernel6[1, 2, 1] = 1
    kernel6[0, 1, 1] = kernel6[2, 1, 1] = 1
    kernel6[1, 1, 0] = kernel6[1, 1, 2] = 1

    lv = (chi_labels == 1).astype(np.int8)
    rv = (chi_labels == 2).astype(np.int8)
    seam = (
        ((ndi.convolve(rv, kernel6, mode="constant") > 0) & (chi_labels == 1)) |
        ((ndi.convolve(lv, kernel6, mode="constant") > 0) & (chi_labels == 2))
    )

    bg_nb = ndi.convolve(
        (chi_labels == 0).astype(np.int8), kernel6, mode="constant", cval=1
    )
    boundary = seam & (bg_nb > 0)
    print(f"  [septal loop] seam boundary voxels: {boundary.sum()}")

    # Skeletonise
    vol = np.zeros(S.shape, dtype=bool)
    coords = np.argwhere(boundary)
    vol[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    vol = binary_dilation(vol, structure=ball(1))
    skel = skeletonize(vol)
    skel_coords = np.argwhere(skel)
    print(f"  [septal loop] skeleton voxels: {len(skel_coords)}")

    # Build adjacency
    tree = cKDTree(skel_coords)
    pairs = tree.query_pairs(r=1.75)
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)

    # Collapse knot clusters (degree > 2 vertices)
    knots = [v for v, ns in adj.items() if len(ns) > 2]
    if knots:
        kcoords = skel_coords[knots]
        ktree = cKDTree(kcoords)
        kpairs = ktree.query_pairs(r=3.0)

        if kpairs:
            rows = [i for i, j in kpairs] + [j for i, j in kpairs]
            cols = [j for i, j in kpairs] + [i for i, j in kpairs]
            mat = coo_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(len(knots), len(knots)),
            ).tocsr()
        else:
            mat = coo_matrix((len(knots), len(knots)))

        n_kc, klabs = connected_components(mat, directed=False)

        for cluster_id in range(n_kc):
            cluster_verts = [
                knots[i] for i in range(len(knots)) if klabs[i] == cluster_id
            ]
            if len(cluster_verts) < 2:
                continue
            center = skel_coords[cluster_verts].mean(axis=0)
            keeper = min(
                cluster_verts,
                key=lambda v: np.linalg.norm(skel_coords[v] - center),
            )
            for kv in cluster_verts:
                if kv == keeper:
                    continue
                for n in list(adj[kv]):
                    adj[n].discard(kv)
                    if n != keeper:
                        adj[keeper].add(n)
                        adj[n].add(keeper)
                adj[kv].clear()
            adj[keeper].discard(keeper)

            ns = list(adj[keeper])
            if len(ns) > 2:
                best = [ns[0]]
                for n in ns[1:]:
                    vn = skel_coords[n] - skel_coords[keeper]
                    cosines = [
                        np.dot(vn, skel_coords[k] - skel_coords[keeper]) /
                        (np.linalg.norm(vn) *
                         np.linalg.norm(skel_coords[k] - skel_coords[keeper]) + 1e-9)
                        for k in best
                    ]
                    if max(cosines) < 0.5:
                        best.append(n)
                    if len(best) == 2:
                        break
                for n in ns:
                    if n not in best:
                        adj[keeper].discard(n)
                        adj[n].discard(keeper)

    active_count = sum(1 for v, ns in adj.items() if ns)
    print(f"  [septal loop] after knot cleanup: {active_count} active vertices")

    # Walk the ordered loop
    active = {v for v, ns in adj.items() if ns}
    start = min(active, key=lambda v: skel_coords[v].sum())
    loop_idx = [start]
    visited = {start}
    curr = start
    prev_v = None
    for _ in range(len(active) + 5):
        nbrs = adj[curr] - visited
        if not nbrs:
            break
        if prev_v is None:
            nxt = next(iter(nbrs))
        else:
            direction = skel_coords[curr] - skel_coords[prev_v]
            nxt = max(
                nbrs,
                key=lambda n: np.dot(skel_coords[n] - skel_coords[curr], direction),
            )
        loop_idx.append(nxt)
        visited.add(nxt)
        prev_v = curr
        curr = nxt

    raw = skel_coords[loop_idx].astype(float)
    smooth = gaussian_filter1d(raw, sigma=4, axis=0, mode="wrap")
    gap = np.linalg.norm(raw[-1] - raw[0])
    print(
        f"  [septal loop] ordered loop: {len(smooth)} vertices, "
        f"close gap: {gap:.2f} vox"
    )
    return smooth


# =============================================================================
# Stage 2a — Interactive: pick apex + AV node, split loop into arcs
# =============================================================================

def pick_septal_landmarks(
    surf_mesh: pv.PolyData,
    loop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Open ONE interactive PyVista window showing the heart surface with the
    seam loop.  User picks APEX (first press P) then AV NODE (second press P).
    Then a second window lets the user confirm which arc is POSTERIOR (Y/C).

    Returns
    -------
    apex_pt      : (3,) voxel coords snapped to loop
    av_pt        : (3,) voxel coords snapped to loop
    anterior_arc : (K,3) ordered 3-D coordinates
    posterior_arc: (M,3) ordered 3-D coordinates
    """
    # ── Pick apex + AV node ────────────────────────────────────────────────
    state = {"picks": []}
    labels = ["APEX", "AV NODE"]
    colors = ["yellow", "magenta"]
    loop_tree = cKDTree(loop)
    loop_line = _to_pyvista_line(loop, close=True)

    plotter = pv.Plotter()
    plotter.add_text(
        "Pick TWO landmarks in order:\n"
        "  1st pick = APEX   2nd pick = AV NODE\n"
        "Hover + press P to pick.  Rotate with left-drag.",
        font_size=11, color="white",
    )
    plotter.add_mesh(surf_mesh, color="lightcoral", opacity=0.55,
                     show_edges=False, smooth_shading=True)
    plotter.add_mesh(loop_line, color="cyan", line_width=4,
                     render_lines_as_tubes=True)

    def on_pick(point):
        if len(state["picks"]) >= 2:
            return
        _, nearest = loop_tree.query(np.asarray(point))
        snapped = loop[nearest]
        label = labels[len(state["picks"])]
        color = colors[len(state["picks"])]
        state["picks"].append(snapped)
        print(f"  Picked {label}: loop vertex {nearest}, "
              f"coords ({snapped[0]:.1f}, {snapped[1]:.1f}, {snapped[2]:.1f})")
        plotter.add_mesh(pv.Sphere(radius=2.5, center=snapped),
                         color=color, name=f"pick_{label}")
        plotter.render()

    plotter.enable_point_picking(callback=on_pick, show_message=True,
                                  show_point=True, color="red", point_size=10,
                                  tolerance=0.025)
    plotter.show()
    try:
        plotter.close()
    except Exception:
        pass

    if len(state["picks"]) < 2:
        raise RuntimeError(
            f"Need 2 picks (apex + AV node), got {len(state['picks'])}. "
            "Re-run and pick both landmarks before closing the window."
        )

    apex_pt, av_pt = state["picks"]

    # ── Split loop into two arcs ───────────────────────────────────────────
    tree = cKDTree(loop)
    _, i_apex = tree.query(apex_pt)
    _, i_av   = tree.query(av_pt)

    if i_apex > i_av:
        i_apex, i_av = i_av, i_apex
        apex_pt, av_pt = av_pt, apex_pt

    arc1 = loop[i_apex : i_av + 1]
    arc2 = np.vstack([loop[i_av:], loop[: i_apex + 1]])

    # ── Confirm which arc is posterior ─────────────────────────────────────
    confirm_state = {"choice": None}

    arc1_mesh = _to_pyvista_line(arc1, close=False)
    arc2_mesh = _to_pyvista_line(arc2, close=False)

    pl2 = pv.Plotter()
    pl2.add_text(
        "Which arc is the POSTERIOR interventricular ridge?\n"
        "  Press Y = YELLOW arc is posterior\n"
        "  Press C = CYAN arc is posterior\n"
        "Rotate freely to inspect both sides.",
        font_size=11, color="white",
    )
    pl2.add_mesh(surf_mesh, color="lightcoral", opacity=0.55,
                 show_edges=False, smooth_shading=True)
    pl2.add_mesh(arc1_mesh, color="yellow", line_width=6,
                 render_lines_as_tubes=True)
    pl2.add_mesh(arc2_mesh, color="cyan", line_width=6,
                 render_lines_as_tubes=True)
    pl2.add_mesh(pv.Sphere(radius=2.5, center=apex_pt), color="white")
    pl2.add_mesh(pv.Sphere(radius=2.5, center=av_pt), color="magenta")

    def _pick_yellow():
        confirm_state["choice"] = "arc1"
        pl2.add_text("YELLOW = posterior  ✓", name="confirm",
                     font_size=14, color="yellow", position="lower_left")
        pl2.render()

    def _pick_cyan():
        confirm_state["choice"] = "arc2"
        pl2.add_text("CYAN = posterior  ✓", name="confirm",
                     font_size=14, color="cyan", position="lower_left")
        pl2.render()

    pl2.add_key_event("y", _pick_yellow)
    pl2.add_key_event("Y", _pick_yellow)
    pl2.add_key_event("c", _pick_cyan)
    pl2.add_key_event("C", _pick_cyan)
    pl2.show()
    try:
        pl2.close()
    except Exception:
        pass

    if confirm_state["choice"] is None:
        print("  WARNING: no key pressed — defaulting to arc1 (yellow) as posterior.")
        confirm_state["choice"] = "arc1"

    if confirm_state["choice"] == "arc1":
        posterior_arc, anterior_arc = arc1, arc2
    else:
        posterior_arc, anterior_arc = arc2, arc1

    return apex_pt, av_pt, anterior_arc, posterior_arc


# =============================================================================
# Stage 2b — Interactive: pick anterior vertex (apical tip of anterior ridge)
# =============================================================================

def pick_anterior_vertex(
    surf_mesh: pv.PolyData,
    ant_arc: np.ndarray,
    post_arc: np.ndarray,
    loop: np.ndarray,
) -> np.ndarray:
    """
    Open a PyVista window showing the anterior ridge (yellow) and posterior
    ridge (cyan).  User hovers over the apical tip of the anterior ridge and
    presses P to pick it.  The pick is snapped to the nearest point on ant_arc.

    Returns
    -------
    anterior_vertex : (3,) voxel coords — the θ = 0 anchor near the apex
    """
    state = {"pick": None}
    ant_tree = cKDTree(ant_arc)

    plotter = pv.Plotter()
    plotter.add_text(
        "Pick the ANTERIOR VERTEX\n"
        "(apical tip of the yellow anterior ridge)\n"
        "Hover + press P  ·  Rotate with left-drag  ·  Close when done",
        font_size=11, color="white",
    )
    plotter.add_mesh(surf_mesh, color="#c0705a", opacity=0.45,
                     show_edges=False, smooth_shading=True)
    plotter.add_mesh(_to_pyvista_line(loop, close=True),
                     color="#888888", line_width=2,
                     render_lines_as_tubes=True, opacity=0.5)
    plotter.add_mesh(_to_pyvista_line(ant_arc),  color="#ffdd44", line_width=5,
                     render_lines_as_tubes=True)
    plotter.add_mesh(_to_pyvista_line(post_arc), color="#44ddff", line_width=5,
                     render_lines_as_tubes=True)

    def on_pick(point):
        if state["pick"] is not None:
            return
        _, idx = ant_tree.query(np.asarray(point))
        snapped = ant_arc[idx]
        state["pick"] = snapped
        print(f"  Picked anterior vertex: arc index {idx}  "
              f"({snapped[0]:.1f}, {snapped[1]:.1f}, {snapped[2]:.1f})")
        plotter.add_mesh(pv.Sphere(radius=3.0, center=snapped),
                         color="#ffffff", name="ant_vertex_sphere")
        plotter.render()

    plotter.enable_point_picking(callback=on_pick, show_message=True,
                                  show_point=True, color="red",
                                  point_size=10, tolerance=0.025)
    plotter.show()
    try:
        plotter.close()
    except Exception:
        pass

    if state["pick"] is None:
        raise RuntimeError(
            "No anterior vertex picked. Re-run and press P before closing."
        )
    return state["pick"]


# =============================================================================
# Stage 1+2 combined: compute loop and split into arcs (cache-aware)
# =============================================================================

def compute_septal_loop_and_arcs(
    S: np.ndarray,
    chi_labels: np.ndarray,
    *,
    mesh_step: int = 2,
    ridge_save_path: str | Path | None = None,
    anterior_vertex_save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Stage 1 (septal loop) + Stage 2a (landmark picking + arc split) +
    Stage 2b (anterior vertex), with optional caching.

    If `ridge_save_path` already exists on disk, the loop + arcs are loaded
    from it and the interactive windows are skipped.  Delete the file to
    force a re-pick.

    Returns
    -------
    loop             : (N, 3) full ordered seam loop
    anterior_arc     : (K, 3)
    posterior_arc    : (M, 3)
    anterior_vertex  : (3,)  apical tip of the anterior ridge
    """
    # ── Load from cache ────────────────────────────────────────────────────
    if ridge_save_path is not None:
        ridge_save_path = Path(ridge_save_path)
    if anterior_vertex_save_path is not None:
        anterior_vertex_save_path = Path(anterior_vertex_save_path)

    ridge_cached = ridge_save_path is not None and ridge_save_path.exists()
    av_cached = (
        anterior_vertex_save_path is not None
        and anterior_vertex_save_path.exists()
    )

    if ridge_cached and av_cached:
        print(f"  [rotational] Loading cached septal ridge from {ridge_save_path}")
        z = np.load(ridge_save_path)
        loop          = z["loop"]
        anterior_arc  = z["anterior_arc"]
        posterior_arc = z["posterior_arc"]
        anterior_vertex = np.load(anterior_vertex_save_path)
        print(f"  [rotational] Loading cached anterior vertex from "
              f"{anterior_vertex_save_path}")
        return loop, anterior_arc, posterior_arc, anterior_vertex

    # ── Stage 1: build loop ────────────────────────────────────────────────
    print("  [rotational] Stage 1: computing septal loop …")
    loop = compute_septal_loop(S, chi_labels)

    # ── Build surface mesh for interactive picking ─────────────────────────
    print("  [rotational] Building surface mesh for interactive picking …")
    surf_mesh = _build_surface_mesh(S, mesh_step=mesh_step)

    # ── Stage 2a: interactive landmarks ───────────────────────────────────
    print("  [rotational] Stage 2a: interactive landmark picking (apex + AV node) …")
    print("    Hover + press P for APEX first, then AV NODE. Close window when done.")
    apex_pt, av_pt, anterior_arc, posterior_arc = pick_septal_landmarks(
        surf_mesh, loop
    )

    # ── Stage 2b: anterior vertex ──────────────────────────────────────────
    print("  [rotational] Stage 2b: pick apical tip of anterior ridge …")
    print("    Hover + press P on the yellow anterior ridge tip. Close when done.")
    anterior_vertex = pick_anterior_vertex(surf_mesh, anterior_arc, posterior_arc, loop)

    # ── Cache ──────────────────────────────────────────────────────────────
    if ridge_save_path is not None:
        ridge_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ridge_save_path,
            loop=loop,
            anterior_arc=anterior_arc,
            posterior_arc=posterior_arc,
            apex_on_loop=apex_pt,
            av_on_loop=av_pt,
        )
        print(f"  [rotational] Saved septal ridge → {ridge_save_path}")

    if anterior_vertex_save_path is not None:
        anterior_vertex_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(anterior_vertex_save_path, anterior_vertex)
        print(f"  [rotational] Saved anterior vertex → {anterior_vertex_save_path}")

    return loop, anterior_arc, posterior_arc, anterior_vertex


# =============================================================================
# Stage 3 — θ computation (ψ-stratified, per-ventricle)
# =============================================================================

def _compute_theta_for_ventricle(
    chi_label_id: int,
    label_name: str,
    chi_labels: np.ndarray,
    psi: np.ndarray,
    long_axis: np.ndarray,
    ant_arc: np.ndarray,
    post_arc: np.ndarray,
    *,
    n_bins: int = 100,
    lv_inner_frac: float = 0.35,
    rv_inner_frac: float = 0.0,
    smooth_sigma: float = 2.0,
    psi_apex_threshold: float = 0.05,
) -> np.ndarray:
    """
    Compute θ for all voxels belonging to chi_label_id (1=LV or 2=RV).

    Reference frame at each ψ bin:
      C(ψ)  = centroid of in-plane voxels (innermost endo_inner_frac for LV;
               all voxels for RV)
      e₁(ψ) = normalise( A_perp(ψ) − C(ψ) )   where A is the anterior ridge
      e₂(ψ) = long_axis × e₁, sign fixed so posterior ridge lands at θ ≈ π

    Returns flat (V,) array of θ values (NaN for apex cap) matching
    np.argwhere(chi_labels == chi_label_id) order.
    """
    endo_inner_frac = lv_inner_frac if chi_label_id == 1 else rv_inner_frac
    shape  = psi.shape
    coords = np.argwhere(chi_labels == chi_label_id).astype(float)
    psi_v  = psi[chi_labels == chi_label_id]
    print(f"  [θ {label_name}] {len(coords):,} voxels")

    # In-plane projection (remove long-axis component)
    def proj(pts):
        return pts - np.outer(pts @ long_axis, long_axis)

    coords_perp = proj(coords)
    ant_perp    = proj(ant_arc)
    post_perp   = proj(post_arc)

    # ψ values along the ridge arcs
    def psi_at(arc):
        idx = np.clip(np.round(arc).astype(int), 0, np.array(shape) - 1)
        return psi[idx[:, 0], idx[:, 1], idx[:, 2]]

    psi_ant  = psi_at(ant_arc)
    psi_post = psi_at(post_arc)

    # Per-bin reference frames
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    C  = np.full((n_bins, 3), np.nan)
    e1 = np.full((n_bins, 3), np.nan)
    e2 = np.full((n_bins, 3), np.nan)

    for i in range(n_bins):
        in_bin = (psi_v >= edges[i]) & (psi_v < edges[i + 1])
        if in_bin.sum() < 5:
            continue

        vox_perp = coords_perp[in_bin]
        ring_c   = vox_perp.mean(axis=0)

        if endo_inner_frac > 0:
            dists  = np.linalg.norm(vox_perp - ring_c, axis=1)
            cutoff = np.quantile(dists, endo_inner_frac)
            C[i]   = vox_perp[dists <= cutoff].mean(axis=0)
        else:
            C[i] = ring_c

        # Anterior-ridge anchor at this ψ level → e₁
        a_idx = np.argmin(np.abs(psi_ant - centers[i]))
        vec   = ant_perp[a_idx] - C[i]
        n     = np.linalg.norm(vec)
        if n < 1e-6:
            continue
        e1[i] = vec / n

        # e₂ via right-hand rule; sign fixed by posterior check
        raw = np.cross(long_axis, e1[i])
        n2  = np.linalg.norm(raw)
        if n2 < 1e-6:
            continue
        e2[i] = raw / n2

        p_idx   = np.argmin(np.abs(psi_post - centers[i]))
        d_post  = post_perp[p_idx] - C[i]
        theta_p = np.arctan2(e2[i] @ d_post, e1[i] @ d_post)
        if theta_p < 0:
            e2[i] = -e2[i]

    # Fill NaN bins by nearest valid bin
    valid = np.where(~np.isnan(C[:, 0]))[0]
    if len(valid) == 0:
        raise RuntimeError(
            f"No valid ψ bins for {label_name}. Check chi_labels / psi."
        )
    for arr in (C, e1, e2):
        for i in range(n_bins):
            if np.isnan(arr[i, 0]):
                arr[i] = arr[valid[np.argmin(np.abs(valid - i))]]

    # Smooth frames along ψ
    if smooth_sigma > 0:
        C  = gaussian_filter1d(C,  sigma=smooth_sigma, axis=0, mode="nearest")
        e1 = gaussian_filter1d(e1, sigma=smooth_sigma, axis=0, mode="nearest")
        e2 = gaussian_filter1d(e2, sigma=smooth_sigma, axis=0, mode="nearest")

    # Re-normalise after smoothing
    for i in range(n_bins):
        for arr in (e1, e2):
            n = np.linalg.norm(arr[i])
            if n > 1e-9:
                arr[i] /= n

    # Interpolate frame to each voxel's ψ level
    psi_c = np.clip(psi_v, centers[0], centers[-1])
    frac  = np.interp(psi_c, centers, np.arange(n_bins, dtype=float))
    lo    = np.floor(frac).astype(int)
    hi    = np.minimum(lo + 1, n_bins - 1)
    t     = (frac - lo)[:, None]

    C_v  = C[lo]  * (1 - t) + C[hi]  * t
    e1_v = e1[lo] * (1 - t) + e1[hi] * t
    e2_v = e2[lo] * (1 - t) + e2[hi] * t

    for v in (e1_v, e2_v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v[:] = np.where(n > 1e-9, v / n, v)

    d = coords_perp - C_v
    theta_v = np.arctan2(
        np.einsum("vi,vi->v", e2_v, d),
        np.einsum("vi,vi->v", e1_v, d),
    )
    theta_v[theta_v < 0] += 2 * np.pi          # → [0, 2π)
    theta_v[psi_v < psi_apex_threshold] = np.nan

    valid_count = (~np.isnan(theta_v)).sum()
    print(f"    valid θ: {valid_count:,}   "
          f"range [{np.nanmin(theta_v):.3f}, {np.nanmax(theta_v):.3f}] rad")
    return theta_v


def compute_theta(
    chi_labels: np.ndarray,
    psi: np.ndarray,
    long_axis: np.ndarray,
    ant_arc: np.ndarray,
    post_arc: np.ndarray,
    *,
    n_bins: int = 100,
    lv_inner_frac: float = 0.35,
    rv_inner_frac: float = 0.0,
    smooth_sigma: float = 2.0,
    psi_apex_threshold: float = 0.05,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Compute θ independently for LV (chi=1) and RV (chi=2) and merge into
    one volume.  Background voxels remain NaN.

    Parameters
    ----------
    chi_labels          : (Nx,Ny,Nz) int    0=bg, 1=LV, 2=RV
    psi                 : (Nx,Ny,Nz) float  apicobasal coordinate [0,1]
    long_axis           : (3,) float         unit vector apex→base
    ant_arc, post_arc   : (N,3) float        ordered 3-D arc coordinates in
                                             voxel-index space
    n_bins              : int   number of ψ bins (default 100)
    lv_inner_frac       : float innermost fraction of LV voxels used for C(ψ)
    rv_inner_frac       : float fraction for RV (0 = use all)
    smooth_sigma        : float Gaussian σ applied to frames along ψ
    psi_apex_threshold  : float voxels below this ψ level get θ = NaN
    save_path           : optional .npy cache path

    Returns
    -------
    theta : (Nx,Ny,Nz) float32   θ ∈ [0, 2π), NaN outside myocardium
    """
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        if save_path.exists():
            print(f"  [rotational] Loading cached theta from {save_path}")
            return np.load(save_path)

    long_axis = np.asarray(long_axis, dtype=float)
    long_axis = long_axis / np.linalg.norm(long_axis)

    theta = np.full(psi.shape, np.nan, dtype=np.float32)

    for chi_id, label, frac in [
        (1, "LV", lv_inner_frac),
        (2, "RV", rv_inner_frac),
    ]:
        theta_v = _compute_theta_for_ventricle(
            chi_id, label, chi_labels, psi, long_axis,
            ant_arc, post_arc,
            n_bins=n_bins,
            lv_inner_frac=lv_inner_frac,
            rv_inner_frac=rv_inner_frac,
            smooth_sigma=smooth_sigma,
            psi_apex_threshold=psi_apex_threshold,
        )
        idx = np.argwhere(chi_labels == chi_id)
        theta[idx[:, 0], idx[:, 1], idx[:, 2]] = theta_v.astype(np.float32)

    valid = ~np.isnan(theta)
    print(f"  [rotational] Done. theta range (inside mask): "
          f"[{theta[valid].min():.4f}, {theta[valid].max():.4f}] rad")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, theta)
        print(f"  [rotational] Saved theta → {save_path}")

    return theta


# =============================================================================
# Convenience wrapper — called by coordinates/rotational.py's compute_theta
# =============================================================================

def generate_rotational_coordinate(
    S: np.ndarray,
    chi_labels: np.ndarray,
    psi: np.ndarray,
    long_axis: np.ndarray,
    *,
    mesh_step: int = 2,
    n_bins: int = 100,
    lv_inner_frac: float = 0.35,
    rv_inner_frac: float = 0.0,
    smooth_sigma: float = 2.0,
    psi_apex_threshold: float = 0.05,
    ridge_save_path: str | Path | None = None,
    anterior_vertex_save_path: str | Path | None = None,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Full rotational-coordinate pipeline in one call:

      1. Compute (or load cached) septal seam loop.
      2. Interactive: user picks apex + AV node, confirms posterior arc.
      3. Interactive: user picks apical tip of anterior ridge.
      4. Compute θ per ventricle and merge.

    Parameters
    ----------
    S              : (Nx,Ny,Nz) bool   myocardium mask
    chi_labels     : (Nx,Ny,Nz) int    0=bg, 1=LV, 2=RV
    psi            : (Nx,Ny,Nz) float  apicobasal coordinate
    long_axis      : (3,) float         unit vector pointing apex→base
    mesh_step      : marching-cubes step size for visualisation surface
    n_bins         : ψ-stratification bins
    lv_inner_frac  : innermost fraction of LV voxels for centroid C(ψ)
    rv_inner_frac  : same for RV (0 = all voxels)
    smooth_sigma   : Gaussian σ for frame smoothing along ψ
    psi_apex_threshold : voxels below this ψ get θ = NaN
    ridge_save_path           : .npz cache for the loop + arcs
    anterior_vertex_save_path : .npy cache for the anterior vertex
    save_path      : .npy cache for the final theta volume

    Returns
    -------
    theta : (Nx,Ny,Nz) float32   θ ∈ [0, 2π), NaN outside myocardium
    """
    # Fast path: theta already cached
    if save_path is not None:
        p = Path(save_path).with_suffix(".npy")
        if p.exists():
            print(f"  [rotational] Loading cached theta from {p}")
            return np.load(p)

    # Stages 1 + 2 — loop, arcs, anterior vertex
    loop, ant_arc, post_arc, _ = compute_septal_loop_and_arcs(
        S, chi_labels,
        mesh_step=mesh_step,
        ridge_save_path=ridge_save_path,
        anterior_vertex_save_path=anterior_vertex_save_path,
    )

    # Stage 3 — θ field
    theta = compute_theta(
        chi_labels, psi, long_axis, ant_arc, post_arc,
        n_bins=n_bins,
        lv_inner_frac=lv_inner_frac,
        rv_inner_frac=rv_inner_frac,
        smooth_sigma=smooth_sigma,
        psi_apex_threshold=psi_apex_threshold,
        save_path=save_path,
    )

    return theta
