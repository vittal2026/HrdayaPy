"""
manual_landmarks.py
====================
Interactive manual picking of the apex and basal-region landmarks, to
replace automatic apex/base orientation detection for the apicobasal
coordinate.

Why this exists
----------------
Automatic apex/base orientation (mesh_labelling.compute_long_axis +
find_pole_cap) turned out to be unreliable in general: several different,
individually-reasonable geometric criteria (transverse cross-sectional
spread, epicardium-to-cavity gap, distance to the combined blood-pool
centroid, mesh-graph hop-distance to the nearest endocardium) were tried
against the same real validated mask and gave *contradictory* answers
about which pole is the apex. This isn't a bug in any one of them so much
as a sign that apex-vs-base orientation isn't reliably recoverable from
unlabelled surface geometry alone, at least not without a much larger
anatomical-prior/statistical-atlas effort than fits this pipeline. So:
let a person look at the mesh and click.

What stays automatic
---------------------
mesh_labelling.label_ventricle_mesh's epi/LV-endo/RV-endo/RV-septal split
(via convex-hull ray tracing) and the resulting chi (biventricular) seeds
were validated as working well and are UNCHANGED -- this module only
replaces the apex_voxels / basal_voxels inputs that feed
generate_apicobasal_coordinate.

Workflow
--------
    landmarks = pick_and_save_landmarks(S, save_path=cfg.PATH_LANDMARKS)

opens two sequential PyVista windows (rotate freely with left-click-drag
in each; picking itself is hover + press P, matching the proven pattern
from plot_voltage_trace_at_picked_node.py -- left-click is left alone for
camera rotation):
  1. "Pick the APEX" -- hover + press P once, on the mesh surface
  2. "Pick BASAL REGION points" -- hover + press P for a sequence of
     points tracing around what you consider the base, in order, rotating
     the mesh as needed; PyVista draws the connecting geodesic path on the
     surface live as you go (built-in -- enable_geodesic_picking; press C
     to clear and restart if needed). Close the window when done.

The clicked basal points are connected into a closed loop (the open path
PyVista already draws between consecutive clicks, plus one more geodesic
segment computed here from the last point back to the first), and a band
of the surface within `basal_band_mm` of that closed loop becomes the
basal Dirichlet region -- this is what "demarcate a region around these
points" means concretely. The apex becomes a small disc of radius
`apex_radius_mm` around the clicked point (a literal single point/voxel
would be too weak a Dirichlet condition for the Laplace solve).

Note: this module previously also picked an AV-node point/region here.
It was removed -- nothing downstream ever consumed it (apicobasal only
uses apex_voxels/basal_voxels, and both the Purkinje root pick and the
rotational-coordinate pick have their own independent AV-node pickers).
If you need the old behaviour back, see version control history.

Everything is done with mesh-geodesic (surface) distance, not straight-
line Euclidean distance, via a single multi-source Dijkstra pass over the
mesh's vertex-adjacency graph -- so the basal band follows the actual
surface even where it curves sharply (e.g. into the inter-ventricular
groove), and never "leaks" through the lumen to the wrong side of a thin
wall the way a Euclidean radius could.

Landmarks are cached to `save_path` after picking, so you only pick once
per dataset; re-running with the same save_path loads the cached points
instead of re-opening the picker (mirrors the rest of this pipeline's
run=True/False caching convention).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .mesh_labelling import extract_surface_mesh, face_geometry, voxelize_face_labels


# =============================================================================
# Mesh <-> PyVista conversion
# =============================================================================

def _to_pyvista(verts, faces):
    import pyvista as pv
    F = faces.shape[0]
    vtk_faces = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(verts, vtk_faces)


def _vertex_adjacency(verts, faces):
    """Sparse, Euclidean-edge-length-weighted vertex adjacency graph."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    w = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    V = verts.shape[0]
    graph = coo_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(V, V),
    ).tocsr()
    return graph


# =============================================================================
# Interactive picking (requires a display -- run this on your own machine)
# =============================================================================

def pick_single_point(verts, faces, mesh=None, title: str = "Pick a point",
                       instructions: str = "Rotate freely with left-click-drag. "
                                            "Hover over a point on the mesh surface "
                                            "and press P to pick (press P again "
                                            "elsewhere to move the pick). Close the "
                                            "window when you're happy with it."):
    """
    Opens one PyVista window; hovering + pressing P selects/re-selects a
    single point on the mesh surface (matches the proven interaction
    pattern from plot_voltage_trace_at_picked_node.py: plain
    enable_point_picking with a single-argument callback and an explicit
    nearest-point snap, rather than enable_surface_point_picking's
    use_picker=True 2-argument callback, and *not* left_clicking=True --
    left-click is already bound to camera rotation by default, so
    enabling it for picking causes accidental picks while rotating).
    Returns the picked vertex index into `verts`, or None if nothing was
    picked.
    """
    import pyvista as pv
    if mesh is None:
        mesh = _to_pyvista(verts, faces)

    state = {"idx": None}
    plotter = pv.Plotter()
    plotter.add_text(f"{title}\n{instructions}", font_size=12)
    plotter.add_mesh(mesh, color="lightcoral", show_edges=False, smooth_shading=True)

    def callback(point):
        idx = mesh.find_closest_point(point)
        state["idx"] = int(idx)
        coords = mesh.points[idx]
        print(f"  Picked vertex {idx} at ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")
        plotter.add_mesh(pv.Sphere(radius=mesh.length * 0.01, center=coords),
                          color="yellow", name="picked_point")
        plotter.add_text(f"vertex {idx}  ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})",
                          name="pick_label", font_size=10, color="white", position="lower_left")
        plotter.render()

    plotter.enable_point_picking(callback=callback, show_message=instructions,
                                  show_point=True, color="red", point_size=10,
                                  tolerance=0.025)
    plotter.show()
    return state["idx"]


def pick_basal_loop(verts, faces, mesh=None):
    """
    Opens one PyVista window with geodesic picking enabled: hover and
    press P for a SEQUENCE of points tracing around the basal region (in
    order; rotate freely with left-click-drag between picks). PyVista
    draws the connecting geodesic path live as you go. Press C at any
    point to clear and restart the loop. Close the window when done.

    Returns the ordered list of picked vertex indices (length >= 2), or
    an empty list if nothing was picked.
    """
    import pyvista as pv
    if mesh is None:
        mesh = _to_pyvista(verts, faces)

    instructions = ("Rotate freely with left-click-drag. Hover and press P for a "
                     "SEQUENCE of points tracing around what you consider the BASE, "
                     "in order -- the path connecting your picks is drawn live. "
                     "Press C to clear and restart the loop. Close the window when "
                     "you've gone all the way around; the loop is closed automatically.")
    plotter = pv.Plotter()
    plotter.add_text(f"Pick BASAL REGION points\n{instructions}", font_size=12)
    plotter.add_mesh(mesh, color="lightcoral", show_edges=False, smooth_shading=True)
    plotter.enable_geodesic_picking(show_message=instructions, color="yellow",
                                     point_size=12, line_width=4, keep_order=True)
    plotter.show()

    geo = plotter.picked_geodesic
    if geo is None or geo.n_points == 0 or "vtkOriginalPointIds" not in geo.point_data:
        return []
    # de-duplicate consecutive repeats while preserving order
    ids = geo.point_data["vtkOriginalPointIds"].tolist()
    ordered = [ids[0]] if ids else []
    for i in ids[1:]:
        if i != ordered[-1]:
            ordered.append(i)
    return ordered


# =============================================================================
# Turning picked points into Dirichlet regions
# =============================================================================

def close_loop(verts, faces, ordered_vertex_ids, graph=None):
    """
    Appends one more geodesic segment closing the loop from the last
    picked point back to the first, returning the full closed-loop vertex
    sequence (as mesh vertex indices, with the path-following points
    in between consecutive clicks -- not just the raw clicks).
    """
    if len(ordered_vertex_ids) < 2:
        return list(ordered_vertex_ids)
    mesh = _to_pyvista(verts, faces)
    closing = mesh.geodesic(ordered_vertex_ids[-1], ordered_vertex_ids[0], keep_order=True)
    closing_ids = closing.point_data["vtkOriginalPointIds"].tolist()
    return list(ordered_vertex_ids) + closing_ids


def region_around_vertices(verts, faces, seed_vertex_ids, radius_mm: float, graph=None):
    """
    Boolean (F,) face mask: faces with at least one vertex within
    `radius_mm` geodesic (surface) distance of any seed vertex. Uses a
    single multi-source Dijkstra pass, so an empty/long seed list (e.g.
    a whole closed loop) costs the same as a single seed point.
    """
    if graph is None:
        graph = _vertex_adjacency(verts, faces)
    dist = dijkstra(graph, indices=np.asarray(seed_vertex_ids, dtype=int),
                     min_only=True, limit=radius_mm * 3)
    in_band = dist <= radius_mm
    face_mask = in_band[faces].any(axis=1)
    return face_mask


# =============================================================================
# Top-level orchestrator
# =============================================================================

def pick_and_save_landmarks(S: np.ndarray, save_path, mesh_step: int = 2,
                              apex_radius_mm: float = 6.0,
                              basal_band_mm: float = 8.0,
                              verbose: bool = True):
    """
    Full interactive pick -> region -> voxelize -> cache pipeline.

    Returns a dict:
        apex_voxels, basal_voxels   : (Nx,Ny,Nz) bool -- feed directly to
                                       generate_apicobasal_coordinate's
                                       apex_mask / basal_mask
        apex_point                  : (3,) picked coordinate (voxel-index space)
        basal_loop_points            : (K,3) the closed basal contour
        verts, faces                 : the mesh these were picked on
    """
    def log(msg):
        if verbose:
            print(f"  [manual_landmarks] {msg}")

    save_path = Path(save_path).with_suffix(".npz")
    if save_path.exists():
        log(f"Loading cached landmarks from {save_path}")
        z = np.load(save_path)
        return {k: z[k] for k in z.files}

    log("Extracting surface mesh ...")
    verts, faces = extract_surface_mesh(S, step_size=mesh_step)
    mesh = _to_pyvista(verts, faces)
    graph = _vertex_adjacency(verts, faces)

    # Smoothed copy for on-screen display only -- same point count, order,
    # and connectivity as `mesh` (Taubin smoothing moves vertices, it never
    # changes topology), so a vertex index picked on this mesh is still a
    # valid index into the original verts/faces/graph below. We deliberately
    # smooth this copy rather than extract_surface_mesh's output directly,
    # since that function is shared with mesh_labelling.py's *automatic*
    # apex/base classification, which does need the raw, unsmoothed
    # geometry (see that module's docstring). Region growing, Dijkstra
    # basal-band distances, and voxelization below all keep using the
    # original, unsmoothed verts/faces/graph -- only what you see while
    # picking is smoothed.
    display_mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1, normalize_coordinates=True)

    log("Opening picker: APEX (hover + press P once) ...")
    apex_idx = pick_single_point(verts, faces, mesh=display_mesh, title="Pick the APEX")
    if apex_idx is None:
        raise RuntimeError("No apex point was picked.")

    log("Opening picker: BASAL REGION (hover + press P for a sequence around the base) ...")
    basal_clicks = pick_basal_loop(verts, faces, mesh=display_mesh)
    if len(basal_clicks) < 3:
        raise RuntimeError("Need at least 3 basal points to define a region "
                            f"(got {len(basal_clicks)}). Re-run and pick more points.")

    log("Closing the basal loop and building Dirichlet regions ...")
    basal_loop_ids = close_loop(verts, faces, basal_clicks, graph=graph)

    apex_face_mask = region_around_vertices(verts, faces, [apex_idx], apex_radius_mm, graph)
    basal_face_mask = region_around_vertices(verts, faces, basal_loop_ids, basal_band_mm, graph)

    log(f"apex region: {apex_face_mask.sum()} faces  |  "
        f"basal region: {basal_face_mask.sum()} faces "
        f"(from a {len(basal_loop_ids)}-point closed loop)")

    vox = voxelize_face_labels(S, verts, faces, {
        "apex": apex_face_mask, "basal": basal_face_mask,
    })

    result = {
        "apex_voxels": vox["apex"], "basal_voxels": vox["basal"],
        "apex_point": verts[apex_idx],
        "basal_loop_points": verts[basal_loop_ids],
    }
    np.savez(save_path, **result)
    log(f"Saved -> {save_path}")
    return result
