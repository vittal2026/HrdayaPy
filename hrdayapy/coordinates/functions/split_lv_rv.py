"""
split_lv_rv.py
===============
Splits a fused biventricular myocardium binary mask into separate
Left Ventricle (LV) and Right Ventricle (RV) myocardium labels, using
only the segmentation mask -- no intensities or trained model required.

Algorithm
---------
1.  Recover the two ventricular cavities by filling enclosed holes in
    the binary myocardium mask, slice-by-slice along the apico-basal
    axis. Where the myocardial ring is closed in cross-section,
    hole-filling reveals the cavity it encloses.
2.  Label the recovered cavity voxels in full 3D. In a clean mask this
    gives exactly two connected components: the LV cavity and the RV
    cavity (small spurious bits are dropped by keeping only the two
    largest components).
3.  Disambiguate which component is LV vs RV using the anatomical fact
    that the cardiac apex is formed almost entirely by the LV:
    whichever cavity component extends furthest toward the apex is
    the LV.
4.  Use the two cavity components as markers/seeds for a marker-based
    watershed restricted to the myocardium mask, which grows each seed
    outward through the muscle to produce the final LV/RV split.

This is a straight port of the standalone prototype, adapted to share
the `axis` / `base_idx` already computed by Step 1 of the pipeline
(find_apico_basal_axis / compute_basal_plane) instead of taking a
separate apex_axis / apex_at_start pair -- the apex is, by definition,
whichever end of `axis` lies farthest from `base_idx`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.segmentation import watershed


def split_lv_rv(S, axis=None, base_idx=None, connectivity: int = 1,
                 lv_seed=None, rv_seed=None):
    """
    Split a fused biventricular myocardium mask into LV / RV labels.

    Two ways to specify the seeds
    ------------------------------
    Preferred: pass `lv_seed` / `rv_seed` directly -- boolean voxel masks
    of LV/RV endocardial surface voxels (e.g. mesh_labelling.
    label_ventricle_mesh's "lv_endo_voxels" / "rv_endo_voxels"). These
    come from the same orientation-independent ray-tracing labelling used
    for surface_label and chi, so all three coordinate fields are
    seeded from one consistent source of truth instead of two
    independent algorithms that can disagree with each other.

    Legacy fallback: if `lv_seed`/`rv_seed` are not given, the original
    cavity-recovery method is used (`axis`, `base_idx` become required).
    This recovers the LV/RV cavities by hole-filling slice-by-slice along
    a single grid axis, which only works cleanly if that axis is close
    to the heart's true apicobasal direction -- prefer the mesh-derived
    seeds above for any non-trivially-oriented heart.

    Parameters
    ----------
    S : (Nx,Ny,Nz) ndarray, bool or {0,1}
        Binary myocardium mask. Must be a single continuous structure
        containing both ventricles fused at the septum (no cavities
        included).
    axis, base_idx : legacy path only -- same convention as
        find_apico_basal_axis / compute_basal_plane.
    connectivity : int, default 1
        Connectivity used for 3D connected-component labeling of the
        recovered cavities (legacy path only).
    lv_seed, rv_seed : (Nx,Ny,Nz) bool, optional
        LV / RV endocardial surface-voxel masks (preferred path).

    Returns
    -------
    labels : (Nx,Ny,Nz) uint8
        0 = background, 1 = LV myocardium, 2 = RV myocardium.
    info : dict
        Diagnostic info.
    """
    mask = np.asarray(S).astype(bool)
    if mask.ndim != 3:
        raise ValueError("Expected a 3D volume.")

    if lv_seed is not None and rv_seed is not None:
        lv_seed_muscle = ndi.binary_dilation(lv_seed) & mask
        rv_seed_muscle = ndi.binary_dilation(rv_seed) & mask
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[lv_seed_muscle] = 1
        markers[rv_seed_muscle] = 2
        elevation = ndi.distance_transform_edt(mask)
        labels = watershed(-elevation, markers=markers, mask=mask).astype(np.uint8)
        info = {"seed_source": "mesh_labelling (orientation-independent)",
                "lv_seed_voxels": int(lv_seed.sum()), "rv_seed_voxels": int(rv_seed.sum())}
        return labels, info

    if axis is None or base_idx is None:
        raise ValueError(
            "Either (lv_seed, rv_seed) or (axis, base_idx) must be provided.")

    # ---- Derive apex_at_start from base_idx -----------------------------
    # The apex is whichever end of `axis` lies farther from the basal plane.
    other_axes = tuple(a for a in range(3) if a != axis)
    occ = np.where(mask.any(axis=other_axes))[0]
    idx_min, idx_max = int(occ.min()), int(occ.max())
    apex_at_start = (base_idx - idx_min) > (idx_max - base_idx)

    # ---- Step 1: recover cavities by slice-wise hole filling ------------
    filled = np.zeros_like(mask)
    slicer = [slice(None)] * mask.ndim
    n_slices = mask.shape[axis]
    for i in range(n_slices):
        slicer[axis] = i
        filled[tuple(slicer)] = ndi.binary_fill_holes(mask[tuple(slicer)])

    cavities = filled & (~mask)

    if not cavities.any():
        raise RuntimeError(
            "No enclosed cavities were recovered. Check that `axis` "
            "matches the slice-stacking axis of the volume, and that "
            "the myocardium ring is actually closed in cross-section."
        )

    # ---- Step 2: label cavity components in full 3D ----------------------
    cav_labels, n_comp = cc_label(cavities, return_num=True,
                                   connectivity=connectivity)
    if n_comp < 2:
        raise RuntimeError(
            f"Expected 2 cavity components (LV + RV), found {n_comp}. "
            "The mask may have gaps, or the two cavities may be fused "
            "into one (try connectivity=1)."
        )

    sizes = ndi.sum(cavities, cav_labels, index=np.arange(1, n_comp + 1))
    order = np.argsort(sizes)[::-1]
    top2 = (order[:2] + 1).tolist()  # component ids, 1-indexed

    # ---- Step 3: disambiguate LV vs RV using apex proximity --------------
    extents = {}
    for comp_id in top2:
        coords_along_axis = np.where(cav_labels == comp_id)[axis]
        extents[comp_id] = (int(coords_along_axis.min()),
                             int(coords_along_axis.max()))

    def apex_reach(comp_id):
        lo, hi = extents[comp_id]
        return lo if apex_at_start else hi

    if apex_at_start:
        lv_comp = min(top2, key=apex_reach)   # smallest index = closest to apex
        rv_comp = max(top2, key=apex_reach)
    else:
        lv_comp = max(top2, key=apex_reach)   # largest index = closest to apex
        rv_comp = min(top2, key=apex_reach)

    lv_seed = (cav_labels == lv_comp)
    rv_seed = (cav_labels == rv_comp)

    # The cavities themselves are NOT part of `mask` (they are background),
    # so seeds must be placed on the thin layer of muscle immediately
    # bordering each cavity for the watershed (which only labels voxels
    # where mask==True) to actually propagate from them.
    lv_seed_muscle = ndi.binary_dilation(lv_seed) & mask
    rv_seed_muscle = ndi.binary_dilation(rv_seed) & mask

    # ---- Step 4: marker-based watershed through the myocardium ----------
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[lv_seed_muscle] = 1
    markers[rv_seed_muscle] = 2

    # Distance transform from outside the muscle gives a smooth surface
    # for the watershed to climb; seeds grow outward through the wall.
    elevation = ndi.distance_transform_edt(mask)
    labels = watershed(-elevation, markers=markers, mask=mask)
    labels = labels.astype(np.uint8)

    info = {
        "apex_at_start": bool(apex_at_start),
        "n_cavity_components_found": int(n_comp),
        "cavity_sizes_top2": {int(c): float(s)
                               for c, s in zip(top2, sizes[np.array(top2) - 1])},
        "lv_component_id": int(lv_comp),
        "rv_component_id": int(rv_comp),
        "apex_extents": extents,
    }
    return labels, info
