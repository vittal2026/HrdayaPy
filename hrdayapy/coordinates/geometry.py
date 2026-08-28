"""
coordinates/geometry.py
========================
Compute/load pair for Stage 1 of the coordinates pipeline: the myocardium
mask (S) and the anatomy labelling derived from it (surface labels,
long axis, apex/basal regions).

This replaces the old run_step_1_mask(cfg, run=True/False) pattern from
the now-removed legacy pipeline runner. The two functions below are
symmetric and independent of each other:

    compute_geometry(...)  -- does the physics/segmentation work, returns
                               (S, anatomy), and saves them if you give it
                               save paths.
    load_geometry(...)     -- reads a previously saved (S, anatomy) back
                               from disk. No recomputation.

Neither function reads or writes any fixed location — every path is an
argument, so a script is free to point at any patient's files.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import load_muscle_mask, label_ventricle_mesh, relabel_lv_rv_with_psi


def compute_geometry(
    mask_path,
    *,
    target_spacing=0.4,
    original_spacing=None,
    target_shape=None,
    pad_width: int = 5,
    mesh_step: int = 2,
    save_path_S=None,
    save_path_anatomy=None,
):
    """
    Load a segmentation NRRD and run the full anatomy-labelling pass.

    Parameters
    ----------
    mask_path : str or Path
        Path to the patient's myocardium segmentation NRRD.
    target_spacing : float or (3,) sequence of float, default 0.4
        Desired output spacing in mm/voxel (isotropic if a single float).
        The mask is resampled so the OUTPUT grid actually has this
        spacing -- this is what every downstream stage (Purkinje growth,
        monodomain solve, ECG) should then use as its own voxel_size, so
        pass the SAME number through rather than re-typing it elsewhere.
    original_spacing : None | "nrrd" | float | (3,) sequence of float, optional
        What the NRRD's native spacing (mm/voxel) actually is, so the
        resample factor can be computed correctly:

          * None (default) -- no information assumed; the loaded array is
            used as-is and simply labelled with target_spacing (NO
            interpolation happens). Only correct if the file already
            happens to be sampled at target_spacing -- this reproduces
            the old, spacing-unaware behaviour, so don't rely on it
            unless you've actually confirmed the file's native spacing.
          * "nrrd" -- read the true spacing from the NRRD header
            (`space directions` / `spacings`) and resample from that.
            This is normally what you want.
          * float or 3-tuple -- state the original spacing explicitly
            (e.g. 1.0 if you know the segmentation was exported at 1 mm),
            overriding the header.
    target_shape : (int, int, int), optional
        DEPRECATED legacy path, kept only for backward compatibility with
        older scripts. Forces the mask to an exact voxel count regardless
        of physical spacing -- this is the behaviour that silently
        rescales the heart if the assumed spacing was wrong. Prefer
        target_spacing (+ original_spacing) for anything new; ignored if
        target_spacing is given (which it is, by default).
    pad_width : int
        Padding (in voxels of the OUTPUT/target spacing) added around the
        mask on each side.
    mesh_step : int
        Marching-cubes step size used for the anatomy mesh (lower = finer,
        slower).
    save_path_S : str or Path, optional
        If given, the boolean mask S is written here as .npy.
    save_path_anatomy : str or Path, optional
        If given, the anatomy dict is written here as .npz. The resolved
        spacing is saved alongside it (key "spacing_mm") so load_geometry
        can hand it back without anyone needing to re-derive or re-type it.

    Returns
    -------
    S       : (Nx, Ny, Nz) bool ndarray -- myocardium mask
    anatomy : dict -- surface_label, surface_label_rv, lv_endo_voxels,
              rv_endo_voxels, apex_voxels, basal_voxels, long_axis,
              basal_region_is_flat_lid (see label_ventricle_mesh docstring
              for the full field list), plus "spacing_mm": (3,) float64 --
              the actual mm/voxel spacing S is sampled at. Use this as
              voxel_size in every downstream stage instead of a separately
              hand-typed constant.
    """
    S, _, spacing_mm = load_muscle_mask(
        mask_path,
        target_spacing=target_spacing,
        original_spacing=original_spacing,
        target_shape=target_shape,
        pad_width=pad_width,
        plot=False,
    )
    anatomy = label_ventricle_mesh(S, mesh_step=mesh_step)
    anatomy["spacing_mm"] = spacing_mm

    if save_path_S is not None:
        save_path_S = Path(save_path_S)
        save_path_S.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path_S, S)

    if save_path_anatomy is not None:
        save_path_anatomy = Path(save_path_anatomy)
        save_path_anatomy.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path_anatomy,
            surface_label=anatomy["surface_label"],
            surface_label_rv=anatomy["surface_label_rv"],
            lv_endo_voxels=anatomy["lv_endo_voxels"],
            rv_endo_voxels=anatomy["rv_endo_voxels"],
            apex_voxels=anatomy["apex_voxels"],
            basal_voxels=anatomy["basal_voxels"],
            long_axis=anatomy["long_axis"],
            basal_region_is_flat_lid=anatomy["basal_region_is_flat_lid"],
            spacing_mm=spacing_mm,
        )

    return S, anatomy


def refine_geometry_with_psi(
    S,
    anatomy,
    psi,
    *,
    mesh_step: int = 2,
    basal_psi_threshold: float = 0.97,
    save_path_anatomy=None,
):
    """
    Redo just the LV/RV endocardial split in `anatomy` using the
    psi-based basal cut (ported from test_psi_basal_cut.py), now that
    psi is available.

    compute_geometry's LV/RV split (Stage 1) runs before psi exists
    (psi -- Stage 3 -- needs its own apex/basal landmarks, not anything
    from `anatomy`, so there's no circular dependency forcing psi to be
    computed earlier). Without psi, the LV/RV split falls back to naive
    connected components + a hull-volume heuristic, which can merge LV
    and RV endocardium into one component (or misassign which is which)
    whenever they're topologically connected through an open/cut base.
    Call this afterwards, once psi is on disk, to get a corrected
    anatomy dict -- it's cheap: it reuses the cached surface_label
    instead of repeating the expensive ray-tracing classify_epi_endo.

    Always recomputes: like the other compute_* functions in this
    package, this always redoes the LV/RV split and overwrites
    save_path_anatomy if given, even if a file is already there.

    Parameters
    ----------
    S                   : (Nx,Ny,Nz) bool -- myocardium mask, same one
                          `anatomy` was computed from.
    anatomy             : dict -- from compute_geometry / load_geometry.
                          Must still have the mesh_step it was built
                          with in sync with `mesh_step` below.
    psi                 : (Nx,Ny,Nz) float -- from compute_psi.
    mesh_step           : marching-cubes step size -- MUST match what
                          compute_geometry used to build `anatomy`.
    basal_psi_threshold : faces with sampled psi >= this are "basal"
                          for the cut (default 0.97).
    save_path_anatomy   : if given, the refined anatomy dict is written
                          here as .npz, in the same format as
                          compute_geometry's save_path_anatomy.

    Returns
    -------
    anatomy : dict -- same shape as compute_geometry's anatomy, with the
              LV/RV-dependent fields (surface_label, surface_label_rv,
              lv_endo_voxels, rv_endo_voxels, long_axis,
              basal_region_is_flat_lid, ...) replaced by the psi-cut
              result. anatomy["spacing_mm"] is carried over unchanged.
    """
    refined = relabel_lv_rv_with_psi(
        S, psi, anatomy["surface_label"],
        mesh_step=mesh_step, basal_psi_threshold=basal_psi_threshold,
        long_axis_hint=anatomy.get("long_axis"),
    )
    refined["spacing_mm"] = anatomy.get("spacing_mm")

    if save_path_anatomy is not None:
        save_path_anatomy = Path(save_path_anatomy)
        save_path_anatomy.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path_anatomy,
            surface_label=refined["surface_label"],
            surface_label_rv=refined["surface_label_rv"],
            lv_endo_voxels=refined["lv_endo_voxels"],
            rv_endo_voxels=refined["rv_endo_voxels"],
            apex_voxels=refined["apex_voxels"],
            basal_voxels=refined["basal_voxels"],
            long_axis=refined["long_axis"],
            basal_region_is_flat_lid=refined["basal_region_is_flat_lid"],
            spacing_mm=refined["spacing_mm"],
        )

    return refined


def load_geometry(path_S, path_anatomy):
    """
    Load a previously saved (S, anatomy) pair. Does no computation.

    Parameters
    ----------
    path_S       : str or Path -- .npy file written by compute_geometry
    path_anatomy : str or Path -- .npz file written by compute_geometry

    Returns
    -------
    S, anatomy -- same shapes/types as compute_geometry
    """
    path_S = Path(path_S)
    path_anatomy = Path(path_anatomy)
    if not path_S.exists():
        raise FileNotFoundError(f"No saved mask at {path_S}")
    if not path_anatomy.exists():
        raise FileNotFoundError(f"No saved anatomy at {path_anatomy}")

    S = np.load(path_S)
    z = np.load(path_anatomy)
    anatomy = {k: z[k] for k in z.files}
    anatomy["long_axis"] = anatomy["long_axis"].astype(float)
    anatomy["basal_region_is_flat_lid"] = bool(anatomy["basal_region_is_flat_lid"])
    return S, anatomy


def load_cut_mask(
    cut_mask_path,
    *,
    target_spacing=0.4,
    original_spacing=None,
    target_shape=None,
    pad_width: int = 5,
):
    """
    Load the Slicer-exported "cut" mask Z used only for visualisation
    (e.g. visualise_coordinates, plot_stimulus_region). There's no
    compute-side pair for this one -- Z comes pre-made from Slicer, this
    package doesn't generate it, so loading is the only path in.

    Parameters
    ----------
    cut_mask_path : str or Path -- path to the cut-mask NRRD
    target_spacing, original_spacing, target_shape, pad_width :
        must match whatever compute_geometry used for S (same meanings as
        there -- see compute_geometry's docstring), so Z lines up
        voxel-for-voxel with the other fields.

    Returns
    -------
    Z : (Nx,Ny,Nz) bool
    """
    Z, _, _ = load_muscle_mask(
        cut_mask_path,
        target_spacing=target_spacing,
        original_spacing=original_spacing,
        target_shape=target_shape,
        pad_width=pad_width,
        plot=False,
    )
    return Z
