"""
coordinates/stimulus_region.py
=================================
Compute/load pairs for a voxel region marking an ectopic stimulus site.
Two independent ways to define the same kind of output (an (Nx,Ny,Nz)
bool mask, usable anywhere a stim_region is, e.g. compute_coupled's
ectopic_region):

    compute_stim_region            -- target windows in (psi, phi, chi,
                                       theta) UVC space, e.g. "near the
                                       base, near the epicardium, on the
                                       RV, at 180 degrees around"
    compute_stim_region_from_point -- click a point on the myocardium
                                       surface and grow a small ball
                                       around it; for when "right about
                                       there" is easier than a coordinate
                                       window

Pick whichever reads more naturally for your case -- both save/load the
same plain boolean .npy format, so they're interchangeable everywhere
downstream.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import select_coordinate_region, region_centroid_voxel, pick_and_save_stim_region


def compute_stim_region(
    *,
    psi=None, psi_target=None, psi_tol: float = 0.05,
    phi=None, phi_target=None, phi_tol: float = 0.05,
    chi=None, chi_target=None, chi_tol: float = 0.5,
    theta=None, theta_target_deg=None, theta_tol_deg: float = 10.0,
    save_path=None,
):
    """
    Select the myocardial voxels whose coordinates fall within target
    +/- tolerance windows. Pass only the fields/targets you want to
    constrain -- the rest are left unconstrained.

    Parameters
    ----------
    psi, phi, chi, theta : (Nx,Ny,Nz) float or None -- the coordinate
                            fields to constrain against
    *_target              : target value for that field (theta in degrees)
    *_tol                 : half-width of the target window
    save_path             : if given, the boolean region is written here
                             as .npy

    Returns
    -------
    region : (Nx,Ny,Nz) bool -- selected voxels
    """
    region = select_coordinate_region(
        psi=psi, psi_target=psi_target, psi_tol=psi_tol,
        phi=phi, phi_target=phi_target, phi_tol=phi_tol,
        chi=chi, chi_target=chi_target, chi_tol=chi_tol,
        theta=theta, theta_target_deg=theta_target_deg,
        theta_tol_deg=theta_tol_deg,
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, region)
    return region


def load_stim_region(path):
    """Load a previously saved stimulus region. No recomputation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved stimulus region at {path}")
    return np.load(path)


def compute_stim_region_from_point(
    S,
    save_path,
    *,
    radius_mm: float = 5.0,
    voxel_size: float = 0.4,
    mesh_step: int = 2,
    target_mm: float | None = 1.0,
):
    """
    Interactively pick a point on the myocardium surface and grow a
    solid ball of radius `radius_mm` around it, as an alternative to
    compute_stim_region's UVC target-window selection.

    Opens a PyVista window: hover over the myocardium surface near
    where you want the ectopic focus and press P to pick (press P again
    elsewhere to move the pick). Close the window when you're happy
    with it. Requires a display -- not for headless use.

    Parameters
    ----------
    S           : (Nx,Ny,Nz) bool -- myocardium mask, from compute_geometry
    save_path   : region is always written here as .npy (required --
                  the picker result is not returned without being saved)
    radius_mm   : radius of the grown region around the picked point, in mm
    voxel_size  : mm per voxel (isotropic) -- match whatever
                  compute_coupled will be called with
    mesh_step   : marching-cubes step size for the picking surface, in
                  voxels. Only takes effect when target_mm=None.
    target_mm   : marching-cubes step size for the picking surface, in mm
                  instead of voxels -- overrides mesh_step. None uses the
                  literal mesh_step value instead. See
                  coordinates.mm_step_size / manual_stim_region.py's
                  pick_and_save_stim_region for why this exists: a
                  voxel-count step_size makes the picking surface (and
                  everything built on it) get denser -- and slower --
                  the finer voxel_size is, unrelated to whether that
                  detail is wanted.

    Returns
    -------
    region : (Nx,Ny,Nz) bool -- same format as compute_stim_region's output
    """
    save_path = Path(save_path).with_suffix(".npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick into a temp file first, same crash-safety pattern as
    # compute_landmarks: only replace the real file once picking fully
    # succeeds, so a cancelled/failed pick never clobbers a good region
    # you already had saved.
    tmp_path = save_path.with_name(save_path.stem + "_tmp_pick.npy")
    if tmp_path.exists():
        tmp_path.unlink()

    region = pick_and_save_stim_region(
        S, save_path=tmp_path, radius_mm=radius_mm,
        voxel_size=voxel_size, mesh_step=mesh_step, target_mm=target_mm,
    )

    tmp_path.replace(save_path)
    print(f"  [compute_stim_region_from_point] Saved -> {save_path}")
    return region


def load_stim_region_from_point(path):
    """
    Load a previously picked point-based stimulus region. No
    recomputation. Identical file format to load_stim_region -- provided
    separately only so a script's load-branch reads symmetrically with
    whichever compute_* it paired with.
    """
    return load_stim_region(path)


def compute_stim_regions_from_points(
    S,
    names,
    save_dir,
    *,
    radius_mm: float = 5.0,
    voxel_size: float = 0.4,
    mesh_step: int = 2,
    target_mm: float | None = 1.0,
):
    """
    Multi-site counterpart of compute_stim_region_from_point: opens one
    PyVista picker window per name in `names`, in order, and returns all
    of the resulting masks together. Use this for protocols that need
    several independent myocardial stimulus sites (e.g. an S1 site and
    an S2 site), instead of calling compute_stim_region_from_point once
    per site by hand.

    Each site is saved separately as <save_dir>/<name>_stim_region.npy
    (same crash-safe temp-file pattern as compute_stim_region_from_point),
    so a later stage can reload any subset of them without re-picking.

    Parameters
    ----------
    S           : (Nx,Ny,Nz) bool -- myocardium mask, from compute_geometry
    names       : sequence of str -- one identifier per site, e.g.
                  ["apex", "rv_bw"]. Picked in this order. Also used as
                  the "region" key in a stim_protocol entry with
                  target="region" downstream.
    save_dir    : directory the per-site .npy files are written into
    radius_mm, voxel_size, mesh_step : as in compute_stim_region_from_point;
                  same value used for every site in this call. Pick one
                  site individually with compute_stim_region_from_point
                  instead if it needs a different radius.

    Returns
    -------
    regions : dict[str, (Nx,Ny,Nz) bool] -- one mask per name, in the
              order given
    """
    save_dir = Path(save_dir)
    regions = {}
    for i, name in enumerate(names, start=1):
        print(f"--- Pick site {i}/{len(names)}: {name!r} ---")
        regions[name] = compute_stim_region_from_point(
            S, save_path=save_dir / f"{name}_stim_region.npy",
            radius_mm=radius_mm, voxel_size=voxel_size, mesh_step=mesh_step,
            target_mm=target_mm,
        )
    return regions


def load_stim_regions(names, save_dir):
    """
    Load several previously saved stim_regions_from_points sites. No
    recomputation/re-picking. Raises the same FileNotFoundError as
    load_stim_region if a name's file is missing.

    Parameters
    ----------
    names    : sequence of str -- same names passed to
               compute_stim_regions_from_points
    save_dir : directory they were saved into

    Returns
    -------
    regions : dict[str, (Nx,Ny,Nz) bool] -- one mask per name
    """
    save_dir = Path(save_dir)
    return {name: load_stim_region(save_dir / f"{name}_stim_region.npy")
            for name in names}
