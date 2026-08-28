"""
ecg/functions/build_torso_grid.py
===================================
Step 10a — Build the coarse torso FDM grid for the Poisson forward solve.

What this module does
---------------------
1. Load the full torso NRRD segmentation.
2. Resample to the coarse working resolution (ECG_DX_COARSE_MM, e.g. 2 mm)
   using nearest-neighbour interpolation to preserve label integrity.
3. Map integer labels → conductivity values σ(x,y,z)  [S/m].
4. Assemble the sparse FDM stiffness matrix A from the 7-point stencil:

       ∇·(σ ∇φ) = b

   with Neumann (zero-flux) boundary conditions on all faces — appropriate
   because no current crosses the air–torso interface.

5. Pin one interior node (arbitrary) to remove the nullspace of the Neumann
   problem (φ is defined up to a constant; we fix the mean implicitly by
   grounding one node).

6. Compute the sparse Cholesky factorisation of A via scikit-sparse (CHOLMOD)
   or fall back to pypardiso / scipy SuperLU.  The factor is reused for every
   snapshot frame — pay the factorisation cost once.

7. Identify surface voxels: label ≥ 1 voxels that are adjacent to label 0
   (air).  These are the BSPM measurement sites.

8. Save everything needed for the per-frame Poisson solve to torso_grid.npz.

Stiffness matrix assembly — 7-point FDM
-----------------------------------------
For a voxel at flat index k with conductivity σ_k, its 6 face-neighbours
share a conductivity σ_j.  The harmonic mean of σ_k and σ_j is used at the
face (standard for cell-centred FDM with discontinuous coefficients):

    σ_face = 2 σ_k σ_j / (σ_k + σ_j)

The off-diagonal entry A[k, j] = -σ_face / h²  (h = dx_coarse in metres).
The diagonal entry A[k, k] = sum of all face conductances for node k.

Air voxels (σ = 0) are excluded from the system: their rows/cols are not
added, reducing the system size and ensuring A is non-singular after pinning.

Output — torso_grid.npz
------------------------
    shape          (3,) int       coarse grid shape (ni, nj, nk)
    dx_mm          scalar float   coarse voxel size [mm]
    origin_mm      (3,) float     physical origin of coarse grid [mm]
    sigma_vol      (ni,nj,nk)     conductivity at each voxel [S/m]
    active_mask    (ni,nj,nk)     bool — True for non-air voxels in the system
    active_ids     (N_active,)    flat indices of active voxels
    surface_ids    (N_surf,)      flat indices of surface voxels (subset of active)
    surface_xyz    (N_surf,3)     physical mm coords of surface voxels
    pin_flat_idx   scalar int     flat index of the pinned node
    # Note: the sparse factor is NOT saved (not serialisable).
    # Call build_torso_grid() at runtime to get the factor object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Smoothed Boundary Method (SBM) — smoothed domain function ψ
# ---------------------------------------------------------------------------
#
# Motivation: the sharp-boundary FDM above (active_mask cuts voxels in/out,
# harmonic-mean face conductance goes exactly to 0 at any face touching air)
# resolves the true anatomical boundary only to the coarse voxel size h --
# a "staircase" -- because active_mask is a binary yes/no per coarse voxel.
# This is the leading suspect for the observed sub-first-order (p≈0.80,
# vs. expected ~1) cross-level convergence.
#
# SBM (Kim & Lowengrub 2005; Yu, Kim & Voorhees / Zhao et al.) replaces the
# sharp indicator with a smooth domain function ψ ∈ [0,1] and solves
#
#     ∇·(ψ σ ∇φ) = ψ b
#
# instead of ∇·(σ∇φ)=b on a hard-clipped domain. As ψ sharpens back into a
# step function this recovers the original PDE with a natural zero-flux
# Neumann condition on ∂{ψ=1/2} -- with NO explicit boundary tracking,
# because the smooth ψ term does the work that hard voxel inclusion/
# exclusion did before.
#
# IMPORTANT: ψ must be built from finer information than the coarse grid
# itself, or this is cosmetic. Smoothing an already-binarized coarse mask
# (naive SBM) leaves the boundary's *position* known only to O(h_coarse) --
# it removes the flux discontinuity but not the geometric staircase error,
# since the input to the smoothing step is exactly as coarse as before.
# Verified on a synthetic sphere (see conversation/notebook): naive
# coarse-smoothing gave essentially the same coverage-fraction RMS error as
# the plain binary mask (0.551 vs 0.551), while building ψ from the FINE
# segmentation and downsampling ψ itself (this implementation) gave ~23%
# lower error (0.426) by capturing sub-coarse-cell boundary position as a
# genuine partial-coverage fraction, not just smoothing the symptom.
#
# CORRECTNESS NOTE (found via a production run showing a suspicious RMS
# spike concentrated on the finest level pair, with L-inf roughly flat --
# see conversation history): ψ_face (the domain function's value at a
# face between two cells) uses a HARMONIC mean of ψ_k and ψ_n, not an
# arithmetic one, mirroring sigma_face just below it. This matters more
# than it looks: sigma is EXTENDED (nonzero) through the SBM band by
# extend_conductivity(), specifically so ψ alone shapes the boundary --
# but that means an arithmetic mean face-ψ lets a tissue cell (ψ=1) retain
# ~50% of full coupling strength to an immediately adjacent, nominally
# "excluded" band cell at ψ=SBM_PSI_CUTOFF, REGARDLESS of how small the
# cutoff is (arithmetic mean of 1 and anything is always >=0.5). That
# band cell's own diagonal is tiny (near-singular row), so this asymmetric
# coupling can pull well-conditioned tissue/surface nodes' solved values
# toward whatever a poorly-determined, warm-start-history-dependent band
# cell settles at. Harmonic mean(1, cutoff) ~ 2*cutoff instead -- correctly
# near-zero, tracking the more restrictive side, same principle already
# used for sigma_face. Confirmed by direct test: harmonic mean drops that
# coupling ratio from 0.500 to 0.002 for cutoff=1e-3. An earlier "SBM with
# step-function ψ reproduces the sharp case exactly" regression test
# passed even with the (buggy) arithmetic mean -- but only because that
# test used sigma=0 on the exterior side, which short-circuits via the
# sigma_k+sigma_n<1e-30 guard before ψ_face is ever evaluated. That guard
# does NOT fire once sigma is extended into the band (the real code path),
# so the test gave false confidence and missed this for one full
# implementation pass.
#
def compute_smoothed_domain(
    volume_fine:     np.ndarray,
    spacing_fine:    np.ndarray,
    dx_coarse_mm:    float,
    coarse_shape:    tuple,
    interface_width_mm: float = 3.0,
    psi_cutoff:      float = 1e-3,
) -> np.ndarray:
    """
    Build the SBM domain function ψ (coarse grid, values in [psi_cutoff, 1])
    from the FINE-resolution torso segmentation.

    Parameters
    ----------
    volume_fine   : (ni,nj,nk) int label volume at fine (native NRRD) res
    spacing_fine  : (3,) fine voxel spacing [mm]
    dx_coarse_mm  : coarse voxel size [mm]
    coarse_shape  : shape of the coarse grid ψ must match (from
                    _resample_labels on the same volume/spacing/dx) --
                    zoom() on fine vs. coarse data can differ by a voxel
                    at each axis edge due to independent rounding, so ψ is
                    cropped/padded to this shape exactly.
    interface_width_mm : ψ transition width (tanh length scale). ~1-2
                    coarse cells is the usual SBM choice: too narrow and
                    you're back to a near-binary mask (no benefit); too
                    wide and you blur real anatomy away from the boundary.
    psi_cutoff    : floor applied to ψ over the SBM-active region (see
                    below) -- keeps every included row's diagonal bounded
                    away from 0, avoiding a near-singular/ill-conditioned
                    system out at the edge of the smoothed band.

    Returns
    -------
    psi_coarse : (coarse_shape) float64, 0 outside the SBM-active band,
                 in [psi_cutoff, 1] inside it.
    """
    fine_mask = (volume_fine != 0).astype(np.float64)

    d_in  = ndi.distance_transform_edt(fine_mask,     sampling=spacing_fine)
    d_out = ndi.distance_transform_edt(1.0 - fine_mask, sampling=spacing_fine)
    d_signed = d_in - d_out                      # +inside, -outside, mm

    psi_fine = 0.5 * (1.0 + np.tanh(3.0 * d_signed / interface_width_mm))

    zoom_factor = spacing_fine / dx_coarse_mm     # same convention as _resample_labels
    psi_coarse = ndi.zoom(psi_fine, zoom_factor, order=1)

    # Reconcile off-by-one shape mismatches vs. the label resample (each
    # zoom() call rounds output size independently) -- crop/pad with 0
    # (i.e. "outside") rather than silently misaligning axes.
    out = np.zeros(coarse_shape, dtype=np.float64)
    sl_src = tuple(slice(0, min(s, c)) for s, c in zip(psi_coarse.shape, coarse_shape))
    sl_dst = tuple(slice(0, min(s, c)) for s, c in zip(psi_coarse.shape, coarse_shape))
    out[sl_dst] = psi_coarse[sl_src]

    out[out < psi_cutoff] = 0.0                   # true exterior: excluded from system
    return out


def extend_conductivity(sigma_vol: np.ndarray, tissue_mask: np.ndarray,
                        sbm_mask: np.ndarray) -> np.ndarray:
    """
    Extend sigma_vol (defined only on tissue_mask) by constant extrapolation
    (nearest-tissue-voxel value) into the SBM band (sbm_mask & ~tissue_mask).

    Why: SBM's boundary shaping comes entirely from ψ; σ itself should be
    smoothly defined (not hard-zeroed) through the band, or ψ=0 and σ=0
    both trying to enforce the same boundary independently just reproduces
    the sharp cutoff SBM was meant to remove.
    """
    if not np.any(sbm_mask & ~tissue_mask):
        return sigma_vol.copy()
    _, nearest_idx = ndi.distance_transform_edt(~tissue_mask, return_indices=True)
    extended = sigma_vol.copy()
    band = sbm_mask & ~tissue_mask
    extended[band] = sigma_vol[tuple(idx[band] for idx in nearest_idx)]
    return extended


# ---------------------------------------------------------------------------
# Flat voxel index -> physical mm coordinates
# ---------------------------------------------------------------------------

def voxel_ids_to_xyz(flat_ids: np.ndarray, shape: tuple,
                     dx_mm: float, origin_mm: np.ndarray) -> np.ndarray:
    """
    Convert flat (i*nj*nk + j*nk + k) voxel indices to physical mm
    coordinates. Shared by surface-node xyz (build_torso_grid) and
    full-active-node xyz (used for whole-volume, not surface-restricted,
    cross-level comparison -- see run_experiment7_dgx.py).
    """
    ni, nj, nk = shape
    i = flat_ids // (nj * nk)
    j = (flat_ids % (nj * nk)) // nk
    k = flat_ids % nk
    return (np.stack([i, j, k], axis=1).astype(np.float64) * dx_mm
            + origin_mm[np.newaxis, :])


# ---------------------------------------------------------------------------
# NRRD loader (shared pattern with register_heart_to_torso.py)
# ---------------------------------------------------------------------------

def _load_nrrd(path: Path):
    try:
        import nrrd
    except ImportError as e:
        raise ImportError("pynrrd required:  pip install pynrrd") from e

    volume, header = nrrd.read(str(path))
    volume = np.asarray(volume, dtype=np.int32)

    if "space directions" in header:
        sd      = np.asarray(header["space directions"], dtype=np.float64)
        spacing = np.array([np.linalg.norm(sd[i]) for i in range(3)])
        origin  = np.asarray(header.get("space origin", [0., 0., 0.]), np.float64)
    elif "spacings" in header:
        spacing = np.asarray(header["spacings"], np.float64)
        origin  = np.zeros(3, np.float64)
    else:
        spacing = np.ones(3, np.float64)
        origin  = np.zeros(3, np.float64)

    return volume, spacing, origin


# ---------------------------------------------------------------------------
# Resample NRRD to coarse resolution
# ---------------------------------------------------------------------------

def _resample_labels(volume:   np.ndarray,
                     spacing:  np.ndarray,
                     dx_coarse_mm: float) -> np.ndarray:
    """
    Downsample label volume to dx_coarse_mm using nearest-neighbour zoom.
    Preserves label integers exactly.
    """
    zoom = spacing / dx_coarse_mm          # per-axis zoom factors
    coarse = ndi.zoom(volume.astype(np.float32), zoom, order=0)
    return coarse.astype(np.int32)


# ---------------------------------------------------------------------------
# Stiffness matrix assembly
# ---------------------------------------------------------------------------

def _assemble_stiffness(sigma_vol: np.ndarray,
                        h_m:       float,
                        active_mask: np.ndarray,
                        active_ids:  np.ndarray,
                        id_map:      np.ndarray,
                        psi_vol:     "np.ndarray | None" = None) -> sp.csr_matrix:
    """
    Build the sparse FDM stiffness matrix for  ∇·(σ ∇φ) = b, or, if
    psi_vol is given, the SBM operator  ∇·(ψ σ ∇φ) = ψ b  (see
    compute_smoothed_domain docstring above for why/when to use this).

    Parameters
    ----------
    sigma_vol   : (ni, nj, nk) conductivity [S/m]. If psi_vol is given,
                  this should already be extended through the SBM band
                  (see extend_conductivity) -- σ itself should NOT be
                  hard-zeroed outside the true tissue mask in that case,
                  since ψ alone is what shapes the boundary; a second,
                  independent hard cutoff via σ=0 just reproduces the
                  staircase SBM is meant to remove.
    h_m         : voxel size [metres]  (uniform isotropic)
    active_mask : (ni, nj, nk) bool, True = included in system. Sharp
                  path: non-air voxels. SBM path: psi_vol > 0.
    active_ids  : (N,) flat indices of active voxels
    id_map      : (ni*nj*nk,) maps flat voxel index → row index in A
                  -1 for inactive voxels
    psi_vol     : (ni,nj,nk) float in [0,1], SBM domain function. None
                  (default) reproduces the original sharp-boundary
                  behaviour exactly (ψ≡1 on active_mask, ψ≡0 elsewhere).

    Returns
    -------
    A : (N, N) sparse CSR matrix, symmetric positive semi-definite
    """
    ni, nj, nk = sigma_vol.shape
    N          = len(active_ids)
    h2         = h_m ** 2

    rows, cols, vals = [], [], []

    # Neighbour offsets in (i,j,k) and corresponding flat strides
    strides = np.array([nj * nk, nk, 1], dtype=np.int64)
    offsets = [( 1, 0, 0), (-1, 0, 0),
               ( 0, 1, 0), ( 0,-1, 0),
               ( 0, 0, 1), ( 0, 0,-1)]

    sigma_flat = sigma_vol.ravel()
    psi_flat = psi_vol.ravel() if psi_vol is not None else None

    for flat_k in active_ids:
        row_k  = id_map[flat_k]
        sigma_k = sigma_flat[flat_k]
        psi_k = psi_flat[flat_k] if psi_flat is not None else 1.0
        diag    = 0.0

        # Unravel flat index
        i = flat_k // (nj * nk)
        j = (flat_k % (nj * nk)) // nk
        k = flat_k % nk

        for di, dj, dk in offsets:
            ni_, nj_, nk_ = i + di, j + dj, k + dk
            if not (0 <= ni_ < ni and 0 <= nj_ < nj and 0 <= nk_ < nk):
                continue                       # boundary — Neumann: skip

            flat_n  = ni_ * nj * nk + nj_ * nk + nk_
            sigma_n = sigma_flat[flat_n]

            if psi_flat is None:
                if sigma_k + sigma_n < 1e-30:  # both air — no coupling
                    continue
                psi_face = 1.0
            else:
                psi_n = psi_flat[flat_n]
                if psi_k <= 0.0 and psi_n <= 0.0:  # both fully exterior
                    continue
                if sigma_k + sigma_n < 1e-30:
                    continue
                # Harmonic (not arithmetic) mean -- tracks the SMALLER of
                # the two psi values, same principle already used for
                # sigma_face below (and for the same reason: flux/coupling
                # across a face should be limited by the more restrictive
                # side, not averaged with it). This matters a lot in
                # practice: with sigma extended into the SBM band (see
                # extend_conductivity -- sigma is nonzero there by design,
                # so the sigma_k+sigma_n<1e-30 guard above does NOT save
                # us here), an ARITHMETIC mean lets a psi=1 tissue face
                # retain ~50% of full coupling strength to an immediately
                # adjacent psi=SBM_PSI_CUTOFF band cell, regardless of how
                # small the cutoff is -- i.e. a "fully excluded" neighbour
                # still couples at half strength. That partially defeats
                # the point of psi shaping the boundary, and lets a
                # poorly-conditioned (near-singular, since its own
                # diagonal is tiny) band cell's solved value pull on its
                # well-conditioned tissue neighbour's equation. Harmonic
                # mean(1, cutoff) ~ 2*cutoff, correctly near-zero.
                if psi_k + psi_n < 1e-30:
                    psi_face = 0.0
                else:
                    psi_face = (2.0 * psi_k * psi_n) / (psi_k + psi_n)

            sigma_face = (2.0 * sigma_k * sigma_n) / (sigma_k + sigma_n)
            conductance = psi_face * sigma_face / h2

            diag += conductance

            if active_mask.ravel()[flat_n]:    # neighbour in system
                row_n = id_map[flat_n]
                rows.append(row_k)
                cols.append(row_n)
                vals.append(-conductance)

        rows.append(row_k)
        cols.append(row_k)
        vals.append(diag)

    A = sp.csr_matrix(
        (np.array(vals, np.float64),
         (np.array(rows, np.int32), np.array(cols, np.int32))),
        shape=(N, N),
    )
    return A


# ---------------------------------------------------------------------------
# Factorisation
# ---------------------------------------------------------------------------

def _factorise(A: sp.csc_matrix, pin_row: int):
    """
    Pin one node (remove nullspace), then factorise.

    Pinning: set row and column of pin_row to identity (diagonal = 1,
    off-diagonal = 0).  This fixes φ[pin_row] = 0 (reference potential).

    Returns the factor object and a function  solve(b) → φ.
    """
    A = A.tolil()
    A[pin_row, :] = 0.0
    A[:, pin_row] = 0.0
    A[pin_row, pin_row] = 1.0
    A = A.tocsc()

    # Try CHOLMOD first (scikit-sparse) — fastest on Linux/Mac
    try:
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        print("  [factor] Using CHOLMOD (scikit-sparse)")

        def solve_fn(b: np.ndarray) -> np.ndarray:
            b[pin_row] = 0.0
            return factor(b)

        return factor, solve_fn

    except ImportError:
        pass

    # Try pypardiso — fastest on Windows (Intel MKL)
    try:
        import pypardiso
        print("  [factor] Using pypardiso (Intel MKL PARDISO)")
        # pypardiso works as a drop-in for scipy.sparse.linalg.spsolve
        # but pre-analyses the matrix for reuse

        from pypardiso import factorized as pf
        solve_pardiso = pf(A)

        def solve_fn(b: np.ndarray) -> np.ndarray:
            b[pin_row] = 0.0
            return solve_pardiso(b)

        return solve_pardiso, solve_fn

    except ImportError:
        pass

    # Fall back to scipy SuperLU — slower but always available
    print("  [factor] Using scipy SuperLU (consider: pip install scikit-sparse or pypardiso)")
    from scipy.sparse.linalg import splu
    factor = splu(A)

    def solve_fn(b: np.ndarray) -> np.ndarray:
        b[pin_row] = 0.0
        return factor.solve(b)

    return factor, solve_fn


# ---------------------------------------------------------------------------
# Surface node identification
# ---------------------------------------------------------------------------

def _find_surface_ids(label_vol:   np.ndarray,
                      active_mask: np.ndarray,
                      id_map:      np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Surface voxels = active (label >= 1) voxels that have at least one
    face-neighbour with label 0 (air).

    Returns
    -------
    surface_flat : (N_surf,) flat indices into the coarse grid
    surface_rows : (N_surf,) row indices into the system matrix A
    """
    is_air     = (label_vol == 0)
    # Dilate air mask by 1 voxel in each axis — overlap = surface
    air_dilated = ndi.binary_dilation(is_air, structure=ndi.generate_binary_structure(3, 1))
    surface_mask = active_mask & air_dilated

    surface_flat = np.where(surface_mask.ravel())[0].astype(np.int32)
    surface_rows = id_map[surface_flat]
    return surface_flat, surface_rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_torso_grid(
    torso_nrrd_path: Path,
    sigma_map:       dict,
    dx_coarse_mm:    float,
    out_path:        Path,
    verbose:         bool = True,
    use_sbm:         bool = False,
    sbm_interface_width_mm: float = 3.0,
    sbm_psi_cutoff:  float = 1e-3,
) -> dict:
    """
    Build, factorise, and save the coarse torso FDM grid.

    Parameters
    ----------
    torso_nrrd_path : path to torso.seg.nrrd
    sigma_map       : dict  label (int) → conductivity (S/m)
    dx_coarse_mm    : coarse voxel size [mm]
    out_path        : path to save torso_grid.npz
    verbose         : print progress
    use_sbm         : if True, use the Smoothed Boundary Method instead of
                      the sharp-cut (staircased) boundary -- solves
                      ∇·(ψσ∇φ)=ψb with ψ built from the FINE segmentation
                      (see compute_smoothed_domain docstring). If False
                      (default), reproduces the original sharp-boundary
                      behaviour exactly, byte-for-byte in the algorithm
                      (ψ is never constructed).
    sbm_interface_width_mm : ψ transition width (only used if use_sbm).
                      ~1-2 coarse cells is the usual choice.
    sbm_psi_cutoff  : ψ floor / inclusion threshold (only used if use_sbm).

    Returns
    -------
    dict with keys:
        shape, dx_mm, origin_mm, sigma_vol, active_mask, active_ids,
        active_id_map, surface_flat, surface_rows, surface_xyz,
        pin_row, solve_fn, psi_vol (SBM only), tissue_mask
    """
    # ── 1. Load NRRD ──────────────────────────────────────────────────────
    if verbose:
        print(f"  Loading {Path(torso_nrrd_path).name} ...")
    volume, spacing, origin = _load_nrrd(Path(torso_nrrd_path))
    if verbose:
        print(f"    Fine shape  : {volume.shape}  spacing={spacing} mm")

    # ── 2. Resample ───────────────────────────────────────────────────────
    if verbose:
        print(f"  Resampling to {dx_coarse_mm} mm ...")
    coarse_vol = _resample_labels(volume, spacing, dx_coarse_mm)
    # Coarse origin: same physical corner as fine grid
    coarse_origin = origin.copy()
    shape = coarse_vol.shape
    if verbose:
        print(f"    Coarse shape: {shape}  "
              f"({shape[0]*shape[1]*shape[2]/1e6:.2f} M voxels)")

    # ── 3. Conductivity volume ────────────────────────────────────────────
    if verbose:
        print("  Assigning conductivities ...")
    sigma_vol = np.zeros(shape, dtype=np.float64)
    for label, sigma in sigma_map.items():
        sigma_vol[coarse_vol == label] = sigma
    if verbose:
        for label, sigma in sigma_map.items():
            n = (coarse_vol == label).sum()
            print(f"    label {label}: {n:>8,} voxels  σ={sigma} S/m")

    # ── 4. Domain mask(s) ───────────────────────────────────────────────────
    # tissue_mask: the TRUE anatomical torso (non-air), used throughout for
    # surface/electrode-site identification regardless of use_sbm -- SBM
    # only changes how the *linear system* treats the boundary, not where
    # BSPM measurement sites physically are.
    tissue_mask = (coarse_vol != 0)

    psi_vol = None
    if use_sbm:
        if verbose:
            print(f"  Building SBM domain function ψ  "
                  f"(interface width={sbm_interface_width_mm} mm, "
                  f"from fine segmentation) ...")
        psi_vol = compute_smoothed_domain(
            volume, spacing, dx_coarse_mm, shape,
            interface_width_mm=sbm_interface_width_mm,
            psi_cutoff=sbm_psi_cutoff,
        )
        active_mask = psi_vol > 0.0
        n_band = int((active_mask & ~tissue_mask).sum())
        if verbose:
            print(f"    ψ range on active band: "
                  f"[{psi_vol[active_mask].min():.4f}, {psi_vol[active_mask].max():.4f}]")
            print(f"    SBM band voxels beyond true tissue: {n_band:,}")
        sigma_vol = extend_conductivity(sigma_vol, tissue_mask, active_mask)
    else:
        active_mask = tissue_mask

    active_flat = np.where(active_mask.ravel())[0].astype(np.int32)
    N_active    = len(active_flat)

    # Map flat index → row in system matrix (-1 = inactive)
    id_map = np.full(sigma_vol.size, -1, dtype=np.int32)
    id_map[active_flat] = np.arange(N_active, dtype=np.int32)

    if verbose:
        label = "SBM-active" if use_sbm else "Active (non-air)"
        print(f"  {label} voxels: {N_active:,}  "
              f"({N_active/sigma_vol.size*100:.1f} % of coarse grid)")

    # ── 5. Assemble stiffness matrix ──────────────────────────────────────
    h_m = dx_coarse_mm * 1e-3   # mm → metres for SI conductivity [S/m]
    if verbose:
        method = "SBM" if use_sbm else "sharp-boundary"
        print(f"  Assembling {method} FDM stiffness matrix  (h={dx_coarse_mm} mm) ...")
        print(f"    System size: {N_active:,} × {N_active:,}")

    A = _assemble_stiffness(sigma_vol, h_m, active_mask, active_flat, id_map,
                             psi_vol=psi_vol)

    if verbose:
        print(f"    Non-zeros  : {A.nnz:,}  "
              f"(fill {A.nnz/N_active**2*100:.4f} %)")

    # ── 6. Pin one interior node (reference potential = 0) ────────────────
    # Choose a deep interior node far from any boundary -- from tissue_mask
    # specifically (ψ≈1 there even under SBM), not the SBM-extended band,
    # so the pinned reference is always a physically solid, well-conditioned
    # choice regardless of use_sbm.
    interior = np.where(tissue_mask.ravel())[0]
    pin_flat  = int(interior[len(interior) // 2])
    pin_row   = int(id_map[pin_flat])
    if verbose:
        print(f"  Pinning node flat={pin_flat}  row={pin_row}")

    # ── 7. Factorise ──────────────────────────────────────────────────────
    if verbose:
        print("  Factorising stiffness matrix ...")
    import time
    t0 = time.perf_counter()
    factor, solve_fn = _factorise(A.tocsc(), pin_row)
    if verbose:
        print(f"    Factorisation done in {time.perf_counter()-t0:.2f} s")

    # ── 8. Surface nodes ──────────────────────────────────────────────────
    # Always identified against tissue_mask (true anatomy), not the
    # SBM-extended active_mask -- BSPM electrode sites are physical torso
    # surface locations regardless of how the solve handles the boundary.
    if verbose:
        print("  Identifying surface nodes ...")
    surface_flat, surface_rows = _find_surface_ids(coarse_vol, tissue_mask, id_map)

    # Physical mm coordinates of surface voxels
    surface_xyz = voxel_ids_to_xyz(surface_flat, shape, dx_coarse_mm, coarse_origin)

    if verbose:
        print(f"    Surface nodes: {len(surface_flat):,}")

    # ── 9. Save metadata (not the factor — not serialisable) ──────────────
    save_kwargs = dict(
        shape        = np.array(shape, np.int32),
        dx_mm        = np.float64(dx_coarse_mm),
        origin_mm    = coarse_origin,
        sigma_vol    = sigma_vol.astype(np.float32),
        active_mask  = active_mask,
        tissue_mask  = tissue_mask,
        active_ids   = active_flat,
        active_id_map= id_map,
        surface_flat = surface_flat,
        surface_rows = surface_rows,
        surface_xyz  = surface_xyz.astype(np.float32),
        pin_row      = np.int32(pin_row),
        use_sbm      = use_sbm,
    )
    if use_sbm:
        save_kwargs["psi_vol"] = psi_vol.astype(np.float32)
    np.savez(str(out_path), **save_kwargs)
    if verbose:
        print(f"  Saved → {out_path}")

    result = dict(
        shape        = shape,
        dx_mm        = dx_coarse_mm,
        origin_mm    = coarse_origin,
        sigma_vol    = sigma_vol,
        active_mask  = active_mask,
        tissue_mask  = tissue_mask,
        active_ids   = active_flat,
        active_id_map= id_map,
        surface_flat = surface_flat,
        surface_rows = surface_rows,
        surface_xyz  = surface_xyz,
        pin_row      = pin_row,
        solve_fn     = solve_fn,
        A_scipy      = A,            # raw CSR — used by GPU PCG solver
        use_sbm      = use_sbm,
    )
    if use_sbm:
        result["psi_vol"] = psi_vol
    return result


def load_torso_grid(
    out_path:        Path,
    torso_nrrd_path: Path,
    sigma_map:       dict,
    dx_coarse_mm:    float,
    verbose:         bool = True,
) -> dict:
    """
    Load a previously saved torso_grid.npz and re-build the factorisation
    (since the sparse factor cannot be serialised).

    The geometry/conductivity are loaded from the npz; only the matrix
    assembly and factorisation are redone (fast: ~10–30 s on 32 GB machine).
    """
    if verbose:
        print(f"  Loading {Path(out_path).name} ...")
    data = np.load(str(out_path))

    sigma_vol    = data["sigma_vol"].astype(np.float64)
    active_mask  = data["active_mask"]
    active_ids   = data["active_ids"]
    id_map       = data["active_id_map"]
    pin_row      = int(data["pin_row"])
    shape        = tuple(data["shape"].tolist())
    dx_mm        = float(data["dx_mm"])
    use_sbm      = bool(data["use_sbm"]) if "use_sbm" in data.files else False
    psi_vol      = data["psi_vol"].astype(np.float64) if "psi_vol" in data.files else None

    h_m = dx_mm * 1e-3
    N   = len(active_ids)
    if verbose:
        print(f"    Grid shape  : {shape}  dx={dx_mm} mm")
        print(f"    Active nodes: {N:,}" + ("  (SBM)" if use_sbm else ""))
        print("  Re-assembling stiffness matrix ...")

    A = _assemble_stiffness(sigma_vol, h_m, active_mask, active_ids, id_map,
                             psi_vol=psi_vol)

    # The CPU factorisation (_factorise) is only needed by callers that
    # actually invoke solve_fn -- compute_bspm's GPU PCG path only reads
    # A_scipy and never touches solve_fn. Factorising here unconditionally
    # cost ~10-30s on every load for zero benefit in that path. Deferred via
    # a small lazy wrapper: the first real call factorises (and prints the
    # same message the eager path used to print up front); every call after
    # that reuses the cached solve_fn. Callers keep doing grid["solve_fn"](b)
    # exactly as before -- no call-site changes needed anywhere else.
    _lazy = {}
    def solve_fn(b: np.ndarray) -> np.ndarray:
        if "fn" not in _lazy:
            if verbose:
                print("  Factorising stiffness matrix (first solve_fn call) ...")
            _, _lazy["fn"] = _factorise(A.tocsc(), pin_row)
        return _lazy["fn"](b)

    result = {k: data[k] for k in data.files}
    result.update(dict(
        shape        = shape,
        dx_mm        = dx_mm,
        active_id_map= id_map,
        solve_fn     = solve_fn,
        pin_row      = pin_row,
        A_scipy      = A,       # raw CSR — used by GPU PCG solver
        use_sbm      = use_sbm,  # normalize from npz 0-d array to plain bool
    ))
    return result
