"""
generate_apicobasal_coordinate.py
===================================
Solves the Laplace equation on the myocardium mask to produce an
apico-basal coordinate  psi in [0, 1]  (0 = apex, 1 = base).

Optimisations
-------------
1. Neighbour sum uses pad+slice instead of torch.roll — halves memory
   traffic per RBGS iteration on large volumes.
2. Valid-neighbour count precomputed once before the loop.
3. EDT warm-start: psi_init = dist_to_apex / (dist_to_apex + dist_to_base).
4. save_path: result cached to .npy; subsequent calls skip all computation.

Public API
----------
    psi = generate_apicobasal_coordinate(S, ab_axis, base_idx,
                                          surface_labels, ...)
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from tqdm import trange
from scipy.ndimage import convolve, binary_dilation, distance_transform_edt


# =============================================================================
# Vectorised neighbour helpers
# =============================================================================

def _neighbour_sum(psi: torch.Tensor) -> torch.Tensor:
    """6-connected neighbour sum via pad+slice (no torch.roll copies)."""
    p = torch.nn.functional.pad(psi, (1, 1, 1, 1, 1, 1))
    return (
        p[:-2, 1:-1, 1:-1] +
        p[2:,  1:-1, 1:-1] +
        p[1:-1, :-2, 1:-1] +
        p[1:-1, 2:,  1:-1] +
        p[1:-1, 1:-1, :-2] +
        p[1:-1, 1:-1, 2: ]
    )


def _valid_neighbour_count(S_t: torch.Tensor) -> torch.Tensor:
    """Number of in-mask face-neighbours per voxel. Computed once."""
    p = torch.nn.functional.pad(S_t.float(), (1, 1, 1, 1, 1, 1))
    return (
        p[:-2, 1:-1, 1:-1] +
        p[2:,  1:-1, 1:-1] +
        p[1:-1, :-2, 1:-1] +
        p[1:-1, 2:,  1:-1] +
        p[1:-1, 1:-1, :-2] +
        p[1:-1, 1:-1, 2: ]
    )


# =============================================================================
# Public API
# =============================================================================

def generate_apicobasal_coordinate(
    S: np.ndarray,
    ab_axis: int | None = None,
    base_idx: int | None = None,
    surface_labels: np.ndarray | None = None,
    apex_percent: float = 0.10,
    max_iter: int = 40,
    tol: float = 1e-5,
    device: str | torch.device | None = None,
    save_path: str | Path | None = None,
    apex_mask: np.ndarray | None = None,
    basal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute apico-basal coordinate psi by solving Laplace's equation.

    psi = 0 at apex,  psi = 1 at base.

    Two ways to specify the Dirichlet regions
    ------------------------------------------
    Preferred: pass `apex_mask` / `basal_mask` directly -- boolean voxel
    arrays (e.g. from mesh_labelling.label_ventricle_mesh's
    "apex_voxels" / "basal_voxels"). These are located via the true,
    continuous long axis of the heart and are not tied to any voxel-grid
    axis, so they remain correct regardless of how the heart sits in the
    image.

    Legacy fallback: if `apex_mask`/`basal_mask` are not given, the
    original grid-axis-based derivation is used (`ab_axis`, `base_idx`,
    `surface_labels` become required in that case). This path assumes the
    apicobasal direction is aligned with one of the three grid axes and
    is kept only for backward compatibility -- prefer the mask-based
    inputs for any non-trivially-oriented heart.

    Parameters
    ----------
    S              : (Nx,Ny,Nz) ndarray  binary myocardium mask
    ab_axis        : int   apico-basal axis index (0, 1, or 2) -- legacy
    base_idx       : int   basal plane index along ab_axis -- legacy
    surface_labels : (Nx,Ny,Nz) ndarray  +1 epi / -1 endo / 0 other --
                     required for the legacy path
    apex_mask      : (Nx,Ny,Nz) bool  Dirichlet psi=0 region (preferred)
    basal_mask     : (Nx,Ny,Nz) bool  Dirichlet psi=1 region (preferred)
    apex_percent   : legacy-path only; unused when apex_mask is given
    max_iter       : RBGS iteration cap
    tol            : convergence tolerance (max absolute change per sweep)
    device         : torch device (auto-detected if None)
    save_path      : if given, save result to this .npy path and load on
                     subsequent calls instead of recomputing.

    Returns
    -------
    psi : (Nx,Ny,Nz) float32  apico-basal coordinate (NaN outside mask)
    """
    # ── Cache load ────────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        if save_path.exists():
            print(f"  [apicobasal] Loading cached psi from {save_path}")
            return np.load(save_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
        print("  [apicobasal] CUDA not available — falling back to CPU")

    S      = S.astype(bool)
    shape  = S.shape
    N_myo  = int(S.sum())
    print(f"  [apicobasal] {N_myo:,} myocardial voxels  |  device: {device}")

    if apex_mask is not None and basal_mask is not None:
        # ── Preferred path: caller supplies the Dirichlet regions directly,
        # e.g. from mesh_labelling.label_ventricle_mesh, which locates them
        # via the true (continuous, non-grid-aligned) cardiac long axis.
        dirichlet_0 = apex_mask.astype(bool) & S
        dirichlet_1 = basal_mask.astype(bool) & S
        print(f"  [apicobasal] Using supplied apex/basal masks "
              f"({int(dirichlet_0.sum())} / {int(dirichlet_1.sum())} voxels)")
    else:
        # ── Legacy path: derive both regions from a single grid axis. Only
        # correct if the heart's apicobasal direction happens to be aligned
        # with that axis -- prefer apex_mask/basal_mask above for anything
        # else. Kept for backward compatibility only.
        if ab_axis is None or base_idx is None or surface_labels is None:
            raise ValueError(
                "Either (apex_mask, basal_mask) or (ab_axis, base_idx, "
                "surface_labels) must be provided.")

        # Apex Dirichlet patch (psi = 0)
        epi_coords   = np.argwhere(surface_labels == 1)
        dist_to_base = np.abs(epi_coords[:, ab_axis] - base_idx)
        apex_voxel   = epi_coords[np.argmax(dist_to_base)]
        epi_dists    = np.linalg.norm(epi_coords - apex_voxel, axis=1)
        k            = max(1, min(30, len(epi_coords)))
        apex_cap     = epi_coords[np.argsort(epi_dists)[:k]]

        dirichlet_0 = np.zeros(shape, dtype=bool)
        dirichlet_0[apex_cap[:, 0], apex_cap[:, 1], apex_cap[:, 2]] = True

        # Basal Dirichlet ridge (psi = 1) — epi/endo contact zone
        epi  = surface_labels ==  1
        endo = surface_labels == -1

        kernel     = np.ones((3, 3, 3), dtype=np.int8)
        kernel[1, 1, 1] = 0
        endo_neigh = convolve(endo.astype(np.int8), kernel, mode="constant", cval=0)
        epi_neigh  = convolve(epi.astype(np.int8),  kernel, mode="constant", cval=0)

        dirichlet_1 = ((epi & (endo_neigh > 0)) | (endo & (epi_neigh > 0)))
        dirichlet_1 = binary_dilation(dirichlet_1,
                                      structure=np.ones((3,3,3), dtype=bool),
                                      iterations=3) & S

    # ── Active masks ──────────────────────────────────────────────────────────
    S_t  = torch.from_numpy(S).to(device)
    d0   = torch.from_numpy(dirichlet_0).to(device)
    d1   = torch.from_numpy(dirichlet_1).to(device)

    active = S_t & (~d0) & (~d1)
    parity = (
        torch.arange(shape[0], device=device)[:, None, None] +
        torch.arange(shape[1], device=device)[None, :, None] +
        torch.arange(shape[2], device=device)[None, None, :]
    ) % 2 == 0
    red_active   = active &  parity
    black_active = active & ~parity

    # ── EDT warm-start ────────────────────────────────────────────────────────
    print("  [apicobasal] Computing EDT initial guess ...", flush=True)
    a   = distance_transform_edt(~dirichlet_0)
    b   = distance_transform_edt(~dirichlet_1)
    den = a + b

    psi_init = np.full(shape, np.nan, dtype=np.float32)
    valid    = S & (den > 0)
    psi_init[valid]       = (a[valid] / den[valid]).astype(np.float32)
    psi_init[dirichlet_0] = 1e-6
    psi_init[dirichlet_1] = 1.0
    psi_init[~S]          = np.nan

    init_vals = psi_init[S]
    print(f"  [apicobasal] EDT init range: "
          f"[{np.nanmin(init_vals):.4f}, {np.nanmax(init_vals):.4f}]")

    psi = torch.from_numpy(np.nan_to_num(psi_init, nan=0.0)).to(device).contiguous()

    # ── Precompute valid-neighbour count ──────────────────────────────────────
    valid_count = _valid_neighbour_count(S_t).clamp(min=1.0)

    # ── RBGS solve ────────────────────────────────────────────────────────────
    print(f"  [apicobasal] RBGS solve  (tol={tol}, max_iter={max_iter}) ...",
          flush=True)

    converged = False
    err       = float("inf")
    it        = 0

    pbar = trange(max_iter, desc="Apicobasal RBGS", leave=True)
    for it in pbar:
        psi_old = psi.clone()

        acc = _neighbour_sum(psi)
        psi[red_active]   = acc[red_active]   / valid_count[red_active]
        psi[d0] = 1e-6
        psi[d1] = 1.0

        acc = _neighbour_sum(psi)
        psi[black_active] = acc[black_active] / valid_count[black_active]
        psi[d0] = 1e-6
        psi[d1] = 1.0

        err = torch.max(torch.abs(psi - psi_old)).item()
        pbar.set_postfix(err=f"{err:.3e}")
        if err < tol:
            converged = True
            pbar.close()
            break

    if converged:
        print(f"  [apicobasal] RBGS converged in {it + 1} / {max_iter} iterations  "
              f"(max change {err:.2e})")
    else:
        print(f"  [apicobasal] WARNING: RBGS did NOT converge in {max_iter} iterations  "
              f"(max change {err:.2e} > tol {tol:.2e})  "
              f"— consider increasing max_iter or loosening tol")

    # ── Reconstruct volume (NaN outside mask) ─────────────────────────────────
    psi_np      = psi.detach().cpu().numpy().astype(np.float32)
    psi_out     = np.full(shape, np.nan, dtype=np.float32)
    psi_out[S]  = psi_np[S]
    np.clip(psi_out, 0.0, 1.0, out=psi_out)

    print(f"  [apicobasal] Done.  psi range (inside mask): "
          f"[{psi_out[S].min():.4f}, {psi_out[S].max():.4f}]")

    # ── Cache save ────────────────────────────────────────────────────────────
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, psi_out)
        print(f"  [apicobasal] Saved psi -> {save_path}")

    return psi_out
