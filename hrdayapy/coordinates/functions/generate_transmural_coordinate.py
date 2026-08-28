"""
generate_transmural_coordinate.py
==================================
Solves the Laplace equation on the myocardium mask to produce a
transmural coordinate  phi in [0, 1]  (0 = endocardium, 1 = epicardium).

Optimisations
-------------
1. Laplacian assembly is fully vectorised (NumPy) — no Python loop over voxels.
2. Sparse matrix built as scipy CSR, converted to torch COO for GPU CG.
3. EDT warm-start: phi_init = dist_to_endo / (dist_to_endo + dist_to_epi).
   Reduces CG iteration count by ~5-10x vs a zero initial guess.
4. Volume reconstruction uses numpy fancy-indexing.
5. save_path: result cached to .npy; subsequent calls skip all computation.

Public API
----------
    phi = generate_transmural_coordinate(S, surface_label, ...)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from tqdm import trange
from scipy.ndimage import distance_transform_edt


# =============================================================================
# CG solver
# =============================================================================

def _conjugate_gradient(
    A,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    maxiter: int = 10_000,
) -> torch.Tensor:
    x      = torch.zeros_like(b) if x0 is None else x0.clone()
    r      = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    p      = r.clone()
    rsold  = torch.dot(r, r)
    r0norm = rsold.sqrt().item()

    converged = False
    rel_res   = 1.0
    it        = 0

    pbar = trange(maxiter, desc="Transmural CG", leave=True)
    for it in pbar:
        Ap      = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
        alpha   = rsold / torch.dot(p, Ap)
        x       = x + alpha * p
        r       = r - alpha * Ap
        rsnew   = torch.dot(r, r)
        rel_res = rsnew.sqrt().item() / (r0norm + 1e-30)
        pbar.set_postfix(rel_res=f"{rel_res:.2e}")
        if rel_res < tol:
            converged = True
            pbar.close()
            break
        p      = r + (rsnew / rsold) * p
        rsold  = rsnew

    if converged:
        print(f"  [transmural] CG converged in {it + 1} / {maxiter} iterations  "
              f"(rel. residual {rel_res:.2e})")
    else:
        print(f"  [transmural] WARNING: CG did NOT converge in {maxiter} iterations  "
              f"(rel. residual {rel_res:.2e} > tol {tol:.2e})  "
              f"— consider increasing maxiter or loosening tol")

    return x


# =============================================================================
# Vectorised Laplacian assembly
# =============================================================================

def _build_laplacian_transmural(
    vox_idx: np.ndarray,
    flat_to_node: np.ndarray,
    surface_label: np.ndarray,
    shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build sparse Laplacian A and RHS b for the transmural Laplace solve.

    Boundary conditions
    -------------------
    surface_label == +1 (epicardium) : Dirichlet phi = 1
    surface_label == -1 (endocardium): Dirichlet phi = 0
    all other myocardial voxels      : Neumann d(phi)/dn = 0
    """
    Nx, Ny, Nz = shape
    N = len(vox_idx)

    surf_vals = surface_label[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]
    is_epi    = surf_vals ==  1
    is_endo   = surf_vals == -1
    is_dir    = is_epi | is_endo
    is_inter  = ~is_dir

    b_np = np.zeros(N, dtype=np.float32)
    b_np[is_epi]  = 1.0

    # Dirichlet rows: A[p,p] = 1
    dir_ids = np.where(is_dir)[0]
    rows_d  = dir_ids
    cols_d  = dir_ids
    vals_d  = np.ones(len(dir_ids), dtype=np.float32)

    # Interior rows: 6-point stencil, Neumann BCs at mask boundary
    inter_ids = np.where(is_inter)[0]
    vi = vox_idx[inter_ids]

    offsets = np.array([
        [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]
    ], dtype=np.int32)

    rows_off_list, cols_off_list, vals_off_list = [], [], []
    degree = np.zeros(N, dtype=np.float32)

    for dx, dy, dz in offsets:
        ni = vi[:, 0] + dx
        nj = vi[:, 1] + dy
        nk = vi[:, 2] + dz

        in_bounds  = ((ni >= 0) & (ni < Nx) &
                      (nj >= 0) & (nj < Ny) &
                      (nk >= 0) & (nk < Nz))
        nflat      = np.where(in_bounds, ni*Ny*Nz + nj*Nz + nk, 0).astype(np.int64)
        neigh_node = np.where(in_bounds, flat_to_node[nflat], -1)
        has_neigh  = in_bounds & (neigh_node >= 0)

        src = inter_ids[has_neigh]
        dst = neigh_node[has_neigh]
        rows_off_list.append(src)
        cols_off_list.append(dst)
        vals_off_list.append(np.full(src.shape, -1.0, dtype=np.float32))

        # Diagonal: +1 only for directions with an actual tissue neighbour.
        # Neumann faces (boundary or outside mask) contribute nothing to the
        # stencil — the flux is zero there, so the diagonal is not inflated.
        np.add.at(degree, inter_ids[has_neigh], 1.0)

    rows_off = np.concatenate(rows_off_list)
    cols_off = np.concatenate(cols_off_list)
    vals_off = np.concatenate(vals_off_list)

    all_rows = np.concatenate([rows_d,  rows_off,  inter_ids])
    all_cols = np.concatenate([cols_d,  cols_off,  inter_ids])
    all_vals = np.concatenate([vals_d,  vals_off,  degree[inter_ids]])

    # The stencil loop already adds symmetric off-diagonal pairs for interior
    # nodes, and Dirichlet boundary rows are written as identity rows (diagonal
    # only).  Explicitly symmetrising with 0.5*(A + A^T) would average the
    # identity diagonal with an off-diagonal contribution from the transpose,
    # corrupting the Dirichlet constraints.  The matrix is assembled directly
    # without post-hoc symmetrisation.
    A_csr = sp.csr_matrix(
        (all_vals, (all_rows.astype(np.int32), all_cols.astype(np.int32))),
        shape=(N, N), dtype=np.float32)

    A_coo   = A_csr.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]),
                           dtype=torch.long, device=device)
    values  = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    A_torch = torch.sparse_coo_tensor(indices, values, (N, N),
                                      device=device).coalesce()
    b_torch = torch.tensor(b_np, dtype=torch.float32, device=device)

    return A_torch, b_torch


# =============================================================================
# Public API
# =============================================================================

def generate_transmural_coordinate(
    S: np.ndarray,
    surface_label: np.ndarray,
    device: str | torch.device = "cuda",
    tol: float = 1e-6,
    maxiter: int = 10_000,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Generate transmural coordinate phi in [0,1] by solving the Laplace equation.

    Epicardium  (surface_label == +1) : Dirichlet phi = 1
    Endocardium (surface_label == -1) : Dirichlet phi = 0
    Elsewhere                         : Neumann d(phi)/dn = 0

    Parameters
    ----------
    S              : (Nx,Ny,Nz) uint8   binary myocardium mask
    surface_label  : (Nx,Ny,Nz) int8   +1 epi / -1 endo / 0 interior
    device         : torch device string or object (default "cuda")
    tol            : CG relative residual tolerance
    maxiter        : CG iteration cap
    save_path      : if given, save result to this .npy path and load it
                     on subsequent calls instead of recomputing.

    Returns
    -------
    phi : (Nx,Ny,Nz) float32  transmural coordinate (NaN outside mask)
    """
    # ── Cache load ────────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".npy")
        if save_path.exists():
            print(f"  [transmural] Loading cached phi from {save_path}")
            return np.load(save_path)

    assert S.shape == surface_label.shape
    shape      = S.shape
    Nx, Ny, Nz = shape

    _dev = torch.device(device) if isinstance(device, str) else device
    if not torch.cuda.is_available() and _dev.type == "cuda":
        _dev = torch.device("cpu")
        print("  [transmural] CUDA not available — falling back to CPU")

    # ── Node index map ────────────────────────────────────────────────────────
    print("  [transmural] Building node index map ...", flush=True)
    vox_idx = np.argwhere(S == 1).astype(np.int32)
    N       = len(vox_idx)
    print(f"  [transmural] {N:,} myocardial voxels  |  device: {_dev}")

    flat_to_node = -np.ones(Nx * Ny * Nz, dtype=np.int32)
    flat_ids     = vox_idx[:, 0]*Ny*Nz + vox_idx[:, 1]*Nz + vox_idx[:, 2]
    flat_to_node[flat_ids] = np.arange(N, dtype=np.int32)

    # ── Laplacian assembly ────────────────────────────────────────────────────
    print("  [transmural] Assembling Laplacian (vectorised) ...", flush=True)
    A, b = _build_laplacian_transmural(
        vox_idx, flat_to_node, surface_label, shape, _dev)

    # ── EDT warm-start ────────────────────────────────────────────────────────
    print("  [transmural] Computing EDT initial guess ...", flush=True)
    dist_endo = distance_transform_edt(surface_label != -1)
    dist_epi  = distance_transform_edt(surface_label !=  1)
    denom     = dist_endo + dist_epi
    phi_vol   = np.where(denom > 0, dist_endo / denom, 0.5).astype(np.float32)
    phi_vol[surface_label == -1] = 0.0
    phi_vol[surface_label ==  1] = 1.0

    phi_init_flat = phi_vol[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]
    x0 = torch.tensor(phi_init_flat, dtype=torch.float32, device=_dev)
    print(f"  [transmural] EDT init range: "
          f"[{phi_init_flat.min():.4f}, {phi_init_flat.max():.4f}]")

    # ── CG solve ──────────────────────────────────────────────────────────────
    print(f"  [transmural] CG solve  (tol={tol}, maxiter={maxiter}) ...",
          flush=True)
    phi_flat = _conjugate_gradient(A, b, x0=x0, tol=tol, maxiter=maxiter)
    phi_flat = phi_flat.detach().cpu().numpy()

    # Re-enforce Dirichlet boundary values on the solution vector.
    # Symmetrising A can slightly perturb the identity rows, allowing CG
    # to drift outside [0,1] at boundary nodes.
    surf_vals = surface_label[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]
    phi_flat[surf_vals == -1] = 0.0
    phi_flat[surf_vals ==  1] = 1.0

    # ── Reconstruct volume ────────────────────────────────────────────────────
    phi = np.full(shape, np.nan, dtype=np.float32)
    phi[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = phi_flat
    np.clip(phi, 0.0, 1.0, out=phi)

    phi_inside = phi[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]
    print(f"  [transmural] Done.  phi range (inside mask): "
          f"[{phi_inside.min():.4f}, {phi_inside.max():.4f}]")

    # ── Cache save ────────────────────────────────────────────────────────────
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, phi)
        print(f"  [transmural] Saved phi -> {save_path}")

    return phi
