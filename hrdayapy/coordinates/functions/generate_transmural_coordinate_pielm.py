"""
generate_transmural_coordinate.py
==================================
Solves the Laplace equation on the myocardium mask to produce a
transmural coordinate  phi in [0, 1]  (0 = endocardium, 1 = epicardium).

Method: PIELM (Physics-Informed Extreme Learning Machine)
-----------------------------------------------------------
Instead of assembling a voxel-graph finite-difference Laplacian and running
Krylov CG on an N-dof system (N = number of myocardial voxels, can be in the
millions), phi is represented as a single-hidden-layer network with FIXED,
randomly-drawn hidden weights ("ELM" features) and a trainable linear output
layer:

    phi(x) = sum_j  beta_j * tanh(w_j . x_hat + b_j),   x_hat in [-1, 1]^3

Because tanh is smooth, both the field and its Laplacian are linear in the
unknowns beta (w_j, b_j are fixed/random, not trained). That turns the PDE
solve into a *linear least-squares* problem in beta:

    * PDE residual rows   : Laplacian(phi)(x_i) = 0   for interior collocation points
    * Dirichlet BC rows    : phi(x_i) = 1              for epicardial points
                              phi(x_i) = 0              for endocardial points

solved via the normal equations (A^T W A) beta = A^T W c, where W up-weights
the boundary rows. The unknown vector beta has only H entries (H = number of
hidden units, a few thousand) instead of N, so the linear system that CG has
to solve is tiny and converges essentially immediately regardless of the
number of voxels. Once beta is found, phi is evaluated at every voxel with a
single forward pass (batched to bound memory).

This trades an exact discrete Laplacian for a smooth, mesh-free, globally
supported approximation — appropriate when a smooth transmural coordinate
(rather than an exact per-voxel harmonic function) is what's needed. Random
features and collocation subsampling use a fixed seed, so results are
reproducible.

Public API
----------
    phi = generate_transmural_coordinate(S, surface_label, ...)

The public function signature, argument names/defaults, return type/shape,
caching behaviour (save_path) and CUDA-fallback behaviour are all identical
to the previous finite-difference/CG implementation, so this is a drop-in
replacement.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from tqdm import trange
from typing import Callable


# =============================================================================
# Internal hyperparameters (not part of the public API — see module docstring)
# =============================================================================

_N_HIDDEN          = 2048     # number of random tanh features (unknowns = H)
_N_COLLOCATION_MAX = 200_000  # cap on interior PDE-residual collocation points
_N_BOUNDARY_MAX    = 100_000  # cap on boundary (Dirichlet) points, per surface
_BC_WEIGHT         = 50.0     # relative weight of Dirichlet rows vs PDE rows
_EVAL_BATCH        = 200_000  # voxels per batch when evaluating the final field
_SEED              = 0        # RNG seed for random features + subsampling


# =============================================================================
# CG solver on the (small, dense) normal-equations system
# =============================================================================

def _conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    maxiter: int = 10_000,
) -> torch.Tensor:
    """Generic CG against a matvec callable (works for dense or sparse A)."""
    x      = torch.zeros_like(b) if x0 is None else x0.clone()
    r      = b - matvec(x)
    p      = r.clone()
    rsold  = torch.dot(r, r)
    r0norm = rsold.sqrt().item()

    converged = False
    rel_res   = 1.0
    it        = 0

    pbar = trange(maxiter, desc="Transmural PIELM CG", leave=True)
    for it in pbar:
        Ap      = matvec(p)
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
# PIELM feature construction
# =============================================================================

def _normalize_coords(vox_idx: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map voxel indices into [-1, 1]^3 based on the mask's bounding box."""
    lo = vox_idx.min(axis=0).astype(np.float32)
    hi = vox_idx.max(axis=0).astype(np.float32)
    center = (lo + hi) / 2.0
    half_extent = np.maximum((hi - lo) / 2.0, 1.0)  # avoid div-by-0 on flat axes
    coords = (vox_idx.astype(np.float32) - center) / half_extent
    return coords, center, half_extent


def _init_random_features(device: torch.device, rng: np.random.Generator):
    """Fixed (untrained) hidden layer: H random tanh neurons over 3D input."""
    W_np = rng.uniform(-1.0, 1.0, size=(_N_HIDDEN, 3)).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=(_N_HIDDEN,)).astype(np.float32)
    W = torch.tensor(W_np, device=device)
    b = torch.tensor(b_np, device=device)
    w_sq_norm = (W ** 2).sum(dim=1)  # |w_j|^2, used in the Laplacian term
    return W, b, w_sq_norm


def _features_and_laplacian(
    X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, w_sq_norm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    sigma  = tanh(X W^T + b)                      -> (n, H)
    lap    = |w_j|^2 * sigma''(z_j)                -> (n, H)
             sigma''(z) = -2*sigma*(1-sigma^2)
    """
    z     = X @ W.T + b                  # (n, H)
    sigma = torch.tanh(z)
    lap   = w_sq_norm.unsqueeze(0) * (-2.0 * sigma * (1.0 - sigma ** 2))
    return sigma, lap


def _subsample(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    sel = rng.choice(len(idx), size=cap, replace=False)
    return idx[sel]


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
    Generate transmural coordinate phi in [0,1] via a PIELM Laplace solve.

    Epicardium  (surface_label == +1) : Dirichlet phi = 1
    Endocardium (surface_label == -1) : Dirichlet phi = 0
    Elsewhere                         : Laplace PDE residual minimised
                                         (soft Neumann emerges naturally from
                                         the unconstrained interior).

    Parameters
    ----------
    S              : (Nx,Ny,Nz) uint8   binary myocardium mask
    surface_label  : (Nx,Ny,Nz) int8   +1 epi / -1 endo / 0 interior
    device         : torch device string or object (default "cuda")
    tol            : CG relative residual tolerance (for the H x H normal-
                     equations solve of the PIELM output weights)
    maxiter        : CG iteration cap (for the same small solve)
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

    rng = np.random.default_rng(_SEED)

    # ── Voxel indices & normalized coordinates ──────────────────────────────
    print("  [transmural] Building node index map ...", flush=True)
    vox_idx = np.argwhere(S == 1).astype(np.int32)
    N       = len(vox_idx)
    print(f"  [transmural] {N:,} myocardial voxels  |  device: {_dev}")

    coords_all, center, half_extent = _normalize_coords(vox_idx, shape)
    surf_vals = surface_label[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]
    is_epi    = surf_vals ==  1
    is_endo   = surf_vals == -1
    is_inter  = ~(is_epi | is_endo)

    epi_idx    = np.where(is_epi)[0]
    endo_idx   = np.where(is_endo)[0]
    inter_idx  = np.where(is_inter)[0]

    # ── Random ELM features (fixed, untrained hidden layer) ─────────────────
    print(f"  [transmural] Initialising {_N_HIDDEN} random PIELM features ...",
          flush=True)
    W, b, w_sq_norm = _init_random_features(_dev, rng)

    # ── Collocation / boundary subsampling ──────────────────────────────────
    coll_idx = _subsample(inter_idx, _N_COLLOCATION_MAX, rng)
    epi_bc   = _subsample(epi_idx,  _N_BOUNDARY_MAX, rng)
    endo_bc  = _subsample(endo_idx, _N_BOUNDARY_MAX, rng)
    print(f"  [transmural] PDE collocation points: {len(coll_idx):,}  |  "
          f"BC points: {len(epi_bc):,} epi + {len(endo_bc):,} endo", flush=True)

    X_coll = torch.tensor(coords_all[coll_idx], device=_dev)
    X_epi  = torch.tensor(coords_all[epi_bc],   device=_dev)
    X_endo = torch.tensor(coords_all[endo_bc],  device=_dev)

    # ── Assemble the (small, H x H) normal-equations system ────────────────
    print("  [transmural] Assembling PIELM design matrix ...", flush=True)
    _, lap_coll = _features_and_laplacian(X_coll, W, b, w_sq_norm)      # (Nc, H)
    sig_epi, _  = _features_and_laplacian(X_epi,  W, b, w_sq_norm)      # (Nb1, H)
    sig_endo, _ = _features_and_laplacian(X_endo, W, b, w_sq_norm)      # (Nb2, H)

    rhs_coll = torch.zeros(X_coll.shape[0], device=_dev)
    rhs_epi  = torch.ones(X_epi.shape[0],  device=_dev)
    rhs_endo = torch.zeros(X_endo.shape[0], device=_dev)

    bc_w = _BC_WEIGHT ** 0.5  # sqrt weight, since it enters A^T A as weight^2
    A = torch.cat([lap_coll, bc_w * sig_epi, bc_w * sig_endo], dim=0)
    c = torch.cat([rhs_coll, bc_w * rhs_epi, bc_w * rhs_endo], dim=0)

    # --- diagnostics: find where NaN first appears ---
    print("  [debug] coords_all has nan:", np.isnan(coords_all).any())
    print("  [debug] X_coll has nan/inf:", not torch.isfinite(X_coll).all().item())
    print("  [debug] lap_coll has nan/inf:", not torch.isfinite(lap_coll).all().item())
    print("  [debug] sig_epi has nan/inf:", not torch.isfinite(sig_epi).all().item())
    print("  [debug] sig_endo has nan/inf:", not torch.isfinite(sig_endo).all().item())
    print("  [debug] A has nan/inf:", not torch.isfinite(A).all().item())
    print("  [debug] c has nan/inf:", not torch.isfinite(c).all().item())

    AtA = A.T @ A
    Atc = A.T @ c
    print("  [debug] AtA has nan/inf:", not torch.isfinite(AtA).all().item())
    print("  [debug] Atc has nan/inf:", not torch.isfinite(Atc).all().item())
    print("  [debug] AtA diag min/max:", AtA.diagonal().min().item(), AtA.diagonal().max().item())
    # Ridge regularization: A^T A from random ELM features is frequently
    # near-singular; without this, CG divides by ~0 and the residual goes NaN.
    ridge = 1e-0 * torch.diagonal(AtA).mean()
    AtA = AtA + ridge * torch.eye(_N_HIDDEN, device=_dev)

    # ── CG solve for beta ────────────────────────────────────────────────────
    print(f"  [transmural] CG solve for {_N_HIDDEN} output weights  "
          f"(tol={tol}, maxiter={maxiter}) ...", flush=True)
    beta = _conjugate_gradient(lambda v: AtA @ v, Atc, x0=None,
                                tol=tol, maxiter=maxiter)

    # ── Evaluate phi at every myocardial voxel (batched forward pass) ──────
    print("  [transmural] Evaluating field at all voxels ...", flush=True)
    phi_flat = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, _EVAL_BATCH):
            end   = min(start + _EVAL_BATCH, N)
            X_b   = torch.tensor(coords_all[start:end], device=_dev)
            sig_b = torch.tanh(X_b @ W.T + b)
            phi_b = sig_b @ beta
            phi_flat[start:end] = phi_b.detach().cpu().numpy()

    # Re-enforce Dirichlet boundary values on the solution vector — the
    # least-squares fit only satisfies the BCs approximately (weighted, not
    # exact), so clamp the known surfaces back to their exact targets.
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
