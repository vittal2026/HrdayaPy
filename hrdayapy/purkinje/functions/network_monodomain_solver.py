"""
network_monodomain_solver.py
============================
Monodomain solver for 1-D cable networks (e.g. Purkinje trees).

Numerical scheme
----------------
Strang (second-order) operator splitting per time step:

    1.  ½ dt  implicit diffusion   — backward Euler, solved via CG
    2.  1  dt  ionic ODE + stimulus — Rush-Larsen gates (in ionic model),
                                      Forward Euler for concentrations
    3.  ½ dt  implicit diffusion   — same CG solve, warm-started

The diffusion sub-problem at each half-step is:

    (I + ½·dt·D·L) · V* = V

where L is the graph Laplacian (dx²-normalised).  The matrix is never
formed explicitly; only sparse matrix-vector products are used, keeping
the solver GPU-friendly for large networks.

With the Rush-Larsen gate integrator the ionic ODE is stable up to
dt ≈ 0.1 ms for TTP06, and the implicit diffusion half-steps are
unconditionally stable, so the default dt is 0.1 ms — a 5× speedup
over the previous explicit-Euler scheme (dt = 0.02 ms).

Public API
----------
solve_monodomain(nodes, elements, activation_times, ionic_model,
                 save_path, **kwargs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import trange


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _discretise_tree(
    nodes_mm: np.ndarray,
    elements: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Subdivide each branch into segments of approximately *dx* mm.

    Original endpoint nodes are reused so that junctions stay connected;
    interior points are appended as new nodes.

    Returns
    -------
    comp_nodes  : (M, 3) float32 – all computational node positions (mm)
    comp_edges  : (E, 2) int32   – pairs of adjacent node indices
    branch_map  : list of lists  – node indices along each branch (ordered)
    """
    n_original = len(nodes_mm)
    extra_nodes: list[list[float]] = []
    comp_edges:  list[list[int]]   = []
    branch_map:  list[list[int]]   = []
    extra_counter = n_original

    for e in elements:
        p0, p1 = nodes_mm[e[0]], nodes_mm[e[1]]
        length  = np.linalg.norm(p1 - p0)
        n_seg   = max(1, int(np.round(length / dx)))

        bmap = [int(e[0])]
        for k in range(1, n_seg):
            t  = k / n_seg
            pt = ((1 - t) * p0 + t * p1).tolist()
            extra_nodes.append(pt)
            bmap.append(extra_counter)
            extra_counter += 1
        bmap.append(int(e[1]))

        branch_map.append(bmap)
        for i in range(len(bmap) - 1):
            comp_edges.append([bmap[i], bmap[i + 1]])

    comp_nodes     = np.array(nodes_mm.tolist() + extra_nodes, dtype=np.float32)
    comp_edges_arr = np.array(comp_edges, dtype=np.int32)
    return comp_nodes, comp_edges_arr, branch_map


def _build_laplacian_sparse(
    N:       int,
    edges:   np.ndarray,
    w_edge:  float,        # conductance weight per edge = sigma_P * S_P / h_P
    A_i:     float,        # membrane loading per node  = A_P * h_P
    device:  torch.device,
) -> torch.Tensor:
    """
    Build the physically correct graph Laplacian  L = D^{-1} K  for the
    Purkinje cable network as a sparse COO tensor.

    K is the symmetric conductance matrix:
        K[i,j] = +w_edge   off-diagonal (connected nodes)
        K[i,i] = -degree_i * w_edge

    D is the diagonal membrane-loading matrix:
        D[i,i] = A_i = A_P * h_P   (uniform for all cable nodes)

    So L = D^{-1}K.

    Parameters
    ----------
    w_edge  : sigma_P * S_P / h_P   [mS]   Purkinje–Purkinje edge conductance
    A_i     : A_P * h_P             [mm^2] membrane loading factor
    """
    import math as _math
    rows_e = edges[:, 0].astype(np.int64)
    cols_e = edges[:, 1].astype(np.int64)

    degree = np.zeros(N, dtype=np.float64)
    np.add.at(degree, rows_e, w_edge)
    np.add.at(degree, cols_e, w_edge)

    r_all = np.concatenate([rows_e, cols_e, np.arange(N, dtype=np.int64)])
    c_all = np.concatenate([cols_e, rows_e, np.arange(N, dtype=np.int64)])
    v_all = np.concatenate([
        np.full(len(rows_e), +w_edge, dtype=np.float64),   # off-diag: +w
        np.full(len(rows_e), +w_edge, dtype=np.float64),
        -degree,                                             # diagonal: −degree·w
    ])
    # Apply D^{-1}: divide by A_i (uniform, so scalar division)
    v_all /= A_i

    indices = torch.tensor(np.stack([r_all, c_all]), dtype=torch.long,    device=device)
    values  = torch.tensor(v_all,                    dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()


# ──────────────────────────────────────────────────────────────────────────────
# Conjugate Gradient solver  (pure PyTorch, runs entirely on GPU)
# ──────────────────────────────────────────────────────────────────────────────

def _cg_solve(
    A_mv: callable,
    b: torch.Tensor,
    x0: torch.Tensor,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Solve the symmetric positive-definite system  A·x = b  via
    Conjugate Gradient, warm-started from *x0*.

    Parameters
    ----------
    A_mv     : callable  v -> A @ v   (matrix-free, stays on GPU)
    b        : (N,) right-hand side
    x0       : (N,) initial guess — pass previous solution for warm start
    tol      : absolute residual tolerance  ‖r‖₂ < tol
    max_iter : safety cap on iterations

    Returns
    -------
    x : (N,) approximate solution
    """
    x      = x0.clone()
    r      = b - A_mv(x)
    p      = r.clone()
    rs_old = (r * r).sum()

    for _ in range(max_iter):
        Ap     = A_mv(p)
        alpha  = rs_old / ((p * Ap).sum() + 1e-30)
        x      = x + alpha * p
        r      = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < tol:
            break
        p      = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Public solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_monodomain(
    nodes: np.ndarray,
    elements: np.ndarray,
    activation_times: np.ndarray,
    ionic_model: str,
    save_path: str,
    *,
    # Spatial / temporal
    voxel_size: float = 0.4,    # mm per voxel (converts nodes from voxel → mm)
    dx: float  = 0.2,           # mm   – Purkinje spatial step
    dt: float  = 0.1,           # ms   – timestep
    T:  float  = 400.0,         # ms   – total simulation duration
    # ── Physical conductivity parameters (mm/ms/mS/uF unit system) ────────────
    sigma_P: float = 0.225,     # mS/mm  Purkinje axial conductivity (CV ~1.5 mm/ms, R_P=0.015 mm)
    Cm:      float = 0.01,      # uF/mm^2 specific membrane capacitance (= 1 uF/cm^2)
    A_P:     float = 40.0,      # mm^-1  Purkinje membrane S/V ratio (= 2/r, r=0.05 mm)
    R_P:     float = 0.015,     # mm     Purkinje fibre radius (~15 um physiological)
    # Stimulus
    stim_amp: float = 12.0,     # uA/mm^2  suprathreshold stimulus
    stim_dur: float = 2.0,      # ms       stimulus duration
    # CG solver
    cg_tol:      float = 1e-5, # absolute residual tolerance
    cg_max_iter: int   = 50,   # iteration cap per half-step
    # Output
    save_every: int    = 40,   # 40 × 0.1 ms = 4 ms temporal resolution
    device: str | None = None, # "cuda", "cpu", or None (auto)
) -> None:
    """
    Integrate the monodomain equation on a Purkinje / cable network using
    Strang operator splitting and save voltage traces to a compressed NPZ.

    Scheme per time step
    --------------------
    1. Half-step implicit diffusion   CG solve of (I + ½·dt·D·L)·V* = V^n
    2. Full-step ionic ODE + stimulus Rush-Larsen gates + Forward Euler
                                      concentrations (inside ionic model)
    3. Half-step implicit diffusion   CG solve of (I + ½·dt·D·L)·V^{n+1} = V**

    Parameters
    ----------
    nodes : (N, 3) array
        Node positions in *voxel* coordinates.
    elements : (E, 2) array
        Branch endpoint indices into *nodes*.
    activation_times : (N,) array
        Local activation time (ms) for each node; sets the stimulus onset.
    ionic_model : str
        Name of the ionic model class in ``ionic_models``.
        Example: ``"TTP06"``.
    save_path : str
        Output path (with or without ``.npz`` extension).
    voxel_size : float
        Physical size of one voxel (mm).
    dx : float
        Spatial discretisation step (mm).
    dt : float
        Time step (ms). Default 0.1 ms is stable for TTP06 with this scheme.
    T : float
        Total simulation duration (ms).
    D : float
        Monodomain diffusion coefficient (mm²/ms).
    stim_amp : float
        Stimulus amplitude (µA/cm²).
    stim_dur : float
        Stimulus duration (ms).
    cg_tol : float
        Absolute CG residual tolerance for each diffusion half-step.
    cg_max_iter : int
        Maximum CG iterations per half-step.
    save_every : int
        Store one voltage snapshot every *save_every* time steps.
    device : str or None
        PyTorch device. Defaults to CUDA when available.

    Output format (NPZ)
    -------------------
    time        : (n_saved,)      – saved time points (ms)
    comp_nodes  : (M, 3)          – computational node positions (mm)
    comp_edges  : (E2, 2)         – computational edge list
    bmap_<k>    : (L_k,)          – node indices for branch k
    branch_<k>  : (n_saved, L_k)  – voltage along branch k (mV)
    """

    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    print(f"Device  : {_device}")

    # ── Ionic model ───────────────────────────────────────────────────────────
    # The network solver uses EndoTTP06Torch directly (same as coupled solver).
    # The legacy getattr(_ionic_models, ionic_model) path is removed because
    # ionic_models no longer exports a top-level TTP06 class.
    from .purkinje_myocardium_pipeline import EndoTTP06Torch

    # ── Discretise tree ───────────────────────────────────────────────────────
    nodes_mm = nodes * voxel_size
    comp_nodes, comp_edges, branch_map = _discretise_tree(nodes_mm, elements, dx)

    N  = len(comp_nodes)
    Nt = int(T / dt)
    print(f"Network : {N} computational nodes, {len(comp_edges)} edges")
    print(f"Steps   : {Nt}  |  dt = {dt} ms  |  Saved frames: {Nt // save_every}")

    # ── Derived geometric quantities ─────────────────────────────────────────
    import math as _math
    S_P   = _math.pi * R_P ** 2      # mm^2   Purkinje fibre cross-sectional area
    w_pp  = sigma_P * S_P / dx        # mS     edge conductance weight (eq. 11)
    # [FIX M1] cell volume = S_P * dx, not dx alone — omitting S_P inflated
    # D_P by 1/S_P ~ 127x.  SIGMA_P corrected by same factor in config.
    Ai_P  = A_P * S_P * dx            # mm^2   membrane loading = A_P * cell_vol

    # ── Graph Laplacian  L = D^{-1}K ─────────────────────────────────────────
    L = _build_laplacian_sparse(N, comp_edges, w_pp, Ai_P, _device)

    # ── CN diffusion operator  A·v = (Cm/dt)*v - 0.5*L*v  ───────────────────
    # The half-step system (eq. 25, theta=0.5):
    #   b  = (Cm/dt)*V + 0.5*L*V
    #   Ax = (Cm/dt)*x - 0.5*L*x = b
    cn_scale = Cm / dt

    def A_mv(v: torch.Tensor) -> torch.Tensor:
        # A = (Cm/dt)*I - 0.5*L  ;  L has negative diagonal, so -0.5*L
        # adds diffusion (positive off-diagonal contribution).
        Lv = torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1)
        return cn_scale * v - 0.5 * Lv

    def rhs_of(v: torch.Tensor) -> torch.Tensor:
        # b = (Cm/dt)*I*v + 0.5*L*v  ;  +0.5*L adds the explicit diffusion.
        Lv = torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1)
        return cn_scale * v + 0.5 * Lv

    # ── Ionic model ───────────────────────────────────────────────────────────
    ionic  = EndoTTP06Torch(_device)
    states = ionic.init_states(N)
    # Cm is the PDE membrane capacitance [uF/mm^2], passed as a parameter.

    # ── Stimulus schedule ─────────────────────────────────────────────────────
    n_orig = len(nodes)
    act_np = np.array(activation_times, dtype=np.float32, copy=True)
    finite = np.isfinite(act_np)
    if not np.any(finite):
        raise ValueError("activation_times are all NaN/inf; cannot choose stimulus root.")

    # Ignore NaN/inf when choosing the stimulation root.
    act_np[~finite] = T  # make sure NaNs are never selected as the minimum
    act_np = np.clip(act_np, 0.0, T - stim_dur)
    orig_act_times = torch.tensor(act_np, dtype=torch.float32, device=_device)

    # ── Storage ───────────────────────────────────────────────────────────────
    n_saved = Nt // save_every
    V_all   = np.empty((n_saved, N), dtype=np.float32)
    time    = np.arange(Nt, dtype=np.float32) * dt

    # ── Stimulus schedule ─────────────────────────────────────────────────────
    min_val = orig_act_times.min()
    root_mask = torch.isclose(orig_act_times, min_val, atol=1e-6)  # AV node(s) only
    n_roots = int(root_mask.sum().item())
    if n_roots == 0:
        raise RuntimeError("Computed root_mask is empty; check activation_times.")
    root_idx = torch.where(root_mask)[0].detach().cpu().numpy().astype(int)
    preview = root_idx[:10].tolist()
    print(
        f"Root stimulus: {n_roots} node(s) with min act_time={float(min_val.item()):.3f} ms; "
        f"indices={preview}{'...' if len(root_idx) > 10 else ''}"
    )


    # ── Time integration  (Strang splitting) ──────────────────────────────────
    for n in trange(Nt, desc="Integrating"):
        t      = float(time[n])
        V_flat = states["V"].squeeze()   # (N,)

        # ── Step 1 : ½ dt implicit diffusion  (CN half-step) ────────────────
        # Solve: (Cm/dt * I - 0.5*L) V* = (Cm/dt * I + 0.5*L) V^n
        b_s1   = rhs_of(V_flat)
        V_star = _cg_solve(A_mv, b_s1, V_flat,
                           tol=cg_tol, max_iter=cg_max_iter)

        # ── Step 2 : full dt ionic ODE + stimulus ─────────────────────────────
        # ── Step 2 : full dt ionic ODE + stimulus ─────────────────────────────
        # EndoTTP06Torch.step() operates on 1-D (N,) tensors; no unsqueeze needed.
        states["V"] = V_star

        I_stim = torch.zeros(N, device=_device)
        active_now = (t >= 0) & (t <= stim_dur)
        if active_now:
            I_stim[:n_orig][root_mask] = stim_amp

        Iion_flat, states = ionic.step(states, dt, I_stim)  # (N,), dict

        V_dstar = V_star + dt * (-Iion_flat / Cm + I_stim / Cm)
        states["V"] = V_dstar

        # ── Step 3 : ½ dt implicit diffusion  (CN half-step) ─────────────────
        b_s3  = rhs_of(V_dstar)
        V_new = _cg_solve(A_mv, b_s3, V_dstar,
                          tol=cg_tol, max_iter=cg_max_iter)

        states["V"] = V_new

        if n % save_every == 0:
            V_all[n // save_every] = V_new.detach().cpu().numpy()

    time_saved = time[::save_every]

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict: dict[str, np.ndarray] = {
        "time":       time_saved,
        "comp_nodes": comp_nodes,
        "comp_edges": comp_edges,
    }
    for b, bmap in enumerate(branch_map):
        bmap_arr = np.array(bmap, dtype=np.int32)
        save_dict[f"bmap_{b}"]   = bmap_arr
        save_dict[f"branch_{b}"] = V_all[:, bmap_arr]

    save_path = str(Path(save_path).with_suffix(".npz"))
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI convenience
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monodomain solver for cable networks")
    parser.add_argument("pkn_path",      help="Path to Purkinje .npz file")
    parser.add_argument("save_path",     help="Output .npz path")
    parser.add_argument("--model",       default="TTP06")
    parser.add_argument("--dt",          type=float, default=0.1)
    parser.add_argument("--T",           type=float, default=400.0)
    parser.add_argument("--D",           type=float, default=0.2)
    parser.add_argument("--dx",          type=float, default=0.2)
    parser.add_argument("--stim_amp",    type=float, default=50.0)
    parser.add_argument("--stim_dur",    type=float, default=2.0)
    parser.add_argument("--cg_tol",      type=float, default=1e-5)
    parser.add_argument("--cg_max_iter", type=int,   default=50)
    parser.add_argument("--save_every",  type=int,   default=40)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    from functions import load_purkinje as _load
    nodes, elements, act_times = _load(args.pkn_path)

    solve_monodomain(
        nodes            = nodes,
        elements         = elements,
        activation_times = act_times,
        ionic_model      = args.model,
        save_path        = args.save_path,
        dt               = args.dt,
        T                = args.T,
        D                = args.D,
        dx               = args.dx,
        stim_amp         = args.stim_amp,
        stim_dur         = args.stim_dur,
        cg_tol           = args.cg_tol,
        cg_max_iter      = args.cg_max_iter,
        save_every       = args.save_every,
        device           = args.device,
    )
