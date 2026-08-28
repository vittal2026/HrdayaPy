"""
ecg/functions/compute_bspm.py
==============================
Steps 11 & 12 — Full GPU pipeline for BSPM computation.

Pipeline (all on GPU per frame)
---------------------------------
  Vm  →  K @ Vm  →  scatter_add  →  b  →  PCG solve  →  phi[surface]  →  BSPM

The stiffness matrix A and preconditioner are uploaded to GPU once before
the loop.  Per-frame host↔device transfer is only Vm_frame in (~5 MB) and
BSPM surface values out (~0.4 MB).

Solver: Preconditioned Conjugate Gradient (PCG)
  Preconditioner: Jacobi (diagonal of A) — simple, effective for this
  problem since A is diagonally dominant.  Converges in ~50–150 iterations
  for smooth torso conductivity.

Stopping criterion: absolute error in mV (not relative residual), on a
  choice of three statistics (`cg_metric`) of z = M_inv_diag * r, the
  Jacobi-preconditioned residual, whose values approximate the remaining
  per-node error in phi itself (A^{-1} r) precisely because A is
  diagonally dominant:
    'rms'  (default) - sqrt(mean(z**2)) < cg_tol_mV -- bulk/typical error
    'p99'             - 99th-percentile(|z|) < cg_tol_mV -- robust
                         worst-case, ignores the single worst ~1% of nodes
    'linf'            - max(|z|) < cg_tol_mV -- true worst-case, every
                         node guaranteed under the target
  All three are fixed, physically interpretable targets ("stop once
  [this statistic of] every node is within cg_tol_mV of converged") that
  don't drift with each frame's source magnitude the way a relative
  ||r||/||b|| test does. See _pcg_gpu docstring for the per-metric
  iteration-cost tradeoff.

OMP fix
-------
Sets KMP_DUPLICATE_LIB_OK=True before importing torch to suppress the
harmless Windows OpenMP conflict between PyTorch and MKL.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Suppress Windows OpenMP duplicate-lib warning from PyTorch + MKL
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _get_device():
    import torch
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [gpu] {name}  ({vram:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print("  [gpu] CUDA not available — using CPU")
    return dev


def _scipy_csr_to_torch(M: sp.csr_matrix, device, dtype):
    """Scipy CSR → torch sparse CSR on device (beta but stable in practice)."""
    import torch, warnings
    M    = M.astype(np.float32)
    crow = torch.from_numpy(M.indptr.astype(np.int32)).to(device)
    col  = torch.from_numpy(M.indices.astype(np.int32)).to(device)
    val  = torch.from_numpy(M.data).to(dtype).to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_csr_tensor(crow, col, val,
                                       size=M.shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Heart graph Laplacian
# ---------------------------------------------------------------------------

def build_heart_laplacian(vox_idx: np.ndarray,
                           n_nodes: int,
                           h_fine_mm: float) -> sp.csr_matrix:
    """Vectorised sparse graph Laplacian of the myocardium fine grid."""
    ijk     = vox_idx.astype(np.int32)
    ijk_off = ijk - ijk.min(axis=0)
    shape   = tuple((ijk_off.max(axis=0) + 2).tolist())

    lookup  = np.full(shape, -1, dtype=np.int32)
    lookup[ijk_off[:, 0], ijk_off[:, 1], ijk_off[:, 2]] = np.arange(n_nodes, dtype=np.int32)

    ni, nj, nk   = shape
    offsets       = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], np.int32)
    all_rows, all_cols = [], []

    for di, dj, dk in offsets:
        ni_ = ijk_off[:, 0] + di
        nj_ = ijk_off[:, 1] + dj
        nk_ = ijk_off[:, 2] + dk
        in_b   = (ni_>=0)&(ni_<ni)&(nj_>=0)&(nj_<nj)&(nk_>=0)&(nk_<nk)
        node_k = np.where(in_b)[0]
        node_n = lookup[ni_[in_b], nj_[in_b], nk_[in_b]]
        valid  = node_n >= 0
        all_rows.append(node_k[valid])
        all_cols.append(node_n[valid])

    rows  = np.concatenate(all_rows)
    cols  = np.concatenate(all_cols)
    K_off = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                           shape=(n_nodes, n_nodes))
    deg   = np.array(K_off.sum(axis=1)).ravel().astype(np.float32)
    return K_off - sp.diags(deg, format="csr")


# ---------------------------------------------------------------------------
# Projection map
# ---------------------------------------------------------------------------

def build_projection_map(coords_mm, R, t, scale,
                          coarse_origin, dx_coarse_mm,
                          active_id_map, coarse_shape):
    torso  = (R @ (coords_mm * scale).T).T + t
    ijk_f  = (torso - coarse_origin) / dx_coarse_mm
    ijk    = np.floor(ijk_f).astype(np.int32)
    ni, nj, nk = coarse_shape

    in_b   = ((ijk[:,0]>=0)&(ijk[:,0]<ni)&
               (ijk[:,1]>=0)&(ijk[:,1]<nj)&
               (ijk[:,2]>=0)&(ijk[:,2]<nk))

    flat            = np.full(len(coords_mm), -1, dtype=np.int64)
    flat[in_b]      = (ijk[in_b,0].astype(np.int64)*nj*nk +
                       ijk[in_b,1].astype(np.int64)*nk +
                       ijk[in_b,2].astype(np.int64))

    proj            = np.full(len(coords_mm), -1, dtype=np.int32)
    ib_idx          = np.where(in_b)[0]
    cand            = active_id_map[flat[in_b]]
    hit             = cand >= 0
    proj[ib_idx[hit]] = cand[hit]

    valid  = proj >= 0
    n_miss = (~valid).sum()
    n_tot  = len(coords_mm)
    if n_miss > 0.05 * n_tot:
        print(f"  [warn] {n_miss:,}/{n_tot:,} nodes missed — check registration.")
    else:
        print(f"  Projection: {valid.sum():,}/{n_tot:,} mapped "
              f"({n_miss} missed, {n_miss/n_tot*100:.2f}%)")
    return proj, valid


# ---------------------------------------------------------------------------
# GPU Preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------

def _pcg_gpu(A_gpu, b: "torch.Tensor", M_inv_diag: "torch.Tensor",
             x0: "torch.Tensor | None" = None,
             tol_abs_mV: float = 0.01, max_iter: int = 500,
             metric: str = "rms"):
    """
    Solve  A x = b  on GPU using Jacobi-preconditioned CG.

    Parameters
    ----------
    A_gpu      : torch sparse CSR  (N×N, float32)
    b          : (N,) float32 tensor on GPU
    M_inv_diag : (N,) float32  1/diag(A)
    x0         : (N,) warm-start initial guess (previous frame solution)
    tol_abs_mV : absolute stopping tolerance, in mV of body-surface
                 potential, applied to whichever statistic `metric`
                 selects of z = M_inv_diag * r, the Jacobi-preconditioned
                 residual. Because A is diagonally dominant here (see
                 module docstring), z ≈ A^{-1} r is a per-node estimate
                 of the remaining solution error in x's own units (Volts;
                 x is the torso potential phi in V, hence the 1e-3
                 conversion below) -- a direct, resolution-independent
                 stand-in for "how far phi still is from converged" that,
                 unlike a relative residual, doesn't get looser or
                 tighter just because the frame's source magnitude ||b||
                 happens to be large or small.
    max_iter   : maximum iterations
    metric     : which statistic of |z| to test against tol_abs_mV:
                   'rms'  - sqrt(mean(z**2))   -- typical/bulk error.
                            Cheapest; can let a small fraction of nodes
                            stay above tol_abs_mV as long as the bulk is
                            well under it.
                   'p99'  - 99th percentile of |z| -- robust worst-case;
                            ignores the single worst ~1% of nodes, so
                            immune to one or two pathological outliers
                            (e.g. right at a pinned/boundary node) but
                            still tight on everything else.
                   'linf' - max(|z|) -- true worst-case, every node
                            (including boundary/interface-adjacent
                            outliers) is guaranteed under tol_abs_mV.
                            Strictest; typically needs the most
                            iterations of the three for the same
                            tol_abs_mV.
                 All three are computed from the same z already needed
                 for the CG recurrence, so switching metrics costs
                 nothing extra except 'p99', which needs one
                 torch.quantile call per iteration (still cheap for the
                 handful of iterations this solver typically takes).

    Returns x, n_iter, err_mV (the converged/final value of the chosen
    metric, in mV, regardless of which metric was used to stop -- so the
    three metrics stay directly comparable across choices)
    """
    import torch

    if metric not in ("rms", "p99", "linf"):
        raise ValueError(f"metric must be one of 'rms', 'p99', 'linf', got {metric!r}")

    N          = b.shape[0]
    tol_abs_v  = tol_abs_mV * 1e-3            # mV -> V (x is in Volts)

    def _stat(z_):
        if metric == "rms":
            return z_.norm() / (N ** 0.5)             # sqrt(mean(z**2))
        elif metric == "linf":
            return z_.abs().max()
        else:  # p99
            try:
                return torch.quantile(z_.abs(), 0.99)
            except RuntimeError:
                # torch.quantile on CUDA has a ~16.8M-element ceiling
                # (2**24); this grid is under it today (largest level is
                # ~12.5M nodes) but fall back to a CPU quantile rather
                # than crash if a future finer grid crosses it.
                return torch.quantile(z_.abs().cpu(), 0.99).to(z_.device)

    x      = x0.clone() if x0 is not None else torch.zeros_like(b)
    r      = b - torch.mv(A_gpu, x) if x0 is not None else b.clone()
    z      = M_inv_diag * r
    p      = z.clone()
    rz     = torch.dot(r, z)

    for i in range(max_iter):
        Ap      = torch.mv(A_gpu, p)
        pAp     = torch.dot(p, Ap)
        if pAp.abs() < 1e-30:
            break
        alpha   = rz / pAp
        x       = x + alpha * p
        r       = r - alpha * Ap
        z       = M_inv_diag * r
        err_v   = _stat(z)                    # V, per chosen metric
        if err_v < tol_abs_v:
            return x, i + 1, float(err_v * 1e3)   # report back in mV
        rz_new  = torch.dot(r, z)
        beta    = rz_new / rz
        p       = z + beta * p
        rz      = rz_new

    z = M_inv_diag * r
    return x, max_iter, float(_stat(z) * 1e3)


# ---------------------------------------------------------------------------
# Full GPU solver closure (built once, called per frame)
# ---------------------------------------------------------------------------

def make_gpu_solver(A_scipy:    sp.csr_matrix,
                    pin_row:    int,
                    proj_rows:  np.ndarray,
                    valid_mask: np.ndarray,
                    K_gpu,
                    sigma_i:    float,
                    h_fine_m:   float,
                    dx_coarse_mm: float,
                    surface_rows: np.ndarray,
                    n_rows:     int,
                    device,
                    cg_tol_mV:  float = 0.01,
                    cg_max:     int   = 500,
                    cg_metric:  str   = "rms",
                    psi_row:    "np.ndarray | None" = None,
                    tissue_rows: "np.ndarray | None" = None):
    """
    Upload A to GPU, build Jacobi preconditioner, return a closure that
    takes Vm_frame (CPU numpy) and returns phi_surface (CPU numpy) [V].

    psi_row : (n_rows,) SBM domain function ψ, in the SAME row order as
              A_scipy/grid["active_ids"] (i.e. psi_vol.ravel()[active_ids]).
              None (default, or when use_sbm=False on the grid) -- the RHS
              is left unweighted, matching the original sharp-boundary
              behaviour exactly. When given, the RHS is weighted by ψ to
              match the SBM operator ∇·(ψσ∇φ)=ψb assembled into A_scipy
              (see build_torso_grid.compute_smoothed_domain /
              _assemble_stiffness). In practice this is a no-op almost
              everywhere the heart source projects to, since those rows
              are deep interior (ψ≈1) -- it only matters right at the
              torso boundary, for consistency with the LHS operator.
    tissue_rows : (N_tissue,) row indices corresponding to TRUE tissue
              voxels (grid["tissue_mask"]), as opposed to the full SBM
              solve domain, which under use_sbm=True also includes an
              exterior "band" of non-tissue voxels with ψ floored near
              the SBM cutoff. Those band rows have a near-vanishing
              diagonal (face conductance ~ ψ_face·σ_face with ψ_face≈
              SBM_PSI_CUTOFF), making them nearly singular/poorly
              constrained -- their solved values are NOT physically
              meaningful extracellular potential and can be numerically
              far off (seen in practice: full-field capture including
              the band showed potentials 5-11x the physically-plausible
              surface range, at every level, tracking the ~10% band
              fraction almost exactly). capture_full restricts to
              tissue_rows when given, so "the complete extracellular
              potential field" means the true anatomical domain, not the
              SBM padding. None (default, and always when use_sbm=False,
              where tissue_mask==active_mask already) captures every
              active row -- no behaviour change for the sharp-boundary
              path.
    """
    import torch, warnings

    print("  Uploading stiffness matrix A to GPU ...")
    t0 = time.perf_counter()

    # Pin and convert A
    A_lil              = A_scipy.tolil()
    A_lil[pin_row, :]  = 0.0
    A_lil[:, pin_row]  = 0.0
    A_lil[pin_row, pin_row] = 1.0
    A_pinned           = A_lil.tocsr().astype(np.float32)

    A_gpu   = _scipy_csr_to_torch(A_pinned, device, torch.float32)
    diag    = np.array(A_pinned.diagonal(), dtype=np.float32)
    diag    = np.where(np.abs(diag) > 1e-30, diag, 1.0)  # avoid /0
    M_inv   = torch.from_numpy(1.0 / diag).to(device)

    del A_pinned        # free CPU copy

    print(f"    Upload done in {time.perf_counter()-t0:.2f} s")

    # Static GPU tensors for the source computation
    valid_gpu = torch.from_numpy(valid_mask).to(device)
    proj_gpu  = torch.from_numpy(proj_rows[valid_mask].astype(np.int64)).to(device)
    surf_gpu  = torch.from_numpy(surface_rows.astype(np.int64)).to(device)
    coeff     = float(sigma_i / h_fine_m**2)   # σᵢ/h²  [A/m³ per V]

    tissue_gpu = None
    if tissue_rows is not None:
        tissue_gpu = torch.from_numpy(tissue_rows.astype(np.int64)).to(device)

    psi_gpu = None
    if psi_row is not None:
        psi_gpu = torch.from_numpy(psi_row.astype(np.float32)).to(device)

    # Fixed (not per-cell!) divisor — expected number of fine heart-grid
    # voxels per coarse torso-grid voxel. A constant divisor here (rather
    # than a per-cell np.add.at(...) count, and rather than the physical
    # volume h_fine_m**3) keeps b in the correct A/m³ density units AND
    # preserves Σb ≈ 0 (no net monopole leak), since every entry gets
    # divided by the same number.
    h_fine_mm  = h_fine_m * 1e3
    n_expected = float((dx_coarse_mm / h_fine_mm) ** 3)
    print(f"  [solver] coeff={coeff:.4e}  n_expected={n_expected:.2f} "
          f"fine voxels / coarse voxel")

    iter_counts = []
    prev_x      = [None]

    warned_leak = [False]

    def solve_frame(Vm_np: np.ndarray, capture_full: bool = False):
        """
        capture_full : if True, also return the full (not surface-
        restricted) active-node φ field [V], for whole-volume cross-level
        convergence comparison (see run_experiment7_dgx.py). False
        (default) for the normal per-frame BSPM loop, since the full
        field is 40-100x the surface subset in size and most frames
        don't need it.
        """
        Vm_gpu  = torch.from_numpy(Vm_np).to(device) * 1e-3   # mV → V

        # Iv = (σᵢ/h²) K Vm  [A/m³]   (K is the *positive* discrete Laplacian,
        # K@Vm ≈ h²∇²Vm, so this is +σᵢ∇²Vm — matches A = -∇·(σ∇φ) together
        # with the Geselowitz relation ∇·(σ∇φ) = -∇·(σᵢ∇Vm). No extra minus.)
        lap = torch.mv(K_gpu, Vm_gpu)
        Iv  = lap[valid_gpu] * coeff                             # (N_valid,) A/m³

        # Σ(K@Vm) == 0 exactly over the *full* heart node set (no net
        # monopole), but valid_mask drops some nodes (outside the coarse
        # grid / projecting onto air) before we ever see them here — that
        # silently breaks the cancellation and injects a spurious net
        # current, which the Neumann solve can only resolve by leaking it
        # out through the single pinned reference node. That leak shows up
        # as a smooth, nearly-uniform (monopole-like) field that dominates
        # every torso location roughly equally — the "same shape everywhere"
        # symptom. Fix: re-zero the mean of the *masked* source so Σsrc≈0
        # is restored regardless of which/how many nodes got dropped.
        leak = float(Iv.mean())
        src  = Iv - leak
        if not warned_leak[0] and abs(leak) > 1e-3 * (float(Iv.abs().mean()) + 1e-30):
            print(f"  [warn] masked source had nonzero mean (leak={leak:.3e} A/m³ "
                  f"vs |Iv| mean={float(Iv.abs().mean()):.3e}) — check valid_mask "
                  f"coverage / registration; auto-corrected by re-centring.")
            warned_leak[0] = True

        # Scatter-add Iv onto coarse RHS, then divide by the *fixed*
        # expected fine-voxels-per-coarse-voxel ratio (not per-cell count,
        # not voxel volume) — keeps units correct and preserves Σb≈0.
        b_gpu   = torch.zeros(n_rows, dtype=torch.float32, device=device)
        b_gpu.scatter_add_(0, proj_gpu, src)
        b_gpu.div_(n_expected)
        if psi_gpu is not None:
            b_gpu.mul_(psi_gpu)   # SBM: RHS matches ∇·(ψσ∇φ)=ψb on the LHS
        # With Σsrc≈0 restored above, the RHS is now compatible with the
        # pure-Neumann system, so eliminating pin_row only fixes the
        # additive gauge constant (reference potential = 0) — it is no
        # longer acting as a hidden current sink/source.
        b_gpu[pin_row] = 0.0

        # PCG solve
        phi_gpu, n_it, res_mV = _pcg_gpu(A_gpu, b_gpu, M_inv,
                                          x0=prev_x[0],
                                          tol_abs_mV=cg_tol_mV, max_iter=cg_max,
                                          metric=cg_metric)
        prev_x[0] = phi_gpu.detach()
        iter_counts.append(n_it)

        if n_it >= cg_max:
            print(f"\n  [warn] PCG hit max_iter={cg_max} on frame 2 "
                  f"({cg_metric} err={res_mV:.4f} mV, target={cg_tol_mV} mV). "
                  f"Increase ECG_CG_MAX_ITER or loosen ECG_CG_TOL_MV.")

        return phi_gpu[surf_gpu].cpu().numpy(), n_it, res_mV, \
               ((phi_gpu[tissue_gpu] if tissue_gpu is not None else phi_gpu).cpu().numpy()
                if capture_full else None)

    return solve_frame, iter_counts


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_bspm_loop(
    vm_snapshots_path: Path,
    grid:              dict,
    registration:      dict,
    sigma_i:           float,
    h_fine_mm:         float,
    out_path:          Path,
    cg_tol_mV:         float = 0.01,
    cg_max_iter:       int   = 500,
    cg_metric:         str   = "rms",
    t_window:          "tuple[float, float] | None" = None,
    verbose:           bool  = True,
    full_field_frames: "np.ndarray | None" = None,
    full_field_out_path: "Path | None" = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    t_window : optional (t_min_ms, t_max_ms). If given, only Vm frames whose
               time_vm falls within [t_min, t_max] (inclusive) are solved —
               everything else is skipped entirely (not loaded into the
               per-frame loop). Useful for fast iteration on a QRS/T window
               instead of the full beat. bspm.npz will only contain frames
               from this window.
    full_field_frames : optional (K,) array of absolute frame indices (into
               the post-t_window Vm_all/time_ms arrays) at which to ALSO
               capture the complete (not surface-restricted) active-node
               phi field, for whole-volume cross-level convergence
               comparison. K should be small (tens, not hundreds) — the
               full field is 40-100x the surface-only BSPM signal in size
               (see run_experiment7_dgx.py sizing comment), so capturing
               it for all frames is generally infeasible at fine
               resolutions. None (default): no full-field capture, no
               change from the surface-only behaviour.
    full_field_out_path : where to save the full-field capture (a
               separate .npz from out_path, since it can be much larger
               and most workflows don't need it at all). Required if
               full_field_frames is given.
    """

    # ── Load snapshots ────────────────────────────────────────────────────
    if verbose:
        print(f"  Loading {Path(vm_snapshots_path).name} ...")
    snap      = np.load(str(vm_snapshots_path))
    Vm_all    = snap["Vm"].astype(np.float32)
    coords_mm = snap["coords_mm"].astype(np.float64)
    vox_idx   = snap["vox_idx"].astype(np.int32)
    time_ms   = snap["time_vm"].astype(np.float64)

    # ── Optional time-window restriction (for fast iteration) ──────────────
    if t_window is not None:
        t_min, t_max = t_window
        keep     = (time_ms >= t_min) & (time_ms <= t_max)
        n_before = len(time_ms)
        Vm_all   = Vm_all[keep]
        time_ms  = time_ms[keep]
        if verbose:
            print(f"  [t_window] Restricting to [{t_min:.1f}, {t_max:.1f}] ms: "
                  f"{n_before} -> {len(time_ms)} frames")
        if len(time_ms) == 0:
            raise ValueError(
                f"t_window={t_window} selected 0 frames out of "
                f"[{snap['time_vm'].min():.1f}, {snap['time_vm'].max():.1f}] ms available."
            )

    n_frames, N_myo = Vm_all.shape
    h_fine_m  = h_fine_mm * 1e-3

    if verbose:
        print(f"    Nodes={N_myo:,}  frames={n_frames}  "
              f"h={h_fine_mm} mm  σᵢ={sigma_i:.4f} S/m")

    device = _get_device()

    # ── Build heart Laplacian → GPU ───────────────────────────────────────
    if verbose:
        print("  Building heart graph Laplacian ...")
    t0    = time.perf_counter()
    K_cpu = build_heart_laplacian(vox_idx, N_myo, h_fine_mm)
    if verbose:
        print(f"    {K_cpu.nnz:,} nnz  ({time.perf_counter()-t0:.2f} s)")
    K_gpu = _scipy_csr_to_torch(K_cpu, device, __import__("torch").float32)
    del K_cpu

    # ── Projection map ────────────────────────────────────────────────────
    if verbose:
        print("  Precomputing projection map ...")
    R, t_reg, scale = (registration["R"],
                       registration["t"],
                       float(registration["scale"]))
    proj_rows, valid_mask = build_projection_map(
        coords_mm, R, t_reg, scale,
        grid["origin_mm"], grid["dx_mm"],
        grid["active_id_map"], grid["shape"],
    )

    # ── SBM: per-row ψ (in the same row order as active_ids), if this
    # grid was built with use_sbm=True. None otherwise -- keeps the
    # RHS unweighted, matching the original sharp-boundary behaviour.
    psi_row = None
    if bool(grid.get("use_sbm", False)) and "psi_vol" in grid:
        psi_row = np.asarray(grid["psi_vol"]).ravel()[grid["active_ids"]]

    # ── Build full GPU solver ─────────────────────────────────────────────
    solve_frame, iter_counts = make_gpu_solver(
        A_scipy      = grid["A_scipy"],
        pin_row      = grid["pin_row"],
        proj_rows    = proj_rows,
        valid_mask   = valid_mask,
        K_gpu        = K_gpu,
        sigma_i      = sigma_i,
        h_fine_m     = h_fine_m,
        dx_coarse_mm = grid["dx_mm"],
        surface_rows = grid["surface_rows"],
        n_rows       = len(grid["active_ids"]),
        device       = device,
        cg_tol_mV    = cg_tol_mV,
        cg_max       = cg_max_iter,
        cg_metric    = cg_metric,
        psi_row      = psi_row,
    )

    N_surf = len(grid["surface_rows"])
    N_active = len(grid["active_ids"])
    if verbose:
        print(f"  Surface nodes : {N_surf:,}")
        print(f"  Running full-GPU loop ({n_frames} frames) ...")

    full_field_set = set(int(f) for f in full_field_frames) if full_field_frames is not None else set()
    if full_field_set and full_field_out_path is None:
        raise ValueError("full_field_frames given but full_field_out_path is None")
    if full_field_set:
        n_missing = len(full_field_set - set(range(n_frames)))
        if n_missing:
            raise ValueError(
                f"full_field_frames contains {n_missing} index/indices outside "
                f"[0, {n_frames-1}] (post-t_window range) — frame indices must "
                f"be computed against the SAME t_window as this call."
            )
        full_field_phi = np.zeros((N_active, len(full_field_set)), dtype=np.float32)
        full_field_frame_order = sorted(full_field_set)
        full_field_col = {f: c for c, f in enumerate(full_field_frame_order)}
        if verbose:
            est_gb = N_active * len(full_field_set) * 4 / 1e9
            print(f"  [full-field] capturing {len(full_field_set)} frames "
                  f"× {N_active:,} active nodes (~{est_gb:.2f} GB) for "
                  f"whole-volume convergence comparison")

    # Warm-up
    import torch
    _ = solve_frame(Vm_all[0])
    torch.cuda.synchronize()

    # ── Per-frame loop ────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        frame_iter = tqdm(range(n_frames), unit="frame",
                          bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} "
                                     "[{elapsed}<{remaining}, {rate_fmt}]")
    except ImportError:
        frame_iter = range(n_frames)

    bspm_signal = np.zeros((N_surf, n_frames), dtype=np.float32)
    t_loop      = time.perf_counter()

    for f in frame_iter:
        capture_full = f in full_field_set
        phi_surf, n_it, res_mV, phi_full = solve_frame(Vm_all[f], capture_full=capture_full)
        bspm_signal[:, f]      = phi_surf * 1e3   # V → mV
        if capture_full:
            full_field_phi[:, full_field_col[f]] = phi_full * 1e3   # V → mV, same units as bspm_signal

        if hasattr(frame_iter, "set_postfix_str"):
            frame_iter.set_postfix_str(
                f"t={time_ms[f]:.0f}ms  "
                f"CG={n_it}it  {cg_metric}={res_mV:.4f}mV  "
                f"[{bspm_signal[:,f].min():.2f}, {bspm_signal[:,f].max():.2f}]mV"
            )

    total = time.perf_counter() - t_loop
    if verbose:
        avg_it = np.mean(iter_counts)
        print(f"  Done in {total:.1f} s  ({total/n_frames:.3f} s/frame)  "
              f"avg CG iters={avg_it:.1f}")

    # ── Save full-field capture (separate file — can be much larger) ───────
    if full_field_set:
        if verbose:
            print(f"  Saving → {full_field_out_path} ...")
        np.savez(
            str(full_field_out_path),
            phi_full      = full_field_phi,                      # (N_active, K), mV
            frame_indices = np.array(full_field_frame_order, np.int64),
            time_ms       = time_ms[full_field_frame_order].astype(np.float64),
            active_ids    = grid["active_ids"].astype(np.int32),  # for cross-checking node identity
            shape         = np.array(grid["shape"], np.int32),
            dx_mm         = np.float64(grid["dx_mm"]),
            origin_mm     = np.asarray(grid["origin_mm"], np.float64),
        )
        if verbose:
            print(f"  phi_full : {full_field_phi.shape}  "
                  f"[{full_field_phi.min():.3f}, {full_field_phi.max():.3f}] mV")

    # ── Save ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Saving → {out_path} ...")
    np.savez(
        str(out_path),
        surface_xyz  = grid["surface_xyz"].astype(np.float32),
        surface_flat = grid["surface_flat"].astype(np.int32),
        dx_mm        = np.float64(grid["dx_mm"]),
        bspm_signal  = bspm_signal,
        time_ms      = time_ms.astype(np.float64),
        reg_R        = R.astype(np.float64),
        reg_t        = t_reg.astype(np.float64),
        reg_scale    = np.float64(scale),
        reg_dice     = np.float64(float(registration["dice"])),
        sigma_i      = np.float64(sigma_i),
        h_fine_mm    = np.float64(h_fine_mm),
    )
    if verbose:
        print(f"  bspm_signal : {bspm_signal.shape}  "
              f"[{bspm_signal.min():.3f}, {bspm_signal.max():.3f}] mV  "
              f"({bspm_signal.nbytes/1e6:.1f} MB)")

    return bspm_signal
