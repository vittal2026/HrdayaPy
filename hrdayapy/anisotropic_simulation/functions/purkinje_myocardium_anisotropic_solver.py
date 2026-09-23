"""
purkinje_myocardium_anisotropic_solver.py
============================================
Coupled Purkinje-tree + anatomical myocardium monodomain solver, with
ANISOTROPIC myocardial conduction (conductivity along/across the local
fibre direction, instead of a single isotropic sigma_M).

This is a near-verbatim copy of
simulation/functions/purkinje_myocardium_pipeline.py -- the Purkinje
discretisation, ionic models, PMJ coupling, time-stepping (Strang
splitting / BiCGSTAB) and I/O are all unchanged. The only real
differences are localised to two functions:
  - build_myo_mesh_anisotropic (was build_myo_mesh): 18-connected mesh
    (6 face + 12 edge-diagonal neighbours) with signed per-edge weights
    derived from a fibre-direction-based conductivity tensor, instead of
    6-connected with a uniform scalar w_mm = sigma_M * voxel_size.
  - _build_global_laplacian_gpu: accepts a per-edge weight ARRAY for
    myocardium edges (not a scalar), and vectorises that edge block
    (np.add.at) since 18-connectivity roughly triples edge count.
Everything downstream of the assembled global Laplacian L = D^{-1}K is
untouched -- BiCGSTAB doesn't assume M-matrix structure, so it needed no
changes, though see build_myo_mesh_anisotropic's docstring for what the
now-signed off-diagonal entries mean for conditioning/monitoring.

Coupled Purkinje-tree + anatomical myocardium monodomain solver.

Numerical scheme
----------------
  Strang splitting per time step:
    1. half-dt  implicit diffusion  — CG + Jacobi preconditioner (GPU)
    2. full-dt  ionic ODE           — Rush-Larsen gates, Forward Euler concs (GPU)
    3. half-dt  implicit diffusion  — CG + Jacobi preconditioner (GPU)

  PMJ coupling: semi-implicit Option-B via sparse P2M / M2P (GPU, torch sparse)

Heterogeneity
-------------
  Homogeneous endocardial TTP06 throughout (Purkinje + myocardium).
  Switch to mcell / epi variants by changing the import below.

Stimulus
--------
  Multi-site stimulus protocol (see `stim_protocol` arg of
  purkinje_myocardium_anisotropic_solver() and
  `_resolve_stim_sites` / `_stim_active` below).  Any number of independent
  stimulus sites can be active in the same run, each with its own target
  location, amplitude, duration, onset time, and (optional) pacing period.
  Two built-in targets are provided out of the box:
    "root"    — the AV-node / His-Purkinje root zone (a sphere of radius
                `stim_len_mm` around the earliest-activated Purkinje node).
    "ectopic" — an arbitrary myocardial focus, supplied as a boolean voxel
                mask via the `ectopic_region` arg (e.g. cfg.PATH_STIM_REGION,
                computed in pipeline.py Step 4c from the UVC coordinates).

GPU strategy
------------
  Everything on GPU (torch):
    - Purkinje + myocardium Laplacians as torch sparse COO tensors
    - Jacobi preconditioner (diagonal of LHS) as a dense tensor
    - Ionic state arrays as torch float32 tensors
    - PMJ P2M / M2P matrices as torch sparse COO tensors
  CPU only: one-time mesh construction, NPZ I/O.

Output NPZ schema  (consumed by animate_coupled.py)
----------------------------------------------------
  time                 (n_frames,)       float32
  comp_nodes           (N_comp, 3)       float32
  comp_edges           (K', 2)           int32    clipped to outside Z
  bmap_0 .. bmap_N     int32 arrays
  branch_0..branch_N   (n_frames, len)   float32  Purkinje voltage
  surf_verts           (V, 3)            float32
  surf_faces           (F, 3)            int32
  vert_to_surf_node    (V,)              int32
  surf_frames          (n_frames, N_surf) float32

Vm snapshot NPZ schema  (vm_snapshots.npz, consumed by load_vm_snapshots.py)
-----------------------------------------------------------------------------
  node_ids             (N_myo,)              int32    global solver index Np+i
  coords_mm            (N_myo, 3)            float32  physical xyz [mm]
  vox_idx              (N_myo, 3)            int32    voxel ijk in mask
  Vm                   (n_vm_frames, N_myo)  float32  membrane voltage [mV]
  time_vm              (n_vm_frames,)         float32  simulation time [ms]
  dt_solver            scalar                float32  solver step [ms]
  vm_save_dt           scalar                float32  snapshot interval [ms]
"""

from __future__ import annotations

import os
import sys
import gc
import tempfile
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

np.seterr(over="ignore", invalid="ignore", divide="ignore")


# =============================================================================
# Re-use tree helpers from network_monodomain_solver
# =============================================================================

# ── GPU ionic model classes ───────────────────────────────────────────────────
from .ionic_models.gpu_ttp06     import EndoTTP06, MidTTP06, EpiTTP06
from .ionic_models.hetero_solver import HeteroIonicSolver

try:
    from network_monodomain_solver import _discretise_tree
except ImportError:
    def _discretise_tree(nodes_mm, elements, dx):
        n_original    = len(nodes_mm)
        extra_nodes   = []
        comp_edges    = []
        branch_map    = []
        extra_counter = n_original
        for e in elements:
            p0, p1 = nodes_mm[e[0]], nodes_mm[e[1]]
            length  = float(np.linalg.norm(p1 - p0))
            n_seg   = max(1, int(np.round(length / dx)))
            bmap    = [int(e[0])]
            for k in range(1, n_seg):
                t = k / n_seg
                extra_nodes.append(((1 - t) * p0 + t * p1).tolist())
                bmap.append(extra_counter)
                extra_counter += 1
            bmap.append(int(e[1]))
            branch_map.append(bmap)
            for i in range(len(bmap) - 1):
                comp_edges.append([bmap[i], bmap[i + 1]])
        comp_nodes_arr = np.array(nodes_mm.tolist() + extra_nodes,
                                  dtype=np.float32)
        comp_edges_arr = np.array(comp_edges, dtype=np.int32)
        return comp_nodes_arr, comp_edges_arr, branch_map


# =============================================================================
# Torch ionic model  (endocardial TTP06, fully on GPU)
# =============================================================================

class EndoTTP06Torch:
    """
    Vectorised endocardial TTP06 ionic model operating on torch tensors.
    All arrays live on `device`; no CPU transfers during time integration.

    To switch cell type:
      - change _g_Ks, _g_to
      - change _s_inf() and _tau_s() formulas (see comments)
    """

    # ── Fixed constants ───────────────────────────────────────────────────────
    _R   = 8314.472
    _T   = 310.0
    _F   = 96485.3415
    _Cm  = 0.185
    _Vc  = 0.016404
    _Vsr = 0.001094
    _Ko  = 5.4
    _Nao = 140.0
    _Cao = 2.0

    # ── Cell-type conductances  (ENDO) ────────────────────────────────────────
    _g_K1  = 5.405
    _g_Kr  = 0.096
    _g_Ks  = 0.245      # M-cell: 0.062  |  epi: 0.245
    _g_Na  = 14.838
    _g_bna = 0.00029
    _g_CaL = 0.000175
    _g_bca = 0.000592
    _g_to  = 0.073      # epi/mid: 0.294
    _g_pCa = 0.825
    _g_pK  = 0.0146
    _K_pCa = 0.0005
    _P_NaK  = 1.362
    _K_mk   = 1.0
    _K_mNa  = 40.0
    _K_NaCa = 1000.0
    _K_sat  = 0.1
    _alpha  = 2.5
    _gamma  = 0.35
    _Km_Ca  = 1.38
    _Km_Nai = 87.5
    _P_kna  = 0.03
    _a_rel    = 0.016464
    _b_rel    = 0.25
    _c_rel    = 0.008232
    _K_up     = 0.00025
    _V_leak   = 8e-5
    _Vmax_up  = 0.000425
    _Buf_c    = 0.15
    _K_buf_c  = 0.001
    _Buf_sr   = 10.0
    _K_buf_sr = 0.3
    _tau_g    = 2.0
    _tau_fCa  = 2.0

    def __init__(self, device: torch.device):
        self.dev = device
        # RTF and FRT are kept as separate named constants to avoid
        # the ambiguity of a combined 'RTF2' factor, which conflated
        # the monovalent and divalent Nernst scalings.
        # scalars used directly at each call site:
        #   RTF = R*T/F  (~26.7 mV)  — used in Nernst potentials
        #   FRT = F/(R*T) (~0.0374 mV^-1) — used in exponential voltage terms
        # The original code defined RTF2 = 2*RTF and then applied *2 factors
        # inconsistently, giving correct results in some expressions by accident
        # and wrong results in others.
        self.RTF = self._R * self._T / self._F   # R*T/F
        self.FRT = self._F / (self._R * self._T) # F/(R*T)

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_states(self, N: int, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        d = self.dev
        def _f(v):
            return torch.full((N,), v, dtype=dtype, device=d)
        return {
            "V":    _f(-86.2),
            "Ki":   _f(138.3),
            "Nai":  _f(11.6),
            "Cai":  _f(0.0002),
            "CaSR": _f(0.2),
            "Xr1":  _f(0.0),
            "Xr2":  _f(1.0),
            "Xs":   _f(0.0),
            "m":    _f(0.0),
            "h":    _f(0.75),
            "j":    _f(0.75),
            "d":    _f(0.0),
            "f":    _f(1.0),
            "fCa":  _f(1.0),
            "s":    _f(1.0),
            "r":    _f(0.0),
            "g":    _f(1.0),
        }

    # ── Rush-Larsen helper ────────────────────────────────────────────────────

    @staticmethod
    def _rl(x, x_inf, tau, dt):
        return x_inf - (x_inf - x) * torch.exp(-dt / tau)

    # ── Gate kinetics ─────────────────────────────────────────────────────────

    def _gates(self, V: torch.Tensor, Cai: torch.Tensor, dt: float):
        rl = self._rl

        # The activation curves below use the physiologically correct
        # sigmoid argument signs (depolarisation increases open probability).
        # negated, flipping each steady-state curve to its mirror image.
        # TTP06 (Ten Tusscher 2006) Table 1:
        #   xr1_inf = 1/(1+exp(-(V+26)/7))   = sigmoid(+(V+26)/7)
        #   xs_inf  = 1/(1+exp(-(V+5)/14))   = sigmoid(+(V+5)/14)
        #   d_inf   = 1/(1+exp(-(V+5)/7.5))  = sigmoid(+(V+5)/7.5)
        # The original code multiplied each argument by (-1).

        # Xr1
        xr1_inf = torch.sigmoid((V + 26.0) / 7.0)
        tau_xr1 = (450.0 / (1 + torch.exp((-45.0 - V) / 10.0))
                   * 6.0  / (1 + torch.exp(( V + 30.0) / 11.5)))

        # Xr2
        xr2_inf = torch.sigmoid(-(V + 88.0) / 24.0)
        tau_xr2 = (3.0   / (1 + torch.exp((-60.0 - V) / 20.0))
                   * 1.12 / (1 + torch.exp(( V - 60.0) / 20.0)))

        # Xs
        xs_inf  = torch.sigmoid((V + 5.0) / 14.0)
        tau_xs  = (1100.0 / torch.sqrt(1 + torch.exp((-10.0 - V) / 6.0))
                   / (1 + torch.exp((V - 60.0) / 20.0)))

        # m
        m_inf  = (1.0 / (1 + torch.exp((-56.86 - V) / 9.03))) ** 2
        tau_m  = (1.0 / (1 + torch.exp((-60.0  - V) / 5.0))
                  * (0.1 / (1 + torch.exp((V + 35.0) / 5.0))
                     + 0.1 / (1 + torch.exp((V - 50.0) / 200.0))))

        # h
        h_inf  = (1.0 / (1 + torch.exp((V + 71.55) / 7.43))) ** 2
        a_h    = torch.where(V < -40.0,
                             0.057 * torch.exp(-(V + 80.0) / 6.8),
                             torch.zeros_like(V))
        b_h    = torch.where(V < -40.0,
                             2.7 * torch.exp(0.079 * V)
                             + 3.1e5 * torch.exp(0.3485 * V),
                             0.77 / (0.13 * (1 + torch.exp((V + 10.66) / -11.1))))
        tau_h  = 1.0 / (a_h + b_h + 1e-30)

        # j
        j_inf  = h_inf.clone()
        a_j    = torch.where(
            V < -40.0,
            ((-25428.0 * torch.exp(0.2444 * V)
              - 6.948e-6 * torch.exp(-0.04391 * V))
             * (V + 37.78))
            / (1 + torch.exp(0.311 * (V + 79.23))),
            torch.zeros_like(V))
        b_j    = torch.where(
            V < -40.0,
            (0.02424 * torch.exp(-0.01052 * V))
            / (1 + torch.exp(-0.1378 * (V + 40.14))),
            (0.6 * torch.exp(0.057 * V))
            / (1 + torch.exp(-0.1 * (V + 32.0))))
        tau_j  = 1.0 / (a_j + b_j + 1e-30)

        # d
        d_inf  = torch.sigmoid((V + 5.0) / 7.5)
        tau_d  = ((1.4 / (1 + torch.exp((-35.0 - V) / 13.0)) + 0.25)
                  * 1.4 / (1 + torch.exp(( V +  5.0) /  5.0))
                  + 1.0 / (1 + torch.exp((50.0 - V) / 20.0)))

        # f
        f_inf  = torch.sigmoid((V + 20.0) / 7.0 * (-1))
        tau_f  = (1125.0 * torch.exp(-(V + 27.0) ** 2 / 240.0)
                  + 80.0
                  + 165.0 / (1 + torch.exp((25.0 - V) / 10.0)))

        # fCa
        a_fCa    = 1.0 / (1 + (Cai / 0.000325) ** 8)
        b_fCa    = 0.1 / (1 + torch.exp((Cai - 0.0005)  / 0.0001))
        g_fCa    = 0.2 / (1 + torch.exp((Cai - 0.00075) / 0.0008))
        fCa_inf  = (a_fCa + b_fCa + g_fCa + 0.23) / 1.46

        # s  ← ENDO kinetics
        # For epi/mcell replace with:
        #   s_inf = 1/(1+exp((V+20)/5))
        #   tau_s = 85*exp(-(V+45)^2/320) + 5/(1+exp((V-20)/5)) + 3
        s_inf = torch.sigmoid(-(V + 28.0) / 5.0)
        tau_s = 1000.0 * torch.exp(-(V + 67.0) ** 2 / 1000.0) + 8.0

        # r
        r_inf  = torch.sigmoid((V - 20.0) / 6.0 * (-1))
        tau_r  = 9.5 * torch.exp(-(V + 40.0) ** 2 / 1800.0) + 0.8

        # g (Ca release inactivation)
        g_inf  = torch.where(
            Cai < 0.00035,
            1.0 / (1 + (Cai / 0.00035) **  6),
            1.0 / (1 + (Cai / 0.00035) ** 16))

        return dict(
            xr1_inf=xr1_inf, tau_xr1=tau_xr1,
            xr2_inf=xr2_inf, tau_xr2=tau_xr2,
            xs_inf=xs_inf,   tau_xs=tau_xs,
            m_inf=m_inf,     tau_m=tau_m,
            h_inf=h_inf,     tau_h=tau_h,
            j_inf=j_inf,     tau_j=tau_j,
            d_inf=d_inf,     tau_d=tau_d,
            f_inf=f_inf,     tau_f=tau_f,
            fCa_inf=fCa_inf,
            s_inf=s_inf,     tau_s=tau_s,
            r_inf=r_inf,     tau_r=tau_r,
            g_inf=g_inf,
        )

    # ── Full ionic step ───────────────────────────────────────────────────────

    def step(
        self,
        states:  dict[str, torch.Tensor],
        dt:      float,
        I_stim:  torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Advance ionic state by dt ms.
        Returns (Iion, new_states).  V in new_states is unchanged
        (caller updates V via the monodomain diffusion solve).
        """
        V    = states["V"]
        Ki   = states["Ki"]
        Nai  = states["Nai"]
        Cai  = states["Cai"]
        CaSR = states["CaSR"]
        Xr1  = states["Xr1"]
        Xr2  = states["Xr2"]
        Xs   = states["Xs"]
        m    = states["m"]
        h    = states["h"]
        j    = states["j"]
        d    = states["d"]
        f    = states["f"]
        fCa  = states["fCa"]
        s    = states["s"]
        r    = states["r"]
        g    = states["g"]

        if I_stim is None:
            I_stim = torch.zeros_like(V)

        # ── Reversal potentials ───────────────────────────────────────────────
        E_Na = self.RTF * torch.log(self._Nao / Nai)
        E_K  = self.RTF * torch.log(self._Ko  / Ki)
        E_Ks = self.RTF * torch.log(
            (self._Ko  + self._P_kna * self._Nao)
            / (Ki + self._P_kna * Nai))
        # Nernst for a divalent cation (Ca²⁺) uses RT/(2F), not RT/F.
        # Original was RTF2*0.5 = (2*RTF)*0.5 = RTF, which is 2x too large.
        E_Ca = (self.RTF / 2.0) * torch.log(self._Cao / Cai)

        # ── Currents ──────────────────────────────────────────────────────────
        a_K1    = 0.1 / (1 + torch.exp(0.06 * (V - E_K - 200.0)))
        b_K1    = ((3.0 * torch.exp(0.0002 * (V - E_K + 100.0))
                    + torch.exp(0.1 * (V - E_K - 10.0)))
                   / (1 + torch.exp(-0.5 * (V - E_K))))
        xK1_inf = a_K1 / (a_K1 + b_K1)
        i_K1    = self._g_K1 * xK1_inf * torch.sqrt(
            torch.tensor(self._Ko / 5.4, device=self.dev)) * (V - E_K)

        i_to  = self._g_to  * r * s * (V - E_K)
        i_Kr  = self._g_Kr  * torch.sqrt(
            torch.tensor(self._Ko / 5.4, device=self.dev)) * Xr1 * Xr2 * (V - E_K)
        i_Ks  = self._g_Ks  * Xs ** 2 * (V - E_Ks)
        i_Na  = self._g_Na  * m ** 3 * h * j * (V - E_Na)
        i_bNa = self._g_bna * (V - E_Na)
        i_bCa = self._g_bca * (V - E_Ca)
        i_pCa = self._g_pCa * Cai / (Cai + self._K_pCa)
        i_pK  = self._g_pK  * (V - E_K) / (1 + torch.exp((25.0 - V) / 5.98))

        # i_CaL  (Goldman-Hodgkin-Katz)
        # FRT = F/(R*T) is used directly to keep Nernst expressions
        # unambiguous across mono- and divalent species.
        # The GHK voltage factor is 4*V*F^2/(R*T) = 4*V*F*FRT.
        # The exponential argument is 2*V*F/(R*T) = 2*V*FRT.
        eV2   = torch.exp(2.0 * V * self.FRT)
        denom = eV2 - 1.0
        denom = torch.where(denom.abs() < 1e-10,
                            torch.full_like(denom, 1e-10), denom)
        i_CaL = (self._g_CaL * d * f * fCa
                 * 4.0 * V * self._F * self.FRT
                 * (Cai * eV2 - 0.341 * self._Cao) / denom)

        # i_NaK
        # Voltage-dependence expressed as V·FRT = V·F/(R·T), dimensionless.
        # Original used V/RTF2*2 = V/(2*RTF)*2 = V/RTF = V*FRT — equivalent
        # but obscured by the spurious factor-of-2 pair.
        i_NaK = (self._P_NaK * self._Ko / (self._Ko + self._K_mk)
                 * Nai / (Nai + self._K_mNa)
                 / (1.0
                    + 0.1245 * torch.exp(-0.1 * V * self.FRT)
                    + 0.0353 * torch.exp(-V       * self.FRT)))

        # i_NaCa
        # Divalent Nernst term uses FRT/2, consistent with the Ca²⁺ Nernst above.
        eVg   = torch.exp( self._gamma       * V * self.FRT)
        eVgm1 = torch.exp((self._gamma - 1.) * V * self.FRT)
        i_NaCa = (self._K_NaCa
                  * (eVg   * Nai ** 3 * self._Cao
                     - eVgm1 * self._Nao ** 3 * Cai * self._alpha)
                  / ((self._Km_Nai ** 3 + self._Nao ** 3)
                     * (self._Km_Ca  + self._Cao)
                     * (1.0 + self._K_sat * eVgm1)))

        Iion = (i_K1 + i_to + i_Kr + i_Ks + i_CaL
                + i_NaK + i_Na + i_bNa + i_NaCa
                + i_bCa + i_pK + i_pCa - I_stim)

        # ── SR dynamics ───────────────────────────────────────────────────────
        i_rel  = ((self._a_rel * CaSR ** 2
                   / (self._b_rel ** 2 + CaSR ** 2))
                  + self._c_rel) * d * g
        i_up   = self._Vmax_up / (1 + self._K_up ** 2 / (Cai ** 2 + 1e-30))
        i_leak = self._V_leak  * (CaSR - Cai)

        bufc   = 1.0 / (1 + self._Buf_c  * self._K_buf_c
                         / (Cai  + self._K_buf_c)  ** 2)
        bufsr  = 1.0 / (1 + self._Buf_sr * self._K_buf_sr
                         / (CaSR + self._K_buf_sr) ** 2)

        # ── Concentration updates (Forward Euler) ─────────────────────────────
        dNai  = (-(i_Na + i_bNa + 3.0 * i_NaK + 3.0 * i_NaCa)
                  * self._Cm / (self._Vc * self._F))
        dKi   = (-(i_K1 + i_to + i_Kr + i_Ks + i_pK + I_stim
                    - 2.0 * i_NaK)
                  * self._Cm / (self._Vc * self._F))
        dCai  = bufc  * (i_leak - i_up + i_rel
                         - (i_CaL + i_bCa + i_pCa - 2.0 * i_NaCa)
                         * self._Cm / (2.0 * self._Vc * self._F))
        dCaSR = bufsr * (self._Vc / self._Vsr) * (i_up - i_rel - i_leak)

        Nai_new  = Nai  + dt * dNai
        Ki_new   = Ki   + dt * dKi
        Cai_new  = torch.clamp(Cai  + dt * dCai,  min=1e-7)
        CaSR_new = torch.clamp(CaSR + dt * dCaSR, min=1e-7)

        # ── Gate updates (Rush-Larsen) ────────────────────────────────────────
        gr = self._gates(V, Cai, dt)
        rl = self._rl

        Xr1_new = rl(Xr1, gr["xr1_inf"], gr["tau_xr1"], dt)
        Xr2_new = rl(Xr2, gr["xr2_inf"], gr["tau_xr2"], dt)
        Xs_new  = rl(Xs,  gr["xs_inf"],  gr["tau_xs"],  dt)
        m_new   = rl(m,   gr["m_inf"],   gr["tau_m"],   dt)
        h_new   = rl(h,   gr["h_inf"],   gr["tau_h"],   dt)
        j_new   = rl(j,   gr["j_inf"],   gr["tau_j"],   dt)
        d_new   = rl(d,   gr["d_inf"],   gr["tau_d"],   dt)
        f_new   = rl(f,   gr["f_inf"],   gr["tau_f"],   dt)
        s_new   = rl(s,   gr["s_inf"],   gr["tau_s"],   dt)
        r_new   = rl(r,   gr["r_inf"],   gr["tau_r"],   dt)

        dfCa    = (gr["fCa_inf"] - fCa) / self._tau_fCa
        fCa_new = torch.where(
            (gr["fCa_inf"] > fCa) & (V > -60.0),
            fCa, fCa + dt * dfCa)
        fCa_new = fCa_new.clamp(0.0, 1.0)

        dg    = (gr["g_inf"] - g) / self._tau_g
        g_new = torch.where(
            (gr["g_inf"] > g) & (V > -60.0),
            g, g + dt * dg)
        g_new = g_new.clamp(0.0, 1.0)

        new_states = {
            "V":    V,
            "Ki":   Ki_new,
            "Nai":  Nai_new,
            "Cai":  Cai_new,
            "CaSR": CaSR_new,
            "Xr1":  Xr1_new,
            "Xr2":  Xr2_new,
            "Xs":   Xs_new,
            "m":    m_new,
            "h":    h_new,
            "j":    j_new,
            "d":    d_new,
            "f":    f_new,
            "fCa":  fCa_new,
            "s":    s_new,
            "r":    r_new,
            "g":    g_new,
        }
        return Iion, new_states


# =============================================================================
# GPU sparse Laplacian builder
# =============================================================================

def _build_laplacian_gpu(
    # NOTE: unused in this file (purkinje_myocardium_anisotropic_solver only
    # calls _build_global_laplacian_gpu below) -- kept only for parity with
    # simulation/functions/purkinje_myocardium_pipeline.py. Still assumes a
    # single scalar w_edge; it is NOT anisotropy-aware and should not be
    # revived for that purpose without the same per-edge-array treatment
    # applied to _build_global_laplacian_gpu above.
    N:        int,
    edges:    np.ndarray,
    w_edge:   float,
    A_i:      float,
    device:   torch.device,
    dtype:    torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the physically correct graph Laplacian  L = D^{-1} K  as a sparse
    COO tensor, along with its diagonal (for the Jacobi preconditioner).

    K is the symmetric conductance matrix:
        K[i,j] = +w_edge   (off-diagonal, connected pair)
        K[i,i] = -degree_i * w_edge

    D is the diagonal membrane-loading matrix:
        D[i,i] = A_i   (same for all nodes in this domain)

    So L[i,j] = K[i,j] / A_i.

    For Purkinje  : w_edge = sigma_P * S_P / h_P,  A_i = A_P * h_P
    For myocardium: w_edge = sigma_M * h_M,         A_i = A_M * h_M^3

    This is the D^{-1}K construction described in Section 5.5 of the model
    description and implemented in the toy solver.  The old code omitted D^{-1},
    effectively solving L = K and producing unphysical conduction velocities.

    Returns
    -------
    L      : (N, N) sparse COO  — weighted graph Laplacian D^{-1}K
    diag_L : (N,)  dense        — diagonal of L (used by Jacobi preconditioner)
    """
    rows_e = edges[:, 0].astype(np.int64)
    cols_e = edges[:, 1].astype(np.int64)

    # Symmetric conductance matrix K (degree * w_edge on diagonal)
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

    # Apply D^{-1}: divide every entry in row i by A_i.
    # Since A_i is uniform within each domain, this is just a scalar division.
    v_all /= A_i

    idx = torch.tensor(np.stack([r_all, c_all]), dtype=torch.long,  device=device)
    val = torch.tensor(v_all,                    dtype=dtype,        device=device)
    L   = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()

    diag_L = torch.tensor(-degree / A_i, dtype=dtype, device=device)
    return L, diag_L ###Line552

def _build_global_laplacian_gpu(
    Np, N_myo,
    comp_edges_p, w_pp, Ai_P,
    myo_edges,    myo_edge_w, Ai_M,
    pmj_comp_ids, pmj_patches, c_pmj,
    device,
    dtype=torch.float64,
):
    """
    Build a single (Np+N_myo) × (Np+N_myo) global Laplacian L = D^{-1}K
    with Purkinje, myocardium, and PMJ coupling all as entries in K.

    Global index convention:
        0 .. Np-1          → Purkinje comp nodes
        Np .. Np+N_myo-1   → myocardial voxel nodes

    PMJ off-diagonal coupling (eq. 13-14 of model description):
        K[p, Np+m] = K[Np+m, p] = +w_pmj   (symmetric)
        K[p, p]   -= w_pmj  (add to diagonal accumulation)
        K[Np+m, Np+m] -= w_pmj

    w_pmj = c_pmj / n_patch_nodes  (distribute equally, total current invariant)

    D[i] = Ai_P for Purkinje nodes, Ai_M for myocardial nodes.
    L[i,:] = K[i,:] / D[i]  — non-symmetric because Ai_P ≠ Ai_M.
    Must use PCG/BiCGSTAB, not CG.

    Anisotropic difference from simulation._build_global_laplacian_gpu:
    `myo_edge_w` is now a per-edge ARRAY (myo["edge_w"] from
    build_myo_mesh_anisotropic), signed, not a uniform scalar w_mm. The
    diagonal is still exactly -sum(row's off-diagonal entries) regardless
    of sign, so conservation (row sums to zero before the D^{-1} scaling)
    holds unchanged. The myocardium-edge block below is also VECTORISED
    (np.add.at) rather than a Python loop, since 18-connectivity gives
    ~3x the edge count of the isotropic 6-connected case and a per-edge
    Python loop over tens of millions of edges at full patient-mesh scale
    would be prohibitively slow. Purkinje and PMJ edge counts are orders
    of magnitude smaller, so those stay as plain Python loops, unchanged.
    """
    N = Np + N_myo
    degree = np.zeros(N, dtype=np.float64)

    # ── Exact nnz up front, so we allocate ONCE instead of growing arrays
    #    via repeated np.concatenate. Every np.concatenate call below made a
    #    brand-new array and copied the old one into it, so at the moment of
    #    each call BOTH the old and the new buffer were resident at once —
    #    at 18-connectivity / ~66M-node scale that transient doubling, spread
    #    across row_arr/col_arr/val_arr/ga/gb/mw all being alive
    #    simultaneously, is what exhausted memory (the reported 9.28 GiB
    #    failure was just the last, largest, of several such copies; peak
    #    resident memory at that point was 3-4x that figure). Filling
    #    pre-sized buffers by slice-assignment does the same job with a
    #    single allocation per array and no transient doubling.
    me = myo_edges                                    # (E,2) int32/int64
    mw = np.asarray(myo_edge_w, dtype=np.float64)      # (E,)  signed
    E_myo   = me.shape[0]
    E_pk    = len(comp_edges_p)
    E_pmj   = sum(len(p) for p in pmj_patches)
    nnz     = 2 * E_pk + 2 * E_pmj + 2 * E_myo + N

    # int32 is sufficient for indices while we're still doing numpy-side
    # bookkeeping (N is ~66M, well under the 2^31 int32 ceiling) -- this
    # halves the footprint of every intermediate array below relative to
    # int64. We only widen to int64 once, in the single torch.tensor() call
    # at the very end, which torch's sparse constructor requires.
    row_arr = np.empty(nnz, dtype=np.int32)
    col_arr = np.empty(nnz, dtype=np.int32)
    val_arr = np.empty(nnz, dtype=np.float64)

    pos = 0

    # ── Purkinje edges (small; unchanged Python loop) ────────────────────────
    for a, b in comp_edges_p:
        row_arr[pos] = a; col_arr[pos] = b; val_arr[pos] = w_pp; pos += 1
        row_arr[pos] = b; col_arr[pos] = a; val_arr[pos] = w_pp; pos += 1
        degree[a] += w_pp; degree[b] += w_pp

    # ── PMJ coupling edges (small; unchanged Python loop) ────────────────────
    for k, p_idx in enumerate(pmj_comp_ids):
        patch = pmj_patches[k]
        w_pmj = c_pmj / len(patch)
        for m_local in patch:
            m_global = Np + m_local
            row_arr[pos] = p_idx;    col_arr[pos] = m_global; val_arr[pos] = w_pmj; pos += 1
            row_arr[pos] = m_global; col_arr[pos] = p_idx;    val_arr[pos] = w_pmj; pos += 1
            degree[p_idx]    += w_pmj
            degree[m_global]  += w_pmj

    # ── Myocardium edges (offset by Np) — VECTORISED, signed weights ────────
    ga = (me[:, 0].astype(np.int32) + Np)
    gb = (me[:, 1].astype(np.int32) + Np)

    np.add.at(degree, ga, mw)
    np.add.at(degree, gb, mw)

    row_arr[pos:pos+E_myo] = ga
    col_arr[pos:pos+E_myo] = gb
    val_arr[pos:pos+E_myo] = mw
    pos += E_myo
    row_arr[pos:pos+E_myo] = gb
    col_arr[pos:pos+E_myo] = ga
    val_arr[pos:pos+E_myo] = mw
    pos += E_myo

    del ga, gb, mw, me  # drop the ~4.4 GB-a-piece intermediates promptly

    # ── Diagonal ─────────────────────────────────────────────────────────────
    diag_idx = np.arange(N, dtype=np.int32)
    row_arr[pos:pos+N] = diag_idx
    col_arr[pos:pos+N] = diag_idx
    val_arr[pos:pos+N] = -degree
    pos += N
    assert pos == nnz

    # ── D^{-1} scaling ───────────────────────────────────────────────────────
    Ai = np.empty(N, dtype=np.float64)
    Ai[:Np]  = Ai_P
    Ai[Np:]  = Ai_M
    val_arr /= Ai[row_arr]   # in place, no extra copy

    # ── Move to GPU one array at a time, widening int32→int64 AFTER the
    #    transfer (i.e. in VRAM, not host RAM). The previous version did
    #    np.stack([row_arr, col_arr]) then .astype(np.int64) then
    #    torch.tensor(...) — three full-size CPU allocations back to back
    #    (~10GB + ~20GB + ~20GB) stacked on top of the ~20GB already
    #    resident (row_arr+col_arr+val_arr), which is what actually
    #    exhausted memory. Transferring int32 first and casting on-device
    #    means the host never holds an int64 copy of either array at all.
    row_t = torch.from_numpy(row_arr).to(device).long()
    del row_arr
    col_t = torch.from_numpy(col_arr).to(device).long()
    del col_arr
    idx = torch.stack([row_t, col_t], dim=0)
    del row_t, col_t

    val = torch.from_numpy(val_arr).to(device=device, dtype=dtype)
    del val_arr
    L   = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()

    diag_L = torch.tensor(-degree / Ai, dtype=dtype, device=device)
    return L, diag_L

# =============================================================================
# CN system matrix + Jacobi-preconditioned BiCGSTAB  (fully on GPU)
#
# Replaces _pcg (Conjugate Gradient).  CG requires the system matrix A to be
# symmetric positive-definite.  Once PMJ coupling is embedded in the global
# Laplacian, L = D^{-1}K is non-symmetric (A_P ≠ A_M gives different D rows),
# so A = (Cm/dt)I − θL is also non-symmetric.  BiCGSTAB handles this correctly.
# =============================================================================

def _build_cn_matrix(
    L:      torch.Tensor,  # (N,N) sparse COO global Laplacian D^{-1}K
    diag_L: torch.Tensor,  # (N,)  diagonal of L  (all negative)
    scale:  float,         # Cm/dt
    theta:  float,         # CN parameter
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build  A = (Cm/dt) I − theta * L  as a sparse CSR tensor (once, pre-loop).

    Returns
    -------
    A_csr : (N,N) sparse CSR — the fixed LHS matrix for every time step
    M_inv : (N,)  Jacobi preconditioner  1 / diag(A)
              diag(A)[i] = scale − theta*diag_L[i] > 0  (diag_L < 0)
    """
    L_coo  = L.coalesce()
    idx    = L_coo.indices()
    val    = (-theta) * L_coo.values().clone()       # start: −theta * L
    is_diag        = idx[0] == idx[1]
    val[is_diag]  += scale                           # add Cm/dt on diagonal

    N     = L_coo.shape[0]
    A_csr = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to_sparse_csr()

    diag_A = scale - theta * diag_L                 # (N,) all positive
    M_inv  = 1.0 / diag_A.clamp(min=1e-30)
    return A_csr, M_inv


def _bicgstab(
    A_csr:   torch.Tensor,  # (N,N) sparse CSR — CN system matrix A
    M_inv:   torch.Tensor,  # (N,)  Jacobi preconditioner
    b:       torch.Tensor,  # RHS
    x0:      torch.Tensor,  # warm-start initial guess
    tol:     float = 1e-5,
    maxiter: int   = 500,
) -> torch.Tensor:
    """
    Jacobi-preconditioned BiCGSTAB.  Solves  A x = b  from initial guess x0.
    Matches solver.py bicgstab() exactly.
    """
    x        = x0.clone()
    r        = b - torch.mv(A_csr, x)
    r_hat    = r.clone()
    rho_prev = alpha = omega = 1.0
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    b_norm = b.norm().item()
    if b_norm == 0.0:
        return x

    for _ in range(maxiter):
        rho = torch.dot(r_hat, r).item()
        if abs(rho) < 1e-300:
            r_hat = r.clone()
            rho   = torch.dot(r_hat, r).item()

        if _ == 0:
            p = r.clone()
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            p    = r + beta * (p - omega * v)

        p_hat = M_inv * p
        v     = torch.mv(A_csr, p_hat)

        denom_alpha = torch.dot(r_hat, v).item()
        if abs(denom_alpha) < 1e-300:
            break
        alpha = rho / denom_alpha

        s = r - alpha * v
        if s.norm().item() < tol * b_norm:
            x = x + alpha * p_hat
            break

        s_hat       = M_inv * s
        t           = torch.mv(A_csr, s_hat)
        denom_omega = torch.dot(t, t).item()
        omega       = (torch.dot(t, s).item() / denom_omega
                       if abs(denom_omega) > 1e-300 else 0.0)

        x        = x + alpha * p_hat + omega * s_hat
        r        = s - omega * t
        rho_prev = rho

        if r.norm().item() / b_norm < tol:
            break
        if abs(omega) < 1e-300:
            break

    return x


# =============================================================================
# Myocardium mesh builder  (CPU — one-time setup)
# =============================================================================

def _build_conductivity_tensor(fibre_dir, vox_idx, sigma_l, sigma_t):
    """
    Per-node transversely-isotropic conductivity tensor
        D(x) = sigma_t * I + (sigma_l - sigma_t) * f(x) f(x)^T
    from a unit fibre-direction field and the two eigen-conductivities.

    fibre_dir : (Nx,Ny,Nz,3) unit fibre vectors in RAW ARRAY-AXIS order
                (component 0 <-> array axis 0, matching vox_idx / this
                module's edge offsets -- e.g. hrdayapy.fibres.compute_fibres'
                output). This is NOT the swapped (axis1,axis0,axis2)
                convention nodes_mm uses for Purkinje coupling below --
                the two are never mixed in this function.
    vox_idx   : (N_myo,3) int array-index coords of myocardial nodes

    Returns six (N_myo,) float64 arrays: D00,D11,D22 (always >0) and
    D01,D02,D12 (signed -- these vanish only where the fibre is aligned
    with a grid axis).
    """
    f = fibre_dir[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2], :].astype(np.float64)
    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
    dsig = sigma_l - sigma_t
    D00 = sigma_t + dsig * f0 * f0
    D11 = sigma_t + dsig * f1 * f1
    D22 = sigma_t + dsig * f2 * f2
    D01 = dsig * f0 * f1
    D02 = dsig * f0 * f2
    D12 = dsig * f1 * f2
    return D00, D11, D22, D01, D02, D12


def build_myo_mesh_anisotropic(
    S:             np.ndarray,
    Z:             np.ndarray,
    voxel_size:    float,
    fibre_dir:     np.ndarray,
    sigma_l:       float,
    sigma_t:       float,
) -> dict:
    """
    Anisotropic counterpart of simulation.build_myo_mesh. Identical in every
    respect (node numbering, Z-surface detection, nodes_mm convention)
    EXCEPT the edge list, which is extended from 6-connected (face
    neighbours only) to 18-connected (+ 12 edge-diagonal neighbours), with
    a signed per-edge conductance weight replacing the old uniform
    sigma_M * voxel_size scalar.

    Why 18-connectivity: the conductivity tensor D(x) built from the fibre
    field has up to 6 independent components (3 diagonal + 3 off-diagonal).
    A 6-connected face-only stencil can only see the 3 diagonal components
    projected onto the grid axes -- it structurally cannot represent the
    off-diagonal (D01,D02,D12) terms that appear whenever the fibre is
    oblique to the grid, which is the normal case for a rotating
    transmural fibre field. Restricting to the diagonal alone was checked
    quantitatively earlier (see conversation history / project notes): at
    45 deg / body-diagonal fibre orientations it can suppress up to
    ~40-65% of the true anisotropy and rotates the propagation ellipse's
    principal axis back toward the grid axes. The 12 edge-diagonal
    neighbours (offset in exactly two of the three axes each) are exactly
    what's needed to recover D01/D02/D12 via the standard central
    mixed-derivative stencil -- see the per-edge weight derivation below.

    Edge weight formulas
    ---------------------
    Face edges (6, e.g. offset (1,0,0)):
        w = D_kk_effective * voxel_size
    where D_kk_effective is the HARMONIC MEAN of the relevant diagonal
    tensor component (D00/D11/D22) at the two endpoint voxels -- harmonic
    averaging is the standard choice for flux continuity across
    heterogeneous conductivity, and is safe here since D00/D11/D22 are
    always positive.

    Edge-diagonal edges (12, e.g. offset (1,1,0) for the D01 term):
    derived from the standard central-difference cross-derivative stencil,
        d/dx(D01 du/dy) + d/dy(D01 du/dx) ~= 2*D01 * d^2u/dxdy
        d^2u/dxdy ~= [u(++) - u(+-) - u(-+) + u(--)] / (4h^2)
    and matched to this codebase's existing K/D^{-1} convention (where the
    face weight sigma*h already absorbs one h^3 into A_i = A_M*h^3, see
    _build_laplacian_gpu's docstring) gives, per unique undirected
    edge-diagonal pair:
        w = +D_offdiag_avg * voxel_size / 2   for a "co-signed" pair,
            e.g. (1,1,0)   (contributes with the SAME sign as D01)
        w = -D_offdiag_avg * voxel_size / 2   for a "counter-signed" pair,
            e.g. (1,-1,0)  (contributes with the OPPOSITE sign)
    where D_offdiag_avg is the ARITHMETIC MEAN of the relevant
    off-diagonal component (D01/D02/D12) at the two endpoint voxels --
    arithmetic, not harmonic, because off-diagonal components are signed
    (harmonic averaging is undefined/inappropriate once a quantity can be
    negative or cross zero between neighbours).

    These are SIGNED weights -- unlike the isotropic 6-connected case,
    the resulting sparse matrix is no longer a pure positive-conductance
    M-matrix. See the accompanying design notes on what that does and
    does not require of the downstream BiCGSTAB solve.

    Returns
    -------
    dict with the same keys as simulation.build_myo_mesh's return, plus:
        edge_w : (E,) float64 -- signed per-edge conductance, same
                 row-order as `edges`. Feed both to
                 _build_global_laplacian_gpu instead of (myo["edges"], w_mm).
    """
    shape      = S.shape
    Nx, Ny, Nz = shape

    vox_idx = np.argwhere(S == 1).astype(np.int32)
    N_myo   = len(vox_idx)
    print(f"  Myocardium: {N_myo:,} nodes")

    flat_to_node           = -np.ones(Nx * Ny * Nz, dtype=np.int32)
    flat_ids               = (vox_idx[:, 0] * Ny * Nz
                               + vox_idx[:, 1] * Nz
                               + vox_idx[:, 2])
    flat_to_node[flat_ids] = np.arange(N_myo, dtype=np.int32)

    D00, D11, D22, D01, D02, D12 = _build_conductivity_tensor(
        fibre_dir, vox_idx, sigma_l, sigma_t)
    diag_comp = {0: D00, 1: D11, 2: D22}
    # off-diagonal component + which two axes it couples, keyed by axis-pair
    offdiag_comp = {(0, 1): D01, (0, 2): D02, (1, 2): D12}

    def _edges_for_offset(offset, weight_of_pair):
        """Vectorised edge construction for one direction, shared by both
        the 6 face offsets and the 12 (as 6 unique pairs) edge-diagonal
        offsets below -- same flat_to_node lookup pattern as the isotropic
        version, generalised to return a per-edge weight alongside."""
        dx_, dy_, dz_ = offset
        ni = vox_idx[:, 0] + dx_
        nj = vox_idx[:, 1] + dy_
        nk = vox_idx[:, 2] + dz_
        inb = ((ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny) & (nk >= 0) & (nk < Nz))
        nf  = np.where(inb, ni * Ny * Nz + nj * Nz + nk, 0).astype(np.int64)
        nb_id = flat_to_node[nf]
        valid = inb & (nb_id >= 0)
        a_idx = np.where(valid)[0].astype(np.int32)
        b_idx = nb_id[valid]
        w = weight_of_pair(a_idx, b_idx)
        return a_idx, b_idx, w

    rows_e_list, cols_e_list, w_list = [], [], []

    # ── 6 face edges: harmonic-mean diagonal conductivity ──
    for axis, offset in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        Dkk = diag_comp[axis]
        def _face_weight(a_idx, b_idx, Dkk=Dkk):
            Da, Db = Dkk[a_idx], Dkk[b_idx]
            D_harm = 2.0 * Da * Db / np.maximum(Da + Db, 1e-30)
            return D_harm * voxel_size
        a_idx, b_idx, w = _edges_for_offset(offset, _face_weight)
        rows_e_list.append(a_idx); cols_e_list.append(b_idx); w_list.append(w)

    # ── 12 edge-diagonal edges (6 unique undirected pairs): signed
    #    arithmetic-mean off-diagonal conductivity ──
    # axis_pair -> (co-signed offset, counter-signed offset)
    diagonal_offsets = {
        (0, 1): ((1, 1, 0), (1, -1, 0)),
        (0, 2): ((1, 0, 1), (1, 0, -1)),
        (1, 2): ((0, 1, 1), (0, 1, -1)),
    }
    for axis_pair, (offset_same, offset_diff) in diagonal_offsets.items():
        Dij = offdiag_comp[axis_pair]

        def _diag_weight_same(a_idx, b_idx, Dij=Dij):
            D_avg = 0.5 * (Dij[a_idx] + Dij[b_idx])   # arithmetic mean -- see docstring
            return D_avg * voxel_size / 2.0

        def _diag_weight_diff(a_idx, b_idx, Dij=Dij):
            D_avg = 0.5 * (Dij[a_idx] + Dij[b_idx])
            return -D_avg * voxel_size / 2.0

        a_idx, b_idx, w = _edges_for_offset(offset_same, _diag_weight_same)
        rows_e_list.append(a_idx); cols_e_list.append(b_idx); w_list.append(w)

        a_idx, b_idx, w = _edges_for_offset(offset_diff, _diag_weight_diff)
        rows_e_list.append(a_idx); cols_e_list.append(b_idx); w_list.append(w)

    edges  = np.column_stack([
        np.concatenate(rows_e_list),
        np.concatenate(cols_e_list),
    ])
    edge_w = np.concatenate(w_list).astype(np.float64)
    print(f"            {len(edges):,} edges (18-connected: 6 face + 12 edge-diagonal)")

    # Z-surface nodes
    offsets_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    is_surf = np.zeros(N_myo, dtype=bool)
    for dx_, dy_, dz_ in offsets_6:
        ni = np.clip(vox_idx[:,0]+dx_, 0, Nx-1)
        nj = np.clip(vox_idx[:,1]+dy_, 0, Ny-1)
        nk = np.clip(vox_idx[:,2]+dz_, 0, Nz-1)
        inb = (((vox_idx[:,0]+dx_)>=0) & ((vox_idx[:,0]+dx_)<Nx) &
               ((vox_idx[:,1]+dy_)>=0) & ((vox_idx[:,1]+dy_)<Ny) &
               ((vox_idx[:,2]+dz_)>=0) & ((vox_idx[:,2]+dz_)<Nz))
        nb_in_Z = inb & (Z[ni, nj, nk] == 1)
        is_surf |= ~nb_in_Z

    surface_node_ids = np.where(is_surf)[0].astype(np.int32)
    print(f"            {len(surface_node_ids):,} Z-surface nodes")

    return dict(
        vox_idx          = vox_idx,
        flat_to_node     = flat_to_node,
        # nodes_mm must share the Purkinje tree's (x,y,z) = (col,row,depth)
        # convention (see create_purkinje.py / pick_av_node_and_biventricular_
        # roots()), NOT vox_idx's raw (row,col,depth) array-index order --
        # otherwise every downstream use that compares myocardium and
        # Purkinje positions (map_pmj's nearest-node search,
        # clip_purkinje_outside_Z, the rendered surface in animate_coupled)
        # silently pairs up the wrong nodes/voxels whenever the geometry
        # isn't symmetric under swapping those two axes. vox_idx itself is
        # left raw -- it's used elsewhere in this module to index directly
        # into S/Z/ectopic_region, which are plain arrays in array-index
        # order.
        nodes_mm         = vox_idx[:, [1, 0, 2]].astype(np.float32) * voxel_size,
        edges            = edges,
        edge_w           = edge_w,   # (E,) signed per-edge conductance -- see docstring
        surface_node_ids = surface_node_ids,
        shape            = shape,
        N_myo            = N_myo,
    )


# =============================================================================
# Surface mesh  (marching cubes, one-time)
# =============================================================================

def build_surface_mesh(Z, myo, voxel_size):
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree

    print("  Building surface mesh (marching cubes)...")
    verts, faces, _, _ = marching_cubes(
        Z.astype(np.float32), level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size))
    # marching_cubes returns verts in raw (row, col, depth) array-index
    # order (scaled by spacing) -- swap to (col, row, depth) to match
    # nodes_mm's convention (see build_myo_mesh) before anything compares
    # the two, e.g. the cKDTree match just below.
    surf_verts = verts[:, [1, 0, 2]].astype(np.float32)
    surf_faces = faces.astype(np.int32)
    print(f"  Surface: {len(surf_verts):,} verts, {len(surf_faces):,} faces")

    surf_ids    = myo["surface_node_ids"]
    surf_pts_mm = myo["nodes_mm"][surf_ids]
    _, nn_idx   = cKDTree(surf_pts_mm).query(surf_verts, k=1, workers=-1)
    return surf_verts, surf_faces, nn_idx.astype(np.int32)


# =============================================================================
# Purkinje clip to outside Z
# =============================================================================

def clip_purkinje_outside_Z(comp_nodes, comp_edges, Z, voxel_size):
    """
    NOTE: fixes a pre-existing bug inherited from
    simulation/functions/purkinje_myocardium_pipeline.py's identical
    function -- the clip bounds below were [Nx-1, Ny-1, Nz-1], applied to
    comp_nodes' own (x,y,z)=(col,row,depth) values BEFORE the axis-0/1
    swap on the next line, when they needed to be [Ny-1, Nx-1, Nz-1] to
    match that pre-swap axis order (x is the column/axis-1 direction,
    bounded by Ny; y is the row/axis-0 direction, bounded by Nx). Silently
    harmless whenever Nx==Ny (e.g. this codebase's existing square-
    cross-section Purkinje-slab benchmark, SLAB_X_MM==SLAB_Y_MM==10.0) --
    surfaces as an IndexError (or, worse, a silent wrong-voxel lookup that
    clips "successfully" to the wrong location without erroring) for any
    non-square cross-section, which includes essentially every real
    anatomical geometry. Flagged for a matching fix in the original
    isotropic file, not just here.
    """
    Nx, Ny, Nz = Z.shape
    vox_xyz = np.clip(
        np.round(comp_nodes / voxel_size).astype(np.int32),
        [0,0,0], [Ny-1, Nx-1, Nz-1])
    # comp_nodes are in (x,y,z) = (col,row,depth) Purkinje convention;
    # Z is a plain array indexed (row,col,depth) -- swap axes 0/1 back
    # before indexing it, or this looks up the wrong voxel entirely
    # whenever the geometry isn't symmetric under that swap.
    vox      = vox_xyz[:, [1, 0, 2]]
    inside   = Z[vox[:,0], vox[:,1], vox[:,2]].astype(bool)
    outside  = ~inside
    keep     = outside[comp_edges[:,0]] & outside[comp_edges[:,1]]
    print(f"  Purkinje: {keep.sum():,} edges outside Z "
          f"({(~keep).sum():,} removed)")
    return comp_edges[keep]


# =============================================================================
# PMJ mapping
# =============================================================================

def map_pmj(comp_nodes_p, terminal_comp_ids, myo, n_pmj=1):
    """
    For each PMJ terminal, return the n_pmj nearest myocardial nodes by
    Euclidean distance.  The conductance c_pmj is later divided equally
    among these nodes so the total current per junction is always c_pmj,
    regardless of n_pmj.

    n_pmj=1  → point coupling (single nearest node)
    n_pmj>1  → distributed coupling to the n_pmj nearest nodes
    """
    from scipy.spatial import cKDTree

    tree      = cKDTree(myo["nodes_mm"])
    k         = min(n_pmj, len(myo["nodes_mm"]))
    dists, nn = tree.query(comp_nodes_p[terminal_comp_ids], k=k, workers=-1)

    # tree.query returns a scalar (not array) when k=1; normalise to 2-D
    if k == 1:
        nn = nn[:, np.newaxis]

    patches = [row.tolist() for row in nn]
    return patches

# =============================================================================
# Terminal finder
# =============================================================================

def _find_terminals(elements, n_nodes):
    deg = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(deg, elements[:,0], 1)
    np.add.at(deg, elements[:,1], 1)
    terms = [i for i in range(1, n_nodes) if deg[i] == 1]
    return terms if terms else [n_nodes - 1]


# =============================================================================
# Stimulus protocol — multiple independent stimulus sites, each with its own
# location, amplitude, duration, onset time, and (optional) pacing period.
# =============================================================================
#
# Each entry of `stim_protocol` is a plain dict:
#
#   target   : "root"    -> AV-node / Purkinje root zone (sphere of radius
#                            stim_len_mm around the earliest-activated node)
#              "ectopic" -> myocardial focus given by `ectopic_region`
#              "custom"  -> explicit global node indices via "node_ids"
#   amp      : stimulus current density [uA/mm^2]
#   dur      : pulse duration [ms]
#   onset    : time of the first pulse [ms]                  (default 0.0)
#   period   : time between repeats [ms]; None/0 -> single pulse (default None)
#   n_pulses : number of repeats; None -> repeat until T if period is set
#   name     : optional label, used only for the printed summary
#
# Example — one ectopic beat at t=0 ms, AV-node escape beat at t=100 ms:
#   stim_protocol = [
#       dict(name="ectopic", target="ectopic", amp=12.0, dur=2.0, onset=0.0),
#       dict(name="av_node", target="root",    amp=12.0, dur=1.0, onset=100.0),
#   ]

def _resolve_stim_sites(
    stim_protocol: list[dict],
    root_mask_p:   np.ndarray,     # (Np,) bool — earliest-activated Purkinje zone
    ectopic_region: np.ndarray | None,   # (Nx,Ny,Nz) bool voxel mask, or None
    vox_idx:       np.ndarray,     # (N_myo, 3) — voxel coords of each myo node
    Np:            int,
    N_myo:         int,
    dev:           torch.device,
    dtype:         torch.dtype,
    stim_regions:  dict[str, np.ndarray] | None = None,   # name -> (Nx,Ny,Nz) bool
) -> list[dict]:
    """
    Turn each `stim_protocol` entry into a ready-to-use stimulus site:
    a precomputed (mask * amp) tensor over the full N_global node vector,
    plus its timing parameters.  Building the tensor once here (instead of
    re-indexing every step) keeps the per-step cost of an arbitrary number
    of sites essentially free.
    """
    N_global = Np + N_myo

    root_node_ids = np.where(root_mask_p)[0]   # already global (Purkinje is [0:Np])

    if ectopic_region is not None:
        in_region = ectopic_region[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].astype(bool)
        ectopic_node_ids = Np + np.where(in_region)[0]
    else:
        ectopic_node_ids = np.array([], dtype=np.int64)

    # Named regions: any number of independent voxel masks, selected per
    # protocol entry with  target="region", region="<name>".  Same mapping
    # as the single "ectopic" mask above (voxel mask -> myocardial nodes ->
    # global solver ids Np + i), just once per name.
    region_node_ids: dict[str, np.ndarray] = {}
    for _name, _mask in (stim_regions or {}).items():
        _in = np.asarray(_mask)[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].astype(bool)
        region_node_ids[_name] = Np + np.where(_in)[0]

    sites = []
    for ev in stim_protocol:
        target = ev.get("target", "root")
        if target == "root":
            node_ids = root_node_ids
        elif target == "ectopic":
            node_ids = ectopic_node_ids
            if node_ids.size == 0:
                print(f"  WARNING: stimulus '{ev.get('name', target)}' targets "
                      f"'ectopic' but no myocardial nodes fall inside "
                      f"ectopic_region (pass a non-empty mask, e.g. "
                      f"cfg.PATH_STIM_REGION, or widen STIM_*_TOL in config.py).")
        elif target == "region":
            rname = ev.get("region")
            if rname not in region_node_ids:
                raise ValueError(
                    f"Stimulus '{ev.get('name', target)}' has target='region' "
                    f"with region={rname!r}, but stim_regions only defines "
                    f"{sorted(region_node_ids)}.")
            node_ids = region_node_ids[rname]
            if node_ids.size == 0:
                print(f"  WARNING: stimulus '{ev.get('name', target)}' targets "
                      f"region {rname!r} but no myocardial nodes fall inside it.")
        elif target == "custom":
            node_ids = np.asarray(ev["node_ids"], dtype=np.int64)
        else:
            raise ValueError(
                f"Unknown stimulus target '{target}' (expected "
                f"'root', 'ectopic', 'region', or 'custom').")

        amp_tensor = torch.zeros(N_global, dtype=dtype, device=dev)
        if node_ids.size:
            idx = torch.as_tensor(node_ids, dtype=torch.long, device=dev)
            amp_tensor[idx] = float(ev["amp"])

        sites.append(dict(
            name       = ev.get("name", target),
            target     = target,
            n_nodes    = int(node_ids.size),
            amp        = float(ev["amp"]),
            amp_tensor = amp_tensor,
            dur        = float(ev["dur"]),
            onset      = float(ev.get("onset", 0.0)),
            period     = ev.get("period", None) or None,
            n_pulses   = ev.get("n_pulses", None),
        ))
    return sites


def _stim_active(t: float, onset: float, dur: float,
                  period: float | None, n_pulses: int | None) -> bool:
    """True if a pulse from this site is firing at time *t* (ms)."""
    if t < onset:
        return False
    if not period:                       # None or 0  -> single pulse, fires once
        return t < onset + dur
    k = int((t - onset) // period)       # which pulse number we're in
    if n_pulses is not None and k >= n_pulses:
        return False
    phase = (t - onset) - k * period
    return phase < dur


# =============================================================================
# Main coupled simulation
# =============================================================================

def purkinje_myocardium_anisotropic_solver(
    nodes:            np.ndarray,
    elements:         np.ndarray,
    activation_times: np.ndarray,
    S:                np.ndarray,
    Z:                np.ndarray,
    phi:              np.ndarray,        # (Nx,Ny,Nz) transmural coord, 0=endo 1=epi
    fibre_dir:        np.ndarray,        # (Nx,Ny,Nz,3) unit fibre direction cosines,
                                          # RAW array-axis order (hp.fibres.compute_fibres output)
    *,
    voxel_size:   float = 0.4,
    dx_p:         float = 0.2,
    sigma_P:      float = 0.225,
    sigma_l:      float = 3.0 * 6.5625e-5,   # myocardial conductivity ALONG fibre [mS/mm]
    sigma_t:      float = 6.5625e-5,          # myocardial conductivity ACROSS fibre [mS/mm]
    Cm:           float = 0.01,
    A_P:          float = None,   # DEPRECATED -- no longer settable, see below
    A_M:          float = 140.0,  # mm^-1 -- corrected from a stale 0.14 default
                                   # (1000x too small; see conversation/commit history --
                                   # this matches Table 3 of Niederer et al. 2011,
                                   # "surface area to volume ratio 140 mm^-1")
    R_P:          float = 0.015,
    dt:           float = 0.1,
    T:            float = 400.0,
    theta:        float = 0.5,
    stim_protocol: list[dict] | None = None,
    ectopic_region: np.ndarray | None = None,
    stim_regions: dict[str, np.ndarray] | None = None,
    stim_len_mm:  float = 2.0,
    c_pmj:        float = 0.05,
    n_pmj:        int   = 1,
    cg_tol:       float = 1e-5,
    cg_max_iter:  int   = 100,
    n_frames:     int   = 100,
    frame_save_dt: float | None = None,  # sampling interval [ms] for the
                                          # animation output (surf_frames /
                                          # branch_* / time in out_npz).
                                          # Overrides n_frames when given, so
                                          # cadence stays fixed as T grows
                                          # instead of spreading n_frames
                                          # evenly across the whole run (which
                                          # gets coarser the longer T is).
                                          # None (default) keeps the old
                                          # n_frames behaviour.
    anim_tmp_dir: str | Path | None = None,  # scratch dir for the animation
                                              # frames' on-disk buffer; None ->
                                              # same directory as out_npz
    device:       str | None = None,
    dtype:        torch.dtype = torch.float64,
    out_npz:      str | Path | None = None,
    # ── Transmural heterogeneity ──────────────────────────────────────────────
    phi_endo_max: float = 0.35,   # phi < this → EndoTTP06
    phi_epi_min:  float = 0.65,   # phi ≥ this → EpiTTP06  (middle → MidTTP06)
    # ── Vm volumetric snapshots ───────────────────────────────────────────────
    vm_save_dt:   float | None = None,   # sampling interval [ms]; None → disabled
    out_vm_npz:   str | Path | None = None,  # path for vm_snapshots.npz
    vm_tmp_dir:   str | Path | None = None,  # scratch dir for the on-disk Vm
                                              # snapshot buffer; None → same
                                              # directory as out_vm_npz
) -> dict:
    """
    Coupled Purkinje + anatomical myocardium monodomain simulation, with
    ANISOTROPIC myocardial conduction along/across the fibre direction.

    This is the anisotropic counterpart of simulation.run_coupled -- same
    Purkinje discretisation, ionic models, PMJ coupling, time-stepping and
    solver, differing only in how myocardium-myocardium conductivity is
    built: a single scalar sigma_M is replaced by two eigen-conductivities
    (sigma_l along the fibre, sigma_t across it) combined with a per-voxel
    fibre direction field into a full conductivity tensor, assembled onto
    an 18-connected (not 6-connected) mesh so the tensor's off-diagonal
    components are actually represented -- see build_myo_mesh_anisotropic's
    docstring for the full derivation and why 6-connectivity alone
    (axis-projected diagonal only) is not adequate for a rotating fibre
    field. sigma_l == sigma_t recovers the isotropic case exactly (up to
    the extra, then-all-zero edge-diagonal terms).

    Stimulus current is injected at one or more independent sites defined by
    `stim_protocol` (see module-level docs above `_resolve_stim_sites`); the
    default reproduces a single AV-node pulse at t=0 if `stim_protocol` is
    omitted. Pass `ectopic_region` (a boolean voxel mask, e.g. the array
    saved at cfg.PATH_STIM_REGION by pipeline.py Step 4c) to enable any
    "ectopic"-target entries in the protocol.

    Transmural-heterogeneous TTP06 ionic model in the myocardium:
        phi < phi_endo_max          → endocardial TTP06
        phi_endo_max..phi_epi_min   → mid-myocardial (M-cell) TTP06
        phi >= phi_epi_min          → epicardial TTP06
    Purkinje nodes always use endocardial TTP06, regardless of phi.
    All diffusion solves and ionic steps run on GPU.

    Unit system: mm / ms / mV / mS / uF / uA  (consistent with toy system).

    Parameters
    ----------
    nodes, elements, activation_times : from load_purkinje()
    S         : binary myocardium mask  (Nx,Ny,Nz) uint8
    Z         : cut mask from Slicer    (Nx,Ny,Nz) uint8
    phi       : transmural coordinate   (Nx,Ny,Nz) float32, 0=endo 1=epi,
                NaN outside S (from generate_transmural_coordinate())
    fibre_dir : unit fibre direction    (Nx,Ny,Nz,3) float32, RAW array-axis
                order (component 0 <-> array axis 0, matching S/phi's own
                axes -- NOT the swapped nodes_mm convention used for
                Purkinje coupling elsewhere in this file). This is exactly
                what hrdayapy.fibres.compute_fibres / load_fibres returns --
                pass its output straight through, no axis manipulation.
    stim_protocol : list of dicts, one per stimulus site (target/amp/dur/
                onset/period/n_pulses/name) — see the comment block above
                `_resolve_stim_sites` for the full schema and an example.
                None -> a single one-shot AV-node pulse at t=0.
    ectopic_region : boolean voxel mask (Nx,Ny,Nz), required only if
                `stim_protocol` includes a "target": "ectopic" entry.
    stim_regions : dict name -> boolean voxel mask (Nx,Ny,Nz). Lets a protocol
                use several independent myocardial sites, e.g.
                    stim_regions  = {"apex": m1, "rv_free_wall": m2}
                    stim_protocol = [dict(target="region", region="apex", ...),
                                     dict(target="region", region="rv_free_wall", ...)]
                Coexists with `ectopic_region` (which is unchanged).
    stim_len_mm : radius [mm] of the "root" stimulus zone around the
                earliest-activated Purkinje node.
    sigma_P   : Purkinje axial conductivity [mS/mm]
    sigma_l   : myocardial conductivity ALONG the local fibre direction [mS/mm]
    sigma_t   : myocardial conductivity ACROSS the local fibre direction [mS/mm]
                (sigma_l == sigma_t reproduces isotropic conduction)
    Cm        : specific membrane capacitance [uF/mm^2] — PDE parameter
    A_M       : myocardial membrane surface-to-volume ratio [mm^-1] --
                see A_M's inline default comment for provenance (Table 3,
                Niederer et al. 2011)
    (A_P is no longer a parameter -- derived internally as 2.0/R_P; see
    R_P's entry below and the [FIX M2] comment at its point of use)
    c_pmj     : lumped PMJ conductance per terminal [mS]
    n_pmj     : number of nearest myocardial nodes coupled per PMJ terminal.
                1 = point coupling; >1 distributes current to the n_pmj
                nearest nodes by Euclidean distance (total current conserved).
    phi_endo_max : phi threshold below which nodes get endocardial TTP06
    phi_epi_min  : phi threshold above which nodes get epicardial TTP06
                   (nodes between the two thresholds get M-cell TTP06)
    n_frames     : number of animation frames (surf_frames/branch_*/time in
                   out_npz), spread evenly across [0, T]. Simple default for
                   short runs; for long runs the resulting cadence (T /
                   n_frames) gets coarser the bigger T is -- use
                   `frame_save_dt` instead when T varies or is long.
    frame_save_dt : animation sampling interval [ms], as an interval instead
                   of a count. When given, overrides n_frames so the cadence
                   is `frame_save_dt` regardless of T (e.g. always every
                   5 ms, whether T is 2 s or 30 min). Backed by the same
                   on-disk memmap pattern as vm_save_dt below, so RAM stays
                   flat even at a fine interval over a long run -- prefer
                   this over raising n_frames for long/fine-cadence runs.
                   None (default) leaves n_frames in charge.
    anim_tmp_dir  : scratch directory for the animation frame buffer during
                   the run. None (default) -> same directory as out_npz.
    vm_save_dt   : Vm snapshot sampling interval [ms].  A snapshot of the full
                   myocardial Vm field is written every `vm_save_dt` ms,
                   independently of `n_frames`/`frame_save_dt` (the animation
                   output). None (default) disables this output entirely.
    out_vm_npz   : path for the Vm snapshot NPZ.  Required when vm_save_dt is
                   set; silently ignored otherwise.
                   NPZ schema — see `_save_vm_snapshots` for full details:
                     node_ids   (N_myo,)             int32   global solver index
                     coords_mm  (N_myo, 3)           float32 physical xyz [mm]
                     vox_idx    (N_myo, 3)           int32   voxel ijk
                     Vm         (n_vm_frames, N_myo) float32 mV
                     time_vm    (n_vm_frames,)        float32 ms
                     dt_solver  scalar                ms
                     vm_save_dt scalar                ms
    """

    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"\nDevice : {dev}")

    # ── Purkinje discretisation ───────────────────────────────────────────────
    nodes_mm = nodes.astype(np.float32) * voxel_size
    comp_nodes_p, comp_edges_p, branch_map = _discretise_tree(
        nodes_mm, elements, dx_p)
    Np = len(comp_nodes_p)
    print(f"Purkinje : {Np:,} comp. nodes, {len(comp_edges_p):,} edges")

    # ── AV-node stimulus zone ─────────────────────────────────────────────────
    act_np  = np.array(activation_times, dtype=np.float32)
    finite  = np.isfinite(act_np)
    act_np[~finite] = T
    min_act = float(act_np[finite].min())
    root_orig    = np.where(np.isclose(act_np, min_act, atol=1e-6))[0]
    root_pos_mm  = nodes_mm[root_orig].mean(axis=0)
    dists        = np.linalg.norm(comp_nodes_p - root_pos_mm[None,:], axis=1)
    stim_mask_np = dists <= stim_len_mm
    if not stim_mask_np.any():
        stim_mask_np[int(dists.argmin())] = True
    stim_mask = torch.tensor(stim_mask_np, dtype=torch.bool, device=dev)
    print(f"  Root stim zone: {int(stim_mask.sum())} nodes within {stim_len_mm} mm of root")

    # ── Derived geometric quantities ─────────────────────────────────────────
    import math as _math
    S_P   = _math.pi * R_P ** 2   # mm^2   Purkinje fibre cross-sectional area
    # [FIX M2] A_P (Purkinje surface-to-volume ratio) is NOT an independent
    # parameter -- for a cylinder (fibre radius R_P, end caps negligible for
    # a long thin cable), A_P = (2*pi*R_P)/(pi*R_P^2) = 2/R_P, mechanically
    # determined by R_P. It used to be a separately-settable kwarg with a
    # default (133.33) that just happened to equal 2/0.015 -- i.e. it was
    # always meant to be derived, just computed by hand once and hardcoded.
    # That let A_P and R_P silently drift apart the moment either one was
    # overridden without remembering to update the other (Ai_P below would
    # then mix two different implied radii -- confirmed to have actually
    # happened at least once: overriding R_P alone while A_P sat at its
    # stale default inflated Ai_P by exactly R_new/R_old). Derived here
    # instead, so this class of bug can't recur.
    if A_P is not None:
        raise TypeError(
            "A_P is no longer a settable parameter -- it's mechanically "
            "determined by R_P (A_P = 2.0/R_P for a cylindrical fibre) and "
            "specifying it independently risks it silently disagreeing with "
            "R_P. Remove the A_P= argument; only set R_P."
        )
    A_P = 2.0 / R_P   # mm^-1
    # Edge conductance weights  w = sigma * face_area / h
    w_pp  = sigma_P * S_P / dx_p           # mS   Purkinje-Purkinje  (eq. 11)
    # NOTE: myocardium-myocardium weights are no longer a single scalar --
    # build_myo_mesh_anisotropic below returns a per-edge array (myo["edge_w"])
    # built from sigma_l/sigma_t + fibre_dir, replacing the isotropic
    # `w_mm = sigma_M * voxel_size` line that was here.
    # Membrane loading  D[i] = A_i * cell_volume_i
    #   Purkinje cell volume = S_P * dx_p  (cross-section x segment length)
    #   [FIX M1] was A_P * dx_p — omitted S_P (~7e-4 mm^2), inflating
    #            D_P by 1/S_P ~ 127x.  SIGMA_P corrected by same factor.
    Ai_P  = A_P * S_P * dx_p              # mm^-1 * mm^2 * mm = mm^2
    Ai_M  = A_M * voxel_size ** 3         # mm^-1 * mm^3 = mm^2  (unchanged)

    # ── Myocardium mesh (18-connected, anisotropic edge weights) ─────────────
    print("Building myocardium mesh (anisotropic, 18-connected)...")
    myo   = build_myo_mesh_anisotropic(S, Z, voxel_size, fibre_dir, sigma_l, sigma_t)
    N_myo = myo["N_myo"]

    # ── Extract phi for each myocardial node ──────────────────────────────────
    # myo["vox_idx"] is (N_myo, 3) — voxel coordinates of each node.
    # phi is a (Nx,Ny,Nz) volumetric array; index it with vox_idx.
    vox = myo["vox_idx"]
    phi_myo = phi[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.float32)
    # Replace NaN (outside mask) with mid-wall value so they don't skew assignment
    phi_myo = np.where(np.isfinite(phi_myo), phi_myo, 0.5)

    # ── Heterogeneous ionic solver (built here; needs Np from Purkinje step) ──
    # Defer construction until after Np is known (below).

    # ── Surface mesh (marching cubes, for animation output) ───────────────────
    surf_verts, surf_faces, vert_to_surf_node = build_surface_mesh(
        Z, myo, voxel_size)
    N_surf = len(surf_verts)

    # ── Clip Purkinje edges to outside Z ─────────────────────────────────────
    comp_edges_clipped = clip_purkinje_outside_Z(
        comp_nodes_p, comp_edges_p, Z, voxel_size)

    # ── PMJ: find terminal comp-nodes, map to myocardial patches ─────────────
    terminal_orig = _find_terminals(elements, len(nodes))
    terminal_set  = set(terminal_orig)
    pmj_comp_ids  = sorted({
        bmap[-1]
        for i, bmap in enumerate(branch_map)
        if elements[i][1] in terminal_set})
    if not pmj_comp_ids:
        pmj_comp_ids = [Np - 1]

    pmj_patches = map_pmj(
        comp_nodes_p, pmj_comp_ids, myo, n_pmj=n_pmj)
    print(f"PMJ      : {len(pmj_comp_ids)} sites, "
          f"{sum(len(p) for p in pmj_patches)} patch nodes, c_pmj={c_pmj} mS")

    # ── Build HeteroIonicSolver (Np known here) ───────────────────────────────
    ionic = HeteroIonicSolver(
        Np           = Np,
        phi_myo      = phi_myo,
        device       = dev,
        dtype        = dtype,
        phi_endo_max = phi_endo_max,
        phi_epi_min  = phi_epi_min,
    )

    # ── Single global Laplacian (Purkinje + myocardium + PMJ) ────────────────
    # PMJ coupling is embedded directly in L as off-diagonal K entries, so
    # the diffusion solve propagates Purkinje↔myocardium coupling implicitly.
    # L is non-symmetric (Ai_P ≠ Ai_M) → BiCGSTAB required (not CG).
    #
    # Use the FULL comp_edges_p here (not comp_edges_clipped).
    # comp_edges_clipped only removes edges for visualisation (so the animator
    # doesn't draw Purkinje lines that pass through the myocardium volume).
    # Using clipped edges in the Laplacian severs the 1D cable wherever it
    # enters Z, creating isolated nodes with no diffusive neighbours — those
    # nodes oscillate (ionic forcing with no smoothing), producing the peaks
    # and valleys visible in the image.
    L_global, diag_L_global = _build_global_laplacian_gpu(
        Np, N_myo,
        comp_edges_p, w_pp, Ai_P,       # ← full edges, not clipped
        myo["edges"],  myo["edge_w"], Ai_M,
        pmj_comp_ids, pmj_patches, c_pmj,
        dev,
        dtype,
    )
    N_global = Np + N_myo

    # ── CN system matrix A = (Cm/dt)I − theta*L  (built once, reused every step)
    # Matches solver.py build_cn_matrix().  A_csr is CSR for fast torch.mv().
    cn_scale        = Cm / dt
    A_csr, M_inv    = _build_cn_matrix(L_global, diag_L_global, cn_scale, theta)
    one_minus_theta = 1.0 - theta

    # ── Timing ───────────────────────────────────────────────────────────────
    n_steps    = int(T / dt)
    if frame_save_dt is not None:
        save_every = max(1, int(round(frame_save_dt / dt)))
        print(f"  Animation frames : every {save_every} steps "
              f"= {save_every * dt:.4g} ms  (requested {frame_save_dt} ms)")
    else:
        save_every = max(1, n_steps // n_frames)

    # ── Ionic state — initialised by HeteroIonicSolver ───────────────────────
    states = ionic.init_states(N_global)

    # ── Stimulus protocol — one or more independent sites ───────────────────
    if stim_protocol is None:
        stim_protocol = [
            dict(name="av_node", target="root", amp=12.0, dur=2.0, onset=0.0),
        ]
    stim_sites  = _resolve_stim_sites(
        stim_protocol, stim_mask_np, ectopic_region, vox,
        Np, N_myo, dev, dtype, stim_regions=stim_regions,
    )
    I_stim_buf = torch.zeros(N_global, dtype=dtype, device=dev)   # reused every step

    # ── Frame storage (on-disk memmap) ─────────────────────────────────────────
    # Same rationale as the Vm snapshot buffer below: at a fine frame_save_dt
    # over a long T, a Python list + np.stack at the end needs 2-3x the final
    # array size in RAM at the moment of stacking. Writing straight into a
    # pre-sized on-disk memmap keeps RAM flat regardless of frame count.
    branch_bmaps       = [np.array(bm, dtype=np.int32) for bm in branch_map]
    _n_anim_frames_max = (n_steps + save_every - 1) // save_every
    _anim_tmp_dir = Path(anim_tmp_dir) if anim_tmp_dir is not None else \
        (Path(out_npz).parent if out_npz is not None else Path("."))
    _anim_tmp_dir.mkdir(parents=True, exist_ok=True)

    def _new_anim_memmap(n_cols: int):
        fh = tempfile.NamedTemporaryFile(
            dir=str(_anim_tmp_dir), suffix=".anim_scratch.npy", delete=False)
        tmp_path = Path(fh.name)
        fh.close()
        arr = np.lib.format.open_memmap(
            str(tmp_path), mode="w+", dtype=np.float32,
            shape=(_n_anim_frames_max, n_cols))
        return arr, tmp_path

    frames_surf, _surf_tmp_path = _new_anim_memmap(len(myo["surface_node_ids"]))
    branch_frames_p, _branch_tmp_paths = [], []
    for bmap in branch_bmaps:
        arr, tmp_path = _new_anim_memmap(len(bmap))
        branch_frames_p.append(arr)
        _branch_tmp_paths.append(tmp_path)
    times           = []
    _anim_frame_count = 0

    # ── Vm volumetric snapshot setup ─────────────────────────────────────────
    # Independent sampling rate: one full-myo Vm snapshot every vm_save_dt ms.
    # vm_save_every is the number of solver steps between consecutive snapshots.
    # myo_node_ids[i] = Np + i  (global index in the [Purkinje | myo] vector).
    #
    # Frames are written directly into a pre-sized on-disk memmap (rather than
    # accumulated in a Python list) so RAM usage stays flat regardless of run
    # length/resolution: for a large N_myo x n_frames field, holding the full
    # history in a list-then-np.stack pattern requires 2-3x the final array
    # size in RAM at the moment of stacking/saving, which is what caused the
    # previous OOM at the very end of long runs. The memmap is written to a
    # scratch .npy on disk during the loop, then streamed (not re-loaded
    # wholesale) into the final compressed .npz by _save_vm_snapshots.
    _run_vm = (vm_save_dt is not None) and (out_vm_npz is not None)
    if _run_vm:
        vm_save_every = max(1, int(round(vm_save_dt / dt)))
        # Actual achieved interval (dt rounding)
        _actual_vm_dt = vm_save_every * dt
        print(f"  Vm snapshots : every {vm_save_every} steps "
              f"= {_actual_vm_dt:.4g} ms  (requested {vm_save_dt} ms)")
        _n_vm_frames_max = (n_steps + vm_save_every - 1) // vm_save_every
        _vm_tmp_dir = Path(vm_tmp_dir) if vm_tmp_dir is not None else \
            (Path(out_vm_npz).parent if out_vm_npz is not None else Path("."))
        _vm_tmp_dir.mkdir(parents=True, exist_ok=True)
        _vm_tmp_fh = tempfile.NamedTemporaryFile(
            dir=str(_vm_tmp_dir), suffix=".vm_scratch.npy", delete=False)
        _vm_tmp_path = Path(_vm_tmp_fh.name)
        _vm_tmp_fh.close()
        vm_frames = np.lib.format.open_memmap(
            str(_vm_tmp_path), mode="w+", dtype=np.float32,
            shape=(_n_vm_frames_max, N_myo))
        vm_times:  list[float] = []
        _vm_frame_count = 0
        # Pre-build coordinate arrays once (static throughout the run)
        _myo_node_ids = (np.arange(N_myo, dtype=np.int32) + Np)  # global ids
        _myo_coords   = myo["nodes_mm"]                           # (N_myo,3) float32
        _myo_vox_idx  = myo["vox_idx"]                            # (N_myo,3) int32
    else:
        vm_save_every = 0

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Coupled Purkinje + Myocardium  |  Transmural Heterogeneous TTP06")
    print(f"  Regions   : {ionic.region_sizes}")
    print(f"  Purkinje  : {Np:,} nodes   dx={dx_p} mm   sigma_P={sigma_P} mS/mm")
    print(f"  Myocardium: {N_myo:,} nodes   dx={voxel_size} mm   "
          f"sigma_l={sigma_l} mS/mm   sigma_t={sigma_t} mS/mm   "
          f"(anisotropy ratio {sigma_l/sigma_t:.2f})")
    print(f"  Surface   : {N_surf:,} nodes")
    print(f"  Stimulus sites:")
    for site in stim_sites:
        rep = (f"every {site['period']} ms" if site["period"]
               else "single pulse")
        npulse = "" if site["n_pulses"] is None else f", n_pulses={site['n_pulses']}"
        print(f"    [{site['name']}] target={site['target']}  "
              f"nodes={site['n_nodes']}  amp={site['amp']} uA/mm^2  "
              f"dur={site['dur']} ms  onset={site['onset']} ms  ({rep}{npulse})")
    print(f"  dt={dt} ms   T={T} ms   steps={n_steps:,}")
    print(f"  Device    : {dev}")
    print(f"{'='*62}\n")

    # ── Time loop — Lie-Trotter operator splitting ────────────────────────────
    #
    #   Step 1 (ionic, explicit):
    #       I_ion        = ionic_current(V^n, gates^n)
    #       gates^{n+1}  = RushLarsen(V^n, gates^n, dt)    ← on V^n
    #       V*           = V^n − (dt/Cm)*I_ion + (dt/Cm)*I_app
    #
    #   Step 2 (diffusion, implicit CN):
    #       b      = (Cm/dt)*V* + (1−theta)*L*V*
    #       A V^{n+1} = b          [A pre-built as CSR, one BiCGSTAB solve]
    #
    # One solve per step (vs three in Strang).  PMJ coupling is embedded in
    # L_global so it is handled correctly in the diffusion step with no
    # separate correction term.
    with tqdm(total=n_steps, desc="Coupled sim", unit="step",
              dynamic_ncols=True) as bar:

        for step_i in range(n_steps):
            t = step_i * dt

            # Sum every currently-firing site's precomputed (mask*amp) tensor.
            # Cheap even for many sites: a handful of full-length adds per step.
            I_stim_buf.zero_()
            for site in stim_sites:
                if _stim_active(t, site["onset"], site["dur"],
                                 site["period"], site["n_pulses"]):
                    I_stim_buf += site["amp_tensor"]
            I_stim = I_stim_buf

            V = states["V"]

            # ── Step 1: ionic update (explicit) ──────────────────────────────
            # EndoTTP06Torch conductances are in nS/pF (TTP06 convention).
            # With V in mV this gives I_ion in pA/pF ≡ µA/µF ≡ mV/ms

            # (current normalised by the ionic model's own Cm = 0.185 µF/cm²).
            #
            # To place I_ion in the same uA/mm² units as I_stim and the PDE:
            #   I_ion [uA/mm²] = I_ion [mV/ms] * Cm [uF/mm²]
            #                  = I_ion [mV/ms] * 0.01
            #
            # Then the unified voltage update is:
            #   V* = V^n + (dt/Cm) * (−I_ion_pde + I_stim)
            #
            # with all terms in uA/mm² and Cm = 0.01 uF/mm², giving mV.
            # stimulus amplitudes are already specified in uA/mm^2 so no conversion needed.
            Iion, states = ionic.step(states, dt, None)
            Iion_pde = Iion * Cm                    # mV/ms → uA/mm²
            V_star   = V + (dt / Cm) * (-Iion_pde + I_stim)
            states["V"] = V_star

            # ── Step 2: diffusion (implicit Crank-Nicolson) ───────────────────
            # b = (Cm/dt)*V* + (1−theta)*L*V*   (single COO matvec on L_global)
            # A*V^{n+1} = b solved with BiCGSTAB (A is CSR, pre-built).
            # Matches solver.py _step() Step 2.
            Lv_star = torch.sparse.mm(L_global, V_star.unsqueeze(1)).squeeze(1)
            b       = cn_scale * V_star + one_minus_theta * Lv_star
            V_new   = _bicgstab(A_csr, M_inv, b, V_star, cg_tol, cg_max_iter)
            V_new   = V_new.clamp(-150., 80.)
            states["V"] = V_new

            # ── Save frames ───────────────────────────────────────────────────
            if step_i % save_every == 0:
                V_cpu   = V_new.cpu().numpy()
                V_p_cpu = V_cpu[:Np]
                V_m_cpu = V_cpu[Np:]
                for b, bmap in enumerate(branch_bmaps):
                    branch_frames_p[b][_anim_frame_count] = V_p_cpu[bmap].astype(np.float32)
                frames_surf[_anim_frame_count] = \
                    V_m_cpu[myo["surface_node_ids"]].astype(np.float32)
                times.append(t)
                _anim_frame_count += 1

            # ── Vm volumetric snapshot ────────────────────────────────────────
            if _run_vm and (step_i % vm_save_every == 0):
                # Slice only the myocardial portion [Np:] to keep memory tight.
                # .cpu() is a no-op when already on CPU. Written straight to
                # the on-disk memmap -- no growing in-RAM list.
                vm_frames[_vm_frame_count] = V_new[Np:].cpu().numpy().astype(np.float32)
                vm_times.append(t)
                _vm_frame_count += 1

            if step_i % max(1, n_steps // 400) == 0:
                bar.set_postfix(
                    t    = f"{t:.1f}",
                    Vp_max = f"{V_new[:Np].max().item():.0f}",
                    Vm_max = f"{V_new[Np:].max().item():.0f}",
                    refresh=False)
            bar.update(1)

    # ── Save Vm volumetric snapshots ──────────────────────────────────────────
    if _run_vm and _vm_frame_count > 0:
        vm_frames.flush()
        # Trim to the frames actually written (n_steps rounding can leave the
        # preallocated buffer 1 row longer than what was filled). This is a
        # view, not a copy -- still backed by the on-disk memmap.
        _vm_data = vm_frames[:_vm_frame_count]
        _save_vm_snapshots(
            path       = out_vm_npz,
            node_ids   = _myo_node_ids,
            coords_mm  = _myo_coords,
            vox_idx    = _myo_vox_idx,
            Vm         = _vm_data,               # memmap view (n_frames, N_myo)
            times      = np.array(vm_times, dtype=np.float32),
            dt_solver  = dt,
            vm_save_dt = vm_save_every * dt,
        )
        del vm_frames, _vm_data
        gc.collect()   # release the memmap's OS file handle (needed on Windows
                        # before the scratch file can be deleted)
        try:
            _vm_tmp_path.unlink()
        except OSError:
            pass   # scratch file cleanup is best-effort; not fatal

    # ── Trim animation frame buffers to what was actually written ──────────────
    # (n_steps rounding can leave the preallocated buffer 1 row longer than
    # what was filled.) Views, not copies -- still backed by the on-disk
    # memmaps until _save_npz streams them out.
    frames_surf.flush()
    for arr in branch_frames_p:
        arr.flush()
    frames_surf     = frames_surf[:_anim_frame_count]
    branch_frames_p = [arr[:_anim_frame_count] for arr in branch_frames_p]

    # ── Pack results ──────────────────────────────────────────────────────────
    results = dict(
        comp_nodes        = comp_nodes_p,
        comp_edges        = comp_edges_clipped,
        branch_map        = branch_map,
        branch_frames_p   = branch_frames_p,
        surf_verts        = surf_verts,
        surf_faces        = surf_faces,
        vert_to_surf_node = vert_to_surf_node,
        frames_surf       = frames_surf,
        times             = times,
    )

    if out_npz is not None:
        _save_npz(results, out_npz)

    # NOTE: unlike the Vm snapshot buffer, frames_surf/branch_frames_p are
    # part of the returned `results` dict (pre-existing behaviour, kept for
    # backward compatibility) -- so, unlike vm_frames, their scratch files
    # are deliberately NOT deleted here: results["frames_surf"] etc. are
    # memmap views still backed by them. They live in anim_tmp_dir (default:
    # next to out_npz) as *.anim_scratch.npy and are yours to delete once
    # you're done with `results` -- they are not cleaned up automatically.

    return results


# =============================================================================
# NPZ writer  (schema matches animate_coupled.py exactly)
# =============================================================================

def _save_npz(results: dict, path: str | Path) -> None:
    path = Path(path).with_suffix(".npz")

    save_dict: dict[str, np.ndarray] = {
        "time":              np.array(results["times"],          dtype=np.float32),
        "comp_nodes":        results["comp_nodes"],
        "comp_edges":        results["comp_edges"],
        "surf_verts":        results["surf_verts"],
        "surf_faces":        results["surf_faces"],
        "vert_to_surf_node": results["vert_to_surf_node"],
        # frames_surf/branch_* are already (n_frames, n_cols) float32 arrays
        # (memmap views, in the long-run/frame_save_dt case) -- np.asarray
        # with copy=False is a no-op when the dtype already matches, same
        # streaming rationale as _save_vm_snapshots below, so a multi-GiB
        # animation buffer isn't doubled in RAM here.
        "surf_frames":       np.asarray(results["frames_surf"], dtype=np.float32, order="C"),
    }

    for b, bmap in enumerate(results["branch_map"]):
        save_dict[f"bmap_{b}"]   = np.array(bmap, dtype=np.int32)
        save_dict[f"branch_{b}"] = np.asarray(
            results["branch_frames_p"][b], dtype=np.float32, order="C")

    np.savez_compressed(str(path), **save_dict)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"\nNPZ -> {path}  ({size_mb:.1f} MB)")
    print(f"  frames       : {len(results['times'])}")
    print(f"  Pkn branches : {len(results['branch_map'])}")
    print(f"  surf_frames  : {save_dict['surf_frames'].shape}\n")


# =============================================================================
# Vm volumetric snapshot writer
# =============================================================================

def _save_vm_snapshots(
    path:       str | Path,
    node_ids:   np.ndarray,   # (N_myo,)       int32  — global solver index Np+i
    coords_mm:  np.ndarray,   # (N_myo, 3)     float32 — physical xyz [mm]
    vox_idx:    np.ndarray,   # (N_myo, 3)     int32  — voxel ijk
    Vm:         np.ndarray,   # (n_frames, N_myo) float32 — membrane voltage [mV]
    times:      np.ndarray,   # (n_frames,)    float32 — simulation time [ms]
    dt_solver:  float,        # solver time step [ms]
    vm_save_dt: float,        # actual snapshot interval [ms]
) -> None:
    """
    Write myocardial Vm snapshots to a compressed NPZ.

    NPZ arrays
    ----------
    node_ids   (N_myo,)             int32   Global index in the coupled solver
                                            vector: node_ids[i] = Np + i.
                                            Use to cross-reference Purkinje nodes.
    coords_mm  (N_myo, 3)           float32 Physical coordinates [mm] (x, y, z).
    vox_idx    (N_myo, 3)           int32   Voxel indices (i, j, k) in the
                                            original mask array.
    Vm         (n_frames, N_myo)    float32 Membrane voltage [mV].
                                            Vm[t_idx, node_i] = voltage at
                                            node node_ids[node_i] at time
                                            time_vm[t_idx].
    time_vm    (n_frames,)          float32 Simulation time [ms] of each frame.
    dt_solver  scalar (float32)     Solver time step [ms].
    vm_save_dt scalar (float32)     Actual snapshot sampling interval [ms]
                                    (= vm_save_every * dt_solver).

    Loading example
    ---------------
    >>> import numpy as np
    >>> d = np.load("vm_snapshots.npz")
    >>> coords  = d["coords_mm"]    # (N_myo, 3)
    >>> Vm      = d["Vm"]           # (n_frames, N_myo)
    >>> t       = d["time_vm"]      # (n_frames,)
    >>> # voltage at node 42 over time:
    >>> v42 = Vm[:, 42]
    >>> # spatial Vm field at frame 10:
    >>> field_t10 = Vm[10, :]
    """
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: Vm may be a large (n_frames, N_myo) array, often backed by an
    # on-disk memmap. `.astype(np.float32)` always copies, even when the
    # dtype already matches -- for a multi-GiB array that copy is exactly
    # what caused OOMs on long/high-resolution runs. `np.asarray(..., copy=False)`
    # is a no-op when the dtype already matches, so np.savez_compressed can
    # stream the data straight from disk/memory without ever materializing
    # a second full copy.
    np.savez_compressed(
        str(path),
        node_ids   = node_ids.astype(np.int32),
        coords_mm  = coords_mm.astype(np.float32),
        vox_idx    = vox_idx.astype(np.int32),
        Vm         = np.asarray(Vm, dtype=np.float32, order="C"),
        time_vm    = times.astype(np.float32),
        dt_solver  = np.float32(dt_solver),
        vm_save_dt = np.float32(vm_save_dt),
    )

    n_frames, N_myo = Vm.shape
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"\nVm snapshots -> {path}  ({size_mb:.1f} MB)")
    print(f"  nodes        : {N_myo:,}  (myocardium only)")
    print(f"  frames       : {n_frames}  "
          f"(every {vm_save_dt:.4g} ms,  t = {float(times[0]):.2f} … "
          f"{float(times[-1]):.2f} ms)")
    print(f"  Vm range     : [{float(Vm.min()):.1f}, {float(Vm.max()):.1f}] mV\n")
