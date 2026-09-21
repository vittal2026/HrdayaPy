"""
ionic_models/gpu_ttp06.py
==========================
GPU-native TTP06 phenotype classes for use in the coupled solver.

Three classes are provided, each fully self-contained (no shared base class):

    EndoTTP06     — endocardial phenotype
    MidTTP06      — mid-myocardial (M-cell) phenotype
    EpiTTP06      — epicardial phenotype

Each satisfies the RegionIonicModel protocol:
    name                            str
    state_names()                   list[str]
    init_state_tensor(N, dev, dt)   (N, n_states) Tensor
    step(V, state, dt, I_stim)      (I_ion, state)

Conductance values (ten Tusscher & Panfilov 2006, Table 1)
--------------------------------------------------------
    Conductance     Endo        MCell       Epi
    g_Ks (nS/pF)   0.392       0.098       0.392
    g_to (nS/pF)   0.073       0.294       0.294    ← NOTE: M-cell shares
                                                       epicardium's large
                                                       g_to, not endocardium's
                                                       small one; see below
    g_Kr (nS/pF)   0.153       0.153       0.153    (shared)
    g_CaL          3.98e-5     3.98e-5     3.98e-5  (shared)

NOTE on g_to
-------------
The original TTP06 paper (ten Tusscher & Panfilov 2006, Am J Physiol) Table 1
gives g_to_epi = g_to_Mcell = 0.294 nS/pF (M-cell shares epicardium's large
Ito) and g_to_endo = 0.073 nS/pF, which produces the prominent phase-1 notch
in epicardial and M-cell -- not endocardial -- cells. An earlier version of
this file incorrectly gave M-cell the same (much smaller) g_to as
endocardium, and additionally scaled both g_Ks and g_to for Endo/M-cell by
factors mis-attributed to "Romero et al. 2009", which do not appear in the
2006 paper's Table 1. All three phenotypes now use the Table 1 values
directly, unscaled. The value 0.073 for g_to,endo (not to be confused with
the unrelated 0.073 CellML-export artefact discussed in some other TTP06
implementations, which stems from the superseded 2004 model's epicardial
value) is the correct endocardial figure per Table 1.

NOTE on s-gate kinetics
------------------------
The s-gate (Ito inactivation) differs between phenotypes:
    Endo / M-cell:  s_inf = 1/(1+exp((V+28)/5))
                    tau_s = 1000*exp(-(V+67)^2/1000) + 8
    Epi:            s_inf = 1/(1+exp((V+20)/5))
                    tau_s = 85*exp(-(V+45)^2/320) + 5/(1+exp((V-20)/5)) + 3

This is the key kinetic difference between phenotypes besides conductances.
This endo/M-cell vs. epi split in s-gate kinetics is asserted by this
module but was not re-verified against the Ito equations of the 2004
ten Tusscher model (which the 2006 paper's appendix references rather
than reproduces); confirm against that source, particularly for the
M-cell case now that M-cell uses epicardium's large g_to (see MidTTP06).

State vector layout (shared across phenotypes, 18 variables)
-------------------------------------------------------------
    col  0  Ki       col  9  d
    col  1  Nai      col 10  f
    col  2  Cai      col 11  f2
    col  3  Xr1      col 12  fCass
    col  4  Xr2      col 13  s
    col  5  Xs       col 14  r
    col  6  m        col 15  CaSR
    col  7  h        col 16  Rprime
    col  8  j        col 17  Cass

Unit convention
---------------
All conductances in nS/pF → currents in pA/pF ≡ µA/µF ≡ mV/ms.
The PDE caller multiplies I_ion [mV/ms] by Cm_PDE [µF/mm²] to obtain
[µA/mm²] before the voltage update.  V is always in mV.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor

# Single source of truth for phenotype conductances -- see that module's
# docstring for why this import exists (it's the fix for a real bug, not
# a style preference). Adjust this import path to wherever the constants
# module actually lives in the package (e.g.
# hrdayapy.simulation.functions.ionic_models.ttp06_phenotype_constants).
from .ttp06_phenotype_constants import TTP06_PHENOTYPES


# =============================================================================
# Shared physical constants (module-level, not repeated per class)
# =============================================================================

_R   = 8314.472
_T   = 310.0
_F   = 96485.3415
_RTF = _R * _T / _F        # ~26.713 mV
_FRT = _F / (_R * _T)      # ~0.03743 mV⁻¹

# Fixed extracellular concentrations
_Ko  = 5.4
_Nao = 140.0
_Cao = 2.0

# Fixed conductances (shared across phenotypes)
_g_Na   = 14.838
_g_K1   = 5.405
_g_bna  = 0.00029
_g_bca  = 0.000592
_g_pCa  = 0.1238
_g_pK   = 0.0146
_P_NaK  = 2.724
_K_mk   = 1.0
_K_mNa  = 40.0
_K_NaCa = 1000.0
_K_sat  = 0.1
_alpha  = 2.5
_gamma  = 0.35
_Km_Ca  = 1.38
_Km_Nai = 87.5
_P_kna  = 0.03
_K_pCa  = 0.0005

# SR / Ca dynamics (shared)
_k1_prime = 0.15
_k2_prime = 0.045
_k3       = 0.06
_k4       = 0.005
_EC       = 1.5
_max_sr   = 2.5
_min_sr   = 1.0
_V_rel    = 0.102
_V_xfer   = 0.0038
_K_up     = 0.00025
_V_leak   = 0.00036
_Vmax_up  = 0.006375
_Buf_c    = 0.2
_K_buf_c  = 0.001
_Buf_sr   = 10.0
_K_buf_sr = 0.3
_Buf_ss   = 0.4
_K_buf_ss = 0.00025
_V_c      = 0.016404
_V_sr     = 0.001094
_V_ss     = 0.00005468
_Cm_model = 1.0     # µF/cm² in TTP06 convention (dimensionless in pA/pF units)

# CellML reference initial conditions (endocardial; same starting point for all)
_INIT = {
    "Ki":     138.4,
    "Nai":    10.355,
    "Cai":    0.00013,
    "Xr1":    0.00448,
    "Xr2":    0.476,
    "Xs":     0.0087,
    "m":      0.00155,
    "h":      0.7573,
    "j":      0.7225,
    "d":      3.164e-5,
    "f":      0.8009,
    "f2":     0.9778,
    "fCass":  0.9953,
    "s":      0.3212,
    "r":      2.235e-8,
    "CaSR":   3.715,
    "Rprime": 0.9068,
    "Cass":   0.00036,
}

# Column order — must match state_names() in every class
_STATE_COLS = [
    "Ki", "Nai", "Cai", "Xr1", "Xr2", "Xs",
    "m",  "h",   "j",   "d",   "f",   "f2",
    "fCass", "s", "r",  "CaSR", "Rprime", "Cass",
]
_N_STATES = len(_STATE_COLS)
_COL = {name: i for i, name in enumerate(_STATE_COLS)}


# =============================================================================
# Shared kinetics (inline functions, no overhead)
# =============================================================================

def _rl(x: Tensor, x_inf: Tensor, tau: Tensor, dt: float) -> Tensor:
    """Rush-Larsen exponential gate integrator."""
    return x_inf - (x_inf - x) * torch.exp(-dt / tau)


def _init_tensor(N: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return (N, N_STATES) initialised state tensor."""
    state = torch.empty((N, _N_STATES), dtype=dtype, device=device)
    for name, col in _COL.items():
        state[:, col] = _INIT[name]
    return state


def _currents_and_gates(
    V:      Tensor,   # (N,)
    state:  Tensor,   # (N, N_STATES)
    dt:     float,
    g_Ks:   float,
    g_to:   float,
    g_Kr:   float  = 0.153,
    g_CaL: float   = 3.98e-5,
    endo_s: bool   = True,    # True → endo/mcell s-gate, False → epi s-gate
) -> tuple[Tensor, Tensor]:
    """
    Core TTP06 kinetics shared by all phenotypes.

    Parameters
    ----------
    endo_s : bool
        True  → endo/M-cell s-gate kinetics (V+28, tau long)
        False → epi s-gate kinetics          (V+20, tau short)

    Returns
    -------
    I_ion   : (N,)         total ionic current  [pA/pF = mV/ms]
    state   : (N, N_STATES) updated state tensor
    """
    Ki     = state[:, _COL["Ki"]]
    Nai    = state[:, _COL["Nai"]]
    Cai    = state[:, _COL["Cai"]]
    Cass   = state[:, _COL["Cass"]]
    CaSR   = state[:, _COL["CaSR"]]
    Xr1    = state[:, _COL["Xr1"]]
    Xr2    = state[:, _COL["Xr2"]]
    Xs     = state[:, _COL["Xs"]]
    m      = state[:, _COL["m"]]
    h      = state[:, _COL["h"]]
    j      = state[:, _COL["j"]]
    d      = state[:, _COL["d"]]
    f      = state[:, _COL["f"]]
    f2     = state[:, _COL["f2"]]
    fCass  = state[:, _COL["fCass"]]
    s      = state[:, _COL["s"]]
    r      = state[:, _COL["r"]]
    Rprime = state[:, _COL["Rprime"]]

    # ── Reversal potentials ───────────────────────────────────────────────────
    E_Na = _RTF * torch.log(torch.tensor(_Nao, dtype=V.dtype, device=V.device) / Nai)
    E_K  = _RTF * torch.log(torch.tensor(_Ko,  dtype=V.dtype, device=V.device) / Ki)
    E_Ks = _RTF * torch.log(
        (_Ko  + _P_kna * _Nao) / (Ki + _P_kna * Nai))
    E_Ca = (_RTF / 2.0) * torch.log(
        torch.tensor(_Cao, dtype=V.dtype, device=V.device) / Cai)

    # ── INa ───────────────────────────────────────────────────────────────────
    m_inf   = (1.0 / (1.0 + torch.exp((-56.86 - V) / 9.03))) ** 2
    alpha_m = 1.0 / (1.0 + torch.exp((-60.0  - V) / 5.0))
    beta_m  = (0.1 / (1.0 + torch.exp((V + 35.0) / 5.0))
               + 0.1 / (1.0 + torch.exp((V - 50.0) / 200.0)))
    tau_m   = alpha_m * beta_m

    h_inf   = (1.0 / (1.0 + torch.exp((V + 71.55) / 7.43))) ** 2
    alpha_h = torch.where(V < -40.0,
                          0.057 * torch.exp(-(V + 80.0) / 6.8),
                          torch.zeros_like(V))
    beta_h  = torch.where(V < -40.0,
                          2.7 * torch.exp(0.079 * V) + 3.1e5 * torch.exp(0.3485 * V),
                          0.77 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
    tau_h   = 1.0 / (alpha_h + beta_h + 1e-30)

    j_inf   = h_inf.clone()
    alpha_j = torch.where(
        V < -40.0,
        ((-25428.0 * torch.exp(0.2444 * V) - 6.948e-6 * torch.exp(-0.04391 * V))
         * (V + 37.78)) / (1.0 + torch.exp(0.311 * (V + 79.23))),
        torch.zeros_like(V))
    beta_j  = torch.where(
        V < -40.0,
        0.02424 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))),
        0.6 * torch.exp(0.057 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
    tau_j   = 1.0 / (alpha_j + beta_j + 1e-30)
    i_Na    = _g_Na * m**3 * h * j * (V - E_Na)

    # ── IKr ───────────────────────────────────────────────────────────────────
    xr1_inf = 1.0 / (1.0 + torch.exp((-26.0 - V) / 7.0))
    tau_xr1 = (450.0 / (1.0 + torch.exp((-45.0 - V) / 10.0))
               * 6.0  / (1.0 + torch.exp((V + 30.0) / 11.5)))
    xr2_inf = 1.0 / (1.0 + torch.exp((V + 88.0) / 24.0))
    tau_xr2 = (3.0  / (1.0 + torch.exp((-60.0 - V) / 20.0))
               * 1.12 / (1.0 + torch.exp((V - 60.0) / 20.0)))
    i_Kr    = g_Kr * torch.sqrt(
        torch.tensor(_Ko / 5.4, dtype=V.dtype, device=V.device)) * Xr1 * Xr2 * (V - E_K)

    # ── IKs ───────────────────────────────────────────────────────────────────
    xs_inf = 1.0 / (1.0 + torch.exp((-5.0 - V) / 14.0))
    axs    = 1400.0 / torch.sqrt(1.0 + torch.exp((5.0 - V) / 6.0))
    bxs    = 1.0 / (1.0 + torch.exp((V - 35.0) / 15.0))
    tau_xs = axs * bxs + 80.0
    i_Ks   = g_Ks * Xs**2 * (V - E_Ks)

    # ── IK1 ───────────────────────────────────────────────────────────────────
    ak1     = 0.1 / (1.0 + torch.exp(0.06 * (V - E_K - 200.0)))
    bk1     = ((3.0 * torch.exp(2e-4 * (V - E_K + 100.0))
                + torch.exp(0.1 * (V - E_K - 10.0)))
               / (1.0 + torch.exp(-0.5 * (V - E_K))))
    xk1_inf = ak1 / (ak1 + bk1)
    i_K1    = _g_K1 * torch.sqrt(
        torch.tensor(_Ko / 5.4, dtype=V.dtype, device=V.device)) * xk1_inf * (V - E_K)

    # ── Ito — s-gate kinetics differ by phenotype ─────────────────────────────
    if endo_s:
        # Endo / M-cell: slower inactivation, more depolarised half-inactivation
        s_inf = 1.0 / (1.0 + torch.exp((V + 28.0) / 5.0))
        tau_s = 1000.0 * torch.exp(-(V + 67.0)**2 / 1000.0) + 8.0
    else:
        # Epi: faster inactivation, produces prominent phase-1 notch
        s_inf = 1.0 / (1.0 + torch.exp((V + 20.0) / 5.0))
        tau_s = (85.0 * torch.exp(-(V + 45.0)**2 / 320.0)
                 + 5.0 / (1.0 + torch.exp((V - 20.0) / 5.0)) + 3.0)

    r_inf = 1.0 / (1.0 + torch.exp((20.0 - V) / 6.0))
    tau_r = 9.5 * torch.exp(-(V + 40.0)**2 / 1800.0) + 0.8
    i_to  = g_to * r * s * (V - E_K)

    # ── ICaL ──────────────────────────────────────────────────────────────────
    d_inf = 1.0 / (1.0 + torch.exp((-8.0 - V) / 7.5))
    tau_d = ((1.4 / (1.0 + torch.exp((-35.0 - V) / 13.0)) + 0.25)
             * 1.4 / (1.0 + torch.exp((V + 5.0) / 5.0))
             + 1.0 / (1.0 + torch.exp((50.0 - V) / 20.0)))
    f_inf  = 1.0 / (1.0 + torch.exp((V + 20.0) / 7.0))
    tau_f  = (1102.5 * torch.exp(-((V + 27.0)**2) / 225.0)
              + 200.0 / (1.0 + torch.exp((13.0 - V) / 10.0))
              + 180.0 / (1.0 + torch.exp((V + 30.0) / 10.0)) + 20.0)
    f2_inf = 0.67 / (1.0 + torch.exp((V + 35.0) / 7.0)) + 0.33
    tau_f2 = (562.0 * torch.exp(-((V + 27.0)**2) / 240.0)
              + 31.0 / (1.0 + torch.exp((25.0 - V) / 10.0))
              + 80.0 / (1.0 + torch.exp((V + 30.0) / 10.0)))
    fCass_inf = 0.6 / (1.0 + (Cass / 0.05)**2) + 0.4
    tau_fCass = 80.0 / (1.0 + (Cass / 0.05)**2) + 2.0

    exp2V  = torch.exp(2.0 * (V - 15.0) * _FRT)
    denom  = exp2V - 1.0
    denom  = torch.where(denom.abs() < 1e-10, torch.full_like(denom, 1e-10), denom)
    i_CaL  = (g_CaL * d * f * f2 * fCass
               * 4.0 * (V - 15.0) * _F**2 / (_R * _T)
               * (0.25 * Cass * exp2V - _Cao) / denom)

    # ── Background, pump, exchanger ───────────────────────────────────────────
    i_bNa = _g_bna * (V - E_Na)
    i_bCa = _g_bca * (V - E_Ca)
    i_pCa = _g_pCa * Cai / (Cai + _K_pCa)
    i_pK  = _g_pK  * (V - E_K) / (1.0 + torch.exp((25.0 - V) / 5.98))

    i_NaK = (_P_NaK * _Ko / (_Ko + _K_mk) * Nai / (Nai + _K_mNa)
             / (1.0 + 0.1245 * torch.exp(-0.1 * V * _FRT)
                    + 0.0353 * torch.exp(-V      * _FRT)))

    eVg   = torch.exp( _gamma       * V * _FRT)
    eVgm1 = torch.exp((_gamma - 1.) * V * _FRT)
    i_NaCa = (_K_NaCa
              * (eVg   * Nai**3 * _Cao - eVgm1 * _Nao**3 * Cai * _alpha)
              / ((_Km_Nai**3 + _Nao**3) * (_Km_Ca + _Cao)
                 * (1.0 + _K_sat * eVgm1)))

    # ── Total ionic current ───────────────────────────────────────────────────
    I_ion = (i_K1 + i_to + i_Kr + i_Ks + i_CaL + i_NaK
             + i_Na + i_bNa + i_NaCa + i_bCa + i_pCa + i_pK)

    # ── Ca2+ SR dynamics ──────────────────────────────────────────────────────
    kcasr  = _max_sr - (_max_sr - _min_sr) / (1.0 + (_EC / CaSR)**2)
    k1     = _k1_prime / kcasr
    k2     = _k2_prime * kcasr
    O      = k1 * Cass**2 * Rprime / (_k3 + k1 * Cass**2 + 1e-30)
    i_rel  = _V_rel  * O          * (CaSR - Cass)
    i_up   = _Vmax_up / (1.0 + _K_up**2 / (Cai**2 + 1e-30))
    i_leak = _V_leak  * (CaSR - Cai)
    i_xfer = _V_xfer  * (Cass - Cai)

    bufc  = 1.0 / (1.0 + _Buf_c  * _K_buf_c  / (Cai  + _K_buf_c )**2)
    bufsr = 1.0 / (1.0 + _Buf_sr * _K_buf_sr / (CaSR + _K_buf_sr)**2)
    bufss = 1.0 / (1.0 + _Buf_ss * _K_buf_ss / (Cass + _K_buf_ss)**2)

    # ── Concentration updates (Forward Euler) ─────────────────────────────────
    dNai  = -(i_Na + i_bNa + 3.0 * i_NaK + 3.0 * i_NaCa) * _Cm_model / (_V_c * _F)
    dKi   = -(i_K1 + i_to + i_Kr + i_Ks + i_pK - 2.0 * i_NaK) * _Cm_model / (_V_c * _F)
    dCai  = bufc  * ((i_leak - i_up) * _V_sr / _V_c + i_xfer
                     - (i_bCa + i_pCa - 2.0 * i_NaCa) * _Cm_model / (2.0 * _V_c * _F))
    dCaSR = bufsr * (i_up - i_rel - i_leak)
    dCass = bufss * (-i_CaL * _Cm_model / (2.0 * _V_ss * _F)
                     + i_rel * _V_sr / _V_ss - i_xfer * _V_c / _V_ss)

    # ── Gate updates (Rush-Larsen) ────────────────────────────────────────────
    new_state = state.clone()
    new_state[:, _COL["m"]]      = _rl(m,     m_inf,     tau_m,     dt)
    new_state[:, _COL["h"]]      = _rl(h,     h_inf,     tau_h,     dt)
    new_state[:, _COL["j"]]      = _rl(j,     j_inf,     tau_j,     dt)
    new_state[:, _COL["Xr1"]]    = _rl(Xr1,   xr1_inf,   tau_xr1,   dt)
    new_state[:, _COL["Xr2"]]    = _rl(Xr2,   xr2_inf,   tau_xr2,   dt)
    new_state[:, _COL["Xs"]]     = _rl(Xs,    xs_inf,    tau_xs,    dt)
    new_state[:, _COL["d"]]      = _rl(d,     d_inf,     tau_d,     dt)
    new_state[:, _COL["f"]]      = _rl(f,     f_inf,     tau_f,     dt)
    new_state[:, _COL["f2"]]     = _rl(f2,    f2_inf,    tau_f2,    dt)
    new_state[:, _COL["fCass"]]  = _rl(fCass, fCass_inf, tau_fCass, dt)
    new_state[:, _COL["s"]]      = _rl(s,     s_inf,     tau_s,     dt)
    new_state[:, _COL["r"]]      = _rl(r,     r_inf,     tau_r,     dt)

    # Rprime: forward Euler
    new_state[:, _COL["Rprime"]] = (Rprime
                                     + dt * (-k2 * Cass * Rprime + _k4 * (1.0 - Rprime)))

    # Concentrations: forward Euler, clamped to physical range
    new_state[:, _COL["Nai"]]  = (Nai  + dt * dNai).clamp(min=1e-3)
    new_state[:, _COL["Ki"]]   = (Ki   + dt * dKi ).clamp(min=1e-3)
    new_state[:, _COL["Cai"]]  = (Cai  + dt * dCai ).clamp(min=1e-7)
    new_state[:, _COL["CaSR"]] = (CaSR + dt * dCaSR).clamp(min=1e-7)
    new_state[:, _COL["Cass"]] = (Cass + dt * dCass).clamp(min=1e-7)

    return I_ion, new_state


# =============================================================================
# Public phenotype classes
# =============================================================================

class EndoTTP06:
    """
    Endocardial TTP06  (ten Tusscher & Panfilov 2006, Table 1, endo s-gate).

    APD ≈ 290 ms at BCL 1000 ms [unverified estimate, see Results].
    No phase-1 notch (small Ito).
    """
    name = "TTP06_Endo"

    _c = TTP06_PHENOTYPES["endo"]
    _g_Ks, _g_to, _g_Kr, _g_CaL, _endo_s = _c.g_Ks, _c.g_to, _c.g_Kr, _c.g_CaL, _c.endo_s

    def state_names(self) -> list[str]:
        return list(_STATE_COLS)

    def init_state_tensor(self, N: int, device: torch.device,
                          dtype: torch.dtype) -> Tensor:
        return _init_tensor(N, device, dtype)

    def step(self, V: Tensor, state: Tensor,
             dt: float, I_stim: Tensor | float) -> tuple[Tensor, Tensor]:
        return _currents_and_gates(
            V, state, dt,
            g_Ks=self._g_Ks, g_to=self._g_to,
            g_Kr=self._g_Kr, g_CaL=self._g_CaL,
            endo_s=self._endo_s,
        )


class MidTTP06:
    """
    Mid-myocardial (M-cell) TTP06  (ten Tusscher & Panfilov 2006, Table 1).

    Reduced IKs relative to endo/epi -> longest APD of the three layers.
    Shares epicardium's large Ito conductance (Table 1), but uses
    endo-type (slow) s-gate kinetics per this module's existing gate
    assignment -- whether this combination reproduces a visible phase-1
    notch, or whether M-cell should instead use epi-type s-gate kinetics,
    depends on the Ito gating equations of the 2004 model (ten Tusscher,
    Noble, Noble & Panfilov, Am J Physiol 2004;286:H1573), which are
    referenced but not reproduced in the 2006 paper's appendix and were
    not available to verify this against directly. Confirm against the
    2004 paper before relying on the notch/no-notch claim.
    M-cells are the primary driver of T-wave morphology.
    """
    name = "TTP06_MCell"

    _c = TTP06_PHENOTYPES["mid"]
    _g_Ks, _g_to, _g_Kr, _g_CaL, _endo_s = _c.g_Ks, _c.g_to, _c.g_Kr, _c.g_CaL, _c.endo_s

    def state_names(self) -> list[str]:
        return list(_STATE_COLS)

    def init_state_tensor(self, N: int, device: torch.device,
                          dtype: torch.dtype) -> Tensor:
        return _init_tensor(N, device, dtype)

    def step(self, V: Tensor, state: Tensor,
             dt: float, I_stim: Tensor | float) -> tuple[Tensor, Tensor]:
        return _currents_and_gates(
            V, state, dt,
            g_Ks=self._g_Ks, g_to=self._g_to,
            g_Kr=self._g_Kr, g_CaL=self._g_CaL,
            endo_s=self._endo_s,
        )


class EpiTTP06:
    """
    Epicardial TTP06  (TTP06 2006 paper Table 1 conductances, epi s-gate).

    Largest IKs + Ito -> shortest APD of the three layers [specific
    ms figure removed pending validation, see Results]. Prominent
    phase-1 notch from the fast epi s-gate combined with large g_to.
    """
    name = "TTP06_Epi"

    _c = TTP06_PHENOTYPES["epi"]
    _g_Ks, _g_to, _g_Kr, _g_CaL, _endo_s = _c.g_Ks, _c.g_to, _c.g_Kr, _c.g_CaL, _c.endo_s

    def state_names(self) -> list[str]:
        return list(_STATE_COLS)

    def init_state_tensor(self, N: int, device: torch.device,
                          dtype: torch.dtype) -> Tensor:
        return _init_tensor(N, device, dtype)

    def step(self, V: Tensor, state: Tensor,
             dt: float, I_stim: Tensor | float) -> tuple[Tensor, Tensor]:
        return _currents_and_gates(
            V, state, dt,
            g_Ks=self._g_Ks, g_to=self._g_to,
            g_Kr=self._g_Kr, g_CaL=self._g_CaL,
            endo_s=self._endo_s,
        )
