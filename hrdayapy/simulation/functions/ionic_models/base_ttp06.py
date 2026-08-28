"""
base_ttp06.py
=============
Shared kinetics, physical constants, and core step function for the
ten Tusscher & Panfilov (2006) human ventricular ionic model.

This module is NOT imported directly by the solver.  It is imported by
the phenotype modules (endo.py, mcell.py, epi.py), each of which
supplies its own conductance constants and re-exports init_states / step.

Public API (same as original ttp06.py)
---------------------------------------
    states       = init_states(shape)
    Iion, states = step(states, dt, I_stim=0.,
                        g_Ks=..., g_to=..., g_Kr=..., g_CaL=...)

All conductances that differ between phenotypes are keyword arguments to
_step_core().  Phenotype modules call _step_core with their own values.
"""

import numpy as np

# ── Backend shim (swap for torch if needed) ───────────────────────────────────
_exp   = np.exp
_log   = np.log
_sqrt  = np.sqrt
_where = np.where

def _zeros(shape): return np.zeros(shape, dtype=np.float64)
def _full(shape, v): return np.full(shape, v, dtype=np.float64)

# ── Physical constants ────────────────────────────────────────────────────────
R   = 8314.472
T   = 310.0
F   = 96485.3415
Cm  = 1.0
V_c = 0.016404
V_sr= 0.001094
V_ss= 0.00005468
RTF = R * T / F

# ── Extracellular ion concentrations (fixed) ──────────────────────────────────
K_o  = 5.4
Na_o = 140.0
Ca_o = 2.0

# ── Conductances common to ALL phenotypes (not heterogeneous) ─────────────────
g_Na    = 14.838
g_K1    = 5.405
g_bna   = 0.00029
g_bca   = 0.000592
g_pCa   = 0.1238
g_pK    = 0.0146
P_NaK   = 2.724
K_mk    = 1.0
K_mNa   = 40.0
K_NaCa  = 1000.0
K_sat   = 0.1
alpha_n = 2.5
gamma   = 0.35
Km_Ca   = 1.38
Km_Nai  = 87.5
P_kna   = 0.03

# ── Ca2+ dynamics parameters (shared) ────────────────────────────────────────
k1_prime = 0.15
k2_prime = 0.045
k3       = 0.06
k4       = 0.005
EC       = 1.5
max_sr   = 2.5
min_sr   = 1.0
V_rel    = 0.102
V_xfer   = 0.0038
K_up     = 0.00025
V_leak   = 0.00036
Vmax_up  = 0.006375
Buf_c    = 0.2
K_buf_c  = 0.001
Buf_sr   = 10.0
K_buf_sr = 0.3
Buf_ss   = 0.4
K_buf_ss = 0.00025

# ── Resting initial values (CellML reference) ─────────────────────────────────
_V0      = -86.709
_Ki0     =  138.4
_Nai0    =  10.355
_Cai0    =  0.00013
_Xr1_0   =  0.00448
_Xr2_0   =  0.476
_Xs0     =  0.0087
_m0      =  0.00155
_h0      =  0.7573
_j0      =  0.7225
_Cass0   =  0.00036
_d0      =  3.164e-5
_f0      =  0.8009
_f2_0    =  0.9778
_fCass0  =  0.9953
_s0      =  0.3212
_r0      =  2.235e-8
_CaSR0   =  3.715
_Rprime0 =  0.9068


def _RL(x, x_inf, tau, dt):
    """Rush-Larsen exponential gate integrator."""
    return x_inf + (x - x_inf) * _exp(-dt / tau)


def init_states(shape):
    """
    Initialise all state variables to CellML resting values.
    shape : tuple, e.g. (N_nodes,)
    Returns a dict of float64 arrays.
    """
    c = lambda v: _full(shape, v)
    return {
        "V":      c(_V0),
        "Ki":     c(_Ki0),
        "Nai":    c(_Nai0),
        "Cai":    c(_Cai0),
        "Xr1":    c(_Xr1_0),
        "Xr2":    c(_Xr2_0),
        "Xs":     c(_Xs0),
        "m":      c(_m0),
        "h":      c(_h0),
        "j":      c(_j0),
        "Cass":   c(_Cass0),
        "d":      c(_d0),
        "f":      c(_f0),
        "f2":     c(_f2_0),
        "fCass":  c(_fCass0),
        "s":      c(_s0),
        "r":      c(_r0),
        "CaSR":   c(_CaSR0),
        "Rprime": c(_Rprime0),
    }


def _step_core(
    states, dt,
    I_stim  = 0.0,
    # ── Heterogeneous conductances — supplied by phenotype module ─────────────
    g_Ks    = 0.392,    # epi default
    g_to    = 0.073,    # epi default
    g_Kr    = 0.153,    # same across phenotypes in TTP06
    g_CaL   = 3.98e-5,  # same across phenotypes in TTP06
):
    """
    Core TTP06 step.  All kinetics live here.  Phenotype-specific
    conductances are passed as arguments — phenotype modules call this
    with their own constants.

    Parameters
    ----------
    states  : dict of arrays, modified in place and returned
    dt      : float (ms)
    I_stim  : array or scalar (pA/pF, positive = depolarising)
    g_Ks, g_to, g_Kr, g_CaL : phenotype conductances

    Returns
    -------
    Iion    : (shape,) float64  total ionic current (pA/pF), excl. stimulus
    states  : same dict, updated
    """
    V      = states["V"]
    Ki     = states["Ki"]
    Nai    = states["Nai"]
    Cai    = states["Cai"]
    Xr1    = states["Xr1"]
    Xr2    = states["Xr2"]
    Xs     = states["Xs"]
    m      = states["m"]
    h      = states["h"]
    j      = states["j"]
    Cass   = states["Cass"]
    d      = states["d"]
    f      = states["f"]
    f2     = states["f2"]
    fCass  = states["fCass"]
    s      = states["s"]
    r      = states["r"]
    CaSR   = states["CaSR"]
    Rprime = states["Rprime"]

    # ── Reversal potentials ───────────────────────────────────────────────────
    E_Na = RTF * _log(Na_o / Nai)
    E_K  = RTF * _log(K_o  / Ki)
    E_Ks = RTF * _log((K_o + P_kna * Na_o) / (Ki + P_kna * Nai))
    E_Ca = 0.5 * RTF * _log(Ca_o / Cai)

    # ── INa ───────────────────────────────────────────────────────────────────
    m_inf   = 1.0 / (1.0 + _exp((-56.86 - V) / 9.03)) ** 2
    alpha_m = 1.0 / (1.0 + _exp((-60.0  - V) / 5.0))
    beta_m  = (0.1 / (1.0 + _exp((V + 35.0) / 5.0))
               + 0.1 / (1.0 + _exp((V - 50.0) / 200.0)))
    tau_m   = alpha_m * beta_m

    h_inf   = 1.0 / (1.0 + _exp((V + 71.55) / 7.43)) ** 2
    alpha_h = _where(V < -40.0, 0.057 * _exp(-(V + 80.0) / 6.8), 0.0)
    beta_h  = _where(V < -40.0,
                     2.7 * _exp(0.079 * V) + 3.1e5 * _exp(0.3485 * V),
                     0.77 / (0.13 * (1.0 + _exp((V + 10.66) / -11.1))))
    tau_h   = 1.0 / (alpha_h + beta_h)

    j_inf   = h_inf
    alpha_j = _where(V < -40.0,
                     ((-2.5428e4 * _exp(0.2444 * V)
                       - 6.948e-6 * _exp(-0.04391 * V))
                      * (V + 37.78))
                     / (1.0 + _exp(0.311 * (V + 79.23))),
                     0.0)
    beta_j  = _where(V < -40.0,
                     0.02424 * _exp(-0.01052 * V)
                     / (1.0 + _exp(-0.1378 * (V + 40.14))),
                     0.6 * _exp(0.057 * V)
                     / (1.0 + _exp(-0.1 * (V + 32.0))))
    tau_j   = 1.0 / (alpha_j + beta_j)
    i_Na    = g_Na * m**3 * h * j * (V - E_Na)

    # ── IKr ───────────────────────────────────────────────────────────────────
    xr1_inf = 1.0 / (1.0 + _exp((-26.0 - V) / 7.0))
    axr1    = 450.0 / (1.0 + _exp((-45.0 - V) / 10.0))
    bxr1    = 6.0   / (1.0 + _exp((V + 30.0) / 11.5))
    tau_xr1 = axr1 * bxr1

    xr2_inf = 1.0 / (1.0 + _exp((V + 88.0) / 24.0))
    axr2    = 3.0  / (1.0 + _exp((-60.0 - V) / 20.0))
    bxr2    = 1.12 / (1.0 + _exp((V - 60.0) / 20.0))
    tau_xr2 = axr2 * bxr2
    i_Kr    = g_Kr * _sqrt(K_o / 5.4) * Xr1 * Xr2 * (V - E_K)

    # ── IKs ───────────────────────────────────────────────────────────────────
    xs_inf = 1.0 / (1.0 + _exp((-5.0 - V) / 14.0))
    axs    = 1400.0 / _sqrt(1.0 + _exp((5.0 - V) / 6.0))
    bxs    = 1.0 / (1.0 + _exp((V - 35.0) / 15.0))
    tau_xs = axs * bxs + 80.0
    i_Ks   = g_Ks * Xs**2 * (V - E_Ks)

    # ── IK1 ───────────────────────────────────────────────────────────────────
    ak1     = 0.1 / (1.0 + _exp(0.06 * (V - E_K - 200.0)))
    bk1     = (3.0 * _exp(2e-4 * (V - E_K + 100.0))
               + _exp(0.1 * (V - E_K - 10.0))) \
              / (1.0 + _exp(-0.5 * (V - E_K)))
    xk1_inf = ak1 / (ak1 + bk1)
    i_K1    = g_K1 * _sqrt(K_o / 5.4) * xk1_inf * (V - E_K)

    # ── Ito ───────────────────────────────────────────────────────────────────
    s_inf  = 1.0 / (1.0 + _exp((V + 28.0) / 5.0))
    tau_s  = 1000.0 * _exp(-(V + 67.0)**2 / 1000.0) + 8.0
    r_inf  = 1.0 / (1.0 + _exp((20.0 - V) / 6.0))
    tau_r  = 9.5  * _exp(-(V + 40.0)**2 / 1800.0) + 0.8
    i_to   = g_to * r * s * (V - E_K)

    # ── ICaL ──────────────────────────────────────────────────────────────────
    d_inf   = 1.0 / (1.0 + _exp((-8.0 - V) / 7.5))
    ad      = 1.4 / (1.0 + _exp((-35.0 - V) / 13.0)) + 0.25
    bd      = 1.4 / (1.0 + _exp((V + 5.0) / 5.0))
    gd      = 1.0 / (1.0 + _exp((50.0 - V) / 20.0))
    tau_d   = ad * bd + gd

    f_inf   = 1.0 / (1.0 + _exp((V + 20.0) / 7.0))
    tau_f   = (1102.5 * _exp(-((V + 27.0)**2) / 225.0)
               + 200.0 / (1.0 + _exp((13.0 - V) / 10.0))
               + 180.0 / (1.0 + _exp((V + 30.0) / 10.0))
               + 20.0)

    f2_inf  = 0.67 / (1.0 + _exp((V + 35.0) / 7.0)) + 0.33
    tau_f2  = (562.0 * _exp(-((V + 27.0)**2) / 240.0)
               + 31.0 / (1.0 + _exp((25.0 - V) / 10.0))
               + 80.0 / (1.0 + _exp((V + 30.0) / 10.0)))

    fCass_inf = 0.6 / (1.0 + (Cass / 0.05)**2) + 0.4
    tau_fCass = 80.0 / (1.0 + (Cass / 0.05)**2) + 2.0

    exp_term = _exp(2.0 * (V - 15.0) * F / (R * T))
    i_CaL    = (g_CaL * d * f * f2 * fCass
                * 4.0 * (V - 15.0) * F**2 / (R * T)
                * (0.25 * Cass * exp_term - Ca_o)
                / (exp_term - 1.0))

    # ── Background + pump currents ────────────────────────────────────────────
    i_bNa = g_bna * (V - E_Na)
    i_bCa = g_bca * (V - E_Ca)

    i_NaK = (P_NaK * K_o / (K_o + K_mk)
             * Nai / (Nai + K_mNa)
             / (1.0 + 0.1245 * _exp(-0.1 * V * F / (R * T))
                    + 0.0353 * _exp(-V * F / (R * T))))

    i_NaCa = (K_NaCa
              * (_exp(gamma * V * F / (R * T)) * Nai**3 * Ca_o
                 - _exp((gamma - 1.0) * V * F / (R * T))
                   * Na_o**3 * Cai * alpha_n)
              / ((Km_Nai**3 + Na_o**3) * (Km_Ca + Ca_o)
                 * (1.0 + K_sat * _exp((gamma - 1.0) * V * F / (R * T)))))

    i_pCa = g_pCa * Cai / (Cai + K_pCa)
    i_pK  = g_pK  * (V - E_K) / (1.0 + _exp((25.0 - V) / 5.98))

    # ── Ca2+ dynamics ─────────────────────────────────────────────────────────
    kcasr  = max_sr - (max_sr - min_sr) / (1.0 + (EC / CaSR)**2)
    k1     = k1_prime / kcasr
    k2     = k2_prime * kcasr
    O      = k1 * Cass**2 * Rprime / (k3 + k1 * Cass**2)
    i_rel  = V_rel  * O * (CaSR - Cass)
    i_up   = Vmax_up / (1.0 + K_up**2 / Cai**2)
    i_leak = V_leak  * (CaSR - Cai)
    i_xfer = V_xfer  * (Cass - Cai)

    bufc  = 1.0 / (1.0 + Buf_c  * K_buf_c  / (Cai  + K_buf_c )**2)
    bufsr = 1.0 / (1.0 + Buf_sr * K_buf_sr / (CaSR + K_buf_sr)**2)
    bufss = 1.0 / (1.0 + Buf_ss * K_buf_ss / (Cass + K_buf_ss)**2)

    # ── Total ionic current ───────────────────────────────────────────────────
    Iion = (i_K1 + i_to + i_Kr + i_Ks + i_CaL + i_NaK
            + i_Na + i_bNa + i_NaCa + i_bCa + i_pCa + i_pK)

    # ── State updates — Rush-Larsen gates ─────────────────────────────────────
    states["m"]      = _RL(m,     m_inf,     tau_m,     dt)
    states["h"]      = _RL(h,     h_inf,     tau_h,     dt)
    states["j"]      = _RL(j,     j_inf,     tau_j,     dt)
    states["Xr1"]    = _RL(Xr1,   xr1_inf,   tau_xr1,   dt)
    states["Xr2"]    = _RL(Xr2,   xr2_inf,   tau_xr2,   dt)
    states["Xs"]     = _RL(Xs,    xs_inf,    tau_xs,    dt)
    states["d"]      = _RL(d,     d_inf,     tau_d,     dt)
    states["f"]      = _RL(f,     f_inf,     tau_f,     dt)
    states["f2"]     = _RL(f2,    f2_inf,    tau_f2,    dt)
    states["fCass"]  = _RL(fCass, fCass_inf, tau_fCass, dt)
    states["s"]      = _RL(s,     s_inf,     tau_s,     dt)
    states["r"]      = _RL(r,     r_inf,     tau_r,     dt)

    # Rprime: forward Euler
    states["Rprime"] = Rprime + dt * (-k2 * Cass * Rprime + k4 * (1.0 - Rprime))

    # Ion concentrations: forward Euler
    dNai = (-(i_Na + i_bNa + 3.0 * i_NaK + 3.0 * i_NaCa)
             * Cm / (V_c * F))
    dKi  = (-(i_K1 + i_to + i_Kr + i_Ks + i_pK - 2.0 * i_NaK - I_stim)
             * Cm / (V_c * F))
    dCai = bufc * ((i_leak - i_up) * V_sr / V_c + i_xfer
                   - (i_bCa + i_pCa - 2.0 * i_NaCa) * Cm / (2.0 * V_c * F))
    dCaSR = bufsr * (i_up - i_rel - i_leak)
    dCass = bufss * (-i_CaL * Cm / (2.0 * V_ss * F)
                     + i_rel * V_sr / V_ss
                     - i_xfer * V_c  / V_ss)

    states["Nai"]  = Nai  + dt * dNai
    states["Ki"]   = Ki   + dt * dKi
    states["Cai"]  = Cai  + dt * dCai
    states["CaSR"] = CaSR + dt * dCaSR
    states["Cass"] = Cass + dt * dCass

    return Iion, states


# Convenience: also export K_pCa for phenotype modules that need it
K_pCa = 0.0005
