"""
mcell.py
========
Mid-myocardial (M-cell) TTP06 phenotype.

Conductance values from Romero et al. (2009) / ten Tusscher & Panfilov (2004):
    g_Ks  = 0.062 * g_Ks_epi   →  0.0243 nS/pF  ← KEY: ~4x smaller than endo
    g_to  = 0.059 * g_to_epi   →  0.0043 nS/pF    (same small Ito as endo)
    g_Kr  = 0.153              (unchanged)
    g_CaL = 3.98e-5            (unchanged)

M-cells have dramatically reduced IKs.  This prolongs the action potential
plateau because IKs is the primary repolarising current during the plateau
phase.  M-cells produce the longest APD in the wall, repolarising last —
which is what generates the T-wave peak in the ECG.

The absence of M-cells (or treating them as a ramp) is the most common reason
forward-model T-waves are too flat.

Public API (same as original ttp06.py)
---------------------------------------
    states       = init_states(shape)
    Iion, states = step(states, dt, I_stim=0.)
"""

from . import base_ttp06 as _base

# ── Phenotype-specific conductances (Romero et al. 2009) ─────────────────────
G_Ks  = 0.062 * 0.392   # = 0.024304 nS/pF  — ~4x less than endo, ~16x less than epi
G_to  = 0.059 * 0.073   # = 0.004307 nS/pF  — same as endo (no phase-1 notch)
G_Kr  = 0.153            # same as epi
G_CaL = 3.98e-5          # same as epi


def init_states(shape):
    """Initialise all state variables to CellML resting values."""
    return _base.init_states(shape)


def step(states, dt, I_stim=0.0):
    """
    Advance M-cell ionic model by dt (ms).

    Parameters
    ----------
    states  : dict of arrays (modified in place)
    dt      : float (ms)
    I_stim  : array or scalar (pA/pF)

    Returns
    -------
    Iion    : total ionic current (pA/pF)
    states  : updated dict
    """
    return _base._step_core(
        states, dt,
        I_stim = I_stim,
        g_Ks   = G_Ks,
        g_to   = G_to,
        g_Kr   = G_Kr,
        g_CaL  = G_CaL,
    )
