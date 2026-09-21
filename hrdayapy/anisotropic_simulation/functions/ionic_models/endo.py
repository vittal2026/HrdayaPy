"""
endo.py
=======
Endocardial TTP06 phenotype.

Conductance values from Romero et al. (2009) / ten Tusscher & Panfilov (2004):
    g_Ks  = 0.245 * g_Ks_epi   →  0.0962 nS/pF
    g_to  = 0.059 * g_to_epi   →  0.0043 nS/pF
    g_Kr  = 0.153              (unchanged across phenotypes)
    g_CaL = 3.98e-5            (unchanged across phenotypes)

Endocardial cells have the smallest Ito (no phase-1 notch in AP)
and intermediate IKs, giving an intermediate APD — longer than epi,
shorter than M-cells.

Public API (same as original ttp06.py)
---------------------------------------
    states       = init_states(shape)
    Iion, states = step(states, dt, I_stim=0.)
"""

from . import base_ttp06 as _base

# ── Phenotype-specific conductances (Romero et al. 2009) ─────────────────────
G_Ks  = 0.245 * 0.392   # = 0.09604 nS/pF
G_to  = 0.059 * 0.073   # = 0.004307 nS/pF
G_Kr  = 0.153            # same as epi
G_CaL = 3.98e-5          # same as epi


def init_states(shape):
    """Initialise all state variables to CellML resting values."""
    return _base.init_states(shape)


def step(states, dt, I_stim=0.0):
    """
    Advance endocardial ionic model by dt (ms).

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
