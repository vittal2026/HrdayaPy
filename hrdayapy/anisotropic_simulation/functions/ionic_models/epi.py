"""
epi.py
======
Epicardial TTP06 phenotype.

Conductances are the original TTP06 published values — epi is the reference
phenotype against which endo and M-cell are scaled:
    g_Ks  = 0.392  nS/pF   (full)
    g_to  = 0.073  nS/pF   (full — produces phase-1 notch in AP)
    g_Kr  = 0.153           (unchanged)
    g_CaL = 3.98e-5         (unchanged)

Epi has the largest Ito (phase-1 notch visible in AP morphology) and
the largest IKs, giving the shortest APD.  Epi repolarises first —
the start of the T-wave in the ECG.

Public API (same as original ttp06.py)
---------------------------------------
    states       = init_states(shape)
    Iion, states = step(states, dt, I_stim=0.)
"""

from . import base_ttp06 as _base

# ── Phenotype-specific conductances — original TTP06 (epi reference) ─────────
G_Ks  = 0.392    # nS/pF
G_to  = 0.073    # nS/pF
G_Kr  = 0.153
G_CaL = 3.98e-5


def init_states(shape):
    """Initialise all state variables to CellML resting values."""
    return _base.init_states(shape)


def step(states, dt, I_stim=0.0):
    """
    Advance epicardial ionic model by dt (ms).

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
