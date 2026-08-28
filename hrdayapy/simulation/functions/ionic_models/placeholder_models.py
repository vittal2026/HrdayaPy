"""
ionic_models/placeholder_models.py
====================================
Placeholder ionic model classes — one per intended cardiac region.

Each class is a *complete, independent* implementation skeleton that
satisfies the RegionIonicModel protocol.  None share a base class or
inherit conductance values from each other.  This is intentional: when
you implement the real kinetics for each region you edit that class in
isolation without touching any other.

Current state
-------------
All classes currently use a minimal FitzHugh-Nagumo-like excitable system
as a stand-in.  This is enough to produce action potentials and test the
dispatcher framework end-to-end before the full TTP06 / O'Hara-Rudy /
custom kinetics are implemented.

To replace a placeholder:
    1. Keep the class name and the ``name`` attribute.
    2. Rewrite ``state_names()``, ``init_state_tensor()``, and ``step()``
       with the real kinetics.
    3. Nothing else in the pipeline needs to change.

Placeholder kinetics (FHN)
--------------------------
    dv/dt = v - v³/3 - w + I_stim
    dw/dt = (v + a - b*w) / tau

Parameters differ per class to give distinguishable AP shapes:

    Class              a      b     tau    V_rest
    ─────────────────────────────────────────────
    TTP06_Endo_Ph     0.7    0.8   12.5   -1.2
    TTP06_MCell_Ph    0.7    0.8   15.0   -1.2   (longer APD)
    TTP06_Epi_Ph      0.7    0.8   10.0   -1.2   (shorter APD)
    TTP06_Purkinje_Ph 0.5    0.8    8.0   -1.2   (fastest)
    InfarctBorderZone 0.9    0.8   20.0   -1.2   (remodelled, slow)

These are *not* physiologically calibrated — they are scaffolding.
"""

from __future__ import annotations
import torch
from torch import Tensor


# ── Shared FHN stepper ────────────────────────────────────────────────────────

def _fhn_step(
    V:      Tensor,   # (N,)  normalised voltage
    state:  Tensor,   # (N, 1)  recovery variable w
    dt:     float,
    I_stim: Tensor | float,
    a: float, b: float, tau: float,
) -> tuple[Tensor, Tensor]:
    w = state[:, 0]
    dv = V - V**3 / 3.0 - w + (I_stim if isinstance(I_stim, Tensor) else
                                 torch.full_like(V, I_stim))
    dw = (V + a - b * w) / tau
    # Forward Euler on the recovery variable only;
    # V update is returned as I_ion so the PDE solver owns it.
    w_new     = (w + dt * dw).clamp(-2.0, 2.0)
    I_ion     = -dv            # sign convention: outward current opposes dV
    state_new = w_new.unsqueeze(1)
    return I_ion, state_new


def _fhn_init(N, v_rest, device, dtype):
    state = torch.zeros((N, 1), dtype=dtype, device=device)
    return state


# ── TTP06_Endo_Ph ─────────────────────────────────────────────────────────────

class TTP06_Endo_Ph:
    """
    Placeholder endocardial model.

    Replace the FHN kinetics below with the real TTP06 endocardial
    equations once ready.  The class interface must not change.
    """
    name = "TTP06_Endo_Ph"

    # FHN parameters (placeholder only)
    _a   = 0.7
    _b   = 0.8
    _tau = 12.5

    def state_names(self) -> list[str]:
        return ["w"]   # FHN recovery variable
        # Replace with full TTP06 list when implementing real kinetics:
        # return ["Ki","Nai","Cai","Xr1","Xr2","Xs","m","h","j",
        #         "Cass","d","f","f2","fCass","s","r","CaSR","Rprime"]

    def init_state_tensor(self, N, device, dtype) -> Tensor:
        return _fhn_init(N, v_rest=-1.2, device=device, dtype=dtype)

    def step(self, V, state, dt, I_stim) -> tuple[Tensor, Tensor]:
        return _fhn_step(V, state, dt, I_stim,
                         self._a, self._b, self._tau)


# ── TTP06_MCell_Ph ────────────────────────────────────────────────────────────

class TTP06_MCell_Ph:
    """
    Placeholder mid-myocardial (M-cell) model.

    M-cells have a longer APD than endo or epi due to smaller IKs.
    The placeholder reproduces this with a longer tau.
    """
    name = "TTP06_MCell_Ph"

    _a   = 0.7
    _b   = 0.8
    _tau = 15.0   # longer recovery → longer APD

    def state_names(self) -> list[str]:
        return ["w"]

    def init_state_tensor(self, N, device, dtype) -> Tensor:
        return _fhn_init(N, v_rest=-1.2, device=device, dtype=dtype)

    def step(self, V, state, dt, I_stim) -> tuple[Tensor, Tensor]:
        return _fhn_step(V, state, dt, I_stim,
                         self._a, self._b, self._tau)


# ── TTP06_Epi_Ph ──────────────────────────────────────────────────────────────

class TTP06_Epi_Ph:
    """
    Placeholder epicardial model.

    Epicardial cells have the shortest APD (largest Ito, prominent
    phase-1 notch).  Placeholder uses a shorter tau.
    """
    name = "TTP06_Epi_Ph"

    _a   = 0.7
    _b   = 0.8
    _tau = 10.0   # shorter recovery → shorter APD

    def state_names(self) -> list[str]:
        return ["w"]

    def init_state_tensor(self, N, device, dtype) -> Tensor:
        return _fhn_init(N, v_rest=-1.2, device=device, dtype=dtype)

    def step(self, V, state, dt, I_stim) -> tuple[Tensor, Tensor]:
        return _fhn_step(V, state, dt, I_stim,
                         self._a, self._b, self._tau)


# ── TTP06_Purkinje_Ph ─────────────────────────────────────────────────────────

class TTP06_Purkinje_Ph:
    """
    Placeholder Purkinje cell model.

    Used for in-wall Purkinje fibres in the myocardial domain (if
    separately segmented).  The Purkinje cable in the 1-D network uses
    the pipeline's EndoTTP06Torch directly; this class is for any
    sub-endocardial Purkinje region in the 3-D mesh.

    Purkinje cells have a distinctive long plateau and fast conduction.
    """
    name = "TTP06_Purkinje_Ph"

    _a   = 0.5
    _b   = 0.8
    _tau = 8.0

    def state_names(self) -> list[str]:
        return ["w"]

    def init_state_tensor(self, N, device, dtype) -> Tensor:
        return _fhn_init(N, v_rest=-1.2, device=device, dtype=dtype)

    def step(self, V, state, dt, I_stim) -> tuple[Tensor, Tensor]:
        return _fhn_step(V, state, dt, I_stim,
                         self._a, self._b, self._tau)


# ── InfarctBorderZone_Ph ──────────────────────────────────────────────────────

class InfarctBorderZone_Ph:
    """
    Placeholder peri-infarct border zone model.

    Border zone cells are electrically remodelled: reduced Na⁺ current
    (depolarised resting potential, slow upstroke), reduced IKr, and
    prolonged APD.  None of this is captured by a simple modification of
    a healthy-cell model — which is precisely why this is a separate class.

    The placeholder uses a high tau (slow, long-APD) to signal that
    something pathological is happening here.
    """
    name = "InfarctBorderZone_Ph"

    _a   = 0.9
    _b   = 0.8
    _tau = 20.0   # very long recovery

    def state_names(self) -> list[str]:
        return ["w"]

    def init_state_tensor(self, N, device, dtype) -> Tensor:
        return _fhn_init(N, v_rest=-1.2, device=device, dtype=dtype)

    def step(self, V, state, dt, I_stim) -> tuple[Tensor, Tensor]:
        return _fhn_step(V, state, dt, I_stim,
                         self._a, self._b, self._tau)
