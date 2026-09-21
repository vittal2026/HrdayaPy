"""
ionic_models/hetero_solver.py
==============================
HeteroIonicSolver — transmural-heterogeneous ionic model dispatcher
for the coupled Purkinje + myocardium solver.

Design
------
This class owns all per-node ionic state and exposes a single method
that matches the EndoTTP06Torch calling convention used by run_coupled():

    I_ion, states = solver.step(states, dt, I_stim=None)

Inside, it splits the global node vector into three groups
(endo / mcell / epi) using pre-built index masks, calls the appropriate
GPU TTP06 class on each group, and scatters the results back.

Purkinje nodes (indices 0 … Np-1) are always assigned EndoTTP06 regardless
of the transmural coordinate — this is the intended behaviour and can be
changed by replacing PurkinjeTTP06 below.

Myocardial nodes (indices Np … Np+N_myo-1) are assigned by phi:
    phi < phi_endo_max            → EndoTTP06
    phi_endo_max ≤ phi < phi_epi  → MidTTP06
    phi ≥ phi_epi_min             → EpiTTP06

State storage
-------------
Each group stores a (N_group, N_STATES) tensor.  The global "states" dict
passed in and out of .step() contains only {"V": Tensor(N_global,)}, which
is the quantity the PDE solver owns.  All other state lives inside this
class and is updated in-place each call.

Unit convention
---------------
.step() returns I_ion in mV/ms (pA/pF), exactly as EndoTTP06Torch does,
so run_coupled()'s existing scaling logic requires no change:

    Iion_pde = Iion * Cm_PDE      # mV/ms → µA/mm²
    V_star   = V + (dt/Cm) * (-Iion_pde + I_stim)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .gpu_ttp06 import EndoTTP06, MidTTP06, EpiTTP06, _N_STATES


# Default transmural thresholds (identical to region_map.py)
_PHI_ENDO_MAX = 0.35
_PHI_EPI_MIN  = 0.65


class HeteroIonicSolver:
    """
    Transmural-heterogeneous TTP06 ionic model for the coupled solver.

    Parameters
    ----------
    Np            : int          number of Purkinje computational nodes
    phi_myo       : (N_myo,) np.ndarray
                    transmural coordinate for each myocardial node,
                    0 = endocardium, 1 = epicardium
    device        : torch.device
    dtype         : torch.dtype  (float32 or float64)
    phi_endo_max  : float        upper phi boundary of the endo layer
    phi_epi_min   : float        lower phi boundary of the epi layer
    V0            : float        initial voltage (mV)
    """

    def __init__(
        self,
        Np:           int,
        phi_myo:      np.ndarray,
        device:       torch.device,
        dtype:        torch.dtype  = torch.float64,
        phi_endo_max: float        = _PHI_ENDO_MAX,
        phi_epi_min:  float        = _PHI_EPI_MIN,
        V0:           float        = -86.709,
    ):
        self.Np     = Np
        self.N_myo  = len(phi_myo)
        self.N      = Np + self.N_myo
        self.dev    = device
        self.dtype  = dtype
        self.V0     = V0

        # ── Model instances ────────────────────────────────────────────────────
        self._pkn_model  = EndoTTP06()   # Purkinje → endo for now
        self._endo_model = EndoTTP06()
        self._mid_model  = MidTTP06()
        self._epi_model  = EpiTTP06()

        # ── Region masks for myocardial nodes (local, 0-based within myo) ─────
        phi = phi_myo.astype(np.float32)
        myo_endo = np.where(phi <  phi_endo_max)[0].astype(np.int64)
        myo_epi  = np.where(phi >= phi_epi_min)[0].astype(np.int64)
        myo_mid  = np.where((phi >= phi_endo_max) & (phi < phi_epi_min))[0].astype(np.int64)

        # Global indices (shift myo indices by Np)
        pkn_idx  = np.arange(Np,        dtype=np.int64)
        endo_idx = myo_endo + Np
        mid_idx  = myo_mid  + Np
        epi_idx  = myo_epi  + Np

        self._pkn_idx  = torch.tensor(pkn_idx,  dtype=torch.long, device=device)
        self._endo_idx = torch.tensor(endo_idx, dtype=torch.long, device=device)
        self._mid_idx  = torch.tensor(mid_idx,  dtype=torch.long, device=device)
        self._epi_idx  = torch.tensor(epi_idx,  dtype=torch.long, device=device)

        # ── Per-group state tensors (N_group, N_STATES) ────────────────────────
        self._pkn_state  = self._pkn_model .init_state_tensor(len(pkn_idx),  device, dtype)
        self._endo_state = self._endo_model.init_state_tensor(len(endo_idx), device, dtype)
        self._mid_state  = self._mid_model .init_state_tensor(len(mid_idx),  device, dtype)
        self._epi_state  = self._epi_model .init_state_tensor(len(epi_idx),  device, dtype)

        # ── Print region breakdown ─────────────────────────────────────────────
        total = self.N
        print(f"\n[HeteroIonicSolver] Transmural region assignment:")
        print(f"  phi thresholds: endo < {phi_endo_max:.2f} ≤ mid < {phi_epi_min:.2f} ≤ epi")
        print(f"  Purkinje  (endo TTP06) : {len(pkn_idx):>8,}  ({100*len(pkn_idx)/total:.1f}%)")
        print(f"  Endo      (endo TTP06) : {len(endo_idx):>8,}  ({100*len(endo_idx)/total:.1f}%)")
        print(f"  Mid-myo   (mcell TTP06): {len(mid_idx):>8,}  ({100*len(mid_idx)/total:.1f}%)")
        print(f"  Epi       (epi TTP06)  : {len(epi_idx):>8,}  ({100*len(epi_idx)/total:.1f}%)")
        print(f"  Total nodes            : {total:>8,}\n")

    # ── Public API matching EndoTTP06Torch.step() ─────────────────────────────

    def init_states(self, N: int, dtype: torch.dtype = None) -> dict[str, Tensor]:
        """
        Return the initial global states dict.
        N and dtype are accepted for API compatibility but ignored —
        this solver owns its own state tensors.
        """
        return {"V": torch.full((self.N,), self.V0,
                                dtype=self.dtype, device=self.dev)}

    def step(
        self,
        states:  dict[str, Tensor],
        dt:      float,
        I_stim:  Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Advance all ionic groups by dt ms.

        Parameters
        ----------
        states  : {"V": (N_global,) Tensor}  — V is the only externally
                  owned state variable; all gate/conc state lives here.
        dt      : timestep (ms)
        I_stim  : ignored (stimulus applied by PDE caller); accepted for
                  API compatibility.

        Returns
        -------
        I_ion   : (N_global,) Tensor  total ionic current [mV/ms = pA/pF]
        states  : updated {"V": ...}  V unchanged (PDE caller owns update)
        """
        V     = states["V"]
        I_ion = torch.zeros(self.N, dtype=self.dtype, device=self.dev)

        # ── Purkinje nodes ────────────────────────────────────────────────────
        if self._pkn_idx.numel() > 0:
            V_g = V[self._pkn_idx]
            Ig, self._pkn_state = self._pkn_model.step(
                V_g, self._pkn_state, dt, 0.0)
            I_ion[self._pkn_idx] = Ig

        # ── Endo myocardial nodes ─────────────────────────────────────────────
        if self._endo_idx.numel() > 0:
            V_g = V[self._endo_idx]
            Ig, self._endo_state = self._endo_model.step(
                V_g, self._endo_state, dt, 0.0)
            I_ion[self._endo_idx] = Ig

        # ── Mid (M-cell) myocardial nodes ─────────────────────────────────────
        if self._mid_idx.numel() > 0:
            V_g = V[self._mid_idx]
            Ig, self._mid_state = self._mid_model.step(
                V_g, self._mid_state, dt, 0.0)
            I_ion[self._mid_idx] = Ig

        # ── Epi myocardial nodes ──────────────────────────────────────────────
        if self._epi_idx.numel() > 0:
            V_g = V[self._epi_idx]
            Ig, self._epi_state = self._epi_model.step(
                V_g, self._epi_state, dt, 0.0)
            I_ion[self._epi_idx] = Ig

        return I_ion, states

    # ── Inspection helpers ────────────────────────────────────────────────────

    @property
    def region_sizes(self) -> dict[str, int]:
        return {
            "purkinje": self._pkn_idx.numel(),
            "endo":     self._endo_idx.numel(),
            "mid":      self._mid_idx.numel(),
            "epi":      self._epi_idx.numel(),
        }

    def get_group_V(self, group: str, V_global: Tensor) -> Tensor:
        """Extract voltage for one region group (for diagnostics)."""
        idx = {"purkinje": self._pkn_idx, "endo": self._endo_idx,
               "mid": self._mid_idx, "epi": self._epi_idx}[group]
        return V_global[idx]
