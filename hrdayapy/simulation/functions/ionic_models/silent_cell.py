"""
ionic_models/silent_cell.py
============================
``SilentCell`` — the default ionic model for inexcitable nodes.

Assigned automatically by NodeIonicDispatcher to any region label that
has no entry in the model_registry.  Typical use: infarct core, scar,
connective tissue, or any region the user has not yet assigned a model to.

SilentCell has zero state variables and returns zero ionic current.
It satisfies the RegionIonicModel protocol.
"""

from __future__ import annotations
import torch
from torch import Tensor


class SilentCell:
    """
    Inexcitable cell — zero ionic current, no state.

    This is the correct physical model for dense scar tissue (no
    transmembrane ion channels) and serves as a safe default for
    any region not yet assigned a real model.
    """

    name = "SilentCell"

    def state_names(self) -> list[str]:
        return []   # no state variables

    def init_state_tensor(
        self,
        N:      int,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> Tensor:
        # (N, 0) — zero columns, correct shape for a no-state model
        return torch.empty((N, 0), dtype=dtype, device=device)

    def step(
        self,
        V:      Tensor,
        state:  Tensor,
        dt:     float,
        I_stim: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        I_ion = torch.zeros(V.shape[0], dtype=V.dtype, device=V.device)
        return I_ion, state   # state unchanged (still (N, 0))
