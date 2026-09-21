"""
ionic_models/dispatcher.py
===========================
``NodeIonicDispatcher`` — assigns an ionic model to every node in the
simulation domain and dispatches ionic steps efficiently by grouping
nodes that share the same model instance.

Overview
--------
At construction the dispatcher receives a ``region_map``: a integer
array of length N (one entry per node) where each integer identifies
a region, and a ``model_registry``: a dict mapping region integer to a
``RegionIonicModel`` instance.  Nodes whose region integer has no entry
in the registry are assigned a ``SilentCell`` (zero ionic current, no
state), which is the correct behaviour for inexcitable regions such as
scar or connective tissue.

During the time loop the solver calls a single method::

    I_ion, = dispatcher.step(V, dt, I_stim)

The dispatcher:

1. Iterates over each unique (region, model) group.
2. Gathers the group's V slice and state tensor.
3. Calls ``model.step(V_group, state_group, dt, I_stim_group)``.
4. Scatters I_ion and updated state back to the full arrays.

All state is stored in a single dict ``{region_id: Tensor(N_group, n_states)}``.
The solver never touches state directly — it only reads and writes V.

Region map
----------
The region map is a 1-D integer array of length N_nodes.  It is produced
externally (e.g. from a multi-label segmentation mask) and passed in at
construction.  The dispatcher does not know or care how regions were
defined.  Conventional labels used in HrdayaPy:

    0  Infarct / scar          → SilentCell (no current)
    1  Endocardium             → e.g. TTP06_Endo
    2  Mid-myocardium (M-cell) → e.g. TTP06_MCell
    3  Epicardium              → e.g. TTP06_Epi
    4  Purkinje (in-wall)      → e.g. TTP06_Purkinje
    …  (user-defined)

These are *conventions*, not hard-coded values.  The user builds a
``model_registry`` dict and passes it in; the dispatcher is agnostic
about what the integers mean.

State union
-----------
Different models may have different state variable sets.  The dispatcher
does NOT maintain a single global state tensor — each group has its own
(N_group, n_states_for_this_model) tensor.  This means there is no
need to align state variable names across models.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Any

from .protocol   import RegionIonicModel
from .silent_cell import SilentCell


class NodeIonicDispatcher:
    """
    Assigns ionic models to nodes and dispatches ionic steps by region group.

    Parameters
    ----------
    region_map     : (N,) int array   integer region label per node
    model_registry : dict[int, RegionIonicModel]
                     maps region integer → model instance.
                     Regions absent from the registry are assigned SilentCell.
    device         : torch.device
    dtype          : torch.dtype  (float32 or float64)

    Attributes
    ----------
    N              : total number of nodes
    region_summary : str  printed at construction for logging
    """

    def __init__(
        self,
        region_map:      np.ndarray,
        model_registry:  dict[int, RegionIonicModel],
        device:          torch.device,
        dtype:           torch.dtype = torch.float32,
    ):
        self.N      = len(region_map)
        self.device = device
        self.dtype  = dtype

        # ── Build groups ──────────────────────────────────────────────────────
        # For each unique region label, resolve the model (defaulting to
        # SilentCell) and store the integer index array for fast gather/scatter.
        _silent = SilentCell()

        self._groups: list[tuple[
            int,                # region label
            RegionIonicModel,   # model instance
            Tensor,             # idx  (N_group,) long
        ]] = []

        unique_labels = np.unique(region_map)
        summary_lines = []

        for label in unique_labels:
            model = model_registry.get(int(label), _silent)
            idx   = torch.tensor(
                np.where(region_map == label)[0],
                dtype=torch.long, device=device,
            )
            self._groups.append((int(label), model, idx))
            summary_lines.append(
                f"  region {label:>3d} : {model.name:<30s}  "
                f"n={idx.numel():>8,}"
            )

        self.region_summary = "\n".join(summary_lines)

        # ── Initialise per-group state tensors ────────────────────────────────
        # Keyed by region label.  SilentCell returns a (N, 0) tensor.
        self._state: dict[int, Tensor] = {}
        for label, model, idx in self._groups:
            self._state[label] = model.init_state_tensor(
                idx.numel(), device, dtype
            )

        print("[NodeIonicDispatcher] Region → model assignment:")
        print(self.region_summary)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        V:      Tensor,                       # (N,)  mV
        dt:     float,                        # ms
        I_stim: Tensor | float = 0.0,         # (N,) or scalar
    ) -> Tensor:
        """
        Advance all ionic models by dt ms.

        Parameters
        ----------
        V      : (N,) voltage tensor on ``self.device``
        dt     : timestep in ms
        I_stim : (N,) stimulus current tensor, or scalar

        Returns
        -------
        I_ion  : (N,) total ionic current
        """
        I_ion = torch.zeros(self.N, dtype=self.dtype, device=self.device)

        for label, model, idx in self._groups:
            if idx.numel() == 0:
                continue

            V_g     = V[idx]
            state_g = self._state[label]

            if isinstance(I_stim, Tensor):
                stim_g = I_stim[idx]
            else:
                stim_g = I_stim   # scalar broadcast

            I_ion_g, state_new = model.step(V_g, state_g, dt, stim_g)

            I_ion[idx]          = I_ion_g
            self._state[label]  = state_new

        return I_ion

    def get_state(self, region_label: int) -> Tensor:
        """Return the (N_group, n_states) state tensor for one region."""
        return self._state[region_label]

    def set_state(self, region_label: int, state: Tensor) -> None:
        """
        Replace the state tensor for one region.

        Used for pre-pacing: run a single-cell simulation to steady state,
        then broadcast the result over all nodes in the group.
        """
        self._state[region_label] = state

    def region_labels(self) -> list[int]:
        """Return the list of unique region labels known to this dispatcher."""
        return [label for label, _, _ in self._groups]

    def model_for_region(self, region_label: int) -> RegionIonicModel:
        """Return the model instance assigned to a region label."""
        for label, model, _ in self._groups:
            if label == region_label:
                return model
        raise KeyError(f"Region label {region_label} not found in dispatcher.")

    def node_indices_for_region(self, region_label: int) -> Tensor:
        """Return the (N_group,) index tensor for a region."""
        for label, _, idx in self._groups:
            if label == region_label:
                return idx
        raise KeyError(f"Region label {region_label} not found in dispatcher.")
