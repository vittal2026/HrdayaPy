"""
ionic_models/protocol.py
========================
Defines the ``RegionIonicModel`` protocol — the interface that every ionic
model class in HrdayaPy must satisfy.

Design rationale
----------------
The old architecture expressed cell-type differences as *parameterisations*
of a single base class (endo/mcell/epi as scaled variants of TTP06).  This
breaks down as soon as a region cannot be expressed that way — an infarct
zone, a fibrotic region, a Purkinje-like phenotype, or a completely
different ionic model (HH, Luo-Rudy, O'Hara-Rudy) in the same simulation.

The new architecture treats each ionic model as a fully independent class.
The only contract is this protocol.  The dispatcher (NodeIonicDispatcher)
batches nodes that share the same model and calls the protocol methods —
it never inspects model internals.

Protocol
--------
Every ionic model class must implement:

    ``name : str``
        Human-readable identifier used in logs and saved metadata.

    ``state_names() -> list[str]``
        Ordered list of state variable names.  The dispatcher allocates
        one (N_nodes, n_states) tensor per model group; the ordering here
        defines column indices.

    ``init_state_tensor(N, device, dtype) -> Tensor``
        Return a (N, n_states) tensor of resting / initial state values
        for N nodes on the given device.

    ``step(V, state, dt, I_stim) -> tuple[Tensor, Tensor]``
        Advance the model by dt milliseconds.

        Parameters
        ----------
        V      : (N,)   transmembrane voltage  [mV]
        state  : (N, n_states)  current state
        dt     : float  timestep  [ms]
        I_stim : (N,) or scalar  applied stimulus  [uA/mm² or pA/pF,
                 consistent with the model's current unit convention]

        Returns
        -------
        I_ion  : (N,)          total ionic current  [same units as I_stim]
        state  : (N, n_states) updated state  (may be the same tensor,
                 modified in-place, or a new tensor)

Constraints
-----------
*  ``V`` is *not* a state variable managed by the ionic model.  The PDE
   solver owns V and passes a snapshot to ``step()``.  The model must not
   store V internally between calls.
*  ``state`` columns must match ``state_names()`` in order.
*  Models are free to use numpy or torch internally; the dispatcher
   converts as needed before calling ``step()``.
*  Thread/process safety is not required — the dispatcher calls models
   sequentially on one device.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import torch
from torch import Tensor


@runtime_checkable
class RegionIonicModel(Protocol):
    """
    Structural protocol for all HrdayaPy ionic model classes.

    A class satisfies this protocol if it has the right method signatures;
    it does NOT need to explicitly inherit from RegionIonicModel.
    """

    name: str   # class-level attribute, e.g. "TTP06_Endo"

    def state_names(self) -> list[str]:
        """
        Return the ordered list of state variable names (excluding V).

        Example for TTP06:
            ["Ki", "Nai", "Cai", "Xr1", ..., "CaSR", "Rprime"]
        """
        ...

    def init_state_tensor(
        self,
        N:      int,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> Tensor:
        """
        Return a (N, n_states) tensor of initial values for N nodes.

        All nodes in a freshly-initialised group start from the same
        resting state.  Per-node variation (e.g. pre-pacing) is applied
        by the caller after this method returns.
        """
        ...

    def step(
        self,
        V:      Tensor,   # (N,)         mV
        state:  Tensor,   # (N, n_states)
        dt:     float,    # ms
        I_stim: Tensor,   # (N,) or scalar
    ) -> tuple[Tensor, Tensor]:
        """
        Advance by dt ms.

        Returns
        -------
        I_ion  : (N,)          total ionic current
        state  : (N, n_states) updated state
        """
        ...
