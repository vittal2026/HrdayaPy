"""
functions/ionic_models/__init__.py
====================================
Public API for the HrdayaPy ionic model package.

Architecture
------------
The package uses a *per-node model assignment* design:

    1.  Every cardiac region is represented by an independent class that
        satisfies the ``RegionIonicModel`` protocol (protocol.py).
        Classes do not share a base class or inherit conductances from
        each other.

    2.  A ``NodeIonicDispatcher`` (dispatcher.py) owns the per-node
        assignment.  At construction it receives a ``region_map`` (one
        integer label per node) and a ``model_registry`` (dict mapping
        label → model instance).  During the time loop the solver calls
        a single method::

            I_ion = dispatcher.step(V, dt, I_stim)

        The dispatcher groups nodes by region, calls each model's
        ``step()`` on its subset, and scatters results back.

    3.  Regions with no entry in the registry are assigned ``SilentCell``
        (zero ionic current) — the correct behaviour for infarct core,
        scar, or connective tissue.

    4.  The ``region_map`` is built externally from a multi-label
        segmentation mask (region_map.py).  If no segmentation is
        available, ``make_transmural_region_map`` reproduces the old
        three-layer phi-based assignment.

Conventional region labels
--------------------------
    LABEL_INFARCT  = 0   → SilentCell
    LABEL_ENDO     = 1   → TTP06_Endo (or placeholder)
    LABEL_MCELL    = 2   → TTP06_MCell (or placeholder)
    LABEL_EPI      = 3   → TTP06_Epi (or placeholder)
    LABEL_BORDER   = 4   → InfarctBorderZone (or placeholder)

These are conventions, not requirements.  The dispatcher is agnostic
about label values.

Typical usage
-------------
::

    from functions.ionic_models import (
        NodeIonicDispatcher,
        make_transmural_region_map,
        region_map_from_segmentation,
        validate_region_map,
        LABEL_ENDO, LABEL_MCELL, LABEL_EPI, LABEL_INFARCT,
        TTP06_Endo_Ph, TTP06_MCell_Ph, TTP06_Epi_Ph, SilentCell,
    )

    # Option A: derive region map from transmural coordinate
    region_map = make_transmural_region_map(phi_nodes)

    # Option B: derive from a multi-label segmentation mask
    region_map = region_map_from_segmentation(seg_mask, vox_idx)

    # Build the model registry — one instance per region
    registry = {
        LABEL_ENDO:    TTP06_Endo_Ph(),
        LABEL_MCELL:   TTP06_MCell_Ph(),
        LABEL_EPI:     TTP06_Epi_Ph(),
        LABEL_INFARCT: SilentCell(),   # explicit, or omit (default)
    }

    validate_region_map(region_map, registry)

    dispatcher = NodeIonicDispatcher(
        region_map, registry, device=dev, dtype=torch.float32
    )

    # In the time loop:
    I_ion = dispatcher.step(V, dt, I_stim)

Backward compatibility
----------------------
The old ``assign_phenotypes``, ``init_states_hetero``, ``step_hetero``,
``ENDO``, ``MCELL``, ``EPI`` names are still exported for code that has
not yet migrated to the dispatcher.  They will be removed in a future
version.
"""

# ── New framework ─────────────────────────────────────────────────────────────
from .protocol          import RegionIonicModel
from .silent_cell       import SilentCell
from .dispatcher        import NodeIonicDispatcher
from .region_map        import (
    region_map_from_segmentation,
    make_transmural_region_map,
    validate_region_map,
    LABEL_INFARCT,
    LABEL_ENDO,
    LABEL_MCELL,
    LABEL_EPI,
    LABEL_BORDER,
    PHI_ENDO_MAX,
    PHI_EPI_MIN,
)
from .placeholder_models import (
    TTP06_Endo_Ph,
    TTP06_MCell_Ph,
    TTP06_Epi_Ph,
    TTP06_Purkinje_Ph,
    InfarctBorderZone_Ph,
)

# ── GPU-native TTP06 phenotype classes (used in the coupled solver) ───────────
from .gpu_ttp06 import EndoTTP06, MidTTP06, EpiTTP06

# ── Heterogeneous ionic solver (transmural endo/mid/epi dispatch) ─────────────
from .hetero_solver import HeteroIonicSolver

# ── Backward-compatible shim ──────────────────────────────────────────────────
# These names match the old ionic_models/__init__.py API.
# New code should use NodeIonicDispatcher + region_map instead.
import numpy as np
from . import endo  as _endo
from . import mcell as _mcell
from . import epi   as _epi

ENDO  = LABEL_ENDO    # = 1
MCELL = LABEL_MCELL   # = 2
EPI   = LABEL_EPI     # = 3


def assign_phenotypes(phi_nodes: np.ndarray) -> np.ndarray:
    """
    Backward-compatible wrapper around make_transmural_region_map.

    .. deprecated::
        Use ``make_transmural_region_map`` and ``NodeIonicDispatcher``
        instead.
    """
    import warnings
    warnings.warn(
        "assign_phenotypes() is deprecated.  Use make_transmural_region_map() "
        "and NodeIonicDispatcher instead.",
        DeprecationWarning, stacklevel=2,
    )
    return make_transmural_region_map(phi_nodes)


def init_states_hetero(shape: tuple, labels: np.ndarray) -> dict:
    """Backward-compatible state initialiser. Deprecated."""
    from .base_ttp06 import init_states
    return init_states(shape)


def step_hetero(
    states:  dict,
    labels:  np.ndarray,
    dt:      float,
    I_stim:  np.ndarray | float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Backward-compatible heterogeneous step. Deprecated."""
    N_myo = labels.shape[0]
    Iion  = np.empty(N_myo, dtype=np.float64)
    for label, module in ((ENDO, _endo), (MCELL, _mcell), (EPI, _epi)):
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        sub      = {k: v[idx] for k, v in states.items()}
        sub_stim = I_stim[idx] if isinstance(I_stim, np.ndarray) else I_stim
        Iion_sub, sub = module.step(sub, dt, I_stim=sub_stim)
        Iion[idx] = Iion_sub
        for k in states:
            states[k][idx] = sub[k]
    return Iion, states


__all__ = [
    # Protocol
    "RegionIonicModel",
    # Core framework
    "SilentCell",
    "NodeIonicDispatcher",
    # Region map
    "region_map_from_segmentation",
    "make_transmural_region_map",
    "validate_region_map",
    "LABEL_INFARCT", "LABEL_ENDO", "LABEL_MCELL", "LABEL_EPI", "LABEL_BORDER",
    "PHI_ENDO_MAX", "PHI_EPI_MIN",
    # Placeholder model classes
    "TTP06_Endo_Ph", "TTP06_MCell_Ph", "TTP06_Epi_Ph",
    "TTP06_Purkinje_Ph", "InfarctBorderZone_Ph",
    # GPU TTP06 phenotypes
    "EndoTTP06", "MidTTP06", "EpiTTP06",
    # Heterogeneous solver
    "HeteroIonicSolver",
    # Backward-compatible
    "ENDO", "MCELL", "EPI",
    "assign_phenotypes", "init_states_hetero", "step_hetero",
]
