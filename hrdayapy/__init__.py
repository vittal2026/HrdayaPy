"""
hrdayapy/__init__.py
======================
HrdayaPy: patient-specific cardiac electrophysiology pipeline.

This top-level package does not re-export individual functions itself --
each pipeline stage is its own sub-package with its own public API
(see that sub-package's __init__.py docstring for details). Import the
stage(s) you need off of this package, e.g.:

    import hrdayapy as hp

    S, anatomy = hp.coordinates.compute_geometry(...)
    f = hp.fibres.compute_fibres(...)
    nodes, elements, act_times = hp.purkinje.compute_network(...)
    results = hp.simulation.compute_coupled(...)
    # -- OR, for anisotropic myocardial conduction along/across fibre:
    results = hp.anisotropic_simulation.compute_anisotropic_coupled(...)
    bspm = hp.ecg.compute_bspm(...)
    comparison = hp.inspect_pipeline.pick_and_compare_unipolar(...)

Pipeline stages, in order:

    coordinates      -- geometry / UVC coordinates (S, chi, phi, psi, theta, ...)
    fibres           -- rule-based myocardial fibre direction field, derived
                         from phi/psi (no additional inputs needed). Feeds
                         anisotropic_simulation below.
    purkinje         -- Purkinje network growth + monodomain activation
    simulation       -- coupled Purkinje-myocardium monodomain simulation,
                         isotropic myocardial conduction (single sigma_M)
    anisotropic_simulation
                     -- same coupled simulation, but myocardial conduction
                         is anisotropic: sigma_l/sigma_t (along/across the
                         local fibre direction from hp.fibres) replace
                         sigma_M. A separate stage rather than a flag on
                         `simulation`, so the two solvers can evolve
                         independently -- see that stage's __init__.py and
                         its solver's module docstring for exactly what
                         differs from `simulation`.
    ecg              -- forward ECG solve (torso registration, grid, BSPM)
    inspect_pipeline -- cross-check simulated BSPM against ground-truth
                         patient recordings ("Part 5 -- Inspect"; not
                         named `inspect` since that would shadow the
                         stdlib module of the same name)
"""

from . import coordinates
from . import fibres
from . import purkinje
from . import simulation
from . import anisotropic_simulation
from . import ecg
from . import inspect_pipeline

__all__ = [
    "coordinates",
    "fibres",
    "purkinje",
    "simulation",
    "anisotropic_simulation",
    "ecg",
    "inspect_pipeline",
]
