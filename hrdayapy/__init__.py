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
    nodes, elements, act_times = hp.purkinje.compute_network(...)
    results = hp.simulation.compute_coupled(...)
    bspm = hp.ecg.compute_bspm(...)
    comparison = hp.inspect_pipeline.pick_and_compare_unipolar(...)

Pipeline stages, in order:

    coordinates      -- geometry / UVC coordinates (S, chi, phi, psi, theta, ...)
    purkinje         -- Purkinje network growth + monodomain activation
    simulation       -- coupled Purkinje-myocardium monodomain simulation
    ecg              -- forward ECG solve (torso registration, grid, BSPM)
    inspect_pipeline -- cross-check simulated BSPM against ground-truth
                         patient recordings ("Part 5 -- Inspect"; not
                         named `inspect` since that would shadow the
                         stdlib module of the same name)
"""

from . import coordinates
from . import purkinje
from . import simulation
from . import ecg
from . import inspect_pipeline

__all__ = [
    "coordinates",
    "purkinje",
    "simulation",
    "ecg",
    "inspect_pipeline",
]
