"""
ecg/__init__.py
==================
Public API for the ECG forward-solve stage (Step 7 of HrdayaPy).

Three sub-steps, each with its own compute/load pair:

    registration = compute_registration(...) / load_registration(...)
        Heart-to-torso registration (Step 9): aligns the coupled
        simulation's heart point cloud to the torso NRRD's myocardium.

    grid = compute_torso_grid(...) / load_torso_grid(...)
        Coarse torso FDM grid (Step 10): resampled/conductivity-mapped
        torso volume with a factorised stiffness matrix, ready to solve.

    bspm = compute_bspm(...) / load_bspm(...)
        Body surface potential map (Steps 11-12): per-frame impressed-
        current + Poisson solve, mapping myocardial Vm onto the torso
        surface over time.

Typical order: compute_registration and compute_torso_grid are
independent of each other and can run in either order; compute_bspm
needs both of their outputs plus the vm_snapshots.npz that
hrdayapy.simulation.compute_coupled produces (vm_save_path).
"""

from .registration import compute_registration, load_registration
from .torso_grid import compute_torso_grid, load_torso_grid
from .bspm import compute_bspm, load_bspm

# Visualisation (no compute/load pair -- renders, doesn't produce a new
# saveable output)
from .functions import visualise_registration

__all__ = [
    "compute_registration", "load_registration",
    "compute_torso_grid", "load_torso_grid",
    "compute_bspm", "load_bspm",
    "visualise_registration",
]
