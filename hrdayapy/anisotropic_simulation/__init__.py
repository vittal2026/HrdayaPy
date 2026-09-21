"""
anisotropic_simulation/__init__.py
=====================================
Public API for the anisotropic_simulation stage -- otherwise identical to
hrdayapy.simulation (see purkinje_myocardium_anisotropic_solver.py's
module docstring for exactly what differs and why). The only interface
difference is sigma_M -> sigma_l + sigma_t + fibre_dir.

    results = compute_anisotropic_coupled(...) / load_anisotropic_coupled(...)
    vm_snapshots = load_vm_snapshots(...)   -- a byproduct of
                   compute_anisotropic_coupled, only produced if you asked
                   for it (vm_save_dt / vm_save_path); there's no separate
                   compute_ for it since it isn't independently computable,
                   only saved alongside the main run.

compute_anisotropic_coupled needs nodes/elements/act_times from
hrdayapy.purkinje, S/Z/phi from hrdayapy.coordinates, and fibre_dir from
hrdayapy.fibres.
"""
from .coupled import compute_anisotropic_coupled, load_anisotropic_coupled
# Volumetric Vm snapshots -- load-only, see module docstring above
from .functions import load_vm_snapshots, VmSnapshots
# Static activation (depolarisation) / deactivation (repolarisation) maps,
# derived from the Vm snapshots above. Multi-beat protocols keep one column
# of timing per beat -- see activation_maps.py docstring for the schema.
from .functions import (
    compute_activation_maps, load_activation_maps,
    save_activation_maps, ActivationMaps,
)
# Visualisation (no compute/load pair -- renders, doesn't produce a new
# saveable output)
from .functions import (
    animate_coupled, plot_activation_snapshot_grid,
    plot_voltage_trace_at_picked_node, plot_activation_maps,
)
__all__ = [
    "compute_anisotropic_coupled", "load_anisotropic_coupled",
    "load_vm_snapshots", "VmSnapshots",
    "compute_activation_maps", "load_activation_maps",
    "save_activation_maps", "ActivationMaps",
    "animate_coupled", "plot_activation_snapshot_grid",
    "plot_voltage_trace_at_picked_node",
    "plot_activation_maps",
]
