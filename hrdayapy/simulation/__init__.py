"""
simulation/__init__.py
========================
Public API for the simulation stage.
    results = compute_coupled(...) / load_coupled(...)
    vm_snapshots = load_vm_snapshots(...)   -- a byproduct of compute_coupled,
                   only produced if you asked for it (vm_save_dt / vm_save_path);
                   there's no separate compute_ for it since it isn't
                   independently computable, only saved alongside the main run.
compute_coupled needs nodes/elements/act_times from hrdayapy.purkinje, and
S/Z/phi from hrdayapy.coordinates.
"""
from .coupled import compute_coupled, load_coupled
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
    "compute_coupled", "load_coupled",
    "load_vm_snapshots", "VmSnapshots",
    "compute_activation_maps", "load_activation_maps",
    "save_activation_maps", "ActivationMaps",
    "animate_coupled", "plot_activation_snapshot_grid",
    "plot_voltage_trace_at_picked_node",
    "plot_activation_maps",
]
