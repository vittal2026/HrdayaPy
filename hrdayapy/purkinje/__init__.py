"""
purkinje/__init__.py
======================
Public API for the Purkinje stage.

    nodes, elements, act_times = compute_network(...) / load_network(...)
    vm                         = compute_vm(...)       / load_vm(...)

compute_network needs S, phi, psi from hrdayapy.coordinates, plus a root
point -- typically from pick_av_node_and_biventricular_roots(), which picks
the AV node once and finds the LV/RV root candidates nearest it.
compute_vm needs nodes/elements/act_times from compute_network.
"""

from .network import compute_network, load_network, compute_biventricular_networks
from .monodomain import compute_vm, load_vm

# Visualisation (no compute/load pair -- renders, doesn't produce a new
# saveable output)
from .functions import visualise_purkinje, animate_on_purkinje_in_time

# Biventricular assembly -- grow LV/RV (/His) trees separately with
# compute_network() (or both at once with compute_biventricular_networks()),
# then stitch them into one network with this and save the result with
# save_purkinje (same .npz format load_network expects).
from .functions.create_purkinje import merge_purkinje_networks
from .functions import save_purkinje, load_purkinje

# Single-pick AV node -> automatic LV/RV root finding (replaces picking
# the LV and RV origins separately -- see create_purkinje.py docstring).
from .functions import pick_av_node_and_biventricular_roots

__all__ = [
    "compute_network", "load_network", "compute_biventricular_networks",
    "compute_vm", "load_vm",
    "visualise_purkinje", "animate_on_purkinje_in_time",
    "merge_purkinje_networks", "save_purkinje", "load_purkinje",
    "pick_av_node_and_biventricular_roots",
]
