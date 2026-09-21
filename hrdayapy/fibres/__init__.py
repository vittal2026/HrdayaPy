"""
fibres/__init__.py
=====================
Public API for the fibres stage.

    f = compute_fibres(...) / load_fibres(...)

compute_fibres needs S, phi, psi from hrdayapy.coordinates (phi =
transmural, psi = apicobasal) plus anatomy["spacing_mm"]. Produces a
(Nx,Ny,Nz,3) unit fibre-direction-cosine field via the standard rule-based
/ Streeter-type helix-angle prescription (Bayer, Blake, Plank & Trayanova
2012) -- no additional inputs (e.g. DT-MRI) needed.

Downstream, this field is the input to the anisotropic-conductivity
extension of the myocardium monodomain solve in hrdayapy.simulation (see
that stage's docs once the extension lands): at each voxel, the local
conductivity tensor is built from sigma_l/sigma_t (eigen-conductivities
along/across the fibre) and this direction field, replacing the current
scalar sigma_M.
"""

from .direction import compute_fibres, load_fibres

# Visualisation (no compute/load pair -- just renders, doesn't produce a
# new saveable output)
from .functions import visualise_fibres

__all__ = [
    "compute_fibres", "load_fibres",
    "visualise_fibres",
]
