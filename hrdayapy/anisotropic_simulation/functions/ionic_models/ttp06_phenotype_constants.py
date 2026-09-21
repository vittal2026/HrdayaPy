"""
ttp06_phenotype_constants.py
===================================================================
SINGLE SOURCE OF TRUTH for TTP06 phenotype-specific conductances.

Every module that needs endocardial / mid-myocardial / epicardial
TTP06 conductances -- whether it operates on PyTorch tensors
(hrdayapy.simulation.functions.ionic_models.gpu_ttp06, used by the
production monodomain solver) or on plain Python floats
(experiments/experiment_1/run_experiment1.py, a hand-translated
scalar reimplementation used for single-cell pacing because
per-operation PyTorch overhead makes N=1 tensor simulation
impractically slow -- roughly 60x slower in a direct comparison run
during manuscript preparation) -- must import from HERE rather than
hardcoding its own copy of these numbers.

This file exists specifically because it didn't exist before: an
earlier version of both gpu_ttp06.py and run_experiment1.py
independently hardcoded the same incorrect, mis-scaled conductances,
misattributed in both places to a source ("Romero et al. 2009",
"TTP06 Table 2") that does not actually contain them. Fixing the bug
required editing two files by hand and had no mechanism to guarantee
they'd stayed in sync -- they hadn't (see git history / manuscript
revision notes for details). This module is the fix for the
underlying architectural problem, not just the numbers.

Source: ten Tusscher & Panfilov (2006), Am J Physiol Heart Circ
Physiol 291(3):H1088-H1100, Table 1.

Any future change to these values should be made ONLY here, and
should be accompanied by re-running
tests/test_ttp06_implementations_agree.py to confirm the tensor and
scalar implementations still agree numerically.
"""

from __future__ import annotations

from typing import NamedTuple


class TTP06PhenotypeConductances(NamedTuple):
    g_Ks: float   # nS/pF -- slow delayed rectifier K+ conductance
    g_to: float   # nS/pF -- transient outward current conductance
    g_Kr: float   # nS/pF -- rapid delayed rectifier K+ conductance (shared across phenotypes)
    g_CaL: float  # cm/ms/uF -- L-type Ca2+ current conductance (shared across phenotypes)
    endo_s: bool  # True -> slow (endo/M-cell) s-gate kinetics; False -> fast (epi) s-gate kinetics


# ten Tusscher & Panfilov (2006), Table 1.
# NOTE: M-cell shares epicardium's LARGE g_to (0.294), not
# endocardium's small one (0.073) -- this pairing is easy to get
# wrong (it was gotten wrong here previously) because it's
# counter-intuitive to assume phenotypes vary "monotonically".
TTP06_PHENOTYPES: dict[str, TTP06PhenotypeConductances] = {
    "endo": TTP06PhenotypeConductances(
        g_Ks=0.392, g_to=0.073, g_Kr=0.153, g_CaL=3.98e-5, endo_s=True),
    "mid": TTP06PhenotypeConductances(
        g_Ks=0.098, g_to=0.294, g_Kr=0.153, g_CaL=3.98e-5, endo_s=True),
    "epi": TTP06PhenotypeConductances(
        g_Ks=0.392, g_to=0.294, g_Kr=0.153, g_CaL=3.98e-5, endo_s=False),
}

PHENOTYPE_LABELS: dict[str, str] = {
    "endo": "Endocardial",
    "mid": "Mid-myocardial (M-cell)",
    "epi": "Epicardial",
}
