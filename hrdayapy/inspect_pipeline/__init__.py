"""
Inspect sub-package of HrdayaPy ("Part 5 -- Inspect").

Cross-checks the simulated BSPM against the ground-truth patient recording
that the anatomy was derived from (Bratislava / EDGAR 2026 ECGI Challenge
dataset).

    registration = compute_torso_registration(...) / load_torso_registration(...)
    result       = pick_and_compare(...)             -- interactive, one-shot;
                   no load counterpart, since the point is picking a new
                   location each time, not replaying a saved one.
    result       = pick_and_compare_multi(...)       -- same, but for up to
                   six points at once, with averaged-beat normalization and
                   a per-point correlation coefficient (Step 14b).
    result       = pick_and_compare_unipolar(...)    -- same interactive
                   picking, but compares the REPRESENTATIVE AVERAGE BEAT at
                   a single picked electrode instead of a raw trace or a
                   normalized multi-point overlay (Step 14c).
    result       = pick_and_compare_bipolar(...)     -- same, but for the
                   representative average beat of an A-B (two-electrode)
                   bipolar difference trace (Step 14d).
    result       = pick_and_compare_unipolar_raw(...) -- same electrode-pick
                   UI/CAR convention as pick_and_compare_unipolar, but plots
                   the FULL raw trace at the picked electrode instead of a
                   representative average beat -- no beat detection or
                   averaging at all (Step 14c, non-averaging variant).
    result       = pick_and_compare_bipolar_raw(...)  -- same, but for the
                   full raw A-B (two-electrode) bipolar difference trace
                   (Step 14d, non-averaging variant).

Note on the folder name: this package is deliberately NOT called
`inspect/`, even though the other sub-packages are named after their
pipeline stage (`ecg/`, `simulation/`, ...). `inspect` is a Python stdlib
module name, and a package literally named `inspect` sitting on
sys.path would shadow the stdlib module for every other import in the
process (numpy/pyvista/matplotlib all use `inspect` internally), causing
hard-to-diagnose failures far from this code. The CLI flags and pipeline
stage are still called "inspect" everywhere a user sees them.
"""

from .torso_registration import compute_torso_registration, load_torso_registration

# Interactive comparison + visualisation (no compute/load pair -- these
# render/pick, they don't produce a new saveable output on their own)
from .functions import pick_and_compare_traces as pick_and_compare
from .functions import pick_and_compare_multi_traces as pick_and_compare_multi
from .functions import pick_and_compare_unipolar_beat_traces as pick_and_compare_unipolar
from .functions import pick_and_compare_bipolar_beat_traces as pick_and_compare_bipolar
from .functions import pick_and_compare_unipolar_raw_traces as pick_and_compare_unipolar_raw
from .functions import pick_and_compare_bipolar_raw_traces as pick_and_compare_bipolar_raw
from .functions import visualise_torso_registration

__all__ = [
    "compute_torso_registration", "load_torso_registration",
    "pick_and_compare",
    "pick_and_compare_multi",
    "pick_and_compare_unipolar",
    "pick_and_compare_bipolar",
    "pick_and_compare_unipolar_raw",
    "pick_and_compare_bipolar_raw",
    "visualise_torso_registration",
]
