"""
purkinje/monodomain.py
========================
Compute/load pair for the Purkinje-only monodomain voltage simulation
(Step 6) -- runs the ionic model + cable diffusion along the tree from
compute_network / load_network.

solve_monodomain (the underlying physics function) always writes its
result to disk -- there's no in-memory-only mode -- so compute_vm here
just runs it and immediately hands you back what it wrote, saving you
a separate load call.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .functions import solve_monodomain
from .._spacing_resolve import resolve_voxel_size


def compute_vm(
    nodes,
    elements,
    act_times,
    save_path,
    *,
    anatomy=None,
    ionic_model: str = "TTP06",
    voxel_size=0.4,
    dx: float = 0.2,
    dt: float = 0.1,
    T: float = 400.0,
    sigma_P: float = 0.225,
    stim_amp: float = 12.0,
    stim_dur: float = 2.0,
    save_every: int = 40,
    device=None,
):
    """
    Run the Purkinje-only monodomain simulation and save the voltage
    traces (this function always writes to save_path -- required, not
    optional, same as the underlying solver).

    Parameters
    ----------
    nodes, elements, act_times : from compute_network / load_network
    save_path    : output .npz path (required)
    anatomy      : dict, optional -- from coordinates.compute_geometry /
                   load_geometry. Only needed (and only used) when
                   voxel_size="from_geometry"; ignored otherwise.
    ionic_model  : e.g. "TTP06"
    voxel_size   : float | "from_geometry" -- mm per voxel (converts
                   nodes from voxel -> mm). Pass a float to hand-pick a
                   value as before, or "from_geometry" to pull it from
                   anatomy["spacing_mm"] instead, so it can't drift out
                   of sync with what compute_geometry actually resampled
                   the mask to (requires anatomy=...).
    dx, dt, T    : spatial step / time step / total duration (mm, ms, ms)
    sigma_P      : Purkinje axial conductivity
    stim_amp, stim_dur : stimulus amplitude / duration
    save_every   : save every N timesteps
    device       : "cuda", "cpu", or None (auto)

    Returns
    -------
    vm : dict -- time, comp_nodes, comp_edges, and per-branch voltage
         arrays (bmap_*, branch_*), i.e. exactly what was saved
    """
    voxel_size = resolve_voxel_size(voxel_size, anatomy, param_name="voxel_size")
    solve_monodomain(
        nodes=nodes, elements=elements, activation_times=act_times,
        ionic_model=ionic_model, save_path=str(save_path),
        voxel_size=voxel_size, dx=dx, dt=dt, T=T,
        sigma_P=sigma_P, stim_amp=stim_amp, stim_dur=stim_dur,
        save_every=save_every, device=device,
    )
    return load_vm(save_path)


def load_vm(path):
    """Load a previously saved Purkinje monodomain result. No recomputation."""
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved Purkinje simulation at {path}")
    z = np.load(path)
    return {k: z[k] for k in z.files}
