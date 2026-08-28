"""
downsample_vm_snapshots.py
===========================
Spatially downsamples P001_vm_snapshots.npz by 0.5x per axis (8x volume
reduction) via 2x2x2 block-averaging, WITHOUT ever loading the full
(n_frames, N_myo) Vm array into RAM. Run this on the DGX box, then scp
just the (much smaller) output file to your PC.

Vm is stored via np.savez_compressed, so it can't be memory-mapped --
this script reads it directly out of the underlying zip archive one
frame at a time (streaming decompression), which keeps peak RAM roughly
constant regardless of how many frames/nodes there are.
"""
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
IN_PATH = SCRIPT_DIR / "outputs" / "P001" / "P001_vm_snapshots.npz"
OUT_PATH = SCRIPT_DIR / "outputs" / "P001" / "P001_vm_snapshots_ds4.npz"
BLOCK = 2  # 2x2x2 voxel blocks -> 0.5x per axis -> 8x volume reduction

print(f"Loading small arrays (node_ids, coords_mm, vox_idx, time_vm) ...")
with np.load(IN_PATH) as d:
    vox_idx = d["vox_idx"]        # (N_myo, 3) int32
    coords_mm = d["coords_mm"]    # (N_myo, 3) float32
    time_vm = d["time_vm"]        # (n_frames,) float32
    dt_solver = float(d["dt_solver"])
    vm_save_dt = float(d["vm_save_dt"])

N_myo = vox_idx.shape[0]
n_frames = time_vm.shape[0]
print(f"N_myo={N_myo:,}  n_frames={n_frames:,}")

# --- group nodes into 2x2x2 voxel blocks ---
block_ijk = vox_idx // BLOCK                      # (N_myo, 3)
_, inverse, counts = np.unique(
    block_ijk, axis=0, return_inverse=True, return_counts=True)
n_blocks = counts.shape[0]
inverse = inverse.astype(np.int64).ravel()
print(f"Downsampled node count: {n_blocks:,}  "
      f"({N_myo / n_blocks:.2f}x average nodes per block)")

# block-averaged coords/vox_idx/node_ids for the output file
new_coords = np.stack([
    np.bincount(inverse, weights=coords_mm[:, a].astype(np.float64), minlength=n_blocks)
    for a in range(3)
], axis=1)
new_coords /= counts[:, None]
new_coords = new_coords.astype(np.float32)

new_vox_idx = np.stack([
    np.bincount(inverse, weights=block_ijk[:, a].astype(np.float64), minlength=n_blocks)
    for a in range(3)
], axis=1)
new_vox_idx = np.round(new_vox_idx / counts[:, None]).astype(np.int32)

new_node_ids = np.arange(n_blocks, dtype=np.int32)

# --- stream Vm frame-by-frame straight out of the zip archive ---
Vm_ds = np.empty((n_frames, n_blocks), dtype=np.float32)

with zipfile.ZipFile(IN_PATH) as zf:
    with zf.open("Vm.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        assert not fortran_order, "expected C-order Vm array"
        assert shape == (n_frames, N_myo), f"unexpected Vm shape {shape}"

        frame_nbytes = N_myo * dtype.itemsize
        for t in range(n_frames):
            buf = stream.read(frame_nbytes)
            if len(buf) < frame_nbytes:
                raise IOError(f"truncated read at frame {t}")
            frame = np.frombuffer(buf, dtype=dtype)

            block_sum = np.bincount(inverse, weights=frame.astype(np.float64), minlength=n_blocks)
            Vm_ds[t] = (block_sum / counts).astype(np.float32)

            if t % 50 == 0:
                print(f"  frame {t}/{n_frames}  t={time_vm[t]:.1f} ms", flush=True)

print(f"Saving downsampled snapshots -> {OUT_PATH}")
np.savez_compressed(
    str(OUT_PATH),
    node_ids=new_node_ids,
    coords_mm=new_coords,
    vox_idx=new_vox_idx,
    Vm=Vm_ds,
    time_vm=time_vm,
    dt_solver=np.float32(dt_solver),
    vm_save_dt=np.float32(vm_save_dt),
)
print("Done.")
