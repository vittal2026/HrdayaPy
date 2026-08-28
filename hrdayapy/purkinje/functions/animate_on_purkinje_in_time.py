import os
import ctypes

# Force NVIDIA GPU before any VTK/PyVista imports
try:
    ctypes.CDLL("nvapi64.dll")
except Exception:
    pass

# Suppress VTK warnings
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"

import time as time_module
import numpy as np
import pyvista as pv
import vtk

# Silence VTK log output
vtk.vtkObject.GlobalWarningDisplayOff()


def animate_on_purkinje_in_time(
    npz_path,
    vmin=-85.0,
    vmax=35.0,
    cmap="jet",
    speed=0.010,
    notebook=False,
):
    # ── Load data ────────────────────────────────────────────
    data       = np.load(npz_path, allow_pickle=False)
    t_arr      = data["time"]        # (n_frames,)
    comp_nodes = data["comp_nodes"]  # (N, 3)
    comp_edges = data["comp_edges"]  # (K, 2)
    n_frames   = len(t_arr)
    N          = len(comp_nodes)

    dt_frame = float(t_arr[1] - t_arr[0]) if n_frames > 1 else 1.0
    sleep_s  = (dt_frame / 1000.0) / speed

    print(f"Simulated dt per frame : {dt_frame:.4f} ms")
    print(f"Playback speed         : {speed}x real cardiac time")
    print(f"Sleep between frames   : {sleep_s*1000:.2f} ms wall clock")
    print(f"Total animation time   : {sleep_s * n_frames:.1f} s  "
          f"(simulated {t_arr[-1]:.1f} ms)")

    # ── Reconstruct branch map ───────────────────────────────
    bmap_keys  = sorted(
        [k for k in data.files if k.startswith("bmap_")],
        key=lambda k: int(k.split("_")[1])
    )
    branch_map = [data[k].tolist() for k in bmap_keys]

    # ── Reconstruct flat voltage array (n_frames, N) ─────────
    branch_keys = sorted(
        [k for k in data.files if k.startswith("branch_")],
        key=lambda k: int(k.split("_")[1])
    )
    V_all = np.full((n_frames, N), float(vmin), dtype=np.float32)
    for b_idx, key in enumerate(branch_keys):
        bh       = data[key]
        node_ids = branch_map[b_idx]
        V_all[:, node_ids] = bh

    # ── Build PyVista line mesh ──────────────────────────────
    lines = np.hstack([
        np.full((len(comp_edges), 1), 2, dtype=np.int_),
        comp_edges
    ]).ravel()

    mesh        = pv.PolyData()
    mesh.points = comp_nodes.astype(np.float32)
    mesh.lines  = lines

    # ── Set up plotter ───────────────────────────────────────
    pl = pv.Plotter(notebook=notebook)
    pl.add_text("t = 0.00 ms", name="time_label", font_size=12)

    mesh.point_data["V"] = V_all[0]
    pl.add_mesh(
        mesh,
        scalars="V",
        cmap=cmap,
        clim=[vmin, vmax],
        line_width=2,
        scalar_bar_args={"title": "Vm (mV)"},
    )

    pl.show(auto_close=False, interactive_update=True)

    # ── Animation loop ───────────────────────────────────────
    for f in range(n_frames):
        # Exit cleanly if user closed the window
        try:
            if not pl.render_window or not pl.render_window.GetNeverRendered() == 0:
                pass  # still alive
        except Exception:
            break

        # Check if window is still open via VTK
        try:
            rw = pl.ren_win
            if rw is None or not rw.GetInteractor():
                break
        except Exception:
            break

        mesh.point_data["V"] = V_all[f]
        pl.add_text(f"t = {t_arr[f]:.2f} ms", name="time_label", font_size=12)

        try:
            pl.render()
            pl.update()
        except Exception:
            break

        time_module.sleep(sleep_s)

    # ── Clean shutdown ───────────────────────────────────────
    try:
        pl.close()
    except Exception:
        pass
