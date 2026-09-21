import os
import ctypes

try:
    ctypes.CDLL("nvapi64.dll")
except Exception:
    pass

os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()


def plot_voltage_trace_at_picked_node(
    npz_path,
    vmin=-85.0,
    vmax=35.0,
    cmap="jet",
    initial_frame=0,
):
    # ── Load data ────────────────────────────────────────────
    data       = np.load(npz_path, allow_pickle=False)
    t_arr      = data["time"]
    comp_nodes = data["comp_nodes"]
    comp_edges = data["comp_edges"]
    N          = len(comp_nodes)

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
    V_all = np.full((len(t_arr), N), float(vmin), dtype=np.float32)
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
    mesh.point_data["V"] = V_all[initial_frame]

    # ── State shared with callback ───────────────────────────
    state = {"picked_node": None}

    # ── Picking callback ─────────────────────────────────────
    def on_pick(point):
        dists    = np.linalg.norm(comp_nodes - np.array(point), axis=1)
        node_idx = int(np.argmin(dists))
        state["picked_node"] = node_idx

        coords = comp_nodes[node_idx]
        print(f"Picked node {node_idx} | "
              f"x={coords[0]:.3f}, y={coords[1]:.3f}, z={coords[2]:.3f}")

        sphere = pv.Sphere(radius=0.3, center=coords)
        pl.add_mesh(sphere, color="white", name="picked_sphere")
        pl.add_text(
            f"Node {node_idx}  ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})",
            name="pick_label",
            font_size=10,
            color="white",
        )
        pl.render()

    # ── Set up plotter ───────────────────────────────────────
    pl = pv.Plotter(notebook=False)
    pl.add_text(
        "Hover over a node and press P to pick, then close the window",
        name="instructions",
        font_size=10,
        color="yellow",
    )
    pl.add_mesh(
        mesh,
        scalars="V",
        cmap=cmap,
        clim=[vmin, vmax],
        line_width=2,
        scalar_bar_args={"title": "Vm (mV)"},
    )
    pl.enable_point_picking(
        callback=on_pick,
        show_message=True,
        show_point=True,
        color="red",
        point_size=10,
        tolerance=0.025,
    )

    pl.show()

    # ── Clean shutdown ───────────────────────────────────────
    try:
        pl.close()
    except Exception:
        pass

    # ── Plot voltage trace ───────────────────────────────────
    node_idx = state["picked_node"]

    if node_idx is None:
        print("No node was picked. Exiting.")
        return

    coords  = comp_nodes[node_idx]
    V_trace = V_all[:, node_idx]

    # Find activation time (first crossing above 0 mV)
    crossings  = np.where(V_trace > 0.0)[0]
    t_activate = t_arr[crossings[0]] if len(crossings) > 0 else None

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(t_arr, V_trace, color="crimson", linewidth=1.5, label=f"Node {node_idx}")
    ax.axhline(0.0,  color="gray",      linewidth=0.8, linestyle="--", label="0 mV")
    ax.axhline(vmin, color="steelblue", linewidth=0.8, linestyle="--",
               label=f"Rest ({vmin} mV)")

    if t_activate is not None:
        ax.axvline(t_activate, color="orange", linewidth=1.0, linestyle=":",
                   label=f"Activation ≈ {t_activate:.1f} ms")

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Membrane Voltage (mV)", fontsize=12)
    ax.set_title(
        f"Action Potential — Node {node_idx}\n"
        f"(x={coords[0]:.2f}, y={coords[1]:.2f}, z={coords[2]:.2f})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(vmin - 5, vmax + 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


