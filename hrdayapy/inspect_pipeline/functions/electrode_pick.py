"""
inspect_pipeline/functions/electrode_pick.py
===============================================
Shared electrode-picking PyVista window, backing both
pick_and_compare_unipolar.py (pick ONE electrode) and
pick_and_compare_bipolar.py (pick TWO electrodes, A then B).

Ported from the experimental scripts' ecg_picker_common.py
(pick_electrodes_blocking), adapted to pick against the electrode
positions already loaded/registered by the caller (this module does no
loading of its own -- registration, torso mesh, and BSPM are all read
once by the calling pick_and_compare_* function and passed in here).

This is a single BLOCKING pl.show() call -- whatever's picked when the
window is closed is what gets returned. There is no incremental/live
plotting while picking; the representative (averaged) beat is only
plotted once, after the window closes.
"""

from __future__ import annotations

import numpy as np


def pick_electrodes(
    gt_mesh,
    electrode_points: np.ndarray,
    max_picks: int,
    snap_warn_mm: float = 15.0,
    window_title: str = "Torso -- pick electrode(s), 'c' to clear",
) -> list[int]:
    """
    Opens ONE blocking PyVista window showing `gt_mesh` (the registered
    dataset torso surface) with `electrode_points` overlaid. Click near an
    electrode to pick it (up to max_picks); picking past max_picks drops
    the oldest pick, 'c' clears everything. Returns only once the window
    is closed, giving back the indices (into electrode_points) selected at
    that point, in pick order (oldest first).

    Returns [] if the window was closed with nothing picked.
    """
    import pyvista as pv

    pv.set_plot_theme("dark")
    pl = pv.Plotter(title=window_title)
    pl.add_mesh(gt_mesh, color="peachpuff", opacity=0.95, smooth_shading=True, show_edges=False)
    pl.add_points(electrode_points, color="blue", point_size=6, render_points_as_spheres=True)
    pl.add_axes()
    pl.add_text(
        "peachpuff = torso    blue = electrodes    magenta = picked",
        position="lower_left", font_size=10, color="white",
    )
    pl.add_title(f"Pick up to {max_picks} electrode(s) -- close this window "
                 f"when done to see the representative-beat plot")

    picked_idx: list = []
    picked_xyz: list = []

    def on_pick(point, *_):
        click_xyz = np.asarray(point, dtype=np.float64)
        d = np.linalg.norm(electrode_points - click_xyz, axis=1)
        idx = int(np.argmin(d))
        if idx in picked_idx:
            print(f"  Electrode #{idx} is already picked -- ignoring.")
            return
        picked_idx.append(idx)
        picked_xyz.append(electrode_points[idx])
        if len(picked_idx) > max_picks:
            picked_idx.pop(0)
            picked_xyz.pop(0)

        labels = [chr(ord("A") + i) for i in range(len(picked_idx))]
        pl.add_point_labels(
            np.array(picked_xyz), labels, point_color="magenta", point_size=16,
            render_points_as_spheres=True, font_size=20, name="picks",
        )
        click_snap = d[idx]
        print(f"  Clicked {np.round(click_xyz, 1)} mm -> electrode #{idx} "
              f"({click_snap:.1f} mm away) -- currently picked: {picked_idx}")
        if click_snap > snap_warn_mm:
            print(f"    [warn] {click_snap:.1f} mm click-to-electrode distance "
                  f"is large -- you may be clicking between two electrodes.")

    message = (
        f"Left-click near an electrode (blue dot) to pick it "
        f"(up to {max_picks}).\nPress 'c' to clear.\n"
        f"Close this window when done to plot the representative beat."
    )
    if hasattr(pl, "enable_surface_point_picking"):
        pl.enable_surface_point_picking(
            callback=on_pick, show_message=message, color="magenta", point_size=14,
        )
    else:  # older pyvista fallback -- nearest-vertex picking
        pl.enable_point_picking(
            callback=on_pick, picker="point", left_clicking=True,
            show_message=message, color="magenta", point_size=14,
        )

    def clear_picks():
        picked_idx.clear()
        picked_xyz.clear()
        try:
            pl.remove_actor("picks")
        except Exception:
            pass
        print("  Cleared picks.")

    pl.add_key_event("c", clear_picks)

    print(f"\nReady -- click up to {max_picks} blue electrode dot(s), then "
          f"CLOSE the window to see the representative-beat plot.")
    pl.show()   # BLOCKING -- nothing is plotted until this returns

    return list(picked_idx)
