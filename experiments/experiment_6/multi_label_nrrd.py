"""
multi_label_nrrd.py

Voxelize ALL of a patient's ECGI surface meshes (torso, heart muscle,
lungs, and the LV/AO + RV/PV chamber blood-pools) onto one shared,
isotropic voxel grid and write a single multi-LABEL NRRD volume
(float-typed, integer-valued) using the `nrrd` (pynrrd) package.

LABEL SCHEME
------------
    0 = outside torso
    1 = torso interior (body tissue not covered by any named organ)
    2 = heart_muscle   (EndoEpi.mat, falls back to Epi.mat)
    3 = left_lung      (otherStructures.Llung)
    4 = right_lung     (otherStructures.Rlung)
    5 = lv_ao          (otherStructures.LV_AO  -- LV + aorta blood pool)
    6 = rv_pv          (otherStructures.RV_PV  -- RV + pulm. trunk blood pool)

All structures are voxel-rasterized onto the *same* grid (same origin,
spacing, dimensions), built from the torso mesh's bounds so that the
output is physically correct -- 'space directions' in the NRRD header is
set to spacing*identity and 'space origin' to the grid origin, so the
volume lines up in physical (mm) space with the original meshes.

PAINTING ORDER (later structures overwrite earlier ones where they
overlap on the grid):
    torso interior -> left_lung -> right_lung -> heart_muscle
        -> lv_ao -> rv_pv
This puts the heart chambers (blood pools) on top of the myocardium
where they're nested inside it, and puts the organs on top of the
generic torso fill. After painting, every voxel outside the torso
surface is forced back to 0 regardless of what (if anything) got
painted there, so "outside torso" always means label 0.

CAVEAT -- meshes must be watertight for a correct inside/outside test;
see the heart_to_binary_nrrd.py docstring for details. The same
auto-fill-holes behaviour (FILL_HOLES_SIZE) is used here for every
structure.

Usage:
    python multi_label_nrrd.py <data_dir> <patient_id> [--spacing 1.0] [--out labels.nrrd]

Example:
    python multi_label_nrrd.py /mnt/user-data/uploads P001 --spacing 1.0
"""

import os
import json
import argparse
from collections import OrderedDict

import numpy as np
import scipy.io as sio
import pyvista as pv
import vtk
from vtk.util import numpy_support
import nrrd


# ============================================================================
# DEFAULTS -- used when run with no command-line arguments. Edit to match
# your setup, or just pass data_dir/patient_id on the command line.
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, 'geometry')
DEFAULT_PATIENT_ID = 'P001'
DEFAULT_SPACING = 1.0
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'inputs', 'torso_labels.nrrd')  # None -> '<patient_id>_labels.nrrd'

# Auto-fill holes up to this size (mesh units, e.g. mm) on any
# non-watertight surface before voxelizing. Set to 0 to disable.
FILL_HOLES_SIZE = 50.0

# How many voxels of padding to add around the torso bounds when sizing
# the shared grid.
PAD_VOXELS = 1


# Label ids, in their final numeric meaning.
LABEL_IDS = OrderedDict([
    ('torso_interior', 1),
    ('heart_muscle',   2),
    ('left_lung',      3),
    ('right_lung',     4),
    ('lv_ao',          5),
    ('rv_pv',          6),
])

# Order in which structures are painted onto the grid -- later entries
# overwrite earlier ones wherever they overlap. 'torso_interior' is
# painted first (it's the whole body cavity), organs next, chambers last
# (so they win over the myocardium they sit inside).
PAINT_ORDER = [
    'torso_interior',
    'left_lung',
    'right_lung',
    'heart_muscle',
    'lv_ao',
    'rv_pv',
]


# ---------------------------------------------------------------- loading --

def _load_top_struct(matfile_path):
    data = sio.loadmat(matfile_path, struct_as_record=False, squeeze_me=True)
    keys = [k for k in data if not k.startswith('__')]
    if not keys:
        raise ValueError(f"No data found in {matfile_path}")
    return data[keys[0]]


def _normalize_mesh(points, faces):
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces)

    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
    if faces.shape[0] == 3 and faces.shape[1] != 3:
        faces = faces.T

    faces = faces.astype(np.int64)
    if faces.min() == 1:        # MATLAB 1-indexing -> convert to 0-indexed
        faces = faces - 1

    return points, faces


def load_mesh(struct):
    fields = set(getattr(struct, '_fieldnames', []))
    if {'node', 'face'} <= fields:
        return _normalize_mesh(struct.node, struct.face)
    if {'pts', 'fac'} <= fields:
        return _normalize_mesh(struct.pts, struct.fac)
    return None


def load_meshes_from_file(path):
    top = _load_top_struct(path)
    single = load_mesh(top)
    if single is not None:
        name = os.path.splitext(os.path.basename(path))[0]
        return {name: single}
    meshes = {}
    for field_name in getattr(top, '_fieldnames', []):
        sub = getattr(top, field_name)
        mesh = load_mesh(sub)
        if mesh is not None:
            meshes[field_name] = mesh
    return meshes


def _to_polydata(node, face):
    n_tris = face.shape[0]
    pv_faces = np.hstack([np.full((n_tris, 1), 3, dtype=np.int64), face]).ravel()
    return pv.PolyData(node, pv_faces)


def load_heart_surface(data_dir, patient_id):
    """EndoEpi preferred, Epi fallback."""
    heart_path = os.path.join(data_dir, f'{patient_id}_EndoEpi.mat')
    if not os.path.exists(heart_path):
        heart_path = os.path.join(data_dir, f'{patient_id}_Epi.mat')
    if not os.path.exists(heart_path):
        raise FileNotFoundError(
            f'No EndoEpi or Epi geometry file found for {patient_id} in {data_dir}'
        )
    node, face = next(iter(load_meshes_from_file(heart_path).values()))
    return _to_polydata(node, face)


def load_torso_surface(data_dir, patient_id):
    torso_path = os.path.join(data_dir, f'{patient_id}_torso.mat')
    if not os.path.exists(torso_path):
        raise FileNotFoundError(f'No torso geometry file found at {torso_path}')
    node, face = next(iter(load_meshes_from_file(torso_path).values()))
    return _to_polydata(node, face)


def load_other_structures(data_dir, patient_id):
    """Returns dict: {'Llung': PolyData, 'LV_AO': PolyData, 'Rlung': ..., 'RV_PV': ...}"""
    other_path = os.path.join(data_dir, f'{patient_id}_otherStructures.mat')
    if not os.path.exists(other_path):
        raise FileNotFoundError(f'No otherStructures file found at {other_path}')
    meshes = load_meshes_from_file(other_path)
    return {name: _to_polydata(node, face) for name, (node, face) in meshes.items()}


# ------------------------------------------------------------- voxelizing --

def build_shared_grid(bounds, spacing, pad_voxels=PAD_VOXELS):
    """
    Compute the shared (nx, ny, nz) dims and origin that every structure
    will be rasterized onto, from a single set of bounds
    (xmin, xmax, ymin, ymax, zmin, zmax) -- normally the torso's bounds.
    """
    pad = spacing * pad_voxels
    xmin, xmax = bounds[0] - pad, bounds[1] + pad
    ymin, ymax = bounds[2] - pad, bounds[3] + pad
    zmin, zmax = bounds[4] - pad, bounds[5] + pad

    nx = int(np.ceil((xmax - xmin) / spacing)) + 1
    ny = int(np.ceil((ymax - ymin) / spacing)) + 1
    nz = int(np.ceil((zmax - zmin) / spacing)) + 1

    origin = np.array([xmin, ymin, zmin])
    return (nx, ny, nz), origin


def _ensure_watertight(surface, name='structure'):
    surface = surface.triangulate()
    if not surface.is_manifold:
        if FILL_HOLES_SIZE > 0:
            print(f'[info] {name}: surface not watertight -- attempting '
                  f'fill_holes(hole_size={FILL_HOLES_SIZE}).')
            surface = surface.fill_holes(FILL_HOLES_SIZE)
            if not surface.is_manifold:
                print(f'[warn] {name}: still not fully watertight after '
                      f'fill_holes -- voxels near remaining gaps may be '
                      f'misclassified.')
        else:
            print(f'[warn] {name}: surface not watertight and '
                  f'FILL_HOLES_SIZE=0 -- voxels near holes may be '
                  f'misclassified.')
    return surface


def _select_inside_stencil(origin, spacing, dims, surface):
    """
    Fast stencil-based inside test using vtkPolyDataToImageStencil.
    Works per-slice (scan-fill), so it is very fast but sensitive to
    mesh gaps -- any open hole lets the fill leak out. Use for large,
    well-closed surfaces (torso, lungs, blood-pool chambers).
    Returns a boolean (nx, ny, nz) array, True = inside the surface.
    """
    nx, ny, nz = dims
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing, spacing, spacing)
    img.SetOrigin(origin)
    img.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    fg = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    fg[:] = 1  # start all-foreground; the stencil carves out the exterior

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(surface)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing, spacing, spacing)
    pol2stenc.SetOutputWholeExtent(img.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(img)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    out_arr = numpy_support.vtk_to_numpy(imgstenc.GetOutput().GetPointData().GetScalars())
    out_arr = out_arr.reshape(nz, ny, nx)              # vtk: X fastest, then Y, then Z
    out_arr = np.transpose(out_arr, (2, 1, 0))          # -> (nx, ny, nz)
    return out_arr.astype(bool)


def _select_inside_enclosed(origin, spacing, dims, surface):
    """
    Robust per-voxel inside test using pyvista select_enclosed_points /
    select_interior_points. Tests every voxel centre independently against
    the mesh, so small holes do NOT cause a flood-fill leak -- the test
    simply classifies each voxel on its own. Slower than the stencil
    method (O(N_voxels) ray-casts vs O(N_slices) scan-fills), but gives
    clean results even on meshes that are not fully watertight, matching
    the behaviour of nrrdise.py. Use for the heart mesh.
    Returns a boolean (nx, ny, nz) array, True = inside the surface.
    """
    nx, ny, nz = dims
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(spacing, spacing, spacing),
        origin=origin,
    )
    if hasattr(grid, 'select_interior_points'):
        selection = grid.select_interior_points(surface, check_surface=False)
        inside_flat = selection['selected_points']
    else:
        selection = grid.select_enclosed_points(surface, tolerance=0.0, check_surface=False)
        inside_flat = selection['SelectedPoints']

    # pyvista ImageData points: X varies fastest, then Y, then Z.
    inside = inside_flat.reshape(nz, ny, nx)
    inside = np.transpose(inside, (2, 1, 0))   # -> (nx, ny, nz)
    return inside.astype(bool)


def _select_inside(origin, spacing, dims, surface, method='stencil'):
    """Dispatcher: 'stencil' (fast, leak-prone) or 'enclosed' (robust, slower)."""
    if method == 'enclosed':
        return _select_inside_enclosed(origin, spacing, dims, surface)
    return _select_inside_stencil(origin, spacing, dims, surface)


def rasterize_on_grid(origin, spacing, dims, surface, name='structure', method='stencil'):
    """
    Rasterize one closed pyvista surface onto the FULL shared grid.
    Returns a boolean (nx, ny, nz) array, True = inside the surface.
    Used for the torso (which legitimately spans the whole grid) --
    see rasterize_local_into() for everything smaller.
    """
    surface = _ensure_watertight(surface, name=name)
    inside = _select_inside(origin, spacing, dims, surface, method=method)
    n_inside = int(inside.sum())
    print(f'[info] {name}: {n_inside} / {inside.size} voxels inside (full grid, method={method})')
    return inside


def rasterize_local_into(label_volume, surface, global_origin, global_dims,
                          spacing, label_value, name='structure',
                          pad_voxels=PAD_VOXELS, method='stencil'):
    """
    Rasterize `surface` on a small sub-grid sized to just its own bounds
    (padded), aligned to the same voxel lattice as the global grid, then
    stamp `label_value` into label_volume wherever it's inside. This is
    far cheaper than testing the whole-body grid for small structures
    like the heart, a lung, or a chamber blood pool.

    method : 'stencil' (fast, default) or 'enclosed' (robust for leaky meshes).
    """
    surface = _ensure_watertight(surface, name=name)

    bounds = surface.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    pad = spacing * pad_voxels

    lo = np.empty(3, dtype=int)
    hi = np.empty(3, dtype=int)
    for d in range(3):
        bmin = bounds[2 * d] - pad
        bmax = bounds[2 * d + 1] + pad
        i_min = int(np.floor((bmin - global_origin[d]) / spacing))
        i_max = int(np.ceil((bmax - global_origin[d]) / spacing))
        lo[d] = max(i_min, 0)
        hi[d] = min(i_max, global_dims[d] - 1)

    sub_nx, sub_ny, sub_nz = (hi - lo + 1)
    sub_origin = tuple(global_origin + lo * spacing)
    sub_dims = (int(sub_nx), int(sub_ny), int(sub_nz))

    inside_local = _select_inside(sub_origin, spacing, sub_dims, surface, method=method)

    n_inside = int(inside_local.sum())
    print(f'[info] {name}: {n_inside} / {inside_local.size} voxels inside '
          f'(local sub-grid {sub_nx}x{sub_ny}x{sub_nz}, method={method})')

    sl = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1))
    region = label_volume[sl]
    region[inside_local] = label_value
    label_volume[sl] = region


# ------------------------------------------------------------------ saving --

def save_label_nrrd(volume, origin, spacing, out_path):
    """Write a float-typed, integer-valued, binary-encoded NRRD labelmap."""
    header = {
        'type': 'float',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': list(volume.shape),
        'space directions': [
            [spacing, 0, 0],
            [0, spacing, 0],
            [0, 0, spacing],
        ],
        'space origin': list(origin),
        'kinds': ['domain', 'domain', 'domain'],
        'encoding': 'raw',
    }
    nrrd.write(out_path, volume.astype(np.float32), header)
    print(f'Saved multi-label NRRD volume to {out_path}  '
          f'(shape={volume.shape}, spacing={spacing} mm, type=float, encoding=raw)')


def save_label_legend(out_path):
    legend_path = os.path.splitext(out_path)[0] + '_labels.json'
    with open(legend_path, 'w') as f:
        json.dump({'0': 'outside_torso', **{str(v): k for k, v in LABEL_IDS.items()}},
                   f, indent=2)
    print(f'Saved label legend to {legend_path}')


# --------------------------------------------------------------------- core --

def build_multi_label_volume(data_dir, patient_id, spacing):
    torso = load_torso_surface(data_dir, patient_id)
    (nx, ny, nz), origin = build_shared_grid(torso.bounds, spacing)
    global_dims = (nx, ny, nz)
    print(f'[info] shared voxel grid: {nx} x {ny} x {nz} @ {spacing} mm')

    # Load every other structure up front.
    other = load_other_structures(data_dir, patient_id)
    surfaces = {
        'heart_muscle': load_heart_surface(data_dir, patient_id),
        'left_lung': other['Llung'],
        'right_lung': other['Rlung'],
        'lv_ao': other['LV_AO'],
        'rv_pv': other['RV_PV'],
    }

    label_volume = np.zeros((nx, ny, nz), dtype=np.float32)

    # 1) Whole-body fill: this genuinely spans the full grid.
    torso_inside = rasterize_on_grid(tuple(origin), spacing, global_dims, torso,
                                      name='torso_interior')
    label_volume[torso_inside] = LABEL_IDS['torso_interior']

    # 2) Everything else is much smaller than the torso -- rasterize each
    #    on its own local sub-grid (same lattice) and stamp it in.
    #    heart_muscle uses the robust 'enclosed' method (per-voxel ray-cast)
    #    to avoid flood-fill leaks through valve-ring gaps in the mesh.
    #    All other structures use the fast stencil method.
    ENCLOSED_STRUCTURES = {'heart_muscle'}
    for name in PAINT_ORDER:
        if name == 'torso_interior':
            continue
        method = 'enclosed' if name in ENCLOSED_STRUCTURES else 'stencil'
        rasterize_local_into(
            label_volume, surfaces[name], origin, global_dims, spacing,
            LABEL_IDS[name], name=name, method=method,
        )

    # Safety net: anything outside the torso surface is always 0, no
    # matter what any other structure's rasterization did out there.
    label_volume[~torso_inside] = 0

    return label_volume, origin


# --------------------------------------------------------------------- cli --

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?', default=DEFAULT_DATA_DIR,
                         help=f'directory containing the .mat files '
                              f'(default: {DEFAULT_DATA_DIR})')
    parser.add_argument('patient_id', nargs='?', default=DEFAULT_PATIENT_ID,
                         help=f'e.g. P001 (default: {DEFAULT_PATIENT_ID})')
    parser.add_argument('--spacing', type=float, default=DEFAULT_SPACING,
                         help=f'isotropic voxel size in mm (default: {DEFAULT_SPACING})')
    parser.add_argument('--out', default=DEFAULT_OUT,
                         help='output .nrrd path (default: <patient_id>_labels.nrrd)')
    args = parser.parse_args()

    out_path = args.out or f'{args.patient_id}_labels.nrrd'

    print(f'[info] data_dir={args.data_dir}  patient_id={args.patient_id}  '
          f'spacing={args.spacing} mm  out={out_path}')

    volume, origin = build_multi_label_volume(args.data_dir, args.patient_id, args.spacing)
    save_label_nrrd(volume, origin, args.spacing, out_path)
    save_label_legend(out_path)

    # Quick summary of label counts.
    print('[info] label voxel counts:')
    print(f'    0  outside_torso     : {int(np.sum(volume == 0))}')
    for name, val in LABEL_IDS.items():
        print(f'    {val}  {name:<16}: {int(np.sum(volume == val))}')


if __name__ == '__main__':
    main()
