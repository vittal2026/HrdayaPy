# HrdayaPy

An open-source, GPU-accelerated, voxel-based pipeline for patient-specific cardiac
electrophysiology, implemented entirely in Python/PyTorch. HrdayaPy computes
transmural and apicobasal ventricular coordinates, generates a patient-specific
Purkinje network, runs a coupled Purkinje-myocardium monodomain simulation
(ten Tusscher-Panfilov TTP06 ionic model with explicit resistive PMJ coupling),
and solves a body-surface forward problem to produce simulated ECG / BSPM signals.

See the accompanying manuscript for full methodological details and validation
results: *HrdayaPy: an open-source Python/PyTorch pipeline for GPU-accelerated,
voxel-based cardiac electrophysiology simulation with coupled Purkinje-myocardium
conduction and a body-surface forward solve* (Rao & Krishna Kumar).

**This repository is code only.** No patient data, geometry files, or simulation
outputs are included. The patient geometry used for validation in the manuscript
(Patient P001) is from the Bratislava Dataset, available via the EDGAR repository:
https://edgar.sci.utah.edu

## Repository structure

- `hrdayapy/` — the installable Python package (four sub-packages: `coordinates`,
  `purkinje`, `simulation`, `ecg`). This is what you get from `pip install hrdayapy`.
- `experiments/` — the scripts used to generate every table and figure in the
  manuscript (Experiments 1-7). Not part of the pip package; clone the repo to
  access these.

## Installation

### 1. Install PyTorch first (do this yourself, before installing hrdayapy)

HrdayaPy's numerically intensive steps (Laplace/Poisson solves, ionic-model ODE
integration, monodomain time-stepping) run on GPU via PyTorch. `pip install hrdayapy`
will pull in *some* version of `torch`, but for GPU acceleration you should install
the build matching your own CUDA version **first**, following the official
instructions at https://pytorch.org/get-started/locally/

A CPU-only PyTorch install will also work, just much more slowly.

### 2. Install hrdayapy

```bash
pip install hrdayapy
```

### 3. A note on two dependencies that can be awkward to install

- **`pypardiso`** requires Intel MKL underneath it.
- **`scikit-sparse`** requires the SuiteSparse C libraries, which are not pulled
  in automatically by pip.

If either fails to build from a plain `pip install`, the most reliable fix on
Windows/macOS/Linux alike is to install them via conda instead:

```bash
conda install -c conda-forge pypardiso scikit-sparse
```

then `pip install hrdayapy` afterward in the same environment.

## Quickstart

```python
import hrdayapy as hp

# See the docstrings in hrdayapy.coordinates, hrdayapy.purkinje,
# hrdayapy.simulation, and hrdayapy.ecg for the four pipeline stages.
```

*(TODO: add a short worked code example here once the public API is finalised —
e.g. loading a segmentation, computing coordinates, generating a Purkinje network.)*

## Reproducing the paper's experiments

Each `experiments/experiment_N/` folder contains the exact script(s) used to
produce the corresponding table/figure in the manuscript. These require the
patient geometry (EDGAR, Patient P001) and a torso segmentation, which are not
included in this repository — see the manuscript's Data Availability statement.

Running the experiments requires the same dependencies as the package itself;
install hrdayapy as above first.

## Citation

If you use HrdayaPy in your research, please cite:

```
Rao, V.L., Krishna Kumar, R. HrdayaPy: an open-source Python/PyTorch pipeline
for GPU-accelerated, voxel-based cardiac electrophysiology simulation with
coupled Purkinje-myocardium conduction and a body-surface forward solve.
(manuscript in preparation)
```

## License

MIT License — see [LICENSE](LICENSE).