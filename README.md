***ViscoReg***: Neural Signed Distance Functions via Viscosity Solutions
---
Created by [Meenakshi Krishnan](https://mkrishnan9.github.io/) and [Ramani Duraiswami](https://www.cs.umd.edu/people/ramanid) from the [Perceptual Interfaces and Reality Lab (PIRL)](https://pirl.umd.edu/), University of Maryland, College Park.

**[[Arxiv]](http://arxiv.org/abs/2507.00412)**

## Introduction

This repository contains the code for **ViscoReg**, a regularization method for training neural signed distance functions (SDFs) for 3D surface reconstruction from point clouds. We propose a viscosity-based regularizer derived from the theory of viscosity solutions to the eikonal equation, which stabilizes neural SDF training and produces state-of-the-art results.

We evaluate on four benchmarks: ShapeNet (NSP split), the Surface Reconstruction Benchmark (SRB), SCUT real scans, and scene reconstruction.

This codebase builds on [DiGS](https://github.com/Chumbyte/DiGS) and [StEik](https://github.com/steik-neurips/steik).

## Installation

Our codebase uses [PyTorch](https://pytorch.org/).

The code was tested with Python 3.9, PyTorch 2.0.1, tensorboardX 2.3, CUDA 11.8.
For a full list of requirements see [`requirements.txt`](requirements.txt).

```sh
conda create -n viscoreg python=3.9
conda activate viscoreg
conda install pip

# Install PyTorch 2.0.1 with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

## Data

### Surface Reconstruction Benchmark (SRB)
Provided in the [Deep Geometric Prior repository](https://github.com/fwilliams/deep-geometric-prior). Download via:
```sh
bash data/scripts/download_srb.sh   # ~1.1 GB
```

### ShapeNet (NSP split)
We use the Neural Splines subset of ShapeNet (first 20 shapes from the test set of 13 categories). Download via:
```sh
bash data/scripts/download_shapenet.sh   # ~784 MB
```

### Scene Reconstruction
Interior room scene from the [SIREN paper](https://github.com/vsitzmann/siren). Download via:
```sh
bash data/scripts/download_scene.sh   # ~56 MB
```

### SCUT Real Scans
Download from the [SCUT-Surf repository](https://github.com/scutsurf/scut-surf) and place under `data/ScutSurf_Data/`.

Please cite the respective datasets when using them in your research.

## Running Experiments

All scripts train, test, and compute metrics in one go. Edit the `GPUS` variable at the top of each script to match your available GPUs.

### ShapeNet
```sh
bash surface_reconstruction/scripts/run_shapenet.sh
```

### Surface Reconstruction Benchmark (SRB)
```sh
bash surface_reconstruction/scripts/run_srb.sh
```

### SCUT Real Scans
```sh
bash surface_reconstruction/scripts/run_scut.sh
```

### Scene Reconstruction
```sh
bash surface_reconstruction/scripts/run_scene.sh
```

## Sanity Checks (2D / 1D)

We provide 2D and 1D shape datasets for quickly verifying the method without any external data.

**2D shapes** (Mandelbrot fractal):
```sh
cd sanitychecks
bash scripts/run_train_test_basic_shape.sh
```

**1D shapes**:
```sh
cd sanitychecks
bash scripts/run_train_test_1d.sh
```

## Code Overview

| Path | Description |
|---|---|
| `models/losses.py` | **ViscoReg loss** (`siren_wo_n_w_visc`) and viscosity eikonal term |
| `models/Net.py` | Network architecture (linear and quadratic neuron variants) |
| `models/SIREN.py` | SIREN network module |
| `surface_reconstruction/train_surface_reconstruction.py` | Training entry point |
| `surface_reconstruction/test_surface_reconstruction.py` | Mesh extraction |
| `surface_reconstruction/compute_metrics_*.py` | Benchmark evaluation |
| `sanitychecks/` | 2D (Mandelbrot fractal) / 1D shape experiments |

## License and Citation

If you find our work useful in your research, please cite:

```bibtex
@article{krishnan2025viscoreg,
  title={ViscoReg: Neural Signed Distance Functions via Viscosity Solutions},
  author={Krishnan, Meenakshi and Duraiswami, Ramani},
  journal={arXiv preprint arXiv:2507.00412},
  year={2025}
}
```

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) file.

## Acknowledgements

This codebase builds on [DiGS](https://github.com/Chumbyte/DiGS) and [StEik](https://github.com/steik-neurips/steik). We thank the authors for releasing their code.

This work was supported by ONR Award N00014-23-1-2086.
