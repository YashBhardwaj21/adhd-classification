# Execution Environments & Dependency Specifications

This directory provides isolated environment dependency specifications for reproducing the analytical and modeling tracks of the ADHD-200 study.

---

## 1. Environment Architecture & Modular Tracks

Due to distinct deep learning, quantum computing, and connectomics software stacks across the experimental pipeline, dependencies are segregated into three modular tracks:

| Track | Experiments | Focus & Modalities | Python Version | Primary Packages | Requirements File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Exp 01–05 | Sliding-window dynamic FC, graph metrics (MST+PT), null models, ComBat harmonization, $k$-means micro-state discovery | 3.11 / 3.13 | `scipy`, `bctpy`, `neuroCombat`, `scikit-learn` | [`track_a/requirements.txt`](track_a/requirements.txt) |
| **Track B** | Exp 06–07 | Semi-supervised pseudo-labeling, Classical GCN, and Quantum Graph Neural Networks (QGCNN) | 3.12 | `torch==2.5.1`, `torch-geometric==2.8.0`, `pennylane==0.44.1`, `pennylane-lightning-gpu` | [`track_b/requirements.txt`](track_b/requirements.txt) |
| **Track C** | Exp 09 | Leave-One-Site-Out (LOSO) population graph learning & classical machine learning baselines | 3.12 | `torch==2.5.1`, `torch-geometric==2.8.0`, `scikit-learn==1.8.0`, `scipy==1.17.1` | [`track_c/requirements.txt`](track_c/requirements.txt) |

> [!NOTE]
> **Experiment 08 (NeuroSTORM / Volumetric Baselines)**:
> The NeuroSTORM foundation model requires PyTorch 2.7.1, CUDA 12.6, and custom compilation for `causal-conv1d` (v1.5.0.post8) and `mamba-ssm` (v2.2.2). See [`docs/reproduction.md`](../docs/reproduction.md) for Docker setup instructions.

---

## 2. Installation Guidelines

### Track A (Connectomics & Harmonization)
```bash
conda create -n adhd_track_a python=3.11 -y
conda activate adhd_track_a
pip install -r environment/track_a/requirements.txt
```

### Track B (Semi-Supervised & Quantum Graph Learning)
```bash
conda create -n adhd_track_b python=3.12 -y
conda activate adhd_track_b
# Recommended CUDA 12.1 PyTorch install:
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r environment/track_b/requirements.txt
```

### Track C (Leave-One-Site-Out Population Graphs)
```bash
conda create -n adhd_track_c python=3.12 -y
conda activate adhd_track_c
# Recommended CUDA 12.1 PyTorch install:
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r environment/track_c/requirements.txt
```

---

## 3. Audited Environment Manifests

Complete frozen package manifests from the audited cluster runs are retained under `configs/`:
- `configs/exp09/execution_environment.json`: Exact Python 3.12.13 runtime, PyTorch 2.5.1, PyG 2.8.0, OS kernel, GPU hardware, and `pip freeze` metadata from the canonical Exp 09 LOSO run.
- `configs/exp09/dataset_manifest.json`: Verified input dataset file hashes, paths, and sample counts.
