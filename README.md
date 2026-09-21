# ADHD-200 Connectomics

Code and supporting results for *ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical--Quantum Graph Learning, and Cross-Site Generalization*.

---

## Overview

This repository contains research code, configuration records, lightweight experimental results, and reproduction workflows for a multi-site resting-state fMRI study using the ADHD-200 dataset. The study evaluates dynamic functional connectivity (dFC), dual-constraint topological graph modeling, empirical Bayes (ComBat) scanner harmonization, semi-supervised graph convolutional networks, baseline deep architectures, and Leave-One-Site-Out (LOSO) population graph generalization across clinical acquisition sites.

---

## Experiments

1. **Dynamic Functional Connectivity**: Sliding-window correlation generation and temporal stability analysis ($W=30, S=5$).
2. **Graph Construction and Null Models**: Minimum Spanning Tree (MST) + proportional thresholding ($\rho=0.20$) and signed null models.
3. **Graph Features**: Extraction of dynamic topological metrics and cross-site scanner variation ANOVA.
4. **ComBat Harmonization**: Scanner batch-effect removal via empirical Bayes and classification trade-off evaluation.
5. **Dynamic States**: Unsupervised $K$-Means clustering ($K=3$) of recurring connectome states and transition dynamics.
6. **Semi-Supervised Labeling**: Self-training pseudo-label generation on the AAL-116 atlas.
7. **Classical GCN and QGCNN**: Evaluation of classical Graph Convolutional Networks and hybrid quantum-classical GCNNs on held-out test data.
8. **Deep-Learning Baselines**: Evaluation of volumetric 3D CNN, NeuroSTORM spatio-temporal transformer, and temporal GNN models.
9. **Leave-One-Site-Out Population Graphs**: 7-fold LOSO cross-validation across 497 subjects using GCN, GAT, GraphSAGE, and GIN.

---

## Data

ADHD-200 neuroimaging data are not redistributed in this repository. To obtain the dataset, consult [`data/README.md`](data/README.md). For details on excluded large arrays and checkpoints, see [`data/provenance.md`](data/provenance.md).

---

## Installation

```bash
# 1. Create a Python 3.12 environment
conda create -n adhd200 python=3.12.13 -y
conda activate adhd200

# 2. Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric and experiment dependencies
pip install torch-geometric==2.8.0
pip install -r environment/track_b/requirements.txt
pip install -e .
```

---

## Reproduction

For end-to-end execution sequences, reproducibility status, and detailed environment configurations, see [`docs/reproduction.md`](docs/reproduction.md). Scientific details and discrepancy notes are documented in [`docs/experiment_notes.md`](docs/experiment_notes.md) and [`docs/provenance.md`](docs/provenance.md).

---

## Repository Structure

```text
adhd-classification/
├── configs/          # Experiment configurations and execution metadata
├── data/             # Data requirements and provenance documentation
├── docs/             # Reproduction instructions, experiment notes, and provenance
├── environment/      # Track-specific dependency requirements
├── notebooks/        # Executed Jupyter notebooks for Experiments 01–09
├── results/          # Lightweight result tables, confusion matrices, and manifest
├── scripts/          # Structural audit and result verification utilities
├── src/              # Python source code for graph utilities and models
├── tests/            # Automated test suite
└── third_party/      # Third-party notices and license attributions
```

---

## Citation

```bibtex
@article{bhardwaj2026evaluating,
  title={ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical--Quantum Graph Learning, and Cross-Site Generalization},
  author={Khare, Shantanu and Deepthi, Bhavana and Bhardwaj, Yash},
  year={2026},
  note={Manuscript in preparation}
}
```
