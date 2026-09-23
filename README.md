# ADHD-200 Connectomics

Code and supporting results for *ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical--Quantum Graph Learning, and Cross-Site Generalization*.

---

## Overview

This repository contains research code, configuration records, lightweight experimental results, and reproduction workflows for a multi-site resting-state fMRI study using the ADHD-200 dataset. The study evaluates dynamic functional connectivity (dFC), dual-constraint topological graph modeling, empirical Bayes (ComBat) scanner harmonization, semi-supervised pseudo-labeling and graph neural network classification, baseline deep architectures, and Leave-One-Site-Out (LOSO) population graph generalization across clinical acquisition sites.

---

## Experiments

1. **Dynamic Functional Connectivity**: Sliding-window correlation generation and temporal stability analysis ($W=30, S=5$).
2. **Graph Construction and Null Models**: Minimum Spanning Tree (MST) + proportional thresholding ($\rho=0.20$) and signed null models.
3. **Graph Features**: Extraction of dynamic topological metrics and cross-site scanner variation ANOVA.
4. **ComBat Harmonization**: Scanner batch-effect removal via empirical Bayes and classification trade-off evaluation.
5. **Dynamic States**: Unsupervised $K$-Means clustering ($K=3$) of recurring connectome states and transition dynamics.
6. **Semi-Supervised Labeling**: Evaluation of multiple pseudo-labeling and ensemble-selection approaches on AAL-116 functional-connectivity features.
7. **Classical GCN and QGCNN**: Evaluation on a separately prepared cohort of 162 clean and 713 selected pseudo-labeled subjects, with pseudo-labeled subjects used only for training.
8. **Deep-Learning Baselines**: Evaluation of volumetric 3D CNN, NeuroSTORM spatio-temporal transformer, and temporal GNN models.
9. **Leave-One-Site-Out Population Graphs**: 7-fold LOSO cross-validation across 497 subjects using GCN, GAT, GraphSAGE, and GIN.

---

## Data

ADHD-200 neuroimaging data are not redistributed in this repository. To obtain the dataset, consult [`data/README.md`](data/README.md). For details on excluded large arrays and checkpoints, see [`data/provenance.md`](data/provenance.md).

---

## Installation & Environments

Dependency requirements are organized into track-specific environments under `environment/`:
- **Track A (Experiments 01–05)**: Uses `environment/track_a/requirements.txt`, which includes `bctpy` for the Experiment 02 BCT-dependent graph null-model functionality.
- **Track B (Experiments 06–08)**: Uses `environment/track_b/requirements.txt` for deep learning and hybrid quantum-classical GNN pipelines. Experiment 07 itself does not require `bctpy` merely because BCT exists elsewhere in the repository.
- **Track C (Experiment 09)**: Uses `environment/track_c/requirements.txt` for population-level graph neural network learning.

```bash
# 1. Create a dedicated Python 3.12 environment
conda create -n adhd200 python=3.12.13 -y
conda activate adhd200

# 2. Install PyTorch with CUDA 12.1 support (for Track B or C)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric and track dependencies (e.g. Track B)
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
├── LICENSES/         # Official third-party license texts (GPL-3.0-or-later, Apache-2.0)
├── notebooks/        # Executed Jupyter notebooks for Experiments 01-09
├── results/          # Lightweight result tables, confusion matrices, and manifest
├── scripts/          # Structural audit and result verification utilities
├── src/              # Python source code for graph utilities and models
├── tests/            # Automated test suite
└── third_party/      # Third-party notices and license attributions
```

---

## License

Original project code is released under the MIT License. Bundled third-party source components retain their respective upstream licenses as documented in NOTICE and third_party/.

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
