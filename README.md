# ADHD-200 Connectomics

Code and supporting results for *ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical--Quantum Graph Learning, and Cross-Site Generalization*.

---

## Overview

This repository contains research code, configuration records, lightweight experimental results, and reproduction workflows for a multi-site resting-state fMRI study using the ADHD-200 dataset. The study evaluates dynamic functional connectivity (dFC), dual-constraint topological graph modeling, empirical Bayes (ComBat) scanner harmonization, semi-supervised pseudo-labeling and graph neural network classification, baseline deep architectures, and Leave-One-Site-Out (LOSO) population graph generalization across clinical acquisition sites.

---

## Experiments

1. **Dynamic Functional Connectivity**: Sliding-window correlation generation and temporal stability analysis ($W=30, S=5$).
2. **Graph Construction and Null Models**: Minimum Spanning Tree (MST) + proportional thresholding ($\rho=0.20$) and signed null models.
3. **Graph Features**: Extraction of dynamic topological metrics, cross-site scanner variation ANOVA, and Welch unequal-variance diagnostic group comparisons.
4. **ComBat Harmonization**: Scanner batch-effect removal via empirical Bayes and classification trade-off evaluation.
5. **Dynamic States**: Dynamic micro-state clustering ($K=3$ selected via multi-index ranking across $K \in [2, 10]$) and transition dynamics.
6. **Semi-Supervised Labeling**: Evaluation of multiple pseudo-labeling and ensemble-selection approaches on AAL-116 functional-connectivity features.
7. **Classical GCN and QGCNN**: Evaluation on a separately prepared cohort of 162 clean and 713 selected pseudo-labeled subjects (117-d node features, unweighted degree, unweighted GCN message passing), with pseudo-labels restricted to training.
8. **Deep-Learning Baselines**: Evaluation of 4D volumetric 3D CNN and NeuroSTORM spatio-temporal transformer on 626 scans, alongside temporal GNN on 764-subject CC200 timeseries.
9. **Leave-One-Site-Out Population Graphs**: Leave-One-Site-Out subject-level functional-connectivity graph learning across 497 subjects and 7 sites using top-10% positive-FC thresholded graphs, alongside 7 classical ML classifier types evaluated across 6 feature families with fold-nested feature selection.

> [!NOTE]
> For classical machine learning baselines in Experiment 09, `results/exp09/w1_model_summary.csv` contains historical recorded model rows (including intermediate and duplicate evaluation runs), while `results/exp09/w1_family_winners.csv` provides the canonical family-level summary.

---

## Data

ADHD-200 neuroimaging data are not redistributed in this repository. To obtain the dataset, consult [`data/README.md`](data/README.md). For details on excluded large arrays and checkpoints, see [`data/provenance.md`](data/provenance.md).

---

## Installation & Environments

Due to conflicting PyTorch, CUDA, and specialized library dependencies across experimental generations, dependencies are strictly separated into track-specific environments under `environment/`:
- **Track A (Experiments 01–05)**: Python 3.12 CPU environment (`environment/track_a/requirements.txt`), including `scikit-learn`, `scipy`, `statsmodels`, `neuroCombat`, and `bctpy` (GPL-3.0 null-model algorithms).
- **Track B (Experiments 06–07)**: Python 3.12 GPU environment (`environment/track_b/requirements.txt`) with PyTorch 2.5.1+cu121, PyTorch Geometric 2.8.0, PennyLane 0.44.1, and PennyLane-Lightning-GPU 0.44.0.
- **Track B (Experiment 08 NeuroSTORM)**: Brev A100 environment requiring Python 3.12, PyTorch 2.7.1, CUDA 12.6, and pinned hardware kernels (`causal-conv1d v1.5.0.post8`, `mamba v2.2.2`).
- **Track C (Experiment 09)**: Azure A100 environment (`environment/track_c/requirements.txt`) with Python 3.12, PyTorch 2.5.1+cu121, and PyG 2.8.0.

> [!IMPORTANT]
> Do **not** attempt to install all tracks into a single monolithic Python environment. For exact setup instructions, dependency pins, and reproduction categories, see [`docs/reproduction.md`](docs/reproduction.md).

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
  author={Khare, Shantanu and Deepthi, Bhavana and Bhardwaj, Yash and Shridevi, S. and Won, Daehan},
  year={2026},
  note={Manuscript in preparation}
}
```
