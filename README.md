# ADHD-200 Connectomics

Code, experiment configurations, and lightweight result artifacts for:

> **"ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical–Quantum Graph Learning, and Cross-Site Generalization"**  
> *Yash Bhardwaj, Bhavana Deepthi, Shantanu Khare, Shridevi S, and Daehan Won*

---

## Study Overview

This study investigates resting-state functional connectomics, machine learning representations, and cross-site diagnostic generalizability using the international **ADHD-200 Sample**. The research is organized into three independent experimental tracks:

- **Track A — Dynamic Connectivity and Graph Analysis (Experiments 01–05)**: Evaluates sliding-window functional connectivity stability, Minimum Spanning Tree (MST) graph construction, ComBat harmonization, and unsupervised micro-state discovery on the Craddock-200 (CC200) atlas.
- **Track B — Semi-Supervised and Quantum Graph Classification (Experiments 06–08)**: Investigates pseudo-labeling, benchmarks an isotropic Classical GCN against a parameterized 6-qubit Hybrid Quantum GCNN (QGCNN) on the AAL-116 atlas, and assesses 4D volumetric CNNs.
- **Track C — Cross-Site Generalization (Experiment 09)**: Assesses out-of-distribution generalizability across 497 subjects from 7 clinical imaging centers under strict 7-fold Leave-One-Site-Out (LOSO) cross-validation.

> **Methodological Independence**: The three tracks evaluate distinct cohorts, atlases, graph representations, and validation protocols; they represent independent empirical inquiries rather than a single unified benchmark pipeline.

---

## Experiments

| ID | Experiment | Brain Atlas / Input | Cohort Size | Primary Evaluation Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **01** | Dynamic FC Temporal Stability | CC200 (190 ROIs) | 764 subjects / 1,193 runs | Autocorrelation decay & Frobenius distance across lags 1–4 |
| **02** | MST+PT Graph Construction | CC200 (190 ROIs) | 764 subjects / 31,060 windows | Connectedness & degree-conserving signed null rewiring |
| **03** | Dynamic Graph Biomarkers | CC200 (190 ROIs) | 534 complete subjects | Welch $t$-tests with BH-FDR correction & cross-site ANOVA |
| **04** | ComBat Scanner Harmonization | CC200 (190 ROIs) | 534 complete subjects | Site prediction accuracy & topology preservation |
| **05** | Dynamic State Discovery | CC200 (190 ROIs) | 764 subjects / 1,193 acquisitions | Unsupervised $k$-means clustering ($K=3$) & dwell dynamics |
| **06** | Semi-Supervised Pseudo-Labeling | AAL-116 (116 ROIs) | 955 subjects (391 clean) | Self-training ($\tau \ge 0.75$) vs 4-model ensemble consensus |
| **07** | Classical GCN vs Hybrid QGCNN | AAL-116 (116 ROIs) | 162 clean + 713 pseudo | Matched benchmark on 33 held-out test clean subjects |
| **08** | Deep Learning Baselines | 4D BOLD / CC200 | 626 / 764 subjects | 3D CNN, NeuroSTORM spatiotemporal transformer, temporal GNN |
| **09** | LOSO Cross-Site Generalization | CC200 (190 ROIs) | 497 subjects / 7 sites | 7-fold LOSO: 42 classical ML baselines & 4 GNN architectures |

---

## Main Empirical Findings

| Experiment | Key Finding | Primary Metric / Evidence Artifact |
| :--- | :--- | :--- |
| **Exp 01** | Smooth, continuous temporal autocorrelation decay | Lag 1: **$0.8990$** $\to$ Lag 4: **$0.5467$** (`temporal_similarity_validation.csv`) |
| **Exp 02** | Guaranteed 100% graph connectedness | Density $\rho=0.20$, 3,591 edges (`graph_construction_configuration.csv`) |
| **Exp 03** | Massive site variation dominates over diagnostic effect | Scanner site effect: **$F > 4600$**, $p < 10^{-15}$ (`site_effect_anova.csv`) |
| **Exp 04** | ComBat substantially attenuates scanner bias | Site prediction drops from **58.64%** $\to$ **31.29%** (`site_prediction_combat_comparison.csv`) |
| **Exp 05** | 3 recurring micro-states; State 0 dominates dwell time | State 0 dwell time = **6.24 windows** (`dynamic_state_biomarkers.csv`) |
| **Exp 06** | Pseudo-label augmentation improves holdout accuracy | Baseline **67.09%** $\to$ Augmented **72.15%** (`exp06_verified_results.json`) |
| **Exp 07** | Matched GCN achieves higher observed AUC than 6-qubit QGCNN | Classical GCN: **0.7293 AUC** vs QGCNN: **0.6429 AUC** (`gcn_qgcnn_test_results.json`) |
| **Exp 08** | 3D spatial CNN achieves strongest baseline accuracy | 3D CNN: **76.19%**, NeuroSTORM 5-fold: **59.10%** (`baseline_model_results.json`) |
| **Exp 09** | Classical ML with phenotype achieves higher observed AUC than GNNs under LOSO | SVM (Graph+Pheno): **0.6493 AUC** vs GAT: **0.5752 AUC** (`gnn_loso_fold_results.csv`) |

---

## Reproduction & Execution

The repository categorizes execution reproducibility into three tiers:
- **Category A (Fully Rerunnable)**: Experiments 01, 02, 03, 04, and 05 can be executed from repository code and public connectome inputs.
- **Category B (Requires Excluded Intermediates)**: Experiments 06, 08, and 09 require unbundled intermediate arrays or 4D functional NIfTI volumes not distributed in the public repository.
- **Category C (Archived)**: Experiment 07 preserves exact configurations and verified test metrics, while historical training arrays are archived externally.

Detailed conda environments, execution workflows, and verification steps are provided in [`docs/reproduction.md`](docs/reproduction.md).

---

## Data Availability

Raw and preprocessed neuroimaging data from the ADHD-200 Consortium are not redistributed in this Git repository. Users may download the Athena CC200 and AAL-116 connectomes via the official [NITRC ADHD-200 Portal](https://www.nitrc.org/projects/fcon_1000/).

Full data policies and exclusion ledgers are documented in [`data/README.md`](data/README.md) and [`data/availability_and_exclusions.md`](data/availability_and_exclusions.md).

---

## Documentation

Comprehensive scientific, technical, and forensic documentation is organized under [`docs/`](docs/):

- **[Study Design](docs/study_design.md)**: Research questions, multi-track architecture, cohort partitions, and lineage.
- **[Methods](docs/methods.md)**: Detailed algorithms, graph construction, edge weight semantics, and model architectures.
- **[Results](docs/results.md)**: Verified numerical result tables, cross-site ANOVA statistics, and interpretation boundaries.
- **[Reproduction](docs/reproduction.md)**: Conda environment setup, execution commands, and test verification suite.
- **[Provenance](docs/provenance.md)**: Forensic evidence classification, cohort lineage ($162 \subset 391$), and cryptographic SHA-256 ledger.
- **[Limitations](docs/limitations.md)**: Methodological constraints, quantum information bottlenecks, and future research directions.

---

## Repository Structure

```text
adhd-classification/
├── README.md                           # Compact project overview
├── CITATION.cff                        # Citation metadata
├── LICENSE                             # MIT License
├── NOTICE                              # Third-party notices and BCT attributions
├── pyproject.toml                      # Package specifications & test configuration
│
├── configs/                            # Canonical execution configurations
│   ├── exp02/                          # CC200 MST+PT graph construction config
│   ├── exp07/                          # Classical GCN and QGCNN run hyperparameters
│   └── exp09/                          # LOSO GNN configs, thresholds, and manifests
│
├── data/                               # Data guidelines and exclusion manifests
│   ├── README.md                       # Data requirements and download instructions
│   └── availability_and_exclusions.md  # Detailed ledger of excluded large files
│
├── docs/                               # Core scientific companion documentation
│   ├── README.md                       # Documentation sitemap
│   ├── study_design.md                 # Research questions, tracks, and cohorts
│   ├── methods.md                      # Mathematical formulations & protocols
│   ├── results.md                      # Verified empirical metrics & tables
│   ├── reproduction.md                 # Execution commands & environment matrix
│   ├── provenance.md                   # Forensic lineage & historical reconciliations
│   └── limitations.md                  # Scientific boundaries & future work
│
├── environment/                        # Isolated Conda environment specifications
│   ├── track_a/                        # Python 3.11/3.13 Connectomics environment
│   ├── track_b/                        # PyTorch 2.5.1 / PennyLane 0.44.1 environment
│   └── track_c/                        # PyTorch Geometric 2.6.1 LOSO environment
│
├── notebooks/                          # Self-contained research notebooks & index (Exp 01–09)
├── results/                            # Lightweight canonical result artifacts
│   ├── README.md                       # Results directory index
│   ├── manifest.csv                    # Cryptographic SHA-256 evidence manifest
│   └── exp01/ ... exp09/               # Per-experiment CSV tables & JSON metrics
│
├── scripts/                            # Operational validation scripts
│   └── validate_results.py             # Numerical results integrity checker
├── src/                                # Reusable library code (common, exp02, exp07, exp08)
├── tests/                              # Pytest test suite
└── third_party/                        # Third-party code documentation & BCT notices
```

---

## Citation

If you use this codebase or benchmark results in your research, please cite:

```bibtex
@misc{bhardwaj2026adhd,
  title={ADHD Classification from Resting-State fMRI: A Multi-Track Study of Dynamic Connectivity, Classical--Quantum Graph Learning, and Cross-Site Generalization},
  author={Bhardwaj, Yash and Deepthi, Bhavana and Khare, Shantanu and S, Shridevi and Won, Daehan},
  year={2026},
  howpublished={Manuscript in preparation},
  url={https://github.com/YashBhardwaj21/adhd-classification}
}
```
