# Reproduction & Execution Guide

This document provides operational instructions for reproducing, rerunning, and verifying the experiments across the repository.

---

## 1. Reproducibility Categories

Each experiment is categorized by its execution status:

### Category A — Rerunnable
Experiments whose source code, configurations, and verification routines are fully contained in the repository and can be regenerated end-to-end from obtainable public ADHD-200 raw/preprocessed data:
- **Experiment 01**: Dynamic functional connectivity autocorrelation decay and Frobenius stability.
- **Experiment 02**: CC200 graph construction (MST+PT) and degree-conserving signed null models.
- **Experiment 03**: Dynamic graph metric extraction and cross-site ANOVA testing.
- **Experiment 04**: ComBat harmonization and topology preservation evaluation.
- **Experiment 05**: Unsupervised dynamic micro-state clustering and biomarker extraction.

### Category B — Requires Excluded Intermediates
Experiments whose complete source code is preserved in the repository, but whose execution requires large intermediate arrays, raw 4D functional NIfTI volumes, or pretrained checkpoints that exceed Git quotas:
- **Experiment 06**: Semi-supervised pseudo-labeling on the AAL-116 atlas (requires unbundled AAL-116 correlation arrays for 955 subjects).
- **Experiment 08**: Volumetric 3D CNN, NeuroSTORM, and temporal graph baselines (requires external 4D functional BOLD NIfTI volumes).
- **Experiment 09**: Leave-One-Site-Out (LOSO) cross-validation and baseline model evaluation (full evaluation tables are retained in `results/exp09/`; executing the notebook from scratch requires intermediate feature parquets not distributed in the public repository).

### Category C — Archived
Experiments for which exact configurations and verified empirical test metrics are retained, but whose historical input arrays and checkpoints are archived externally, preventing end-to-end retraining from within the public Git tree:
- **Experiment 07**: Classical GCN versus Quantum QGCNN benchmark ($N=33$ held-out test evaluation). Running `src/exp07/classical_models/run_gcn_experiment.py` or `src/exp07/quantum_models/run_qgcnn_experiment.py` without externally restoring the combined input arrays will raise an informative `FileNotFoundError`.

---

## 2. Environment Matrix

| Track | Experiments | Python | PyTorch | PyTorch Geometric | PennyLane | Environment File | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Exp 01–05 | 3.10–3.13 | $\ge 2.2.0$ | $\ge 2.5.0$ | N/A | `environment/track_a/requirements.txt` | CPU or Single GPU |
| **Track B** | Exp 06–07 | 3.10–3.12 | 2.5.1+cu121 | 2.8.0 | 0.44.1 | `environment/track_b/requirements.txt` | NVIDIA GPU (CUDA 12.1) |
| **Track B** | Exp 08 | 3.10–3.12 | 2.7.1+cu126 | N/A | N/A | `environment/track_b/requirements.txt` (NeuroSTORM env separate) | NVIDIA GPU (CUDA 12.6) |
| **Track C** | Exp 09 | 3.10–3.12 | 2.5.1+cu121 | 2.8.0 | N/A | `environment/track_c/requirements.txt` | NVIDIA GPU (CUDA 12.1) |

---

## 3. Installation Procedures

Create an isolated conda or virtual environment for each track:

### Track A (Experiments 01–05)
```bash
pip install -r environment/track_a/requirements.txt
```

### Track B (Experiments 06–07 Core & Exp 08)
```bash
pip install -r environment/track_b/requirements.txt
```

### Track C (Experiment 09)
```bash
pip install -r environment/track_c/requirements.txt
```

---

## 4. Execution Order

Execute experimental workflows in the following logical sequence:

```text
Track A (Dynamic Connectomics)
  Step 1: notebooks/exp01/dynamic_fc_temporal_validation.ipynb
  Step 2: notebooks/exp02/mst_proportional_graph_construction.ipynb
  Step 3: notebooks/exp02/graph_metrics_null_model_validation.ipynb
  Step 4: notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb
  Step 5: notebooks/exp04/combat_site_effects_harmonization.ipynb
  Step 6: notebooks/exp05/dynamic_state_discovery.ipynb

Track B (Semi-Supervised & Baselines)
  Step 7: notebooks/exp06/semi_supervised_pseudolabeling.ipynb
  Step 8: notebooks/exp08/volumetric_3d_cnn.ipynb
  Step 9: notebooks/exp08/neurostorm_spatiotemporal_baseline.ipynb
  Step 10: notebooks/exp08/temporal_graph_learning.ipynb

Track C (Cross-Site Generalization Benchmark)
  Step 11: notebooks/exp09/loso_classical_ml_baselines.ipynb
  Step 12: notebooks/exp09/loso_gnn_population_graphs.ipynb
```

---

## 5. Experiment-Specific Commands

### Running Experiment 02 Graph Construction via Public Utility
```bash
python -c "
import numpy as np
from exp02.graph_utils import prepare_graphs
X = np.random.randn(2, 17955).astype(np.float32)
graphs = prepare_graphs(X)
print(f'Constructed {len(graphs)} graphs with {graphs[0].num_nodes} nodes and {graphs[0].num_edges} edges')
"
```

### Testing Experiment 07 Model Instantiation
```bash
python -c "
import torch
from exp07.classical_models.gcn_model import ClassicalGCN
model = ClassicalGCN(input_dim=117, hidden_dim=32)
x = torch.randn(116, 117)
edge_index = torch.zeros((2, 200), dtype=torch.long)
out = model(x, edge_index, batch=torch.zeros(116, dtype=torch.long))
print('Classical GCN forward pass output shape:', out.shape)
"
```

---

## 6. Required External Data

To rerun from scratch, download the following from the [ADHD-200 Consortium NITRC Portal](https://www.nitrc.org/projects/fcon_1000/):
1. **Athena CC200 Connectomes**: Resting-state time series for 764 subjects (`data/preprocessed/cc200/`).
2. **Athena AAL-116 Connectomes**: Resting-state time series for 955 subjects (`data/preprocessed/aal116/`).
3. **Phenotypic Table**: `ADHD200_phenotypic.csv` (`data/phenotypic/`).

---

## 7. Data, Paths, and Non-Distributed Artifacts

### Repository Path Policy
This repository separates source code, canonical research results, external input data, generated intermediate data, and model checkpoints into four distinct artifact classes:

| Class | Definition | Repository Status | Code Resolution |
| :--- | :--- | :--- | :--- |
| **Repository Artifact** | Small, final, or audit-relevant records defining the retained scientific record (`results/`, `configs/`). | Committed to Git | `REPO_ROOT / "results" / ...` |
| **External Source Data** | Raw and preprocessed neuroimaging scans and phenotypic manifests. | External (Not Committed) | `DATA_ROOT / ...` |
| **Regenerable Intermediate** | Large generated `.parquet` tables, dynamic `.npy` FC arrays, and workflow stages. | Local / Regenerable | `INTERMEDIATE_ROOT / ...` |
| **Model Checkpoint** | Pretrained neural foundation models, GNN weights, and classical estimators. | External / Regenerable | `CHECKPOINT_ROOT / ...` |

Paths used by executable code do not depend on machine-specific user accounts, cluster mount paths, or working directories. All code deterministically resolves paths from the repository root through [`src/common/paths.py`](../src/common/paths.py) and supports environment-variable overrides:

```text
REPO_ROOT/
├── data/
│   ├── raw/               # External source scans (NITRC ADHD-200), not committed
│   ├── external/          # Externally obtained atlas parcellations, not committed
│   ├── intermediate/      # Regenerable pipeline parquets/arrays (workflow2/v6, v7, v8)
│   └── checkpoints/       # Pretrained foundation weights (neurostorm_mae.pth)
├── results/               # Committed canonical scientific results (results/manifest.csv)
├── configs/               # Committed hyperparameters and pipeline configurations
├── notebooks/             # Research notebooks (code synchronized; historical outputs preserved)
├── src/                   # Production modular Python packages
├── docs/                  # Architectural, methodological, and provenance documentation
├── scripts/               # Validation and replication test scripts
└── tests/                 # Unit and repository integrity test suite
```

### Environment Variable Overrides
The default paths are designed to execute from a clean repository checkout. Cluster mounts, scratch disks, or HPC storage arrays can be configured using environment variables:

| Environment Variable | Canonical Default | Description |
| :--- | :--- | :--- |
| `ADHD200_PROJECT_ROOT` | Repository Root | Root directory containing `pyproject.toml`, `results/`, `notebooks/`. |
| `ADHD200_DATA_DIR` | `REPO_ROOT / "data"` | Root storage for raw, external, and intermediate imaging data. |
| `ADHD200_RESULTS_DIR` | `REPO_ROOT / "results"` | Authoritative directory for promoted small scientific result artifacts. |
| `ADHD200_WORK_DIR` | `DATA_ROOT / "intermediate"` | Scratch directory for regenerable workflow intermediates (`workflow2/v6`, `v7`, `v8`). |
| `ADHD200_CHECKPOINT_DIR` | `DATA_ROOT / "checkpoints"` | Directory for large pretrained model checkpoints (`.pth`, `.pt`, `.ckpt`). |

### Canonical Repository Artifacts
Small, final, and audit-verified result files that define the retained research record are stored under `results/` and `configs/`. These files are the authoritative repository artifacts referenced throughout this documentation.

### External Source Data
The ADHD-200 source dataset and other large neuroimaging inputs are not committed to Git due to size (>120 GB) and licensing constraints. This includes raw 4D fMRI NIfTI files, preprocessed Athena voxel timeseries, and phenotypic source tables. Users must obtain these inputs from the [ADHD-200 NITRC Portal](https://www.nitrc.org/projects/fcon_1000/) and place them under `data/`.

### Generated Intermediate Artifacts
Pipeline stages generate intermediate artifacts (e.g., sliding-window correlation tensors, topological metric parquets, ComBat-harmonized tables). When rerunning pipelines, intermediate data are placed under `data/intermediate/workflow2/` (`v6`, `v7`, `v8`). A missing intermediate file indicates an unmet pipeline dependency to be regenerated, not a missing repository file.

### Model Checkpoints Policy
Model checkpoints (`.pt`, `.pth`, `.ckpt`) are managed under three distinct policies:
1. **Case A (Not Required for Canonical Evaluation)**: Checkpoints created during exploratory runs are discarded once summary metrics are recorded; training code and configuration are preserved.
2. **Case B (Required for Direct Inference)**: Pretrained foundation models (such as NeuroSTORM MAE, 1.4 GB) have documented download sources from public repositories (HuggingFace Hub / Zenodo) and are mapped to `data/checkpoints/`.
3. **Case C (Trained Model Producing Committed Results)**: For models evaluated on held-out test sets (e.g., Classical GCN and QGCNN in Exp 07), the authoritative evaluation metrics are committed in `results/exp07/`. The historical training script and config are preserved; exact bitwise reproduction of stochastic gradient descent weights is not guaranteed.

### Three Levels of Reproduction
This repository explicitly distinguishes three levels of scientific reproduction:

1. **Level 1 — Inspect**: Full audit and verification of scientific code, hyperparameters, methodological documentation, provenance chains, and canonical results directly from a clean Git clone without downloading raw imaging data (`pytest tests/ -v`, `python scripts/validate_results.py`).
2. **Level 2 — Rerun**: Execution of individual pipeline stages after acquiring external source data or generating intermediate parquets as documented in the artifact registry.
3. **Level 3 — Historical Exact Reproduction**: Bitwise exact recreation of historical runs requiring identical hardware (NVIDIA GPU clusters), original Python environments, exact historical random seeds, and specific intermediate caches.

### Historical Execution Paths & Output Policy
Original executed notebook output cells contain historical execution evidence, such as `/home/nvidia/23BRS1236/adhd_data/`, `/mnt/ADHD200/`, `workflow2/v6/`, and `workflow2/v7/`.

> [!IMPORTANT]
> **Repository Policy on Historical Outputs:**
> *Never modify historical outputs solely to reflect later renames. Modify executable code and current documentation; preserve historical outputs and explicitly label their provenance.*

---

## 8. Non-Distributed Artifact Registry

| Artifact | Artifact Type | Status | Produced By | Required By | How Obtained / Regenerated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ADHD-200 Raw NIfTI** | Raw Imaging Data | External | ADHD-200 Consortium | Exp 08 (Volumetric) | Download from [NITRC Portal](https://www.nitrc.org/projects/fcon_1000/) |
| **Athena CC200 Timeseries** | Preprocessed Data | External | Athena Pipeline | Exp 01–05, Exp 09 | Download CC200 ROI timeseries from NITRC |
| **Athena AAL-116 Timeseries** | Preprocessed Data | External | Athena Pipeline | Exp 06, Exp 07 | Download AAL-116 ROI timeseries from NITRC |
| **`03_fc_matrices/*.npy`** | Intermediate Data | Regenerable | Exp 01 | Exp 02, Exp 09 | Run `notebooks/exp01/dynamic_fc_temporal_validation.ipynb` |
| **`workflow2/v6/*.parquet`** | Intermediate Data | Regenerable | Exp 03 | Exp 04 | Run `notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb` |
| **`workflow2/v7/*.parquet`** | Intermediate Data | Regenerable | Exp 04 | Exp 05 | Run `notebooks/exp04/combat_site_effects_harmonization.ipynb` |
| **`workflow2/v8/*.parquet`** | Intermediate Data | Regenerable | Exp 05 | Downstream Analysis | Run `notebooks/exp05/dynamic_state_discovery.ipynb` |
| **NeuroSTORM MAE Checkpoint** | Model Weights | External | Foundation Model Pretraining | Exp 08 | Download via HuggingFace Hub / Zenodo to `data/checkpoints/` |
| **Classical GCN Checkpoint** | Model Weights | Historical / Unavailable | Exp 07 Training | Historical Inference | Retrain using `src/exp07/classical_models/run_gcn_experiment.py` |
| **QGCNN Quantum Checkpoint** | Model Weights | Historical / Unavailable | Exp 07 Training | Historical Inference | Retrain using `src/exp07/quantum_models/run_qgcnn_experiment.py` |

---

## 9. Canonical Filename & Historical Execution Mapping

The executable code and documentation reference the canonical repository filenames below, while historical outputs retain original execution names:

| Category | Historical Execution Name (in Outputs) | Canonical Repository Name (in Code & Storage) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Notebooks** | `01_fc_generation_and_validation.ipynb` | [`notebooks/exp01/dynamic_fc_temporal_validation.ipynb`](../notebooks/exp01/dynamic_fc_temporal_validation.ipynb) | Verified Present |
| | `02_graph_construction_and_validation.ipynb` | [`notebooks/exp02/mst_proportional_graph_construction.ipynb`](../notebooks/exp02/mst_proportional_graph_construction.ipynb) | Verified Present |
| | `03_graph_metrics_and_null_models.ipynb` | [`notebooks/exp02/graph_metrics_null_model_validation.ipynb`](../notebooks/exp02/graph_metrics_null_model_validation.ipynb) | Verified Present |
| | `04_dynamic_graph_feature_extraction.ipynb` | [`notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb`](../notebooks/exp03/dynamic_graph_features_diagnosis_effects.ipynb) | Verified Present |
| | `(historical w2c_athena_2.ipynb)` | [`notebooks/exp04/combat_site_effects_harmonization.ipynb`](../notebooks/exp04/combat_site_effects_harmonization.ipynb) | Verified Present |
| | `dynamic_transformer.ipynb` | [`notebooks/exp05/dynamic_state_discovery.ipynb`](../notebooks/exp05/dynamic_state_discovery.ipynb) | Verified Present |
| | `exp06_semi_supervised_pseudolabeling.ipynb` | [`notebooks/exp06/semi_supervised_pseudolabeling.ipynb`](../notebooks/exp06/semi_supervised_pseudolabeling.ipynb) | Verified Present |
| | `neuro.ipynb` | [`notebooks/exp08/volumetric_3d_cnn.ipynb`](../notebooks/exp08/volumetric_3d_cnn.ipynb) | Verified Present |
| | `true_neuro.ipynb` | [`notebooks/exp08/neurostorm_spatiotemporal_baseline.ipynb`](../notebooks/exp08/neurostorm_spatiotemporal_baseline.ipynb) | Verified Present |
| | `09_temporal_graph_learning.ipynb` | [`notebooks/exp08/temporal_graph_learning.ipynb`](../notebooks/exp08/temporal_graph_learning.ipynb) | Verified Present |
| | `11_population_graph_learning.ipynb` | [`notebooks/exp09/loso_gnn_population_graphs.ipynb`](../notebooks/exp09/loso_gnn_population_graphs.ipynb) | Verified Present |
| | `12_classical_ml_baseline.ipynb` | [`notebooks/exp09/loso_classical_ml_baselines.ipynb`](../notebooks/exp09/loso_classical_ml_baselines.ipynb) | Verified Present |
| **Source Scripts** | `null_model_und_sign_fixed.py` | [`src/exp02/signed_null_model.py`](../src/exp02/signed_null_model.py) | Verified Present |
| | `randmio_und_signed_fast.py` | [`src/exp02/signed_edge_rewiring.py`](../src/exp02/signed_edge_rewiring.py) | Verified Present |
| | `model_classical_gcn.py` | [`src/exp07/classical_models/gcn_model.py`](../src/exp07/classical_models/gcn_model.py) | Verified Present |
| | `train_classical_gcn.py` | [`src/exp07/classical_models/gcn_training.py`](../src/exp07/classical_models/gcn_training.py) | Verified Present |
| | `run_experiment.py` (GCN) | [`src/exp07/classical_models/run_gcn_experiment.py`](../src/exp07/classical_models/run_gcn_experiment.py) | Verified Present |
| | `quantum_embedding_broadcast.py` | [`src/exp07/quantum_models/qgcnn_quantum_embedding.py`](../src/exp07/quantum_models/qgcnn_quantum_embedding.py) | Verified Present |
| | `train_qgcnn_vectorized.py` | [`src/exp07/quantum_models/qgcnn_training.py`](../src/exp07/quantum_models/qgcnn_training.py) | Verified Present |
| | `run_experiment.py` (QGCNN) | [`src/exp07/quantum_models/run_qgcnn_experiment.py`](../src/exp07/quantum_models/run_qgcnn_experiment.py) | Verified Present |
| **Exp 01 Artifacts** | `dynamic_temporal_validation.csv` | [`results/exp01/temporal_similarity_validation.csv`](../results/exp01/temporal_similarity_validation.csv) | Verified Present |
| | `static_vs_dynamic_validation.csv` | [`results/exp01/static_dynamic_fc_comparison.csv`](../results/exp01/static_dynamic_fc_comparison.csv) | Verified Present |
| **Exp 02 Artifacts** | `graph_construction_strategy.csv` | [`results/exp02/graph_construction_configuration.csv`](../results/exp02/graph_construction_configuration.csv) | Verified Present |
| | `graph_metrics.csv` | [`results/exp02/graph_metrics_window_level.csv`](../results/exp02/graph_metrics_window_level.csv) | Verified Present |
| | `acquisition_graph_metrics.csv` | [`results/exp02/graph_metrics_acquisition_level.csv`](../results/exp02/graph_metrics_acquisition_level.csv) | Verified Present |
| | `subject_graph_metrics.csv` | [`results/exp02/graph_metrics_subject_level.csv`](../results/exp02/graph_metrics_subject_level.csv) | Verified Present |
| | `site_graph_metrics.csv` | [`results/exp02/graph_metrics_site_level.csv`](../results/exp02/graph_metrics_site_level.csv) | Verified Present |
| **Exp 03 Artifacts** | `feature_statistics.csv` | [`results/exp03/dynamic_feature_statistics.csv`](../results/exp03/dynamic_feature_statistics.csv) | Verified Present |
| | `site_anova.csv` | [`results/exp03/site_effect_anova.csv`](../results/exp03/site_effect_anova.csv) | Verified Present |
| | `subject_graph_features.csv` | [`results/exp03/subject_graph_feature_summary.csv`](../results/exp03/subject_graph_feature_summary.csv) | Verified Present |
| **Exp 04 Artifacts** | `comparison_table.csv` | [`results/exp04/combat_harmonization_comparison.csv`](../results/exp04/combat_harmonization_comparison.csv) | Verified Present |
| | `diagnosis_prediction_results.csv` | [`results/exp04/diagnosis_prediction_combat_comparison.csv`](../results/exp04/diagnosis_prediction_combat_comparison.csv) | Verified Present |
| | `effect_size_before_after.csv` | [`results/exp04/site_effect_sizes_before_after_combat.csv`](../results/exp04/site_effect_sizes_before_after_combat.csv) | Verified Present |
| | `graph_topology_preservation.csv` | [`results/exp04/graph_topology_preservation_combat.csv`](../results/exp04/graph_topology_preservation_combat.csv) | Verified Present |
| | `site_prediction_results.csv` | [`results/exp04/site_prediction_combat_comparison.csv`](../results/exp04/site_prediction_combat_comparison.csv) | Verified Present |
| **Exp 05 Artifacts** | `run_dynamic_biomarkers.csv` | [`results/exp05/dynamic_state_biomarkers.csv`](../results/exp05/dynamic_state_biomarkers.csv) | Verified Present |
| | `run_state_sequences.csv` | [`results/exp05/dynamic_state_sequences.csv`](../results/exp05/dynamic_state_sequences.csv) | Verified Present |
| | `run_transition_dynamics.csv` | [`results/exp05/dynamic_state_transitions.csv`](../results/exp05/dynamic_state_transitions.csv) | Verified Present |
| | `subject_dynamic_dataset.csv` | [`results/exp05/subject_dynamic_features.csv`](../results/exp05/subject_dynamic_features.csv) | Verified Present |
| | `subject_feature_correlation.csv` | [`results/exp05/subject_feature_correlations.csv`](../results/exp05/subject_feature_correlations.csv) | Verified Present |
| | `subject_feature_summary.csv` | [`results/exp05/subject_feature_summary.csv`](../results/exp05/subject_feature_summary.csv) | Verified Present |
| **Exp 07 Artifacts** | `checkpoint_analysis.json` | [`results/exp07/gcn_qgcnn_test_results.json`](../results/exp07/gcn_qgcnn_test_results.json) | Verified Present |
| | `report.md` | [`results/exp07/experiment_report.md`](../results/exp07/experiment_report.md) | Verified Present |
| | `clean_subjects_manifest.csv` | [`results/exp07/clean_cohort_lineage.csv`](../results/exp07/clean_cohort_lineage.csv) | Verified Present |
| | `reported_run.json` | [`configs/exp07/experiment_configuration.json`](../configs/exp07/experiment_configuration.json) | Verified Present |
| **Exp 08 Artifacts** | `exp08_verified_results.json` | [`results/exp08/baseline_model_results.json`](../results/exp08/baseline_model_results.json) | Verified Present |
| | `neurostorm_true_results.png` | [`results/exp08/neurostorm_evaluation_summary.png`](../results/exp08/neurostorm_evaluation_summary.png) | Verified Present |
| **Exp 09 Artifacts** | `(historical w1_canonical_model_summary.csv)` | [`results/exp09/classical_ml_canonical_results.csv`](../results/exp09/classical_ml_canonical_results.csv) | Verified Present |
| | `(historical w1_family_winners.csv)` | [`results/exp09/classical_ml_family_summary.csv`](../results/exp09/classical_ml_family_summary.csv) | Verified Present |
| | `(historical w1_model_summary.csv)` | [`results/exp09/classical_ml_historical_results.csv`](../results/exp09/classical_ml_historical_results.csv) | Verified Present |
| | `(historical w2_pareto_front.csv)` | [`results/exp09/representation_pareto_analysis.csv`](../results/exp09/representation_pareto_analysis.csv) | Verified Present |
| | `(historical w2b_error_analysis.csv)` | [`results/exp09/gnn_loso_error_analysis.csv`](../results/exp09/gnn_loso_error_analysis.csv) | Verified Present |
| | `(historical w2b_loso_results.csv)` | [`results/exp09/gnn_loso_fold_results.csv`](../results/exp09/gnn_loso_fold_results.csv) | Verified Present |
| | `(historical w2b_summary_with_ci.csv)` | [`results/exp09/gnn_loso_summary_statistics.csv`](../results/exp09/gnn_loso_summary_statistics.csv) | Verified Present |
| | `(historical w2b_wilcoxon_tests.csv)` | [`results/exp09/gnn_loso_pairwise_tests.csv`](../results/exp09/gnn_loso_pairwise_tests.csv) | Verified Present |
| | `gnn_diagnostic_positive.csv` | [`results/exp09/threshold_positive_fc_diagnostic.csv`](../results/exp09/threshold_positive_fc_diagnostic.csv) | Verified Present |
| | `gnn_diagnostic_absolute.csv` | [`results/exp09/threshold_absolute_fc_diagnostic.csv`](../results/exp09/threshold_absolute_fc_diagnostic.csv) | Verified Present |
| | `graph_statistics.csv` | [`results/exp09/threshold_sweep_graph_statistics.csv`](../results/exp09/threshold_sweep_graph_statistics.csv) | Verified Present |
| | `selected_node_features.json` | [`configs/exp09/node_feature_selection.json`](../configs/exp09/node_feature_selection.json) | Verified Present |
| | `selected_threshold.json` | [`configs/exp09/graph_threshold_selection.json`](../configs/exp09/graph_threshold_selection.json) | Verified Present |
| | `(historical w2b_best_config.json)` | [`configs/exp09/gnn_configuration.json`](../configs/exp09/gnn_configuration.json) | Verified Present |
| | `(historical w2b_dataset_manifest.json)` | [`configs/exp09/dataset_manifest.json`](../configs/exp09/dataset_manifest.json) | Verified Present |
| | `(historical w2b_environment.json)` | [`configs/exp09/execution_environment.json`](../configs/exp09/execution_environment.json) | Verified Present |

---

## 10. Verification & Test Suite

Verify repository integrity and numerical replication without downloading external neuroimaging data:

### Run Results Validation Script
Verifies numerical consistency across all retained results artifacts against ground-truth values:
```bash
python scripts/validate_results.py
```

### Run Pytest Suite
Executes unit tests verifying import paths, config constants, graph dimensions, and model tensor shapes:
```bash
pytest tests/ -v
```

---

## 11. Reproducibility Limitations

1. **Hardware-Dependent Quantum Simulation**: PennyLane quantum simulations rely on CPU/GPU state-vector engines (`default.qubit` / `lightning.gpu`). Minor numerical variation across different BLAS/LAPACK implementations is possible; canonical verified metrics are stored in `results/exp07/gcn_qgcnn_test_results.json`.
2. **Missing Historical Arrays**: Exp 07 cannot be rerun end-to-end without external restoration of the combined 875-subject training arrays.
3. **Non-Retained Producer Code**: The exploratory 8-regime threshold sweep in Exp 09 survives as a recovered empirical data ledger ([`results/exp09/threshold_sweep_graph_statistics.csv`](../results/exp09/threshold_sweep_graph_statistics.csv)); the producer code that generated it was not preserved.
4. **Exp 09 Intermediate Parquets**: The per-subject feature parquets required to execute the LOSO notebook are not distributed in the public repository. Canonical LOSO results are verified in `results/exp09/gnn_loso_summary_statistics.csv`.
