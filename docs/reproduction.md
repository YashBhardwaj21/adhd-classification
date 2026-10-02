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
- **Experiment 09**: Leave-One-Site-Out (LOSO) cross-validation and baseline model evaluation. *(Note: Full evaluation tables are retained in `results/exp09/`; executing the notebook from scratch requires intermediate feature parquets).*

### Category B — Requires Excluded Intermediates
Experiments whose complete source code is preserved in the repository, but whose execution requires large intermediate arrays, raw 4D functional NIfTI volumes, or pretrained checkpoints that exceed Git quotas:
- **Experiment 06**: Semi-supervised pseudo-labeling on the AAL-116 atlas (requires unbundled AAL-116 correlation arrays for 955 subjects).
- **Experiment 08**: Volumetric 3D CNN, NeuroSTORM, and temporal graph baselines (requires external 4D functional BOLD NIfTI volumes).

### Category C — Archived
Experiments for which exact configurations and verified empirical test metrics are retained, but whose historical input arrays and checkpoints are archived externally, preventing end-to-end retraining from within the public Git tree:
- **Experiment 07**: Classical GCN versus Quantum QGCNN benchmark ($N=33$ held-out test evaluation). Running `src/exp07/classical_models/run_gcn_experiment.py` or `src/exp07/quantum_models/run_qgcnn_experiment.py` without externally restoring the combined input arrays will raise an informative `FileNotFoundError`.

---

## 2. Environment Matrix

| Track | Experiments | Python | PyTorch | PyTorch Geometric | PennyLane | Environment File | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Exp 01–05 | 3.10–3.13 | $\ge 2.2.0$ | $\ge 2.5.0$ | N/A | `environment/track_a/environment.yml` | CPU or Single GPU |
| **Track B** | Exp 06–07 | 3.10–3.12 | 2.5.1 | 2.6.1 | 0.44.1 | `environment/track_b/environment.yml` | NVIDIA GPU (CUDA 12.4) |
| **Track B** | Exp 08 | 3.10–3.12 | 2.5.1 | 2.6.1 | N/A | `environment/track_b/environment.yml` | NVIDIA GPU (CUDA 12.4) |
| **Track C** | Exp 09 | 3.10–3.12 | 2.5.1 | 2.6.1 | N/A | `environment/track_c/environment.yml` | NVIDIA GPU (CUDA 12.4) |

---

## 3. Installation Procedures

Create an isolated conda or virtual environment for each track:

### Track A (Experiments 01–05)
```bash
conda env create -f environment/track_a/environment.yml
conda activate adhd200-track-a
pip install -e .
```

### Track B (Experiments 06–07 Core & Exp 08)
```bash
conda env create -f environment/track_b/environment.yml
conda activate adhd200-track-b
pip install -e .
```

### Track C (Experiment 09)
```bash
conda env create -f environment/track_c/environment.yml
conda activate adhd200-track-c
pip install -e .
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

## 7. Excluded and Missing Artifacts

The following files are not redistributed in the repository:
- **`data/raw/`**: Raw 4D fMRI NIfTI files (>120 GB).
- **`src/exp07/` input arrays**: `X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy` (875 subjects).
- **Pretrained Checkpoints**: Checkpoint weights for Classical GCN, QGCNN, and NeuroSTORM.
- **Exp 09 Intermediates**: `w1_pheno_features.parquet`, `w1_graph_features.parquet`.

---

## 8. Verification & Test Suite

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

## 9. Reproducibility Limitations

1. **Hardware-Dependent Quantum Simulation**: PennyLane quantum simulations rely on CPU/GPU state-vector engines (`default.qubit` / `lightning.gpu`) which may exhibit minor numerical drift ($< 10^{-6}$) across different BLAS/LAPACK implementations.
2. **Missing Historical Arrays**: Exp 07 cannot be rerun end-to-end without external restoration of the combined 875-subject training arrays.
3. **Non-Retained Producer Code**: The exploratory 8-regime threshold sweep in Exp 09 survives as a recovered empirical data ledger ([`results/exp09/threshold_sweep_graph_statistics.csv`](../results/exp09/threshold_sweep_graph_statistics.csv)); the producer code that generated it was not preserved.
