# Research Notebooks (Experiments 01–09)

This directory contains the computational research notebooks detailing the analysis, harmonization, graph learning, and validation pipelines across all experimental tracks of the ADHD-200 study.

---

## 1. Notebook Index & Experimental Tracks

| Experiment | Notebook File | Modality / Atlas | Focus & Objective | Output Status |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 01** | [`exp01/dynamic_fc_temporal_validation.ipynb`](exp01/dynamic_fc_temporal_validation.ipynb) | CC200 | Temporal autocorrelation decay across lag 1–4 ($N=764$, 1,193 scans) | Verified outputs retained |
| **Exp 02** | [`exp02/mst_proportional_graph_construction.ipynb`](exp02/mst_proportional_graph_construction.ipynb) | CC200 | Kruskal MST + Proportional Thresholding graph construction ($\rho=0.20$) | Verified outputs retained |
| **Exp 02** | [`exp02/graph_metrics_null_model_validation.ipynb`](exp02/graph_metrics_null_model_validation.ipynb) | CC200 | Topological metric validation against 100 signed null-model permutations | Verified outputs retained |
| **Exp 03** | [`exp03/dynamic_graph_features_diagnosis_effects.ipynb`](exp03/dynamic_graph_features_diagnosis_effects.ipynb) | CC200 | Dynamic topological features across 31,060 windows and ANOVA site tests | Verified outputs retained |
| **Exp 04** | [`exp04/combat_site_effects_harmonization.ipynb`](exp04/combat_site_effects_harmonization.ipynb) | CC200 | Cross-site empirical Bayes (neuroCombat) harmonization & topology preservation | Verified outputs retained |
| **Exp 05** | [`exp05/dynamic_state_discovery.ipynb`](exp05/dynamic_state_discovery.ipynb) | CC200 | Dynamic brain micro-state discovery ($K=3$, ARI stability 0.995) & dwell biomarkers | Verified outputs retained |
| **Exp 06** | [`exp06/semi_supervised_pseudolabeling.ipynb`](exp06/semi_supervised_pseudolabeling.ipynb) | AAL-116 | Semi-supervised pseudo-labeling (Procedure I self-training & Procedure II ensemble) | Verified outputs retained |
| **Exp 08** | [`exp08/volumetric_3d_cnn.ipynb`](exp08/volumetric_3d_cnn.ipynb) | 4D BOLD | Volumetric 3D CNN deep learning baseline ($N=626$) | Verified outputs retained |
| **Exp 08** | [`exp08/neurostorm_spatiotemporal_baseline.ipynb`](exp08/neurostorm_spatiotemporal_baseline.ipynb) | 4D BOLD | NeuroSTORM spatiotemporal foundation model 5-fold cross-validation | Verified outputs retained |
| **Exp 08** | [`exp08/temporal_graph_learning.ipynb`](exp08/temporal_graph_learning.ipynb) | CC200 | Temporal dynamic connectome graph learning baseline ($N=764$) | Verified outputs retained |
| **Exp 09** | [`exp09/loso_classical_ml_baselines.ipynb`](exp09/loso_classical_ml_baselines.ipynb) | CC200 | 7-fold Leave-One-Site-Out classical ML benchmark (42 models across 6 families) | Verified outputs retained |
| **Exp 09** | [`exp09/loso_gnn_population_graphs.ipynb`](exp09/loso_gnn_population_graphs.ipynb) | CC200 | 7-fold Leave-One-Site-Out population graph learning (GCN, GAT, GraphSAGE, GIN) | Verified outputs retained |

---

## 2. Path Parameterization & Environment Portability

To prevent environment-specific lock-in while preserving original execution evidence, all code cells resolve filesystem paths dynamically via environment variables with relative repository defaults:

- `ADHD200_DATA_DIR`: Base directory for raw scans, extracted ROI time series, and static FC matrices (defaults to `data/`).
- `ADHD200_RESULTS_DIR`: Output directory for derived figures, CSV tables, and summary ledgers (defaults to `results/<experiment>/`).
- `ADHD200_WORK_DIR`: Scratch directory for regenerable workflow intermediates (defaults to `data/intermediate/`).
- `W1_DIR`: Intermediate directory for classical ML LOSO feature tables (defaults to `data/intermediate/workflow1/`).
- `W2_DIR`: Intermediate directory for GNN population graph learning (defaults to `data/intermediate/workflow2/`).

---

## 3. Historical Compute Environments & Output Provenance

The executed cell outputs (dataframes, confusion matrices, loss curves, print statements) are intentionally preserved to serve as an immutable, audited scientific record of the original research runs:

1. **Cloud GPU Cluster (`/mnt/ADHD200`)**:
   - Used for compute-intensive experiments (Exp 06, 08, 09). Evidenced by mount paths `/mnt/ADHD200`, `/mnt/ADHD200_WORKFLOW1_RUN`, `/mnt/ADHD200_WORKFLOW2_RUN` visible in executed output cells.
   - Specific hardware was not recorded.
2. **Local Workstation (`/home/nvidia/23BRS1236/`)**:
   - Used for dynamic connectomic time-series processing (Exp 01–05). Evidenced by `/home/nvidia/23BRS1236/adhd_data/` paths in executed output cells.

For detailed instructions on setting up environments and reproducing these experiments, refer to [`docs/reproduction.md`](../docs/reproduction.md).
