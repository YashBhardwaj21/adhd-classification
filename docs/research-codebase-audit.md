# Research Repository Structure, Experiment and Codebase Audit

**Audit Target Repository**: `adhd-classification` (ADHD-200 Functional Connectomics & Deep Learning)  
**Historical Cloud Source of Truth**: `/home/nvidia/23BRS1236` and `/lp-dev/23BRS1236` (captured in `audit_source_files/`)  
**Audit Date**: October 2026 (Updated & Provenance Hardened)  
**Historical Snapshot**: Preserved in [`docs/archive/research-codebase-audit-2026-09.md`](archive/research-codebase-audit-2026-09.md)  

---

## Executive Summary & Core Alignment Verification

### Current Repository Status
The repository contains 175 git-tracked files, 12 canonical Jupyter notebooks across 9 experiments, and 31 registered canonical and historical result artifacts in [`results/manifest.csv`](../results/manifest.csv).

Unlike exploratory research repositories that assume complete end-to-end re-executability, this repository maintains an explicit distinction between:
1. **Fully Executable Research Modules**: Scripts and notebooks reproducible from raw/processed connectomes (Exp 01–05, Exp 09b classical baselines).
2. **Archived / Evaluation Provenance Artifacts**: Experiments where model weights, large binary arrays, or raw 4D scans exceed public repository storage quotas and are permanently archived with cryptographic hashes and diagnostic metrics (Exp 07 quantum/classical benchmarks, Exp 08 volumetric/temporal deep learning).
3. **Recovered Historical Evidence Ledgers**: Artifacts recovered from historical cloud runs whose producing pipeline code was not preserved in the surviving tree, but whose empirical ledgers establish key design choices (Exp 09 eight-regime threshold sweep and diagnostic validations).

### Alignment with Cloud Evidence (`audit_source_files/`)
The `audit_source_files/` directory contains raw development files, cloud scripts, exploratory prototypes, and execution history from the cloud GPU environments (`/home/nvidia/23BRS1236` and `/lp-dev/23BRS1236/mnt/ADHD200`). AST, cryptographic, and unified-diff inspections confirm:
1. **Source Code Lineage**: Production modules in `src/` derive directly from `audit_source_files/mnt/ADHD200/` and `audit_source_files/NeuroSTORM/`, modified only for portable paths (`common.paths`), import normalization, license attribution (`SPDX-License-Identifier`), and static lint compliance.
2. **Canonical Notebook Lineage**: The 12 canonical notebooks in `notebooks/` correspond to the core experimental pipeline stages.
3. **Exploratory Branches**: Multiple exploratory trials evaluated in the cloud were deliberately not promoted to production staging (e.g., DDP distributed training in `train_hybrid_transformer.py`, domain-adversarial adaptation in `w2c_dann.ipynb`, Riemannian tangent-space projections in `w2a_representation_audit.ipynb`, and earlier JAX/vectorized quantum implementations in `quantum_embedding_vmap.py`).

---

# 1. Research Repository Inventory

### Structured File Distribution
- **Git-Tracked Files**: 175
- **Canonical Notebooks**: 12 (Exp 01 to Exp 09)
- **Result Manifest Entries**: 31 registered entries in `results/manifest.csv`
- **Configuration Files**: Stored in `configs/exp02/`, `configs/exp07/`, and `configs/exp09/`

### Canonical Notebook Directory Structure
```text
notebooks/
├── exp01/01_fc_generation_and_validation.ipynb       # Exp 01: dFC sliding-window validation
├── exp02/02_graph_construction_and_validation.ipynb  # Exp 02: MST+PT graph builder
├── exp02/03_graph_metrics_and_null_models.ipynb      # Exp 02: Topological null models
├── exp03/04_dynamic_graph_feature_extraction.ipynb   # Exp 03: Multi-site ANOVA audit
├── exp04/w2c_athena_2.ipynb                          # Exp 04: ComBat harmonization
├── exp05/dynamic_transformer.ipynb                   # Exp 05: Dynamic micro-state clustering
├── exp06/exp06_semi_supervised_pseudolabeling.ipynb  # Exp 06: Semi-supervised pseudo-labeling
├── exp08/09_temporal_graph_learning.ipynb           # Exp 08: Temporal GNN baseline
├── exp08/neuro.ipynb                                 # Exp 08: NeuroSTORM exploratory
├── exp08/true_neuro.ipynb                            # Exp 08: NeuroSTORM foundation model
├── exp09/11_population_graph_learning.ipynb          # Exp 09: 7-fold LOSO GNN benchmark
└── exp09/12_classical_ml_baseline.ipynb              # Exp 09: 7-fold LOSO tabular baselines
```

---

# 2. Experiment-by-Experiment Technical Audit

### Exp 01: Dynamic Functional Connectivity Stability
- **Hypothesis**: Sliding-window dynamic FC correlation decays smoothly across temporal lags, confirming structured temporal transitions rather than stochastic noise.
- **Protocol**: Sliding-window Pearson correlation ($W=30$ TR, step $S=5$ TR, CC200 atlas, 534 subjects, 1,193 acquisitions, 31,060 windows).
- **Verified Findings**: Monotonic decay in mean window correlation across lags: Lag 1 ($r=0.8990$), Lag 2 ($r=0.7789$), Lag 3 ($r=0.6609$), Lag 4 ($r=0.5467$); mean Frobenius distance increases from 32.58 to 70.61 (`results/exp01/dynamic_temporal_validation.csv`).
- **Reproducibility**: Fully reproducible with repository scripts.

### Exp 02: Dual-Constraint Graph Construction & Signed Null Models
- **Hypothesis**: Minimum Spanning Tree (MST) combined with proportional thresholding (PT) prevents graph fragmentation while preserving small-world topology.
- **Protocol**: CC200 functional connectivity ($N=190$ nodes); distance metric $d_{ij} = 1 - |r_{ij}|$; MST backbone augmented with top positive correlations to reach nominal density $D=0.20$. Signed Rubinov-Sporns null model randomized via Numba-accelerated edge rewiring.
- **Verified Findings**: 0 disconnected nodes; average degree $k=38.0$; small-world ratio $\sigma > 1.2$; Numba acceleration reduced window randomization from ~14 hours to <12 minutes (`results/exp02/graph_metrics.csv`).
- **Implementation Note**: `src/exp02/graph_utils.py` and `configs/exp02/graph_config.json` correctly use $1-|r_{ij}|$ and MST+PT.

### Exp 03: Multi-Site Scanner Variation & Confounding Audit
- **Hypothesis**: Topological graph metrics extracted from raw connectomes show negligible scanner hardware variance relative to biological signal.
- **Protocol**: Extracted 6 topological metrics across 534 subjects from 8 imaging sites; one-way ANOVA across acquisition sites.
- **Verified Findings**: Massive scanner confounding: Global efficiency $F=4,609.76$; Characteristic path length $F=4,993.87$ ($p < 10^{-300}$). Reject hypothesis: raw topological features cannot be pooled across sites without harmonization (`results/exp03/site_anova.csv`).

### Exp 04: Empirical Bayes ComBat Harmonization
- **Hypothesis**: ComBat removes site identification while preserving diagnostic classification.
- **Protocol**: Parametric Empirical Bayes ComBat with Age and Gender covariates. Evaluated on both static connectivity features (534 subjects) and dynamic micro-state features (764 subjects).
- **Verified Findings**:
  - **Static Features**: Site prediction accuracy dropped from $58.64\%$ to $31.29\%$ (`results/exp04/site_prediction_results.csv`), while diagnostic accuracy dropped from $64.53\%$ to $59.56\%$ (`results/exp04/diagnosis_prediction_results.csv`).
  - **Dynamic Micro-State Features (Phase E2)**: In contrast, dynamic micro-state features classified with Random Forest (500 trees, 5-fold CV) improved from $0.6256$ to $0.6754$ ($0.626 \to 0.675$, $+0.0497$).
  - *Correction Note*: Stale claims in early draft reports that site identification dropped $95.7\% \to 14.8\%$ represent preliminary exploratory runs; the canonical verified ComBat benchmark in `results/exp04/` records $58.64\% \to 31.29\%$.

### Exp 05: Dynamic Brain Micro-State Clustering
- **Hypothesis**: rs-fMRI transitions between discrete recurrent micro-states with altered dwell times in ADHD.
- **Protocol**: K-means clustering ($K=3$) on sliding-window topological feature vectors (31,060 windows).
- **Verified Findings**: 3 discrete recurring states identified: State 0 (mean dwell 6.24 windows), State 1 (3.82 windows), State 2 (2.45 windows). ADHD subjects show altered state transition dynamics (`results/exp05/run_dynamic_biomarkers.csv`).

### Exp 06: Semi-Supervised Pseudo-Labeling
- **Hypothesis**: Multi-classifier ensemble consensus provides higher-quality pseudo-labels than single-model self-training.
- **Protocol**: 955 candidate AAL-116 FC matrices inner-joined with 691 phenotypic records yielded an aligned clean cohort of $N=391$ subjects (`aligned_subjects.npy`). Evaluated Procedure I (Logistic Regression self-training, $\tau=0.75$) and Procedure II (4-model weighted ensemble: RF, GBDT, LR, Calibrated SVC).
- **Verified Findings**: Procedure I produced 552 pseudo-labels; Procedure II produced 484 consensus pseudo-labels. In a separate 79-subject holdout evaluation, pseudo-label augmentation improved classification accuracy from $0.6709$ to $0.7215$ (`results/exp06/exp06_verified_results.json`).

### Exp 07: Classical GCN vs. Quantum QGCNN
- **Hypothesis**: Parameterized Quantum Circuits (PQC) provide expressive representation advantages over classical GCNs on brain graphs.
- **Protocol**: Evaluated on AAL-116 graphs across 162 clean-labeled subjects and 713 ensemble pseudo-labeled subjects. Clean cohort split deterministically (`random_state=42`, stratified): 103 training, 26 validation, 33 held-out test subjects. Classical model: 3-layer GCN ($117 \to 32 \to 32 \to 16 \to 2$, BatchNorm, Dropout 0.30, 20 epochs). Quantum model: Classical projection ($117 \to 12$), 6-qubit PQC (1 variational layer, 18 parameters), followed by 3 GCN layers.
- **Verified Findings**: Classical GCN achieved **0.7293 AUC** (69.70% accuracy, 15/19 TDC, 8/14 ADHD), decisively outperforming Hybrid QGCNN (**0.6429 AUC**, 60.61% accuracy, 11/19 TDC, 9/14 ADHD). Reject quantum advantage hypothesis: 6-qubit PQC acts as an expressive bottleneck (`results/exp07/checkpoint_analysis.json`).
- **Cohort Lineage Status**: Formally resolved. The 162 clean Exp07 subjects are verified at the subject-ID level as indices 0..161 of the 391 Exp06 aligned cohort (`results/exp07/clean_subjects_manifest.csv`). The historical scientific rationale for selecting the integer 162 prefix was not recovered.

### Exp 08: 4D Spatiotemporal Foundation Model vs. Volumetric Baselines
- **Hypothesis**: A pre-trained 4D Swin4D-Mamba foundation model (NeuroSTORM) outperforms lightweight 3D volumetric CNNs and temporal GNNs.
- **Protocol**: Evaluated on raw 4D BOLD volumes ($96\times96\times96\times80$), 3D mean anatomical volumes, and dynamic connectomes across 626 volumetric and 764 temporal subjects.
- **Verified Findings**: Lightweight 3D CNN achieved **76.19% test accuracy**, whereas NeuroSTORM achieved **59.10% 5-fold accuracy** and Temporal GNN achieved **54.43% test accuracy** (`results/exp08/exp08_verified_results.json`). Massive 4D architectures underperform lightweight baselines when fine-tuned on modest pediatric sample sizes without massive domain-specific pretraining.

### Exp 09: 7-Fold Leave-One-Site-Out (LOSO) Population Graph Learning
- **Hypothesis**: Graph Neural Networks generalize out-of-distribution across unseen clinical sites under strict LOSO evaluation.
- **Protocol**: CC200 atlas ($N=190$ ROIs), 497 subjects across 7 clinical hospital sites (KKI, NYU, NeuroIMAGE, OHSU, Peking_1, Peking_2, Peking_3; Brown excluded due to lack of diagnostic contrast). Evaluated 4 GNN architectures (GAT, SAGE, GCN, GIN) using subject-level graphs constructed with top-10% positive-FC thresholding and self-loops. Evaluated alongside 42 classical ML configurations across 6 feature families.
- **Verified Findings**:
  - Under 7-fold LOSO cross-validation, GAT achieved the highest mean AUC (**0.5752**), followed by SAGE (**0.5502**), GCN (**0.5468**), and GIN (**0.5437**) (`results/exp09/w2b_loso_results.csv`).
  - Non-imaging demographic baseline (Elastic Net on Age, Sex, Handedness) achieved **0.5935 mean AUC** (`results/exp09/w1_family_winners.csv`).
- **Graph Pipeline Distinction**: Exp 02 (dynamic FC $\to$ MST+PT $\to$ null models) and Exp 09 (static FC $\to$ top-10% positive threshold $\to$ self-loops $\to$ GNN $\to$ LOSO) represent two distinct experimental pipelines, not a single chained pipeline.
- **Edge Weight Semantics**: In Exp 09, signed Pearson correlation values are stored as `edge_attr` with self-loops. During message passing, only isotropic GCN consumes `edge_weight=edge_weight`. GAT, SAGE, and GIN perform unweighted message passing over connectivity defined by `edge_index`.
- **GAT Interpretation**: GAT achieved the highest mean AUC among the four tested GNN architectures under 7-fold LOSO. The empirical data do not directly prove that its attention weights suppress scanner noise.

---

# 3. Provenance & Methodological Resolutions

### Summary of Resolved vs. Unrecoverable Items

| Investigation Area | Prior Status | Current Canonical Finding | Retained Evidence |
| :--- | :--- | :--- | :--- |
| **Exp 06 $\to$ Exp 07 Lineage** | Unresolved | Proven: The 162 Exp 07 clean subjects are exactly indices 0..161 of the 391 Exp 06 aligned cohort ($162 \subset 391$). The historical reason for selecting the integer 162 prefix was not recovered. | `results/exp07/clean_subjects_manifest.csv` |
| **Exp 09 Threshold Selection** | Stale / Missing | Partially Resolved: Recovered historical 8-regime topological sweep ledger and downstream diagnostic validation ledgers. The original producer script and formal decision rule were not retained. | `results/exp09/graph_statistics.csv`, `gnn_diagnostic_positive.csv`, `gnn_diagnostic_absolute.csv` |
| **Exp 09 Pareto Front Artifact** | Misattributed | Resolved: `w2_pareto_front.csv` compares feature representation families (ComBat FC vs Raw FC vs GraphPheno), not edge thresholding. Disassociated from threshold metadata. | `configs/exp09/selected_threshold.json`, `results/exp09/README.md` |
| **Exp 06 79-Subject Metrics** | Ambiguous | Resolved: Accuracy metrics 0.6709 and 0.7215 derive from a separate 79-subject holdout evaluation, not the 59-subject validation partition. | `results/exp06/exp06_verified_results.json` |
| **Phenotypic Baseline Equivalence** | Apparent Bug | Resolved Finding: Table XII 'Phenotype' and 'Phenotype without IQ' are identical because `master_cohort.csv` contained no cognitive test columns. | `results/exp09/w1_canonical_model_summary.csv` |
| **Exp 04 Site Prediction Claim** | Stale String | Resolved: Canonical retained results record site accuracy dropping $58.64\% \to 31.29\%$ and diagnosis dropping $64.53\% \to 59.56\%$. Stale $95.7\% \to 14.8\%$ claims purged from current audit. | `results/exp04/site_prediction_results.csv`, `results/exp04/diagnosis_prediction_results.csv` |
| **Exp 09 Edge Semantics** | Ambiguous | Resolved: Signed FC attributes stored in graph; consumed as edge weights only by GCN, while GAT, SAGE, GIN pass unweighted connectivity. | `configs/exp09/w2b_best_config.json`, `configs/exp09/graph_preprocessing_config.json` |

---

# 4. Canonical Reproducibility Classification

```text
Track A (Exp 01 - Exp 05): Fully Reproducible
  - Preprocessed connectomes (CC200) + repository scripts produce exact numerical outputs.
  
Track A (Exp 06): Partially Reproducible
  - 391 aligned cohort defined; pseudo-label generation requires raw unaligned arrays.
  
Track B (Exp 07): Archived Evaluation Provenance
  - Model architectures defined in src/exp07/.
  - Execution from scratch requires external 875-subject arrays (X_combined_full.npy, y_combined.npy).
  - 33-subject clean test performance verified via checkpoint_analysis.json.
  
Track C (Exp 08): Archived Evaluation Provenance
  - Requires raw 4D BOLD volumes (~100 GB) maintained in institutional archives.
  - Baseline metrics verified via exp08_verified_results.json.
  
Track C (Exp 09a GNN LOSO): Historical Execution Lineage
  - Architecture and training defined in 11_population_graph_learning.ipynb.
  - Re-execution from scratch requires missing W2A intermediate correlation matrices.
  - Threshold sweep ledger recovered in results/exp09/graph_statistics.csv.
  
Track C (Exp 09b Classical LOSO): Fully Reproducible
  - 42 model configurations across 6 feature families reproducible via 12_classical_ml_baseline.ipynb.
```

---

# 5. Conclusion & Research Takeaways

1. **Topological Feature Sensitivity**: Raw fMRI graph metrics are overwhelmingly dominated by scanner hardware site variance ($F > 4,600$), requiring robust harmonization or domain-invariant learning before clinical deployment.
2. **Harmonization Trade-off**: Standard ComBat effectively reduces site prediction ($58.64\% \to 31.29\%$), but attenuates static diagnostic signal ($64.53\% \to 59.56\%$) due to site-diagnosis collinearity, while dynamic micro-state features show diagnostic resilience ($0.626 \to 0.675$).
3. **Quantum Expressive Bottleneck**: Constrained NISQ parameter budgets (6 qubits) act as a compression bottleneck, resulting in classical GCN decisively outperforming hybrid quantum architectures ($0.7293$ vs. $0.6429$ AUC).
4. **Foundation Model Overcapacity**: Pre-trained 4D spatiotemporal transformers (NeuroSTORM, 59.10%) fail to surpass lightweight 3D volumetric CNNs (76.19%) when fine-tuned on modest sample cohorts.
5. **Out-of-Distribution Generalization**: Under rigorous Leave-One-Site-Out cross-validation across 7 unseen clinical hospitals, Graph Attention Networks achieve the highest GNN mean AUC (0.5752), while simple demographic baselines achieve 0.5935 AUC, highlighting the clinical ceiling of resting-state fMRI connectomics in pediatric cohorts.
