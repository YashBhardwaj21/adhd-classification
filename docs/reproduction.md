# Reproduction Guide & Execution Environment

This guide describes the software environment, hardware requirements, and reproduction procedures for the experiments in this repository.

---

## 1. Reproduction Status by Experiment

Experiments in this study fall into three distinct reproducibility categories based on input data availability:

### Category A: Rerunnable from Repository + Obtainable External ADHD-200 Data
The following experiments can be executed using the code in this repository once the public ADHD-200 preprocessed connectome data are downloaded:
- **Track A (Connectomic Dynamics & Harmonization)**:
  - Exp 01: Dynamic functional connectivity generation and stability
  - Exp 02: CC200 graph construction and signed null models
  - Exp 03: Topological feature extraction and cross-site ANOVA
  - Exp 04: ComBat scanner harmonization and classification trade-offs
  - Exp 05: Data-driven dynamic connectivity states and Markov transitions
- **Track C (Population Graph Learning & Generalization)**:
  - Exp 09b: Classical machine learning 7-fold LOSO benchmarks (evaluates 7 classifier types across 6 tabular feature families).

### Category B: Requires Reconstruction of Excluded Intermediates
The following experiments can be reproduced after reconstructing intermediate data structures from raw or preprocessed scans:
- **Track B (Semi-Supervised & Baseline Deep Learning)**:
  - Exp 06: Semi-supervised pseudo-labeling (Procedure I & II) generates intermediate pseudo-label assignments from AAL-116 correlation arrays.
  - Exp 08: Volumetric 3D CNN and NeuroSTORM transformer require extracted 4D functional volume sequences ($T=25, 99 \times 117 \times 95$, $>120\text{ GB}$, 626 scans), which are excluded from the repository.
- **Track C (GNN Population Graph Learning)**:
  - Exp 09a: 7-fold Leave-One-Site-Out (LOSO) population graph learning across GNN architectures (`notebooks/exp09/11_population_graph_learning.ipynb`). While final evaluation outputs and `w2_pareto_front.csv` are retained in `results/exp09/`, executing the notebook requires intermediate feature parquets (`w1_pheno_features.parquet`, `w1_graph_features.parquet`) and upstream Workflow 2A representation audit ledgers (`w2_phase2_audit_all_reps.csv`, `w2_site_signal_summary.csv`) which are not bundled in the public repository tree.

### Category C: Archived / Not Currently Rerunnable from Repository
- **Exp 07 (Classical GCN vs Quantum QGCNN)**:
  Experiment 07 is archived rather than currently rerunnable from the public repository because the historical combined input arrays and trained checkpoints are not redistributed. The source code, historical configuration, and verified evaluation results ($N=33$ held-out test subjects, Classical AUC $0.7293$ vs Quantum AUC $0.6429$) are retained in `results/exp07/checkpoint_analysis.json` and `results/exp07/report.md`. The original large combined numpy arrays (`X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy`: 162 clean + 713 pseudo-labelled subjects) and trained checkpoints are absent, so end-to-end reruns are currently unavailable without external restoration of those artifacts. Running `src/exp07/classical_models/run_experiment.py` or `src/exp07/quantum_models/run_experiment.py` without external arrays will explicitly halt with an informative `FileNotFoundError`.

---

## 2. Environment Specifications

### 2.1 Recorded Historical Environments
The experiments in this study were executed across distinct, isolated computing environments:
- **Track A (Experiments 01–05: Connectomics & Harmonization)**:
  - Hardware: x86_64 CPU workstation
  - Python: `3.10`–`3.12`
  - Core dependencies: `bctpy==0.6.1`, `neuroCombat==0.2.12`, `scipy==1.17.1`, `scikit-learn==1.8.0`, `pandas==2.3.3`, `numpy==2.4.6`
- **Track B — Experiments 06 & 07 (Semi-Supervised & Quantum GNN)**:
  - Hardware: NVIDIA GPU with CUDA 12.1 runtime
  - Python: `3.12.13`
  - Core dependencies: `torch==2.5.1+cu121`, `torch-geometric==2.8.0`, `pennylane==0.44.1`, `pennylane-lightning-gpu==0.44.0`
- **Track B — Experiment 08 (NeuroSTORM Spatio-Temporal Baseline)**:
  - Hardware: Brev NVIDIA A100 GPU instance
  - Python: `3.12`
  - Core dependencies: `torch==2.7.1`, CUDA 12.6, Docker-pinned `causal-conv1d==v1.5.0.post8`, `mamba==v2.2.2` built from source
  - Canonical NeuroSTORM commit: `8080b539432862d72f90482da22aa2a19f4edc5d`
- **Track C — Experiment 09 (LOSO Population Graphs & Classical Baselines)**:
  - Hardware: Azure Cloud VM, NVIDIA A100-SXM4-80GB GPU
  - Recorded in `configs/exp09/w2b_environment.json`:
  - Python: `3.12.13`, `torch==2.5.1+cu121` (CUDA 12.1), `torch-geometric==2.8.0`, `scikit-learn==1.8.0`, `pandas==2.3.3`, `numpy==2.4.6`

### 2.2 Track-Specific Environment Installation
Do not attempt to install all tracks into a single unified environment, as NeuroSTORM and PennyLane have conflicting CUDA and C++ extension build requirements. Install the dedicated environment matching the track you wish to reproduce:

#### Track A (Connectomics & Graph Null Models)
```bash
conda create -n adhd200_track_a python=3.12 -y
conda activate adhd200_track_a
pip install -r environment/track_a/requirements.txt
pip install -e .
```

#### Track B — Experiments 06 & 07 (Semi-Supervised & Quantum GNN)
```bash
conda create -n adhd200_track_b python=3.12 -y
conda activate adhd200_track_b
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.8.0
pip install -r environment/track_b/requirements.txt
pip install -e .
```
*Note on NeuroSTORM (Exp 08)*: NeuroSTORM requires a specialized Docker container built with PyTorch 2.7.1, CUDA 12.6, and custom Mamba/causal-conv1d extensions (see `audit_source_files/NeuroSTORM/Dockerfile`). It cannot be reproduced simply via standard pip install on commodity systems.

#### Track C (LOSO Population Graphs & Classical Baselines)
```bash
conda create -n adhd200_track_c python=3.12 -y
conda activate adhd200_track_c
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.8.0
pip install -r environment/track_c/requirements.txt
pip install -e .
```

---

## 3. Step-by-Step Execution Sequence

Execute the canonical notebooks in sequence (paths verified against the repository tree):

### Track A: Connectomic Dynamics & Harmonization (CC200 Atlas)

1. **Exp 01 — Dynamic FC Generation & Temporal Stability**:
   ```bash
   jupyter nbconvert --execute notebooks/exp01/01_fc_generation_and_validation.ipynb --to notebook
   ```

2. **Exp 02 — Graph Construction & Topological Metrics**:
   ```bash
   jupyter nbconvert --execute notebooks/exp02/02_graph_construction_and_validation.ipynb --to notebook
   jupyter nbconvert --execute notebooks/exp02/03_graph_metrics_and_null_models.ipynb --to notebook
   ```

3. **Exp 03 — Topological Feature Extraction & Cross-Site ANOVA**:
   ```bash
   jupyter nbconvert --execute notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb --to notebook
   ```

4. **Exp 04 — ComBat Harmonization & Classification Trade-offs**:
   ```bash
   jupyter nbconvert --execute notebooks/exp04/w2c_athena_2.ipynb --to notebook
   ```

5. **Exp 05 — Dynamic Brain State Modeling**:
   ```bash
   jupyter nbconvert --execute notebooks/exp05/dynamic_transformer.ipynb --to notebook
   ```

### Track B: Semi-Supervised Learning & Baselines

6. **Exp 06 — Semi-Supervised Pseudo-Labeling**:
   ```bash
   jupyter nbconvert --execute notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb --to notebook
   ```

7. **Exp 08 — Volumetric & Temporal Baselines**:
   ```bash
   jupyter nbconvert --execute notebooks/exp08/neuro.ipynb --to notebook
   jupyter nbconvert --execute notebooks/exp08/true_neuro.ipynb --to notebook
   jupyter nbconvert --execute notebooks/exp08/09_temporal_graph_learning.ipynb --to notebook
   ```

### Track C: Population Graph Learning & Generalization (CC200 Atlas)

8. **Exp 09a — Population Graph Learning (GNN Architectures)**:
   *Note*: Requires upstream Workflow 2A intermediate audit artifacts (`w2_phase2_audit_all_reps.csv`, `w2_site_signal_summary.csv`) and Workflow 1 feature parquets:
   ```bash
   jupyter nbconvert --execute notebooks/exp09/11_population_graph_learning.ipynb --to notebook
   ```

9. **Exp 09b — Classical Machine Learning LOSO Benchmarks**:
   ```bash
   jupyter nbconvert --execute notebooks/exp09/12_classical_ml_baseline.ipynb --to notebook
   ```

---

## 4. Methodological & Preprocessing Notes

1. **Exp 09 Graph Representation & Feature Selection**:
   - GNN models operate on the full 190-node connectome thresholded at the 90th percentile of positive correlations (`top_10pct` density, mean 3,782 directed edges) with self-loops and signed edge weights. No ROI feature selection is applied to the graph inputs.
   - For classical machine learning baselines (`12_classical_ml_baseline.ipynb`), feature selection (`SelectKBest(f_classif)`) is strictly nested within each training fold, ensuring held-out scanner sites remain completely unobserved during feature ranking.
   - Node representation format selection (`connectivity` vs `identity` vs `strength`) was compared in an exploratory, non-nested LOSO workflow in Workflow 2A. This selection occurred prior to final GNN hyperparameter training, confirming `connectivity` (unreduced 190-dim correlation rows) as optimal; this exploratory comparison should not be interpreted as an unbiased nested LOSO performance estimate.

2. **External Data Prerequisites**:
   - **Volumetric 4D Scans (Exp 08)**: Raw 4D BOLD fMRI volumes (`>120 GB`) and converted MNI `.npy` arrays (`>30 GB`) exceed repository storage quotas and must be retrieved from institutional ADHD-200 mirrors.
   - **Combined Pseudo-Label Arrays (Exp 07)**: The 875-subject correlation array (`X_combined_full.npy`, 162 clean + 713 pseudo) is preserved externally. The complete cohort composition, split rules, and test evaluation metrics on the 33 held-out subjects are documented in `results/exp07/pseudolabel_provenance.json` and `results/exp07/checkpoint_analysis.json`.
   - **Workflow 2A Representation Audit Artifacts (Exp 09a)**: Intermediate sweep files (`w2_phase2_audit_all_reps.csv`, `w2_site_signal_summary.csv`) and feature parquets (`w1_pheno_features.parquet`, `w1_graph_features.parquet`) used during exploratory representation selection are unbundled from the repository. The canonical Pareto trade-off summary is retained in `results/exp09/w2_pareto_front.csv`, and final LOSO evaluation metrics are retained in `results/exp09/w2b_loso_results.csv` and `architecture_summary.csv`.

---

## 5. Verification & Testing

Verify repository structural integrity and numerical consistency:

```bash
# 1. Check directory structure, notebook validity, and non-empty artifacts
python scripts/audit_repo.py

# 2. Check numerical consistency against retained result tables
python scripts/validate_results.py

# 3. Run automated unit test suite
pytest tests/
```

