# Reproduction Guide & Execution Environment

This guide describes the software environment, hardware requirements, and reproduction procedures for the experiments in this repository.

---

## 1. Reproduction Status by Experiment

Experiments in this study fall into two distinct reproducibility categories based on data availability:

### Category 1: Rerunnable from Repository + Obtainable External ADHD-200 Data
The following experiments can be executed using the code in this repository once the public ADHD-200 preprocessed data are downloaded:
- **Track A (Connectomic Dynamics & Harmonization)**:
  - Exp 01: Dynamic functional connectivity generation and stability
  - Exp 02: CC200 graph construction and signed null models
  - Exp 03: Topological feature extraction and cross-site ANOVA
  - Exp 04: ComBat scanner harmonization and classification trade-offs
  - Exp 05: Dynamic micro-state clustering and Markov transitions
- **Track B (Semi-Supervised & Baseline Deep Learning)**:
  - Exp 06: Semi-supervised pseudo-labelling
  - Exp 08: Lightweight 3D CNN, NeuroSTORM, and temporal GNN baselines
- **Track C (Population Graph Learning & Generalization)**:
  - Exp 09: 7-fold Leave-One-Site-Out (LOSO) population graph learning

### Category 2: Reported Results Retained; Original Execution Inputs Unavailable
- **Exp 07 (Classical GCN vs Quantum QGCNN)**:
  The reported results ($N=33$ held-out test subjects, Classical AUC $0.7293$ vs Quantum AUC $0.6429$) are archived in `results/exp07/checkpoint_analysis.json` and `results/exp07/report.md`. The original large combined numpy arrays (`X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy`) and trained model weights exceed repository quotas and are unavailable. Consequently, Experiment 07 cannot be rerun end-to-end from this repository without external restoration of those arrays. Running `src/exp07/classical_models/run_experiment.py` or `src/exp07/quantum_models/run_experiment.py` without external arrays will explicitly halt with an informative `FileNotFoundError`.

---

## 2. Environment Specifications

### 2.1 Recorded Historical Environment (A100 Execution)
Reconstructed from `configs/exp09/w2b_environment.json`:
- **Hardware**: NVIDIA A100-SXM4-80GB GPU, x86_64 Host
- **Python**: `3.12.13`
- **PyTorch**: `2.5.1+cu121` (CUDA 12.1 runtime)
- **PyTorch Geometric**: `2.8.0`
- **PennyLane**: `0.44.1`
- **PennyLane-Lightning-GPU**: `0.44.0`
- **NumPy**: `2.4.6`, **Pandas**: `2.3.3`, **Scikit-Learn**: `1.8.0`

### 2.2 Suggested Installation

```bash
# 1. Create dedicated Python 3.12 virtual environment
conda create -n adhd200 python=3.12.13 -y
conda activate adhd200

# 2. Install PyTorch with CUDA 12.1 runtime
pip install torch==2.5.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch-geometric==2.8.0

# 4. Install Track B dependencies (PennyLane, MONAI, scientific stack)
pip install -r environment/track_b/requirements.txt

# 5. Install local package in editable mode
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

8. **Exp 09 — Leave-One-Site-Out (LOSO) Cross-Validation**:
   ```bash
   jupyter nbconvert --execute notebooks/exp09/11_population_graph_learning.ipynb --to notebook
   ```

---

## 4. Verification & Testing

Verify repository structural integrity and numerical consistency:

```bash
# 1. Check directory structure, notebook validity, and non-empty artifacts
python scripts/audit_repo.py

# 2. Check numerical consistency against retained result tables
python scripts/validate_results.py

# 3. Run automated unit test suite
pytest tests/
```
