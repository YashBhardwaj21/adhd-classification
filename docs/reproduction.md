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
  - Exp 09: 7-fold Leave-One-Site-Out (LOSO) population graph learning

### Category B: Requires Reconstruction of Excluded Intermediates
The following experiments can be reproduced after reconstructing intermediate data structures from raw or preprocessed scans:
- **Track B (Semi-Supervised & Baseline Deep Learning)**:
  - Exp 06: Semi-supervised pseudo-labeling (Procedure I & II) generates intermediate pseudo-label assignments from AAL-116 correlation arrays.
  - Exp 08: Volumetric 3D CNN and NeuroSTORM transformer require extracted 4D functional volume sequences ($T=25, 99 \times 117 \times 95$), which are excluded from the repository.

### Category C: Archived / Not Currently Rerunnable from Repository
- **Exp 07 (Classical GCN vs Quantum QGCNN)**:
  Experiment 07 is archived rather than currently rerunnable from the public repository because the historical combined input arrays and trained checkpoints are not redistributed. The source code, historical configuration, and verified evaluation results ($N=33$ held-out test subjects, Classical AUC $0.7293$ vs Quantum AUC $0.6429$) are retained in `results/exp07/checkpoint_analysis.json` and `results/exp07/report.md`. The original large combined numpy arrays (`X_combined_full.npy`, `y_combined.npy`, `subjects_combined.npy`: 162 clean + 713 pseudo-labelled subjects) and trained checkpoints are absent, so end-to-end reruns are currently unavailable without external restoration of those artifacts. Running `src/exp07/classical_models/run_experiment.py` or `src/exp07/quantum_models/run_experiment.py` without external arrays will explicitly halt with an informative `FileNotFoundError`.

---

## 2. Environment Specifications

### 2.1 Recorded Historical Environment (A100 Execution)
Recorded from `configs/exp09/w2b_environment.json`:
- **Hardware**: NVIDIA A100-SXM4-80GB GPU, x86_64 Host
- **Python**: `3.12.13`
- **PyTorch**: `2.5.1+cu121` (CUDA 12.1 runtime)
- **PyTorch Geometric**: `2.8.0`
- **PennyLane**: `0.44.1`
- **PennyLane-Lightning-GPU**: `0.44.0`
- **NumPy**: `2.4.6`, **Pandas**: `2.3.3`, **Scikit-Learn**: `1.8.0`

### 2.2 Track-Specific Environment Rationale
Dependency requirements are partitioned into three track environments under `environment/` to minimize package conflicts and maintain historical isolation:
- **`environment/track_a/requirements.txt`**: Lightweight classical connectomics, graph theory, and statistical modeling (`bctpy`, `neuroCombat`, `scipy`, `pandas`, `scikit-learn`). Includes `bctpy` for the Experiment 02 BCT-dependent graph null-model functionality. Does not require GPU or PyTorch.
- **`environment/track_b/requirements.txt`**: Hybrid quantum-classical and spatio-temporal deep learning stack (`pennylane`, `pennylane-lightning-gpu`, `monai`, `torch`, `torch-geometric`). Requires CUDA-enabled PyTorch. Experiment 07 itself operates on reconstructed adjacency and does not require `bctpy` merely because BCT exists elsewhere in the repository.
- **`environment/track_c/requirements.txt`**: Population-level graph neural network learning (`torch`, `torch-geometric`, `scikit-learn`).

### 2.3 Suggested Installation

`pyproject.toml` provides package-level dependencies for local package installation (`pip install -e .`). Track-specific requirements files document the runtime dependency versions and ranges used for each experimental track. The separately recorded environment metadata provides the observed package versions for the historical execution environment.

```bash
# 1. Create dedicated Python 3.12 virtual environment
conda create -n adhd200 python=3.12.13 -y
conda activate adhd200

# 2. Install PyTorch with CUDA 12.1 runtime
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch-geometric==2.8.0

# 4. Install Track-specific dependencies (e.g. Track B for quantum/GNN models)
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
