# Reproduction Guide & Execution Environment

## 1. Verified Software & Hardware Environment

This repository's environment specifications are reconstructed directly from the execution environment metadata recorded in [`configs/exp09/w2b_environment.json`](../configs/exp09/w2b_environment.json).

### 1.1 Tested Hardware Profile
- **Accelerator**: NVIDIA A100-SXM4-80GB (VRAM: $81,920\text{ MiB}$)
- **Host Architecture**: Linux x86_64 / Windows 11 Compatibility
- **Host Memory**: $64\text{ GB} - 128\text{ GB}$ RAM recommended
- **CUDA Capability**: 8.0+ (Ampere / Hopper architecture recommended for TensorFloat-32)

### 1.2 Audited Software Dependencies
| Package | Audited Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.12.13` | Base runtime environment |
| **PyTorch** | `2.5.1+cu121` | Deep learning backend with CUDA 12.1 runtime |
| **torch-geometric (PyG)** | `2.8.0` | Graph neural network architectures (GCN, GAT, SAGE, GIN) |
| **PennyLane** | `0.44.1` | Quantum circuit simulations & parameter broadcasts |
| **PennyLane-Lightning-GPU**| `0.44.1` | High-performance state-vector GPU quantum backend |
| **NumPy** | `2.4.6` | Numerical vectorization & matrix operations |
| **Pandas** | `2.3.3` | Tabular data analysis and manifest management |
| **Scikit-Learn** | `1.8.0` | Classical baselines, clustering, evaluation metrics |
| **MONAI** | `1.5.2` | Medical imaging volumetric data pipelines |
| **bctpy** | `0.5.2` | Brain Connectivity Toolbox algorithms |
| **neuroCombat** | `0.2.1` | Empirical Bayes multi-site scanner harmonization |

---

## 2. Environment Setup Instructions

### 2.1 Conda Environment Creation
```bash
# 1. Create dedicated Python 3.12 virtual environment
conda create -n adhd200 python=3.12.13 -y
conda activate adhd200

# 2. Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric and optional dependencies
pip install torch-geometric==2.8.0

# 4. Install Quantum simulation packages
pip install pennylane==0.44.1 pennylane-lightning-gpu==0.44.1

# 5. Install scientific computing, neuroimaging, and evaluation libraries
pip install numpy==2.4.6 pandas==2.3.3 scikit-learn==1.8.0 monai==1.5.2 bctpy neuroCombat tqdm scipy
```

### 2.2 Verifying PyTorch and GPU Acceleration
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

---

## 3. End-to-End Execution Sequence

The repository is structured into three independent experimental tracks. Execute notebooks or scripts in the prescribed sequence:

### Track A: Connectomic Dynamics & Multi-Site Harmonization (CC200 Atlas)
1. **Dynamic FC Generation**:
   ```bash
   jupyter nbconvert --execute notebooks/exp01/01_fc_generation_and_validation.ipynb --to notebook
   ```
2. **Dual-Constraint Graph Construction & Null Models**:
   ```bash
   jupyter nbconvert --execute notebooks/exp02/02_graph_construction_and_null_models.ipynb --to notebook
   ```
3. **Topological Feature Extraction & ANOVA**:
   ```bash
   jupyter nbconvert --execute notebooks/exp03/03_topological_feature_extraction.ipynb --to notebook
   ```
4. **ComBat Harmonization**:
   ```bash
   jupyter nbconvert --execute notebooks/exp04/04_combat_harmonization.ipynb --to notebook
   ```
5. **Dynamic Brain State Clustering**:
   ```bash
   jupyter nbconvert --execute notebooks/exp05/05_dynamic_state_modeling.ipynb --to notebook
   ```

### Track B: Semi-Supervised Learning & Deep Architectures (AAL-116 Atlas)
1. **Semi-Supervised Pseudo-Labeling**:
   ```bash
   jupyter nbconvert --execute notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb --to notebook
   ```
2. **Classical GCN Training**:
   ```bash
   python src/exp07/classical_models/run_experiment.py
   ```
3. **Quantum GCNN Training (Requires GPU)**:
   ```bash
   python src/exp07/quantum_models/run_experiment.py
   ```
4. **Volumetric & Temporal Baselines**:
   ```bash
   jupyter nbconvert --execute notebooks/exp08/neuro.ipynb --to notebook
   jupyter nbconvert --execute notebooks/exp08/true_neuro.ipynb --to notebook
   jupyter nbconvert --execute notebooks/exp08/09_temporal_graph_learning.ipynb --to notebook
   ```

### Track C: Population Graph Learning & Generalization (CC200 Atlas)
1. **Leave-One-Site-Out Cross-Validation (GCN, GAT, SAGE, GIN)**:
   ```bash
   jupyter nbconvert --execute notebooks/exp09/11_population_graph_learning.ipynb --to notebook
   ```

---

## 4. Verification & Validation Utilities

Run repository verification scripts to assert code and artifact integrity:
```bash
# 1. Audit repository structure and staged artifacts
python scripts/audit_repo.py

# 2. Validate numerical consistency against source-of-truth ledgers
python scripts/validate_results.py

# 3. Run automated unit tests
pytest tests/
```
