# Experiment 8: Volumetric Deep Learning and Temporal Graph Baselines

## 1. Scientific Overview & Objectives
Experiment 8 establishes multimodal and spatio-temporal deep learning baselines to benchmark the graph convolutional and quantum architectures developed in Experiment 7. 

Three distinct architectural families are evaluated:
1. **Volumetric 3D CNN**: Operates directly on four-dimensional BOLD fMRI volume sequences.
2. **NeuroSTORM Spatio-Temporal Transformer**: A state-of-the-art transformer architecture incorporating spatial cross-attention and temporal self-attention.
3. **Temporal Graph Neural Network**: Evaluates dynamic sliding-window graph sequences with recurrent or temporal pooling.

---

## 2. Model Architectures & Methodological Clarifications

### 2.1 Baseline 1: Lightweight 3D CNN on Temporal 4D Volumes
> [!IMPORTANT]
> **Conflation Correction (fMRI Volumes vs Structural T1)**:
> An earlier draft report mistakenly described this model as an anatomical structural/T1 MRI network. Source code inspection of [`notebooks/exp08/neuro.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/neuro.ipynb) confirms that the input data consists of **temporal functional BOLD volume sequences**:
> - Temporal frames are subsampled ($T = 50 \to 25$ volumes).
> - Spatial resolution per volume: $99 \times 117 \times 95$ voxels.
> - Network Architecture: 3D convolutional blocks with 3D batch normalization, max pooling, and dropout (~203,000 learnable parameters).

### 2.2 Baseline 2: NeuroSTORM Spatio-Temporal Transformer
- **Upstream Implementation**: Adapted from the CUHK-AIM-Group repository ([NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)), licensed under Apache-2.0 and staged at [`src/exp08/neurostorm/neurostorm.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp08/neurostorm/neurostorm.py).
- **Design**: Spatial cross-attention over ROI tokens coupled with temporal multi-head self-attention to model long-range dynamic interactions.

### 2.3 Baseline 3: Temporal Graph Neural Network
> [!IMPORTANT]
> **Data Leakage Claim Disproven**:
> An earlier reverse-engineering draft claimed that Experiment 8 suffered from severe data leakage by randomly partitioning overlapping sliding windows between train and test sets. 
> Rigorous audit of [`notebooks/exp08/09_temporal_graph_learning.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/09_temporal_graph_learning.ipynb) disproves this claim:
> - Partitions are strictly constructed at the **subject level**: **534 training subjects**, **115 validation subjects**, and **115 testing subjects**.
> - Disjoint-subject assertions (`assert set(train_subs).isdisjoint(test_subs)`) are explicitly executed in the notebook.
> - All windowed graphs ($N = 4,677$) evaluated in the test set originate exclusively from the 115 held-out test subjects.

---

## 3. Audited Quantitative Results
Traced directly to [`results/exp08/exp08_verified_results.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp08/exp08_verified_results.json):

| Model Architecture | Evaluation Protocol | Test Accuracy | Test AUROC | Source Notebook |
| :--- | :--- | :--- | :--- | :--- |
| **Lightweight 3D CNN** | Single held-out split ($N=63$ test subjects) | **$76.19\%$** ($48/63$) | **$0.7316$** | `notebooks/exp08/neuro.ipynb` |
| **NeuroSTORM** | 5-Fold Cross-Validation | **$59.10\%$** | **$0.5880$** | `notebooks/exp08/true_neuro.ipynb` |
| **NeuroSTORM** | Single held-out split | **$68.25\%$** | **$0.6800$** | `notebooks/exp08/true_neuro.ipynb` |
| **Temporal Graph** | Window-level test ($4,677$ test graphs) | **$54.43\%$** | **$0.5534$** | `notebooks/exp08/09_temporal_graph_learning.ipynb` |

### Temporal Graph Confusion Matrix ($N = 4,677$ test graphs)
$$\begin{bmatrix} \text{TDC Correct} & \text{TDC False Alarm} \\ \text{ADHD Miss} & \text{ADHD Hit} \end{bmatrix} = \begin{bmatrix} 1,661 & 1,242 \\ 839 & 825 \end{bmatrix}$$
- Sensitivity (Recall on ADHD): $825 / (825 + 839) = 49.58\%$
- Specificity (TDC): $1,661 / (1,661 + 1,242) = 57.22\%$

---

## 4. Architectural Separation & Exclusions
- **Hybrid Graph Transformer**: Staged exploratory notebooks (such as `updated_transformer.ipynb`) represent subsequent exploratory research and are not part of the paper-reported Experiment 8 benchmark suite.
- **Pretrained Weights**: Model weight files (`*.pth`, `*.ckpt`) exceeding GitHub file size quotas are excluded from staging; code and verification outputs are fully preserved.

---

## 5. Source Code & Result Provenance
- **Notebooks**:
  - [`notebooks/exp08/neuro.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/neuro.ipynb) (Volumetric 3D CNN)
  - [`notebooks/exp08/true_neuro.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/true_neuro.ipynb) (NeuroSTORM Transformer)
  - [`notebooks/exp08/09_temporal_graph_learning.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp08/09_temporal_graph_learning.ipynb) (Temporal GNN)
- **Source Scripts**:
  - [`src/exp08/neurostorm/neurostorm.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp08/neurostorm/neurostorm.py) (NeuroSTORM architecture, Apache-2.0)
- **Result Artifacts**:
  - [`results/exp08/exp08_verified_results.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp08/exp08_verified_results.json)
  - [`results/exp08/neurostorm_true_results.png`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp08/neurostorm_true_results.png)
