# Experiment 2: Dual-Constraint Graph Construction & Null Model Verification

## 1. Scientific Overview & Objectives
Experiment 2 converts the windowed functional connectivity matrices generated in Experiment 1 into topological brain graphs. A major challenge in functional connectomics is that arbitrary correlation thresholds can fragment the network into disconnected components or introduce spurious edges.

To resolve this, Experiment 2 implements a dual-constraint graph construction strategy:
1. **Minimum Spanning Tree (MST)**: Guarantees full global connectivity without isolated nodes or disconnected components ($N-1 = 189$ edges).
2. **Proportional Thresholding (PT)**: Adds the strongest remaining absolute correlation edges until a target edge density of $\rho = 0.20$ is achieved.

Topological small-world properties ($\sigma = \gamma / \lambda$) are then validated against randomized null models using degree-preserving rewiring.

---

## 2. Graph Topology & Construction Parameters
- **Atlas**: Craddock-200 (CC200), $N = 190$ regions of interest.
- **Edge Density**: $\rho = 0.20$ (20% of all possible pairwise connections).
- **Total Possible Edges**:
  $$\frac{N(N - 1)}{2} = \frac{190 \times 189}{2} = 17,955 \text{ edges}$$
- **Retained Edges per Graph**:
  $$E = \lfloor 0.20 \times 17,955 \rfloor = 3,591 \text{ undirected edges}$$
- **Fixed Mean Node Degree**:
  $$\langle k \rangle = \frac{2E}{N} = \frac{2 \times 3,591}{190} = 37.80$$
- **Graph Invariance**: Every single windowed graph in the 31,060-window dataset shares identical node and edge counts, preventing density-confounded topological distortions.

---

## 3. Mathematical Methodology

### 3.1 Dual-Constraint Algorithm
For an empirical correlation matrix $\mathbf{C} \in \mathbb{R}^{190 \times 190}$:
1. Define edge distance weights $W_{i,j} = 1 - |\mathbf{C}_{i,j}|$.
2. Compute the Minimum Spanning Tree $G_{\text{MST}} = (V, E_{\text{MST}})$ using Kruskal's algorithm, yielding $|E_{\text{MST}}| = 189$ edges.
3. Sort remaining candidate edges $(i, j) \notin E_{\text{MST}}$ in descending order of $|\mathbf{C}_{i,j}|$.
4. Insert candidate edges into $G$ until $|E| = 3,591$.

### 3.2 Null Model Randomization
To compute normalized graph metrics, empirical networks are benchmarked against degree-preserving randomized surrogate networks using the Rubinov & Sporns (2011) algorithm implemented in [`src/exp02/null_model_und_sign_fixed.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/null_model_und_sign_fixed.py):
- **Normalized Clustering ($\gamma$)**: $\gamma = C_{\text{real}} / C_{\text{null}}$
- **Normalized Path Length ($\lambda$)**: $\lambda = L_{\text{real}} / L_{\text{null}}$
- **Small-World Index ($\sigma$)**: $\sigma = \gamma / \lambda$

A network is classified as possessing small-world topology if $\gamma > 1$ and $\lambda \approx 1$, resulting in $\sigma > 1$.

---

## 4. Audited & Verified Quantitative Results
Traced directly to [`results/exp02/graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/graph_metrics.csv) across all 31,060 dynamic windows:

| Topological Metric | Real Network ($\mu \pm \sigma$) | Null Surrogate ($\mu \pm \sigma$) | Ratio / Index | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Clustering ($C$)** | $0.3480 \pm 0.0479$ | $0.3373 \pm 0.0470$ | $\gamma = 1.0318 \pm 0.0048$ | $\gamma > 1$ |
| **Path Length ($L$)** | $2.8327 \pm 0.1004$ | $2.7818 \pm 0.1087$ | $\lambda = 1.0185 \pm 0.0103$ | $\lambda \approx 1$ |
| **Global Efficiency ($E_{\text{glob}}$)** | $0.3929 \pm 0.0157$ | $0.4001 \pm 0.0156$ | — | High integration |
| **Transitivity ($T$)** | $0.3468 \pm 0.0601$ | $0.3471 \pm 0.0618$ | — | Robust triadic closure |
| **Small-World Index ($\sigma$)** | — | — | $\mathbf{\sigma = 1.0132 \pm 0.0103}$ | $\sigma > 1$ Verified |

---

## 5. Implementation Caveats & Provenance Corrections
> [!WARNING]
> **Provenance Correction**: An earlier draft reverse-engineering report misattributed `results/exp04/comparison_table.csv` as evidence for Experiment 2. That file represents ComBat harmonization comparison statistics for Experiment 4. The authoritative metrics for Experiment 2 reside exclusively in [`results/exp02/graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/graph_metrics.csv).

> [!NOTE]
> **Removal of 0-Byte Script**: The repository previously contained an unpopulated 0-byte script `src/exp02/null_model_und_sign_fast.py`. This dead file has been removed; all null-model calculations are executed by the verified implementation in [`src/exp02/null_model_und_sign_fixed.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/null_model_und_sign_fixed.py).

---

## 6. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp02/02_graph_construction_and_null_models.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp02/02_graph_construction_and_null_models.ipynb)
- **Source Scripts**:
  - [`src/exp02/config.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/config.py) (graph hyperparams: $N=190, \rho=0.20$)
  - [`src/exp02/graph_utils.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/graph_utils.py) (PyG conversion utilities)
  - [`src/exp02/null_model_und_sign_fixed.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/null_model_und_sign_fixed.py) (degree-preserving null models)
  - [`src/exp02/randmio_und_signed_fast.py`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/src/exp02/randmio_und_signed_fast.py) (randomized edge swapping)
- **Result Artifacts**:
  - [`results/exp02/graph_construction_strategy.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/graph_construction_strategy.csv)
  - [`results/exp02/graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/graph_metrics.csv)
  - [`results/exp02/subject_graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/subject_graph_metrics.csv)
  - [`results/exp02/site_graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/site_graph_metrics.csv)
  - [`results/exp02/acquisition_graph_metrics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp02/acquisition_graph_metrics.csv)
