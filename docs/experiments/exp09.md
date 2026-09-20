# Experiment 9: Independent Leave-One-Site-Out (LOSO) Population Graph Learning

## 1. Scientific Overview & Objectives
Experiment 9 evaluates the real-world out-of-distribution clinical generalization of Graph Neural Networks (GNNs) using strict **Leave-One-Site-Out (LOSO) cross-validation**. 

Rather than randomly splitting subjects across the consortium (where scanner-specific background distributions are shared between train and test sets), LOSO cross-validation trains models on six acquisition sites and tests exclusively on the seventh unseen site.

The primary objectives are:
1. Benchmark four canonical GNN architectures (GCN, GAT, GraphSAGE, GIN) on whole-brain functional connectivity graphs.
2. Measure true cross-site generalization to completely unobserved scanner hardware and imaging protocols.
3. Quantify the generalization gap between internal validation performance and external site testing.

---

## 2. Cohort & Multi-Site Distribution
Traced directly to [`results/exp09/w2b_loso_results.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/w2b_loso_results.csv) and [`results/exp09/graph_preprocessing_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/graph_preprocessing_summary.csv):

- **Atlas**: Craddock-200 (CC200), $N = 190$ regions of interest.
- **Total Cohort**: **497 subjects** across 7 independent acquisition sites:
  - **KKI** (Kennedy Krieger Institute): 83 subjects
  - **NYU** (New York University Child Study Center): 93 subjects
  - **NeuroIMAGE**: 48 subjects
  - **OHSU** (Oregon Health & Science University): 79 subjects
  - **Peking_1** (Peking University Scan 1): 85 subjects
  - **Peking_2** (Peking University Scan 2): 67 subjects
  - **Peking_3** (Peking University Scan 3): 42 subjects
- **Consortium Diagnosis Ratio**: $43.46\%$ ADHD ($216/497$).

---

## 3. Methodological Discrepancy: Paper Description vs Executed Code
> [!WARNING]
> **Protocol Mismatch in Experiment 9**:
> An unresolved discrepancy exists between the paper methodology text and the executed code in `notebooks/exp09/11_population_graph_learning.ipynb`:
> - **Paper Text Description**:
>   - Graph Construction: Minimum Spanning Tree (MST) + 20% proportional threshold.
>   - Distance Metric: Inverse-distance mapping $d = 1 / |r|$.
>   - Edge Topology: Unweighted edges.
> - **Executed Notebook Configuration**:
>   - Graph Selection: Proportional thresholding selecting the top 10% strongest absolute correlation edges (`top_10pct`).
>   - Edge Attributes: Weighted correlation values with explicit self-loops added.
>   - Node Features: 190-dimensional raw connectivity vectors (`Node_Feature_Dim = 190`).
>   - Empirical Density: Mean edges $= 1,795.5 \pm 0.0$, Mean density $= 0.10003$.
> 
> The repository documentation preserves the executed configuration and its verified artifacts without paper-facing concealment.

---

## 4. Evaluated Architectures
All models process graphs with $N = 190$ nodes, $D = 190$ node features, and two message-passing layers:
- **GCN**: $\text{GCNConv}(190 \to 64) \to \text{GCNConv}(64 \to 32) \to \text{GlobalMeanPool} \to \text{Linear}(32 \to 2)$
- **GAT**: $\text{GATConv}(190 \to 64) \to \text{GATConv}(64 \to 32)$ (4 attention heads) $\to \text{GlobalMeanPool} \to \text{Linear}(32 \to 2)$
- **GraphSAGE**: $\text{SAGEConv}(190 \to 64) \to \text{SAGEConv}(64 \to 32) \to \text{GlobalMeanPool} \to \text{Linear}(32 \to 2)$
- **GIN**: $\text{GINConv}(190 \to 64) \to \text{GINConv}(64 \to 32)$ (MLP backbones) $\to \text{GlobalAddPool} \to \text{Linear}(32 \to 2)$

---

## 5. Audited Quantitative Results (7-Fold LOSO)
Traced directly to [`results/exp09/w2b_loso_results.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/w2b_loso_results.csv):

### 5.1 Architecture Summary (Mean across 7 Unseen Test Sites)
| Architecture | Mean Test AUC | Std AUC | Mean Balanced Accuracy | Mean F1 Score | Best Val AUC (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GAT** | **$0.5752$** | $0.0489$ | **$0.5463$** | $0.4827$ | $0.6538$ |
| **GraphSAGE** | **$0.5502$** | $0.0449$ | $0.5147$ | $0.3323$ | $0.6483$ |
| **GCN** | **$0.5468$** | $0.0691$ | $0.5418$ | $0.4802$ | $0.6568$ |
| **GIN** | **$0.5437$** | $0.0617$ | $0.5313$ | $0.4907$ | $0.6358$ |

### 5.2 Per-Site Test AUC Breakdown
| Held-Out Test Site | $N_{\text{Test}}$ | GCN AUC | GAT AUC | SAGE AUC | GIN AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KKI** | 83 | $0.6021$ | $0.6282$ | $0.5238$ | $0.5119$ |
| **NYU** | 93 | $0.4653$ | $0.5670$ | $0.5427$ | $0.5256$ |
| **NeuroIMAGE** | 48 | $0.5374$ | $0.5548$ | $0.5304$ | $0.4591$ |
| **OHSU** | 79 | $0.4929$ | $0.5142$ | $0.5373$ | $0.5553$ |
| **Peking_1** | 85 | $0.5478$ | $0.5656$ | $0.5225$ | $0.5799$ |
| **Peking_2** | 67 | $0.5027$ | $0.5491$ | $0.5446$ | $0.5152$ |
| **Peking_3** | 42 | $0.6796$ | $0.6476$ | $0.6499$ | $0.6590$ |

---

## 6. Critical Scientific Finding: The Multi-Site Generalization Failure
- **The Generalization Collapse**: While models achieve internal validation AUCs of $0.63 - 0.73$ during training on 6 sites, performance collapses to near-chance levels ($0.54 - 0.58$ AUC) when deployed on an unseen site.
- **Site Heterogeneity**: NYU and OHSU prove especially recalcitrant (GCN AUCs of $0.4653$ and $0.4929$), whereas Peking_3 achieves higher performance ($0.64 - 0.68$).
- **Takeaway**: Standard GNN architectures do not learn scanner-invariant representations without explicit domain adaptation or robust multi-site calibration.

---

## 7. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp09/11_population_graph_learning.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp09/11_population_graph_learning.ipynb)
- **Environment Metadata**: [`configs/exp09/w2b_environment.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/configs/exp09/w2b_environment.json)
- **Result Artifacts**:
  - [`results/exp09/w2b_loso_results.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/w2b_loso_results.csv) (complete 7-fold metric table)
  - [`results/exp09/exp09_verified_results.json`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/exp09_verified_results.json)
  - [`results/exp09/graph_preprocessing_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/graph_preprocessing_summary.csv)
  - [`results/exp09/architecture_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/architecture_summary.csv)
  - [`results/exp09/training_curves.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp09/training_curves.csv)
