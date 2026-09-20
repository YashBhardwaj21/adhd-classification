# Experiment 5: Dynamic Brain State Clustering & Temporal Transition Dynamics

## 1. Scientific Overview & Objectives
Experiment 5 moves beyond static topological averaging to model brain network reconfigurations as discrete recurring micro-states. Brain functional architecture undergoes dynamic switching between states of high modular segregation and states of integrated global efficiency.

The primary objectives are:
1. Identify canonical dynamic connectivity states across the 31,060 temporal windows using unsupervised clustering.
2. Characterize temporal state dynamics, including fractional occupancy, mean dwell times, and Markovian transition probabilities.
3. Extract subject-level dynamic biomarkers (state switching rate, transition entropy) for clinical correlation.

---

## 2. Mathematical Methodology

### 2.1 State Decomposition
For each temporal window $t$, a feature vector $\mathbf{v}^{(t)} = [C^{(t)}, T^{(t)}, E_{\text{glob}}^{(t)}, L^{(t)}, \gamma^{(t)}, \lambda^{(t)}, \sigma^{(t)}]^T$ is standardized and clustered via $K$-Means clustering ($K=3$, determined by elbow and silhouette criteria):
$$\arg\min_{\mathbf{S}} \sum_{k=1}^K \sum_{\mathbf{v}^{(t)} \in S_k} \|\mathbf{v}^{(t)} - \boldsymbol{\mu}_k\|_2^2$$
where $\boldsymbol{\mu}_k$ represents the centroid of state $k \in \{1, 2, 3\}$.

### 2.2 Temporal Biomarkers
From the discrete state sequence $\mathbf{s} = [s_1, s_2, \dots, s_T]$ ($s_t \in \{1, 2, 3\}$) for each scan run:
1. **Fractional Occupancy ($FO_k$)**: Proportion of total scan time spent in state $k$:
   $$FO_k = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(s_t = k)$$
2. **Mean Dwell Time ($MDT_k$)**: Average consecutive window duration spent in state $k$ before transitioning:
   $$MDT_k = \frac{\sum \text{consecutive run lengths in state } k}{\text{number of occurrences of state } k}$$
3. **State Switching Rate ($SR$)**: Frequency of state transitions between consecutive windows:
   $$SR = \frac{1}{T - 1} \sum_{t=1}^{T-1} \mathbb{I}(s_{t+1} \neq s_t)$$
4. **Transition Entropy ($H$)**: Shannon entropy of empirical Markov transition probabilities $P_{ij}$:
   $$H = -\sum_{i=1}^K \sum_{j=1}^K P_{ij} \log(P_{ij} + \epsilon)$$

---

## 3. Audited Empirical State Profiles & Dynamics
Traced directly to [`results/exp05/run_dynamic_biomarkers.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_dynamic_biomarkers.csv) and [`results/exp05/subject_feature_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_feature_summary.csv):

| Dynamic State | Dominant Topological Characteristics | Fractional Occupancy ($FO$) | Mean Dwell Time ($MDT$) |
| :--- | :--- | :--- | :--- |
| **State 1** | **Segregated / High Efficiency**: Elevated local clustering, compact shortest paths | **$49.05\%$** | **$6.24$ windows** (~$62.4\text{ s}$) |
| **State 2** | **Intermediate / Balanced**: Moderate clustering, standard baseline topology | **$27.32\%$** | **$3.44$ windows** (~$34.4\text{ s}$) |
| **State 3** | **Integrated / Long Path Length**: Lower clustering, extended global path length | **$23.63\%$** | **$2.83$ windows** (~$28.3\text{ s}$) |

### Subject-Level Transition Dynamics ($N = 764$ subjects)
- **Mean State Switching Rate**: $0.2048 \pm 0.0891$ transitions per window.
- **Mean Transition Entropy**: $0.6925 \pm 0.2241$ nats.
- **Dwell Time Asymmetry**: State 1 exhibits significantly longer temporal persistence ($6.24$ windows) compared to States 2 ($3.44$) and 3 ($2.83$), indicating that the segregated topology serves as the primary resting-state attractor.

---

## 4. Implementation Details & Artifact Schema
The state analysis produces three hierarchical levels of results:
1. **Window-Level**: Run state sequences in `run_state_sequences.csv`.
2. **Run-Level**: Transition matrices and biomarkers in `run_transition_dynamics.csv` and `run_dynamic_biomarkers.csv`.
3. **Subject-Level**: Pooled clinical feature dataset in `subject_dynamic_dataset.csv`.

---

## 5. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp05/05_dynamic_state_modeling.ipynb`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/notebooks/exp05/05_dynamic_state_modeling.ipynb)
- **Result Artifacts**:
  - [`results/exp05/run_dynamic_biomarkers.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_dynamic_biomarkers.csv)
  - [`results/exp05/run_state_sequences.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_state_sequences.csv)
  - [`results/exp05/run_transition_dynamics.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/run_transition_dynamics.csv)
  - [`results/exp05/subject_dynamic_dataset.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_dynamic_dataset.csv)
  - [`results/exp05/subject_feature_summary.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_feature_summary.csv)
  - [`results/exp05/subject_feature_correlation.csv`](file:///c:/Users/Yash%20Bhardwaj/Downloads/ADHD200_GitHub_Staging/results/exp05/subject_feature_correlation.csv)
