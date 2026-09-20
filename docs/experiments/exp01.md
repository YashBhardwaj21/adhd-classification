# Experiment 1: Dynamic Functional Connectivity Generation & Temporal Stability Validation

## 1. Scientific Overview & Objectives
Experiment 1 establishes the foundational functional connectivity (FC) pipeline for the Track A analysis cohort. Resting-state functional magnetic resonance imaging (rs-fMRI) blood-oxygen-level-dependent (BOLD) time series are mapped into windowed correlation matrices to capture time-varying functional interactions between distributed cortical and subcortical regions.

The primary objective is to generate dynamic functional connectivity (dFC) matrices across the cohort and evaluate the physical and statistical stability of consecutive temporal windows using lag-correlation and Frobenius distance metrics.

---

## 2. Cohort & Preprocessing Specifications
- **Atlas / Parcellation**: Craddock-200 (CC200) parcellation, containing 190 active regions of interest (ROIs) extracted from the Athena preprocessing pipeline.
- **Subject Cohort**: 764 total subjects passing quality control:
  - 456 Typically Developing Controls (TDC)
  - 308 Attention-Deficit/Hyperactivity Disorder (ADHD)
  - 8 acquisition sites: Brown University, KKI, NeuroIMAGE, NYU, OHSU, Peking University, Pittsburgh, WashU.
- **Sliding Window Parameters**:
  - Window length ($W$): 30 TRs (repetition times), corresponding to 60.0 seconds at $\text{TR} = 2.0\text{s}$.
  - Step size ($S$): 5 TRs (10.0 seconds), yielding an 83.3% temporal overlap between consecutive windows.
  - Window function: Rectangular sliding window.
  - Total Dynamic Windows Generated: **31,060** windowed FC matrices across 764 subjects.

---

## 3. Mathematical Methodology

### 3.1 Dynamic Window Correlation
For a subject with $N = 190$ regional time series $\mathbf{X} \in \mathbb{R}^{T \times N}$, the dynamic correlation matrix $\mathbf{C}^{(t)} \in \mathbb{R}^{N \times N}$ at window $t$ spanning time points $[t \cdot S, t \cdot S + W - 1]$ is computed as:
$$\mathbf{C}_{i,j}^{(t)} = \frac{\sum_{\tau=1}^W (X_{\tau, i} - \bar{X}_i)(X_{\tau, j} - \bar{X}_j)}{\sqrt{\sum_{\tau=1}^W (X_{\tau, i} - \bar{X}_i)^2 \sum_{\tau=1}^W (X_{\tau, j} - \bar{X}_j)^2}}$$

### 3.2 Temporal Stability Metrics
To measure the smooth physical evolution of the BOLD signal across sliding windows, two similarity metrics are computed between window $t$ and lagged window $t + k$ ($k \in \{1, 2, 3, 4\}$):
1. **Upper-Triangular Pearson Correlation ($r_{\text{lag-}k}$)**:
   $$r_{\text{lag-}k} = \text{Corr}\left(\text{vech}(\mathbf{C}^{(t)}), \text{vech}(\mathbf{C}^{(t+k)})\right)$$
2. **Frobenius Distance ($D_{\text{Frob}}$)**:
   $$D_{\text{Frob}}(\mathbf{C}^{(t)}, \mathbf{C}^{(t+k)}) = \|\mathbf{C}^{(t)} - \mathbf{C}^{(t+k)}\|_{\text{F}} = \sqrt{\sum_{i=1}^N \sum_{j=1}^N \left(\mathbf{C}_{i,j}^{(t)} - \mathbf{C}_{i,j}^{(t+k)}\right)^2}$$

---

## 4. Audited & Verified Quantitative Results
Values are traced directly to `results/exp01/dynamic_temporal_validation.csv` and verified execution output of Cell 53 in `notebooks/exp01/01_fc_generation_and_validation.ipynb`:

| Lag Distance | Time Delta ($\Delta t$) | Correlation Similarity ($\mu \pm \sigma$) | Frobenius Distance ($\mu \pm \sigma$) |
| :--- | :--- | :--- | :--- |
| **Lag 1** | $10\text{ s}$ ($5\text{ TR}$) | $0.899042 \pm 0.016444$ | $32.578362 \pm 3.449146$ |
| **Lag 2** | $20\text{ s}$ ($10\text{ TR}$) | $0.778905 \pm 0.032205$ | $48.749290 \pm 4.721958$ |
| **Lag 3** | $30\text{ s}$ ($15\text{ TR}$) | $0.660890 \pm 0.045131$ | $60.780884 \pm 5.382166$ |
| **Lag 4** | $40\text{ s}$ ($20\text{ TR}$) | $0.546714 \pm 0.057422$ | $70.605774 \pm 5.968088$ |

### Mathematical Consistency Check
Because consecutive windows share 25 out of 30 time points (83.3% overlap), the theoretical upper bound on correlation decay follows a linear temporal auto-correlation trajectory. The observed monotonic decrease from $0.8990 \to 0.7789 \to 0.6609 \to 0.5467$ precisely mirrors this expectation, validating that no temporal shuffling or indexing misalignment occurred during window generation.

---

## 5. Implementation Caveats & Protocol Discrepancies
- **Diagonal Matrix Treatment**: In static FC generation, diagonal entries were explicitly zeroed, and correlation values underwent Fisher's $z$-transformation ($z = \text{arctanh}(r)$). In dynamic FC generation, the diagonal was retained as $1.0$ and unzeroed. Downstream graph construction modules must zero the diagonal prior to edge extraction to avoid self-loop artifacts.
- **Data Exclusion**: Due to GitHub file-size limits (individual files $>100\text{ MiB}$ blocked), the raw $143.79\text{ MB}$ parquet file containing all raw time series is excluded from the Git staging tree; the summary manifests and validation tables are preserved.

---

## 6. Source Code & Result Provenance
- **Canonical Notebook**: [`notebooks/exp01/01_fc_generation_and_validation.ipynb`](../../notebooks/exp01/01_fc_generation_and_validation.ipynb)
- **Primary Result Artifacts**:
  - [`results/exp01/dynamic_temporal_validation.csv`](../../results/exp01/dynamic_temporal_validation.csv) (lag validation table)
  - [`results/exp01/static_vs_dynamic_validation.csv`](../../results/exp01/static_vs_dynamic_validation.csv) (static vs mean dynamic FC comparison)
  - [`results/exp01/dynamic_manifest.csv`](../../results/exp01/dynamic_manifest.csv) (window indexing manifest)
