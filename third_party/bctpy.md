# Brain Connectivity Toolbox (bctpy) Provenance and Licensing

This document details the provenance, upstream source, attribution, and licensing terms for Brain Connectivity Toolbox (BCT) code incorporated in this repository.

---

## 1. Upstream Project Identification

- **Upstream Project**: Brain Connectivity Toolbox for Python (`bctpy`)
- **Upstream URL**: https://github.com/aestrivex/bctpy
- **Upstream License**: GNU General Public License v3.0 or later (GPL-3.0-or-later)

---

## 2. Affected Repository Paths

The BCT-derived source code bundled in this repository consists of:

1. `src/exp02/null_model_und_sign_fixed.py`
2. `src/exp02/randmio_und_signed_fast.py`

---

## 3. Historical Provenance and Attribution

The historical implementation contains BCT-derived graph-randomization code. `null_model_und_sign_fixed.py` imports utilities and the `randmio_und_signed` routine from the BCTPY package, while `randmio_und_signed_fast.py` is a Numba-accelerated reimplementation of the BCTPY `randmio_und_signed` routine. The relevant historical source is therefore treated as BCT-derived/adapted code rather than as an independently authored graph-randomization algorithm.

Upstream project: aestrivex/bctpy (https://github.com/aestrivex/bctpy). The upstream BCTPY repository is GPL-3.0. The exact BCTPY version and upstream commit used during the historical experiment were not recorded in the available provenance evidence.

The historical `null_model_und_sign_fixed.py` file carries a Rubinov 2011 attribution for the undirected signed null model.

### Component Details

- **`src/exp02/null_model_und_sign_fixed.py`**:
  - Implements `null_model_und_sign_fixed(W, bin_swaps=5, wei_freq=.1, seed=None)`
  - Imports `BCTParamError` and `get_rng` from `bct.utils.miscellaneous_utilities`, and `randmio_und_signed` from `bct.algorithms.reference`
  - Explicitly carries Rubinov attribution: `# @due.dcite(BibTeX(RUBINOV2011), description="Undirected signed null model")`
  - Adapted/derived from `bctpy` for weight-conserving signed matrix randomization with fixed positive and negative strength sequences.

- **`src/exp02/randmio_und_signed_fast.py`**:
  - Implements `randmio_und_signed_fast(R, itr, seed=None)`
  - Numba-accelerated reimplementation of the bctpy randmio_und_signed routine used by the historical project.
  - Explicitly documented as a "Drop-in, Numba-accelerated replacement for bctpy's randmio_und_signed()."
  - Preserves the identical algorithm, rewiring criterion, node-sampling strategy, swap logic, attempt counting, and effective-swap counting as the original `bctpy` routine.

---

## 4. Scientific Citation (Rubinov 2011)

The historical `RUBINOV2011` citation corresponds to:

> Mikail Rubinov and Olaf Sporns.  
> "Weight-conserving characterization of complex functional brain networks."  
> *NeuroImage*, 2011; 56(4): 2068–2079.  
> DOI: [10.1016/j.neuroimage.2011.03.069](https://doi.org/10.1016/j.neuroimage.2011.03.069)  
> PubMed: [21459148](https://pubmed.ncbi.nlm.nih.gov/21459148/)

---

## 5. Historical Execution Status

- **Exact Historical Version Status**: not recorded during original execution
- **Exact Historical Commit Status**: not recorded during original execution

*(Note: The historical records in the original execution environment did not record the specific version or git commit tag used during historical execution.)*

---

## 6. Licensing Notice

The bundled BCT-derived source files (`src/exp02/null_model_und_sign_fixed.py` and `src/exp02/randmio_und_signed_fast.py`) are derived from `bctpy` and are governed by the GNU General Public License v3.0 or later (GPL-3.0-or-later). The full license text is provided in `LICENSES/GPL-3.0-or-later.txt`. They must not be described as MIT-only code. Any distribution or modification of these files must preserve upstream GPL-3.0-or-later notices and Rubinov & Sporns (2011) attributions.
