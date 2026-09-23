# Third-Party Dependencies and Attributions

This document records third-party packages, algorithms, and bundled code utilized within this repository.

---

## 1. NeuroSTORM Spatio-Temporal Transformer
- **Included File**: `src/exp08/neurostorm/neurostorm.py`
- **Upstream Project**: NeuroSTORM (CUHK-AIM-Group)
- **Source**: https://github.com/CUHK-AIM-Group/NeuroSTORM
- **License**: Apache License, Version 2.0 (Apache-2.0; see `LICENSES/Apache-2.0.txt`)
- **Implementation Status**: Adapted from upstream implementation (incorporates architectural blocks derived from MONAI SwinUNETR and SwiFT) with dataloading interfaces for 4D functional fMRI sequences.
- **Publication Reference**: Wang, C., Jiang, Y., Peng, Z., et al. (2026). "Towards a general-purpose foundation model for functional MRI analysis." *Nature Biomedical Engineering*. DOI: 10.1038/s41551-026-01666-y.

---

## 2. Brain Connectivity Toolbox (BCT) Null Models
- **Included Files**:
  - `src/exp02/null_model_und_sign_fixed.py`
  - `src/exp02/randmio_und_signed_fast.py`
- **Upstream Project**: `bctpy` (Brain Connectivity Toolbox for Python)
- **Source**: https://github.com/aestrivex/bctpy
- **License**: GNU General Public License v3.0 or later (GPL-3.0-or-later; see `LICENSES/GPL-3.0-or-later.txt`)
- **Detailed Provenance**: See [third_party/bctpy.md](bctpy.md) for complete details.
- **Implementation Status**: The historical implementation contains BCT-derived graph-randomization code. In the historical source, `null_model_und_sign_fixed.py` depended on BCTPY utilities and the `randmio_und_signed` routine; the current repository version retains the derived null-model implementation and BCT utility imports. `randmio_und_signed_fast.py` is a Numba-accelerated reimplementation of the BCTPY `randmio_und_signed` routine. The relevant historical source is therefore treated as BCT-derived/adapted code rather than as an independently authored graph-randomization algorithm.
- **Upstream Facts & Historical Status**: Upstream project: aestrivex/bctpy (https://github.com/aestrivex/bctpy). The upstream BCTPY repository is GPL-3.0. The exact BCTPY version and upstream commit used during the historical experiment were not recorded in the available provenance evidence.
- **Publication Reference**: Mikail Rubinov and Olaf Sporns. "Weight-conserving characterization of complex functional brain networks." *NeuroImage*, 2011; 56(4): 2068–2079. DOI: [10.1016/j.neuroimage.2011.03.069](https://doi.org/10.1016/j.neuroimage.2011.03.069).
- **Licensing Note**: The bundled BCT-derived source files are governed by GNU General Public License v3.0 or later (GPL-3.0-or-later) as derived works of `bctpy` and must not be described as MIT-only.

---

## 3. neuroCombat (External Dependency)
- **Component**: ComBat harmonization protocol (Experiment 04)
- **Upstream Project**: `neuroCombat` (Python package by Jean-Philippe Fortin)
- **Source**: https://github.com/Jfortin1/neuroCombat
- **License**: MIT License
- **Status in Repository**: External package dependency installed via pip (`pip install neuroCombat`); no source code bundled in-tree.
- **Publication Reference**: Fortin, J. P., et al. (2018). "Harmonization of cortical thickness measurements across scanners and sites." *NeuroImage*, 167, 104-120.
