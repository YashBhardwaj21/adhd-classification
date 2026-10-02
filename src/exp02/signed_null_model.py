# SPDX-License-Identifier: GPL-3.0-or-later
# BCT-derived/adapted implementation.
# Upstream: Brain Connectivity Toolbox for Python (bctpy).
# Modified/adapted from upstream BCT/bctpy implementation; see git history for dates.
# See third_party/bctpy.md for provenance and licensing details.
# @due.dcite(BibTeX(RUBINOV2011), description="Undirected signed null model")
import numpy as np
from bct.utils.miscellaneous_utilities import (
    BCTParamError,
    get_rng,
)


def null_model_und_sign_fixed(W, bin_swaps=5, wei_freq=.1, seed=None):
    rng = get_rng(seed)
    if not np.allclose(W, W.T):
        raise BCTParamError("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat
    An = (W < 0)  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        Ap_r = Ap.copy()
        An_r = An.copy()
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))

    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
            # Positive strengths
            S = np.sum(W * Acur, axis=0)
            # Positive weights
            Wv = np.sort(W[np.where(np.triu(Acur))])
        else:
            Acur = An
            A_rcur = An_r
            # Negative strengths (MATLAB: sum(-W.*An,2))
            S = np.sum((-W) * Acur, axis=0)
            # Negative weight magnitudes (MATLAB: sort(-W(triu(An))))
            Wv = np.sort((-W)[np.where(np.triu(Acur))])

        i, j = np.where(np.triu(A_rcur))
        Lij, = np.where(np.triu(A_rcur).flat)

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq).astype(int)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                R = rng.permutation(m)[:np.min((m, wei_period))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o]

                    if S[i[o]] > 1e-12:
                        f = 1.0 - Wv[r] / S[i[o]]
                        P[i[o], :] *= f
                        P[:, i[o]] *= f

                    # readjust expected weighted probability for j[o]

                    if S[j[o]] > 1e-12:
                        f = 1.0 - Wv[r] / S[j[o]]
                        P[j[o], :] *= f
                        P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])
