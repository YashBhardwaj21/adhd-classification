import numpy as np
from numba import njit


@njit(cache=True)
def _randmio_und_signed_kernel(R, itr, max_attempts, n, n2, n3, n4, seed_val):
    """
    Compiled hot loop. Algorithm, rewiring criterion, node-sampling strategy,
    swap logic, attempt counting, and effective-swap counting are identical
    to the original bctpy randmio_und_signed().

    seed_val: int, -1 means "no seed" (i.e. seed=None in the wrapper).

    NOTE on RNG: Numba maintains its OWN internal random state that is
    separate from NumPy's global state. Calling np.random.seed() in plain
    Python before invoking a jitted function has NO effect on the numbers
    produced inside that jitted function. To get reproducible, seed-correct
    behavior, np.random.seed() must be called from *inside* the compiled
    function, which is what we do here.
    """
    if seed_val >= 0:
        np.random.seed(seed_val)

    eff = 0
    for _ in range(itr):
        att = 0
        while att <= max_attempts:
            # Inlined pick_four_unique_nodes_quickly(), no recursion,
            # same rejection-sampling strategy as the original.
            while True:
                k = np.random.randint(n4)
                a = k % n
                b = (k // n) % n
                c = (k // n2) % n
                d = (k // n3) % n
                if (
                    a != b and
                    a != c and
                    a != d and
                    b != c and
                    b != d and
                    c != d
                ):
                    break

            # Cache row references to avoid repeated 2D indexing
            Ra = R[a]
            Rc = R[c]
            r0_ab = Ra[b]
            r0_ad = Ra[d]
            r0_cb = Rc[b]
            r0_cd = Rc[d]

            # Boolean sign comparisons instead of np.sign()
            sab = r0_ab > 0
            sad = r0_ad > 0
            scb = r0_cb > 0
            scd = r0_cd > 0

            if (
                sab == scd and
                sad == scb and
                sab != sad
            ):
                Ra[d] = r0_ab
                R[d, a] = r0_ab
                Ra[b] = r0_ad
                R[b, a] = r0_ad
                Rc[b] = r0_cd
                R[b, c] = r0_cd
                Rc[d] = r0_cb
                R[d, c] = r0_cb
                eff += 1
                break
            att += 1

    return R, eff


def randmio_und_signed_fast(R, itr, seed=None):
    """
    Drop-in, Numba-accelerated replacement for bctpy's randmio_und_signed().

    Signature, inputs, and outputs are unchanged:
        randmio_und_signed_fast(R, itr, seed=None) -> (R, eff)

    Only the computational kernel is JIT-compiled; this thin Python wrapper
    handles input prep (copy/dtype/shape) and constant precomputation, exactly
    mirroring the original implementation's setup code.
    """
    R = np.ascontiguousarray(R.copy())
    n = R.shape[0]

    itr = int(itr * (n * (n - 1) // 2))
    max_attempts = n // 2

    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n

    seed_val = -1 if seed is None else int(seed)

    R, eff = _randmio_und_signed_kernel(R, itr, max_attempts, n, n2, n3, n4, seed_val)
    return R, eff
