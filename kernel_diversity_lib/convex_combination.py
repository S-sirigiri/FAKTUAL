import numpy as np

class ConvexMatrixCombiner:
    """
    Compute convex combinations of N x N NumPy matrices.

    Parameters
    ----------
    weights : array-like of shape (K,)
        Non-negative coefficients that (approximately) sum to 1.
    rtol, atol : float
        Tolerances used for checking the sum-to-1 constraint.
    renormalize : bool
        If True and the weights don't sum exactly to 1, they will be
        normalized to sum to 1 (as long as their sum is positive).

    Usage
    -----
    >>> import numpy as np
    >>> A = np.eye(3)
    >>> B = np.ones((3, 3))
    >>> comb = ConvexMatrixCombiner([0.25, 0.75])
    >>> C = comb.combine(A, B)           # or comb.combine([A, B])
    """

    def __init__(self, weights, *, rtol=1e-10, atol=1e-12, renormalize=False):
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("`weights` must be a 1D array with at least one element.")
        if not np.all(np.isfinite(w)):
            raise ValueError("All coefficients must be finite.")
        if np.any(w < 0):
            raise ValueError("All coefficients must be non-negative for a convex combination.")
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Weights must have a finite, positive sum.")
        if not np.isclose(s, 1.0, rtol=rtol, atol=atol):
            if renormalize:
                if s <= 0:
                    raise ValueError("Weights sum is non-positive; cannot renormalize.")
                w = w / s
            else:
                raise ValueError(f"Coefficients must sum to 1 (got {s}). "
                                 "Pass renormalize=True to normalize automatically.")
        # Enforce the simplex exactly, including sums accepted within tolerance.
        self.weights = w.copy() / w.sum()

    def combine(self, *matrices):
        """
        Combine matrices using the stored convex weights.

        Accepts either:
          - K separate N x N arrays: combine(A, B, C, ...)
          - A single array/list/tuple of shape (K, N, N): combine([A, B, C]) or combine(stack)

        Returns
        -------
        np.ndarray of shape (N, N)
        """
        # Normalize inputs to a single stacked array of shape (K, N, N)
        if not matrices:
            raise ValueError("At least one matrix is required.")
        if len(matrices) == 1:
            m = matrices[0]
            if isinstance(m, (list, tuple)):
                if not m:
                    raise ValueError("At least one matrix is required.")
                arr = np.stack([np.asarray(x, dtype=float) for x in m], axis=0)
            elif isinstance(m, np.ndarray) and m.ndim == 3:
                arr = m.astype(float, copy=False)
            elif isinstance(m, np.ndarray) and m.ndim == 2:
                arr = m.astype(float, copy=False)[None, ...]
            else:
                raise TypeError("Pass either K NxN arrays, a list/tuple of them, "
                                "or a single array of shape (K, N, N).")
        else:
            arr = np.stack([np.asarray(x, dtype=float) for x in matrices], axis=0)

        if arr.ndim != 3:
            raise ValueError("Input must have shape (K, N, N).")
        K, N1, N2 = arr.shape
        if K != self.weights.size:
            raise ValueError(f"Expected {self.weights.size} matrices (one per weight), got {K}.")
        if N1 != N2:
            raise ValueError(f"Matrices must be square (got {N1}x{N2}).")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Matrices must contain only finite values.")

        # Weighted sum: sum_i w_i * M_i
        # tensordot over the first axis gives an (N, N) result efficiently
        return np.tensordot(self.weights, arr, axes=(0, 0))

    __call__ = combine
