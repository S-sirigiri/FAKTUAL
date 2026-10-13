import cupy as cp

class KernelUtilities:
    """
    A collection of utility functions for kernel-based operations using CuPy.
    """

    def __init__(self):
        # No initialization is needed for static utility methods.
        pass

    @staticmethod
    def compute_bandwidth(X):
        """
        Compute a heuristic bandwidth for kernel operations based on the data X.

        The bandwidth is calculated as the median of the values in X divided by
        the natural logarithm of the number of samples, with a small constant
        added for numerical stability.

        Args:
            X (cp.ndarray): A 3D array of sample values for which to compute the bandwidth.

        Returns:
            float: The computed bandwidth value.
        """
        X = cp.asarray(X)
        return (
                cp.median(X) / cp.log(X.shape[0]) + 1e-3
        )

    @staticmethod
    def __compute_bandwidth(X, *, divide_by_log_n=True, eps=1e-8):
        """
        Compute a heuristic kernel bandwidth from trajectories using pairwise distances.

        The bandwidth is based on the median of the off-diagonal pairwise *squared*
        Euclidean distances between trajectories. If `divide_by_log_n` is True, the
        median is divided by log(n) (a common variant used in practice). A small
        epsilon is added for numerical stability.

        Args:
            X (cp.ndarray): Array of shape (n, T, d) or (n, D) containing n trajectories.
            divide_by_log_n (bool): Whether to divide by log(n). Default: True.
            eps (float): Small constant added for numerical stability. Default: 1e-8.

        Returns:
            float: Bandwidth (sigma^2) suitable for an RBF kernel k(x,y)=exp(-||x-y||^2 / (2*sigma^2)).

        Notes:
            - If you prefer the scikit-learn-style gamma (1 / (2 * median_sqdist)), use:
                  gamma = 1.0 / (2.0 * (bandwidth))
              where `bandwidth` is the value returned by this function with divide_by_log_n=False.
        """
        X = cp.asarray(X)
        n = X.shape[0]
        if n < 2:
            return float(eps)

        # Flatten each trajectory to a single vector
        Xf = X.reshape(n, -1)

        # Pairwise squared distances via ||x||^2 + ||y||^2 - 2 x·y
        # This forms an n x n matrix; for very large n, consider batching or subsampling.
        norms = cp.sum(Xf ** 2, axis=1, keepdims=True)  # (n, 1)
        gram = Xf @ Xf.T  # (n, n)
        sq_dists = norms + norms.T - 2.0 * gram  # (n, n)
        sq_dists = cp.maximum(sq_dists, 0.0)  # clamp tiny negatives

        # Take the upper-triangular (k=1 excludes diagonal) to get unique off-diagonal pairs
        iu = cp.triu_indices(n, k=1)
        vals = sq_dists[iu]
        if vals.size == 0:
            return float(eps)

        med = cp.median(vals)

        if divide_by_log_n:
            # Use log(n+1) to avoid log(1)=0 edge cases for tiny n
            med = med / cp.log(cp.asarray(n + 1.0, dtype=vals.dtype))

        # Add a small epsilon for stability and return as a Python float
        return float((med + eps).item())