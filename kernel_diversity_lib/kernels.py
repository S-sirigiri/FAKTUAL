import numpy as np
import cupy as cp
from collections.abc import Mapping

if __package__:
    from .convex_combination import ConvexMatrixCombiner
else:
    from convex_combination import ConvexMatrixCombiner
#cp.cuda.runtime.setDevice(1)

import ksig
from ksig.kernels import RandomWarpingSeries

import torch
import sigkernel


class RandomFourierSignatureFeaturesKernel:
    """
    A kernel wrapper that uses the Random Fourier Signature Features (RFSF)
    provided by the KSig package (see https://github.com/tgcsaba/KSig)
    to approximate the signature kernel as in [Tóth, Oberhauser, Szabó, 2023].

    This class implements the API:
      K = kernel(X, Y, diag=...)
    where X and Y are arrays of shape (n_seq, T, d) (or a single path of shape (T, d)),
    and returns either the full kernel matrix or only the diagonal (if diag=True).

    Parameters:
      n_levels : int
          The truncation (signature) level, corresponding to the number of signature levels.
      n_components : int
          Number of random Fourier features used for the static feature map.
      method : str, optional (default 'TRP')
          Which projection to use for dimensionality reduction.
          Supported values:
            - 'TRP': Use Tensorized Random Projection.
            - 'DP': Use Diagonal Projection.
    """

    def __init__(self, bandwidth=1., n_levels=5, n_components=100, method='TRP'):
        self.n_levels = n_levels
        self.n_components = n_components
        self.method = method.upper()

        # Instantiate the static feature map using Random Fourier Features.
        self.static_feat = ksig.static.features.RandomFourierFeatures(n_components=n_components,
                                                                      bandwidth=bandwidth,
                                                                      random_state=0
                                                                      )

        # Choose the projection type.
        if self.method == 'TRP':
            self.projection = ksig.projections.TensorizedRandomProjection(n_components=n_components, random_state=0)
        elif self.method == 'DP':
            self.projection = ksig.projections.DiagonalProjection()
        elif self.method == 'TS':
            self.projection = ksig.projections.TensorSketch(n_components=n_components)
        else:
            raise ValueError("Unsupported method. Use 'TRP' or 'DP' or 'TS'.")

        # Create the signature feature map (RFSF kernel) from KSig.
        # Note: The SignatureFeatures class computes a feature map whose inner product
        # approximates the signature kernel.
        self.rfsf = ksig.kernels.SignatureFeatures(
            n_levels=n_levels,
            static_features=self.static_feat,
            projection=self.projection
        )

    def fit(self, X):
        """
        Optionally fit the RFSF feature map to data X.
        This might be needed if the feature map requires data-dependent setup.

        Parameters:
          X : array-like, shape (n_seq, T, d)
        Returns:
          self
        """
        self.rfsf.fit(X)
        return self

    def transform(self, X):
        """
        Compute the low-dimensional features for X.

        Parameters:
          X : array-like, shape (n_seq, T, d)
        Returns:
          P : ndarray of shape (n_seq, F)
              The feature representation of X.
        """
        return self.rfsf.transform(X)

    def __call__(self, X, Y=None, diag=False):
        """
        Evaluate the kernel between sequences X and Y.

        Parameters:
          X : array-like, shape (n_seq_X, T, d) or (T, d)
          Y : array-like, shape (n_seq_Y, T, d) or (T, d), optional.
              If None, then Y is taken as X.
          diag : bool, optional (default False)
              If True, compute only the diagonal entries (i.e. for corresponding pairs).

        Returns:
          K : ndarray
              The computed kernel matrix (or vector if diag=True).
        """
        return self.rfsf(X, Y, diag=diag)




class SignatureKernelTorch:
    """
    A kernel wrapper for the signature kernel that computes a similarity between sequences
    using the provided sigkernel.SigKernel instance.

    The kernel is defined as:
        k(x, y) = signature_kernel.compute_Gram(x, y, max_batch)
    and is normalized so that k(x,x)=1 for every sequence x.

    Parameters:
      sig_kernel : sigkernel.SigKernel
          A pre-initialized signature kernel instance (e.g. created via
          sigkernel.SigKernel(static_kernel, dyadic_order)).
      max_batch : int, optional (default 100)
          Maximum batch size for computation.
      device : torch.device or str, optional
          The device on which computations are performed (default: 'cuda' if available).

    Methods:
      __call__(X, Y=None):
          Computes the normalized kernel matrix between X and Y.
      transform(X):
          Computes the normalized Gram matrix for X.
    """

    def __init__(self, sig_kernel=None, bandwidth=None, max_batch=None, dyadic_order=None, device=None):
        if bandwidth is None:
            self.bandwidth = 1.0
        else:
            self.bandwidth = bandwidth

        if dyadic_order is None:
            self.dyadic_order = 1
        else:
            self.dyadic_order = dyadic_order

        if sig_kernel is None:
            self.sig_kernel = sigkernel.SigKernel(static_kernel=sigkernel.RBFKernel(sigma=self.bandwidth), dyadic_order=self.dyadic_order)
        else:
            self.sig_kernel = sig_kernel

        if max_batch is None:
            self.max_batch = 10
        else:
            self.max_batch = max_batch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device

    def _convert(self, X):
        """
        Convert a cupy array to a torch tensor by first converting to a numpy array.
        Use float64 on CPU (required by the Cython solver), float32 on CUDA.
        """
        if isinstance(X, cp.ndarray):
            X_np = cp.asnumpy(X)
            if self.device == torch.device('cpu'):
                return torch.from_numpy(X_np).double().to(self.device)
            return torch.from_numpy(X_np).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            dtype = torch.float64 if self.device.type == 'cpu' else torch.float32
            return X.to(device=self.device, dtype=dtype)
        else:
            raise ValueError("Input must be a cupy array or a torch tensor.")

    def __call__(self, X, Y=None):
        """
        Compute the normalized kernel matrix between X and Y using the signature kernel.

        Normalization is performed using compute_kernel for the diagonal.

        Parameters:
          X : cupy.ndarray
              First set of sequences.
          Y : cupy.ndarray or None, optional
              Second set of sequences (if None, computes the Gram matrix for X).

        Returns:
          cupy.ndarray: The normalized kernel matrix with k(x,x)=1 for every sequence x.
        """
        # Convert inputs to torch tensors on the specified device.
        self.sig_kernel.static_kernel.sigma = float(self.bandwidth)
        X_torch = self._convert(X)
        if Y is None:
            Y_torch = X_torch
            K = self.sig_kernel.compute_Gram(X_torch, X_torch, sym=True, max_batch=self.max_batch)
            diag = self.sig_kernel.compute_kernel(X_torch, X_torch, max_batch=self.max_batch)
            norm_factor = torch.sqrt(torch.ger(diag, diag))
        else:
            Y_torch = self._convert(Y)
            K = self.sig_kernel.compute_Gram(X_torch, Y_torch, sym=False, max_batch=self.max_batch)
            diag_X = self.sig_kernel.compute_kernel(X_torch, X_torch, max_batch=self.max_batch)
            diag_Y = self.sig_kernel.compute_kernel(Y_torch, Y_torch, max_batch=self.max_batch)
            norm_factor = torch.sqrt(torch.ger(diag_X, diag_Y))
        # Avoid division by zero.
        norm_factor[norm_factor == 0] = 1.0
        K_normalized = K / norm_factor
        #K_normalized = K

        # Convert the result back to a cupy array.
        if K_normalized.device.type == 'cuda':
            cupy_K = cp.fromDlpack(torch.utils.dlpack.to_dlpack(K_normalized))
        else:
            cupy_K = cp.asarray(K_normalized.cpu().detach().numpy())
        return cupy_K

    def transform(self, X):
        """
        Compute the normalized Gram matrix for the given sequences X.

        Parameters:
          X : cupy.ndarray
              The input sequences.

        Returns:
          cupy.ndarray: The normalized Gram matrix.
        """
        return self.__call__(X, Y=None)


class SignatureKernelCupy:
    """
    A wrapper for the signature kernel that ensures compatibility with CuPy.

    This class initializes with a static kernel (default: RBF kernel), a bandwidth parameter,
    and a signature PDE kernel. When called, it computes the kernel matrix using the signature
    kernel and converts the result to a CuPy array.
    """

    def __init__(self, static_kernel=None, bandwidth=None, sig_kernel=None):
        """
        Initialize the SignatureKernelCupy.

        Parameters:
        ----------
        static_kernel : object, optional
            A pre-configured static kernel object. If None, an RBFKernel with the given bandwidth is used.
        bandwidth : float, optional
            The bandwidth parameter for the RBF kernel. Defaults to 1.0 if not provided.
        sig_kernel : object, optional
            A pre-configured signature PDE kernel. If None, a SignaturePDEKernel using the selected static kernel is used.
        """
        # Set default bandwidth if not provided
        if bandwidth is None:
            self._bandwidth = 1.0
        else:
            self._bandwidth = bandwidth

        # If no static kernel is provided, use an RBF kernel with the specified bandwidth
        if static_kernel is None:
            self.static_kernel = ksig.static.kernels.RBFKernel(bandwidth=self._bandwidth)
        else:
            self.static_kernel = static_kernel

        # If no signature kernel is provided, create one using the chosen static kernel
        if sig_kernel is None:
            self.sig_kernel = ksig.kernels.SignaturePDEKernel(static_kernel=self.static_kernel)
        else:
            self.sig_kernel = sig_kernel

    def __call__(self, X, Y=None):
        """
        Compute the signature kernel matrix for inputs X (and Y, if provided).

        Parameters:
        ----------
        X : array-like
            The first set of input data (e.g., paths) for kernel evaluation.
        Y : array-like, optional
            The second set of input data for a cross-kernel evaluation. If None, computes the
            kernel of X with itself.

        Returns:
        -------
        cupy.ndarray
            The computed kernel matrix as a CuPy array.
        """
        # Compute kernel matrix using the signature kernel. Handles both K(X, X) and K(X, Y).
        if Y is None:
            K = self.sig_kernel(X)
        else:
            K = self.sig_kernel(X, Y)

        return K


class GlobalAlignmentKernel:
    """
    A wrapper for the global alignment kernel that ensures compatibility with CuPy.

    This class initializes with a static kernel (default: RBF kernel), a bandwidth parameter,
    and a global alignment kernel. When called, it computes the kernel matrix using the
    kernel and converts the result to a CuPy array.
    """

    def __init__(self, n_features, static_kernel=None, bandwidth=None, ga_kernel=None):
        """
        Initialize the GlobalAlignmentKernel.

        Parameters:
        ----------
        static_kernel : object, optional
            A pre-configured static kernel object. If None, an RBFKernel with the given bandwidth is used.
        bandwidth : float, optional
            The bandwidth parameter for the RBF kernel. Defaults to 1.0 if not provided.
        sig_kernel : object, optional
            A pre-configured signature PDE kernel. If None, a SignaturePDEKernel using the selected static kernel is used.
        """
        # Set default bandwidth if not provided
        if bandwidth is None:
            self._bandwidth = 1.0
        else:
            self._bandwidth = bandwidth

        # If no static kernel is provided, use an RBF kernel with the specified bandwidth
        if static_kernel is None:
            self.static_kernel = ksig.static.kernels.RBFKernel(bandwidth=self._bandwidth)
        else:
            self.static_kernel = static_kernel

        # If no signature kernel is provided, create one using the chosen static kernel
        if ga_kernel is None:
            self.ga_kernel = ksig.kernels.GlobalAlignmentKernel(n_features=n_features, static_kernel=self.static_kernel)
        else:
            self.ga_kernel = ga_kernel

    def __call__(self, X, Y=None):
        """
        Compute the signature kernel matrix for inputs X (and Y, if provided).

        Parameters:
        ----------
        X : array-like
            The first set of input data (e.g., paths) for kernel evaluation.
        Y : array-like, optional
            The second set of input data for a cross-kernel evaluation. If None, computes the
            kernel of X with itself.

        Returns:
        -------
        cupy.ndarray
            The computed kernel matrix as a CuPy array.
        """
        # Compute kernel matrix using the signature kernel. Handles both K(X, X) and K(X, Y).
        if Y is None:
            K = self.ga_kernel(X)
        else:
            K = self.ga_kernel(X, Y)

        return K




class RandomWarpingSeriesKernel:
    """
    A kernel wrapper that uses the Random Warping Series (RWS) features provided
    by the KSig package (https://github.com/tgcsaba/KSig) to construct a kernel
    approximation as described in:

      Wu, L., Yen, I.E.-H., Yi, J., Xu, F., Lei, Q., & Witbrock, M. (2018).
      Random Warping Series: A Random Features Method for Time-Series Embedding.
      In Proceedings of the 35th International Conference on Machine Learning
      (pp. 793–802). PMLR.

    The RWS feature map computes, for each input time series, alignment-based features using
    (approximate) dynamic time warping with a set of random warping series. The kernel is defined as the
    inner product of these feature representations.

    This wrapper provides a scikit-learn-style API:
      - fit(X)
      - transform(X)
      - __call__(X, Y, diag=...)

    In addition, a static (base) kernel can be chosen (via the parameter static_kernel).
    For example, passing static_kernel='rbf' will instruct the underlying KSig RWS to use
    the RBF kernel as the base (static) kernel. This option mimics the setting described in the paper.

    Parameters:
      n_components : int
          Number of random warping series (i.e. the number of features).
      max_warping_length : int, optional
          If provided, specifies the maximum warping length (passed as max_warp).
      stddev : float, optional
          Standard deviation used for generating the random warping series.
      use_gpu : bool, optional
          If True, uses GPU acceleration via CuPy for inner product computation.
      **kwargs : dict
          Additional keyword arguments passed to RandomWarpingSeries.
    """

    def __init__(self, n_components=100, max_warp=None, stdev=1., n_features=None, random_state=None, use_gpu=True, **kwargs):
        self.n_components = n_components
        self.warping_length = max_warp
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self._fitted = False
        stdev = 1.0 if stdev is None else stdev

        # Instantiate the RWS feature map from ksig/kernels.py.
        # Pass the static_kernel parameter to choose a fixed base kernel if desired.
        if max_warp is not None:
            self.rws = RandomWarpingSeries(n_components=n_components,
                                           max_warp=max_warp,
                                           stdev=stdev,
                                           n_features=n_features,
                                           random_state=random_state,
                                           **kwargs)
        else:
            self.rws = RandomWarpingSeries(n_components=n_components,
                                           stdev=stdev,
                                           n_features=n_features,
                                           random_state=random_state,
                                           **kwargs)

    def fit(self, X):
        """
        Optionally fit the RWS feature map to data X.

        Parameters:
          X : array-like, shape (n_seq, T, d)
        Returns:
          self
        """
        self.rws.fit(X)
        self._fitted = True
        return self

    def transform(self, X):
        """
        Compute the low-dimensional RWS features for X.

        Parameters:
          X : array-like, shape (n_seq, T, d)
        Returns:
          P : ndarray of shape (n_seq, F)
              The feature representation of X.
        """
        X = cp.asarray(X, dtype=cp.float64)
        if not self._fitted:
            self.fit(X)
        return self.rws.transform(X)


    def __call__(self, X, Y=None, diag=False):
        """
        Evaluate the kernel between sequences X and Y.

        Parameters:
          X : array-like, shape (n_seq_X, T, d) or (T, d)
          Y : array-like, shape (n_seq_Y, T, d) or (T, d), optional.
              If None, then Y is taken as X.
          diag : bool, optional (default False)
              If True, compute only the diagonal entries (i.e. k(x, x) for each x).

        Returns:
          K : ndarray
              The computed kernel matrix (or a vector if diag=True).
        """
        # Compute the RWS features.
        PX = self.transform(X)
        if Y is None:
            PY = PX
        else:
            PY = self.transform(Y)

        if self.use_gpu:
            PX_gpu = cp.asarray(PX, dtype=cp.float64)
            PY_gpu = cp.asarray(PY, dtype=cp.float64)
            if diag:
                K = cp.sum(PX_gpu * PY_gpu, axis=1)
            else:
                K = cp.dot(PX_gpu, PY_gpu.T)
            return cp.asnumpy(K)
        else:
            if diag:
                return np.sum(PX * PY, axis=1)
            else:
                return PX @ PY.T



class KernelMatrix:
    """
    A unified wrapper to select and evaluate different sequence-based kernels.

    Supported kernel_type strings (exact matches required):
      - "Signature kernel torch"
      - "Signature kernel cupy"
      - "Signature kernel"                  (alias for "Signature kernel torch")
      - "Random fourier signature features kernel"
      - "Random warping series kernel"
      - "Global alignment kernel"

    The constructor takes, in addition to `kernel_type`, all of the positional
    arguments needed by each underlying kernel class. Any argument not relevant
    to the chosen kernel may be passed as None.

    Positional parameters (all default to None):
      sig_kernel         # for SignatureKernelTorch or SignatureKernelCupy
      max_batch          # for SignatureKernelTorch
      device             # for SignatureKernelTorch
      static_kernel      # for SignatureKernelCupy or GlobalAlignmentKernel
      bandwidth          # for SignatureKernelCupy, RandomFourierSignatureFeaturesKernel, or GlobalAlignmentKernel
      n_levels           # for RandomFourierSignatureFeaturesKernel
      n_components       # for RandomFourierSignatureFeaturesKernel or RandomWarpingSeriesKernel
      method             # for RandomFourierSignatureFeaturesKernel
      max_warp           # for RandomWarpingSeriesKernel
      stdev              # for RandomWarpingSeriesKernel
      n_features         # for RandomWarpingSeriesKernel or GlobalAlignmentKernel
      random_state       # for RandomWarpingSeriesKernel
      use_gpu            # for RandomWarpingSeriesKernel
      ga_kernel          # for GlobalAlignmentKernel

    If `kernel_type` is None, defaults to "Signature kernel" (i.e. SignatureKernelTorch())
    and ignores all other arguments.

    Raises
    ------
    TypeError
        If `kernel_type` is not one of the supported strings.
    """

    def __init__(
        self,
        kernel_type=None,
        sig_kernel=None,
        max_batch=None,
        device=None,
        dyadic_order=1,
        static_kernel=None,
        bandwidth=None,
        n_levels=5,
        n_components=100,
        method='TRP',
        max_warp=None,
        stdev=None,
        n_features=None,
        random_state=None,
        use_gpu=None,
        ga_kernel=None
    ):
        # If no kernel_type is provided, default to the Torch implementation of the signature kernel.
        if kernel_type is None:
            # Instantiate SignatureKernelTorch with no arguments (or defaults).
            self.kernel = SignatureKernelTorch()
            return

        # Select and instantiate the appropriate kernel based on the exact string.
        if kernel_type == 'Signature kernel torch':
            # SignatureKernelTorch expects (sig_kernel, bandwidth, max_batch, dyadic_order, device)
            self.kernel = SignatureKernelTorch(sig_kernel, bandwidth, max_batch, dyadic_order, device)

        elif kernel_type == 'Signature kernel cupy':
            # SignatureKernelCupy expects (static_kernel, bandwidth, sig_kernel)
            self.kernel = SignatureKernelCupy(static_kernel, bandwidth, sig_kernel)

        elif kernel_type == 'Signature kernel':
            # Alias for the Torch implementation
            self.kernel = SignatureKernelTorch(sig_kernel, bandwidth, max_batch, dyadic_order, device)

        elif kernel_type == 'Random fourier signature features kernel':
            # RandomFourierSignatureFeaturesKernel expects (bandwidth, n_levels, n_components, method)
            self.kernel = RandomFourierSignatureFeaturesKernel(bandwidth, n_levels, n_components, method)

        elif kernel_type == 'Random warping series kernel':
            # NOTE: use RandomWarpingSeriesKernel (not RandomWarpingSeries)
            # RandomWarpingSeriesKernel expects (n_components, max_warp, stdev, n_features, random_state, use_gpu)
            self.kernel = RandomWarpingSeriesKernel(n_components, max_warp, stdev, n_features, random_state, use_gpu)

        elif kernel_type == 'Global alignment kernel':
            # GlobalAlignmentKernel expects (n_features, static_kernel, bandwidth, ga_kernel)
            self.kernel = GlobalAlignmentKernel(n_features, static_kernel, bandwidth, ga_kernel)

        else:
            raise TypeError(
                f"Unknown kernel type '{kernel_type}'. Valid options are:\n"
                "  'Signature kernel torch', 'Signature kernel cupy', 'Signature kernel',\n"
                "  'Random fourier signature features kernel',\n"
                "  'Random warping series kernel', 'Global alignment kernel'."
            )

    def __call__(self, X, Y=None, diag=False):
        """
        Evaluate the chosen kernel on input sequences, converting inputs to CuPy arrays
        and returning the result as a NumPy array.

        Parameters
        ----------
        X : array-like
            First batch of sequences (e.g., NumPy array or any array-like).
        Y : array-like or None, default=None
            Second batch of sequences. If None, compute the Gram matrix of X with itself.
        diag : bool, default=False
            (Unused—retained only for signature consistency.)

        Returns
        -------
        K : numpy.ndarray
            Kernel matrix (or cross-Kernel matrix) as a NumPy array.
        """

        # Convert inputs to CuPy arrays
        X_cu = cp.asarray(X)
        if Y is None:
            K_cu = self.kernel(X_cu)
        else:
            Y_cu = cp.asarray(Y)
            K_cu = self.kernel(X_cu, Y_cu)

        # Convert result back to NumPy
        K = cp.asnumpy(K_cu)
        return K


class ConvexKernelMatrix(KernelMatrix):
    """Combine independently normalized modality kernels (paper Appendix A.B).

    Each modality is a batch of paths with shape ``(N, T, d_m)``. Inputs
    can be dictionaries of named batches, or the existing concatenated array
    together with ``modality_slices``. Keeping the latter interface lets every
    existing metric and maximizer select the same episode rows unchanged.

    ``weights`` defaults to uniform. A dictionary can specify a subset of
    modalities; omitted names receive zero weight. ``kernel_configs`` maps
    modality names to overrides of any :class:`KernelMatrix` constructor
    option. With no explicit bandwidth, each modality gets its own heuristic
    bandwidth, fixed when fitting the full dataset. Alternatively, provide
    ``modality_kernels`` containing configured callable kernels.

    The result is ``sum_m alpha_m K_m / sqrt(diag(K_m) diag(K_m).T)``.
    In particular, a self Gram matrix has unit diagonal before entropy is
    computed on ``K / N``. Fit once on the full dataset before selecting subsets
    so all comparisons use the same random feature maps.
    """

    def __init__(self, modality_slices=None, weights=None, *,
                 modality_kernels=None, kernel_configs=None,
                 renormalize=False, **kernel_kwargs):
        sources = (modality_slices, modality_kernels, kernel_configs)
        for source in sources:
            if source is not None and not isinstance(source, Mapping):
                raise TypeError("Modality slices, kernels and configs must be mappings.")
        source = next((s for s in sources if s), None)
        if source is None:
            raise ValueError("Provide at least one named modality.")
        self.modality_names = tuple(source)
        if any(not isinstance(name, str) or not name for name in self.modality_names):
            raise ValueError("Modality names must be nonempty strings.")
        self.modality_slices = dict(modality_slices or {})
        self.modality_kernels = dict(modality_kernels or {})
        self.kernel_configs = dict(kernel_configs or {})
        self.kernel_kwargs = dict(kernel_kwargs)
        for label, mapping in (("kernels", self.modality_kernels),
                               ("configs", self.kernel_configs)):
            unknown = set(mapping) - set(self.modality_names)
            if unknown:
                raise ValueError(f"Unknown modality names in {label}: {sorted(unknown)}")
        for name, selector in self.modality_slices.items():
            if (not isinstance(selector, slice) or selector.start is None
                    or selector.stop is None or selector.start < 0
                    or selector.stop <= selector.start or selector.step not in (None, 1)):
                raise ValueError(f"Modality '{name}' needs a nonempty feature slice.")
        for name, config in self.kernel_configs.items():
            if not isinstance(config, Mapping):
                raise TypeError(f"Kernel config for '{name}' must be a mapping.")
            self.kernel_configs[name] = dict(config)
        if any(not callable(kernel) for kernel in self.modality_kernels.values()):
            raise TypeError("Every modality kernel must be callable.")
        if weights is None:
            weights = np.full(len(self.modality_names), 1 / len(self.modality_names))
        elif isinstance(weights, Mapping):
            unknown = set(weights) - set(self.modality_names)
            if unknown:
                raise ValueError(f"Unknown modality names in weights: {sorted(unknown)}")
            weights = [weights.get(name, 0.0) for name in self.modality_names]
        self.weights = ConvexMatrixCombiner(weights, renormalize=renormalize).weights
        if len(self.weights) != len(self.modality_names):
            raise ValueError("There must be one weight per modality.")
        self.active_modalities = tuple(
            name for name, weight in zip(self.modality_names, self.weights) if weight > 0
        )
        self.bandwidths = {}
        self.kernel = self  # Preserve the existing km.kernel.fit(data) interface.
        self._fitted = False

    @staticmethod
    def _numpy(array):
        return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)

    def _split(self, X):
        if isinstance(X, Mapping):
            missing = set(self.active_modalities) - set(X)
            if missing:
                raise ValueError(f"Missing modalities: {sorted(missing)}")
            batches = {name: X[name] for name in self.active_modalities}
        else:
            if not self.modality_slices:
                raise ValueError("Array inputs require modality_slices.")
            if not hasattr(X, 'shape'):
                X = np.asarray(X)
            if X.ndim != 3:
                raise ValueError("Paths must have shape (N, T, d).")
            for name in self.active_modalities:
                if self.modality_slices[name].stop > X.shape[2]:
                    raise ValueError(f"Feature slice for '{name}' exceeds the input width.")
            batches = {name: X[..., self.modality_slices[name]]
                       for name in self.active_modalities}
        count = None
        for name, batch in batches.items():
            if not hasattr(batch, 'shape'):
                batch = np.asarray(batch)
                batches[name] = batch
            if batch.ndim != 3 or batch.shape[1] == 0 or batch.shape[2] == 0:
                raise ValueError(f"Modality '{name}' must have shape (N, T, d), with T,d > 0.")
            if count is not None and batch.shape[0] != count:
                raise ValueError("All modality batches must have the same episode count.")
            count = batch.shape[0]
        return batches, count

    def fit(self, X):
        """Fit each active modality's kernel on the full, aligned dataset."""
        batches, count = self._split(X)
        if count == 0:
            raise ValueError("Cannot fit modality kernels on an empty dataset.")
        for name, batch in batches.items():
            if name not in self.modality_kernels:
                config = dict(self.kernel_kwargs)
                config.update(self.kernel_configs.get(name, {}))
                if config.get('kernel_type') is None:
                    config['kernel_type'] = 'Signature kernel'
                bandwidth = config.get('bandwidth')
                if bandwidth is None:
                    # Same median/log(n) heuristic as the concatenated path,
                    # made positive and defined for one-demo/constant datasets.
                    values = self._numpy(batch)
                    bandwidth = abs(float(np.median(values))) / np.log(max(count, 2)) + 1e-3
                bandwidth = float(bandwidth)
                if not np.isfinite(bandwidth) or bandwidth <= 0:
                    raise ValueError(f"Bandwidth for '{name}' must be finite and positive.")
                config['bandwidth'] = bandwidth
                self.bandwidths[name] = bandwidth
                self.modality_kernels[name] = KernelMatrix(**config)
            kernel = self.modality_kernels[name]
            feature_map = kernel.kernel if isinstance(kernel, KernelMatrix) else kernel
            if hasattr(feature_map, 'fit'):
                feature_map.fit(batch)
        self._fitted = True
        return self

    def _evaluate(self, name, X, Y=None):
        result = self._numpy(self.modality_kernels[name](X, Y))
        expected = (X.shape[0], X.shape[0] if Y is None else Y.shape[0])
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise ValueError(f"Kernel for '{name}' must return a finite matrix of shape {expected}.")
        return np.asarray(result, dtype=float)

    @staticmethod
    def _diagonal(K, name):
        diagonal = np.diag(K)
        if np.any(diagonal <= 0):
            raise ValueError(f"Kernel for '{name}' must have a strictly positive self diagonal.")
        return diagonal

    def __call__(self, X, Y=None, diag=False):
        """Return the normalized convex Gram/cross matrix, or its paired diagonal."""
        batches_x, n = self._split(X)
        self_gram = Y is None or Y is X
        batches_y, m = (batches_x, n) if self_gram else self._split(Y)
        if diag and n != m:
            raise ValueError("Paired diagonal evaluation requires equal episode counts.")
        if n == 0 or m == 0:
            return np.zeros(n if diag else (n, m), dtype=float)
        if not self._fitted:
            self.fit(X)
        combined = np.zeros((n, m), dtype=float)
        for name, weight in zip(self.modality_names, self.weights):
            if weight == 0:
                continue
            x, y = batches_x[name], batches_y[name]
            if x.shape[2] != y.shape[2]:
                raise ValueError(f"X and Y feature widths differ for modality '{name}'.")
            K_x = self._evaluate(name, x)
            dx = self._diagonal(K_x, name)
            if self_gram:
                K, dy = K_x, dx
            else:
                K = self._evaluate(name, x, y)
                dy = self._diagonal(self._evaluate(name, y), name)
            # Taking square roots separately avoids overflow in dx * dy.
            normalized = K / np.sqrt(dx)[:, None] / np.sqrt(dy)[None, :]
            if self_gram:
                normalized = (normalized + normalized.T) / 2
                np.fill_diagonal(normalized, 1.0)
            combined += weight * normalized
        return np.diag(combined).copy() if diag else combined

    def transform(self, X):
        """Return the combined normalized Gram matrix."""
        return self(X)
