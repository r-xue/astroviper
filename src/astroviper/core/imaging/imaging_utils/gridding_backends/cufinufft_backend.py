"""cuFINUFFT-based gridding / degridding backend (GPU).

Inherits all logic from :class:`FinufftBackend` and only overrides the
actual NUFFT calls to use GPU arrays via CuPy + cufinufft.

Requires the ``cufinufft`` and ``cupy`` Python packages.
"""

from astroviper.core.imaging.imaging_utils.gridding_backends.finufft_backend import (
    FinufftBackend,
)


def _check_cufinufft():
    """Import cufinufft and raise a friendly error if not installed."""
    try:
        import cufinufft

        return cufinufft
    except ImportError as exc:
        raise ImportError(
            "The cuFINUFFT backend requires the 'cufinufft' and 'cupy' "
            "packages.  Install with:  pip install cufinufft cupy"
        ) from exc


def _check_cupy():
    """Import cupy and raise a friendly error if not installed."""
    try:
        import cupy

        return cupy
    except ImportError as exc:
        raise ImportError(
            "The cuFINUFFT backend requires the 'cupy' package.  "
            "Install with:  pip install cupy"
        ) from exc


class CufinufftBackend(FinufftBackend):
    """GPU-accelerated NUFFT backend using cuFINUFFT.

    Inherits all gridding/degridding logic from :class:`FinufftBackend`
    and only overrides :meth:`_nufft2d1` and :meth:`_nufft2d2` to
    transfer arrays to/from the GPU.

    Parameters
    ----------
    eps : float
        Requested relative precision for the NUFFT.  Default ``1e-6``.
    **kwargs
        Forwarded to :class:`FinufftBackend`.
    """

    def __init__(self, *args, eps: float = 1e-6, **kwargs):
        # Skip FinufftBackend.__init__'s finufft import; call grandparent
        super(FinufftBackend, self).__init__(*args, **kwargs)
        self.eps = eps
        self.finufft = _check_cufinufft()
        self._cp = _check_cupy()

    def _nufft2d1(self, x, y, strengths, n_modes, eps):
        """Type-1 NUFFT via cuFINUFFT (non-uniform -> uniform)."""
        cp = self._cp
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        c_gpu = cp.asarray(strengths, dtype=cp.complex128)

        result_gpu = self.finufft.nufft2d1(x_gpu, y_gpu, c_gpu, n_modes, eps=eps)
        return cp.asnumpy(result_gpu)

    def _nufft2d2(self, x, y, image_plane, eps):
        """Type-2 NUFFT via cuFINUFFT (uniform -> non-uniform)."""
        cp = self._cp
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        f_gpu = cp.asarray(image_plane, dtype=cp.complex128)

        result_gpu = self.finufft.nufft2d2(x_gpu, y_gpu, f_gpu, eps=eps)
        return cp.asnumpy(result_gpu)
