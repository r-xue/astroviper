"""Backend factory for gridding / degridding.

Provides :func:`get_backend` to obtain a concrete
:class:`~.base.GriddingBackend` instance by name.

Usage::

    from astroviper.core.imaging.imaging_utils.gridding_backends import get_backend

    backend = get_backend(
        "standard",                        # or "finufft"
        image_size=np.array([200, 200]),
        cell_size=np.array([cell, cell]),
    )
    grid, sumwt = backend.grid(vis_data, uvw, weight, freq_chan)
"""

import importlib
from typing import Optional, Type

import numpy as np
import numpy.typing as npt

from astroviper.core.imaging.imaging_utils.gridding_backends.base import GriddingBackend

# Registry stores backend class dotted paths; actual modules (and optional
# dependencies like finufft) are imported lazily only when requested.
_REGISTRY: dict[str, str] = {
    "standard": (
        "astroviper.core.imaging.imaging_utils.gridding_backends."
        "standard_backend.StandardBackend"
    ),
    "finufft": (
        "astroviper.core.imaging.imaging_utils.gridding_backends."
        "finufft_backend.FinufftBackend"
    ),
    "cufinufft": (
        "astroviper.core.imaging.imaging_utils.gridding_backends."
        "cufinufft_backend.CufinufftBackend"
    ),
    "wprojection": (
        "astroviper.core.imaging.imaging_utils.gridding_backends."
        "wprojection_backend.WProjectionBackend"
    ),
    "wstack_finufft": (
        "astroviper.core.imaging.imaging_utils.gridding_backends."
        "wstack_finufft_backend.WStackFinufftBackend"
    ),
    "wstack_cufinufft": (
        "astroviper.core.imaging.imaging_utils.gridding_backends."
        "wstack_cufinufft_backend.WStackCufinufftBackend"
    ),
}


def list_backends():
    """Return the names of all registered backends."""
    return list(_REGISTRY.keys())


def _import_class(dotted_path: str) -> Type[GriddingBackend]:
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_backend(
    name: str = "standard",
    *,
    image_size: Optional[npt.NDArray[np.int_]] = None,
    cell_size: Optional[npt.NDArray[np.float64]] = None,
    chan_mode: str = "continuum",
    **backend_kwargs,
) -> GriddingBackend:
    """Obtain a gridding backend by name.

    Parameters
    ----------
    name : str
        ``"standard"`` (default) or ``"finufft"``, etc.
    image_size : array-like of int, shape (2,)
        Number of pixels ``[ny, nx]``.
    cell_size : array-like of float, shape (2,)
        Pixel size in radians ``[dy, dx]``.
    chan_mode : str
        ``"continuum"`` or ``"cube"``.
    **backend_kwargs
        Extra keyword arguments forwarded to the backend constructor.
        Notable options include:

        - ``oversampling`` *(int)* – Oversampling of the convolution
          kernel.  Used by ``standard`` and ``wprojection`` backends
          (default ``100``).
        - ``support`` *(int)* – Support half-width of the convolution
          kernel.  Used by ``standard`` and ``wprojection`` backends
          (default ``7``).
        - ``eps`` *(float)* – Requested precision for FINUFFT / cuFINUFFT
          backends (default ``1e-6``).
        - ``nthreads`` *(int)* – Number of OpenMP threads for FINUFFT
          backends.  ``0`` (default) uses all available cores; any
          positive integer limits parallelism.  Accepted but ignored by
          GPU (cuFINUFFT) backends.
        - ``wplanes`` *(int)* – Number of w-planes for the w-projection
          and w-stacking backends.

    Returns
    -------
    GriddingBackend
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'.  Available: {list_backends()}")

    if image_size is None or cell_size is None:
        raise ValueError("image_size and cell_size are required")

    cls = _import_class(_REGISTRY[key])
    return cls(
        image_size=np.asarray(image_size),
        cell_size=np.asarray(cell_size, dtype=np.float64),
        chan_mode=chan_mode,
        **backend_kwargs,
    )


def register_backend(name: str, dotted_path: str) -> None:
    """Register a custom backend class.

    Parameters
    ----------
    name : str
        Short name used with :func:`get_backend`.
    dotted_path : str
        Fully-qualified dotted path to the class, e.g.
        ``"mypackage.my_backend.MyBackend"``.
    """
    _REGISTRY[name.lower().strip()] = dotted_path
