"""Pluggable gridding / degridding backends.

Quick start::

    from astroviper.core.imaging.imaging_utils.gridding_backends import get_backend

    # Standard prolate-spheroidal gridder (always available)
    backend = get_backend(
        "standard",
        image_size=[200, 200],
        cell_size=[cell_rad, cell_rad],
    )

    # FINUFFT gridder (requires ``pip install finufft``)
    backend = get_backend(
        "finufft",
        image_size=[200, 200],
        cell_size=[cell_rad, cell_rad],
        eps=1e-6,
    )

    # FINUFFT with explicit thread control (4 OpenMP threads)
    backend = get_backend(
        "finufft",
        image_size=[200, 200],
        cell_size=[cell_rad, cell_rad],
        eps=1e-6,
        nthreads=4,
    )

    # Use identically:
    grid, sumwt = backend.grid(vis_data, uvw, weight, freq_chan)

    # ``vis_ms4`` and ``image_array`` should be provided by the user, e.g. by
    # reading them from a MeasurementSet and allocating an output image array.
    backend.degrid(grid, vis_ms4)
    backend.grid_to_image(vis_ms4, image_array)
"""

from astroviper.core.imaging.imaging_utils.gridding_backends.base import (
    GriddingBackend,
)
from astroviper.core.imaging.imaging_utils.gridding_backends.factory import (
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "GriddingBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
