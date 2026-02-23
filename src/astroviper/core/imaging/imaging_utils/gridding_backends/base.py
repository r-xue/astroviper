"""Abstract base class for gridding / degridding backends.

Every concrete backend should subclass `GriddingBackend` and implement
at least `grid` and `degrid`.  The higher-level helper `grid_to_image`
is provided with a default implementation that calls `grid` followed by
FFT + spheroidal correction, but backends are free to override it.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import xarray


class GriddingBackend(ABC):
    """Backend-agnostic interface for gridding and degridding operations.

    Attributes
    ----------
    image_size : ndarray of int, shape (2,)
        Number of pixels ``[ny, nx]``.
    cell_size : ndarray of float64, shape (2,)
        Pixel size in radians ``[dy, dx]``.
    chan_mode : str
        ``'continuum'`` (all channels -> 1 image) or ``'cube'``
        (one image plane per channel).
    """

    def __init__(
        self,
        image_size: npt.NDArray[np.int_],
        cell_size: npt.NDArray[np.float64],
        *,
        chan_mode: str = "continuum",
    ):
        """Initialize the backend with grid geometry.

        Parameters
        ----------
        image_size : array-like of int, shape (2,)
            Number of pixels ``[ny, nx]``.
        cell_size : array-like of float, shape (2,)
            Pixel size in radians ``[dy, dx]``.
        chan_mode : str, optional
            ``'continuum'`` (all channels -> 1 image) or ``'cube'``
            (one image plane per channel).
        """
        self.image_size = np.asarray(image_size, dtype=int)
        self.cell_size = np.asarray(cell_size, dtype=np.float64)
        self.chan_mode = str(chan_mode)

    # abstract interface

    @abstractmethod
    def grid(
        self,
        vis_data: npt.NDArray[np.complexfloating],
        uvw: npt.NDArray[np.floating],
        weight: npt.NDArray[np.floating],
        freq_chan: npt.NDArray[np.floating],
        *,
        grid: npt.NDArray[np.complexfloating] | None = None,
        sum_weight: npt.NDArray[np.floating] | None = None,
        do_psf: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid visibility data onto a regular UV grid.

        Parameters
        ----------
        vis_data : ndarray of complex, shape (n_time, n_baseline, n_chan, n_pol)
            Complex visibility data.
        uvw : ndarray of float, shape (n_time, n_baseline, 3)
            UVW coordinates in metres.
        weight : ndarray of float, shape (n_time, n_baseline, n_chan, n_pol)
            Visibility weights.
        freq_chan : ndarray of float, shape (n_chan,)
            Channel frequencies in Hz.
        grid : ndarray of complex or None, optional
            Existing grid to accumulate into.  A fresh grid is created
            when *None*.
        sum_weight : ndarray of float or None, optional
            Existing sum-of-weights array to accumulate into.
        do_psf : bool, optional
            If *True*, grid weights only to produce a PSF.

        Returns
        -------
        grid : ndarray of complex, shape (n_image_chan, n_pol, ny, nx)
            Accumulated UV grid.
        sum_weight : ndarray of float, shape (n_image_chan, n_pol)
            Sum of weights per image channel and polarisation.
        """

    @abstractmethod
    def degrid(
        self,
        grid: npt.NDArray[np.complexfloating],
        vis: xarray.core.datatree.DataTree,
        *,
        incremental: bool = False,
    ) -> None:
        """Degrid model visibilities from a UV grid.

        The result is written into ``vis['VISIBILITY_MODEL']``
        **in-place**.

        Parameters
        ----------
        grid : ndarray of complex, shape (n_image_chan, n_pol, ny, nx)
            Gridded model visibilities.
        vis : DataTree
            MS v4 dataset -- modified **in-place**.
        incremental : bool, optional
            When *True*, add to the existing ``VISIBILITY_MODEL``
            instead of replacing it.
        """

    # higher-level interface

    def grid_to_image(
        self,
        vis: xarray.core.datatree.DataTree | list[xarray.core.datatree.DataTree],
        resid_array: npt.NDArray[np.floating],
        *,
        do_psf: bool = False,
        column: str = "VISIBILITY",
    ) -> None:
        """Grid visibilities and form a dirty image or PSF.

        Higher-level helper that typically calls `grid` followed by an
        FFT and spheroidal correction to produce an image-domain
        representation.  Concrete backends may override this for
        efficiency but should preserve the semantics documented here.

        The result is written into *resid_array* **in-place**.

        Parameters
        ----------
        vis : DataTree or list of DataTree
            Input measurement-set dataset(s) containing visibility
            data.  When a list is provided, contributions from all
            datasets are accumulated into the same *resid_array*.
        resid_array : ndarray of float, shape (n_image_chan, n_pol, ny, nx)
            Pre-allocated output array filled **in-place** with the
            resulting image(s).
        do_psf : bool, optional
            If *True*, form the point-spread function by gridding
            weights only.
        column : str, optional
            Name of the visibility column in *vis* to grid
            (e.g. ``'VISIBILITY'``, ``'CORRECTED_DATA'``,
            ``'VISIBILITY_MODEL'``).
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_size={self.image_size.tolist()}, "
            f"cell_size={self.cell_size.tolist()}, "
            f"chan_mode='{self.chan_mode}')"
        )
