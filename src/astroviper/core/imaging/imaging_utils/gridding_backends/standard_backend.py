"""Standard prolate-spheroidal gridding / degridding backend.

This backend wraps the existing numba-jitted gridding and degridding
routines in :mod:`astroviper.core.imaging.imaging_utils.standard_grid` and
:mod:`astroviper.core.imaging.imaging_utils.standard_degrid` behind the
:class:`GriddingBackend` interface.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import xarray

from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel,
    create_prolate_spheroidal_kernel_1D,
)
from astroviper.core.imaging.imaging_utils.gridding_backends.base import GriddingBackend
from astroviper.core.imaging.imaging_utils.standard_degrid import degrid_spheroid_ms4
from astroviper.core.imaging.imaging_utils.standard_grid import (
    standard_grid_numpy_wrap_input_checked,
)


class StandardBackend(GriddingBackend):
    """Prolate-spheroidal convolution-function based gridder/degridder.

    This is the traditional approach used in radio-interferometry imaging
    (similar to CASA's ``StandardFTMachine``).
    """

    def grid(
        self,
        vis_data: npt.NDArray[np.complexfloating],
        uvw: npt.NDArray[np.floating],
        weight: npt.NDArray[np.floating],
        freq_chan: npt.NDArray[np.floating],
        *,
        grid: Optional[npt.NDArray[np.complexfloating]] = None,
        sum_weight: Optional[npt.NDArray[np.floating]] = None,
        do_psf: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw complex UV grid and sum-of-weights.

        The grid-to-image conversion (IFFT, spheroidal correction,
        normalisation) is handled by :meth:`grid_to_image`, NOT here.
        """
        cgk_1D = create_prolate_spheroidal_kernel_1D(self.oversampling, self.support)

        _grid = grid if grid is not None else np.array([])
        _sum_weight = sum_weight if sum_weight is not None else np.array([])

        result_grid, result_sumwt = standard_grid_numpy_wrap_input_checked(
            vis_data,
            _grid,
            _sum_weight,
            uvw,
            weight,
            freq_chan,
            cgk_1D,
            image_size=self.image_size,
            cell_size=self.cell_size,
            oversampling=self.oversampling,
            support=self.support,
            complex_grid=True,
            do_psf=do_psf,
            chan_mode=self.chan_mode,
        )

        return result_grid, result_sumwt

    def grid_to_image(
        self,
        vis: Union[
            xarray.core.datatree.DataTree,
            list[xarray.core.datatree.DataTree],
        ],
        resid_array: npt.NDArray[np.floating],
        *,
        do_psf: bool = False,
        column: str = "VISIBILITY",
    ) -> None:
        """End-to-end: grid visibilities, FFT, and apply spheroidal correction.

        The image is written **in-place** into *resid_array*.

        Parameters
        ----------
        vis : DataTree or list[DataTree]
            Input measurement set(s).
        resid_array : ndarray, shape (n_chan, n_pol, ny, nx)
            Output image array – written in-place.
        do_psf : bool
            Grid weights for PSF instead of visibility data.
        column : str
            Which visibility column to grid (ignored when *do_psf* is True).
        """
        if isinstance(vis, xarray.core.datatree.DataTree):
            list_vis = [vis]
        else:
            list_vis = vis

        ny, nx = resid_array.shape[-2:]
        grid_accum: Optional[np.ndarray] = None
        sumwt: Optional[np.ndarray] = None

        for ds in list_vis:
            if not isinstance(ds, xarray.core.datatree.DataTree):
                raise TypeError("Each element of vis must be an xarray DataTree")
            vis_data = ds[column].data
            uvw = ds.UVW.data

            weight = ds.WEIGHT.data.copy()
            flag = ds.FLAG.data

            weight[flag] = 0.0
            nan_uvw = np.isnan(uvw[:, :, 0]) | np.isnan(uvw[:, :, 1])
            weight[nan_uvw] = 0.0

            freq_chan = ds.coords["frequency"].values

            grid_accum, sumwt = self.grid(
                vis_data,
                uvw,
                weight,
                freq_chan,
                grid=grid_accum,
                sum_weight=sumwt,
                do_psf=do_psf,
            )

        if grid_accum is None or sumwt is None:
            raise RuntimeError("No data was gridded")

        if grid_accum.shape != resid_array.shape:
            raise RuntimeError(
                f"Grid shape {grid_accum.shape} does not match image shape {resid_array.shape}"
            )

        _, corr_term = create_prolate_spheroidal_kernel(
            self.oversampling,
            self.support,
            np.array([ny, nx], dtype=int),
        )

        for chan in range(resid_array.shape[0]):
            for corr in range(resid_array.shape[1]):
                resid_array[chan, corr, :, :] = (
                    np.real(
                        np.fft.fftshift(
                            np.fft.ifft2(np.fft.ifftshift(grid_accum[chan, corr, :, :]))
                        )
                    )
                    / corr_term
                    * ny
                    * nx
                    / sumwt[chan, corr]
                )

    def degrid(
        self,
        grid: npt.NDArray[np.complexfloating],
        vis: xarray.core.datatree.DataTree,
        *,
        incremental: bool = False,
    ) -> None:
        degrid_spheroid_ms4(
            vis=vis,
            grid=grid,
            pixelincr=self.cell_size,
            support=self.support,
            sampling=self.oversampling,
            incremental=incremental,
        )
