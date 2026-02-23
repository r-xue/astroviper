"""FINUFFT-based gridding / degridding backend.

Uses the Flatiron Institute Non-Uniform FFT library (``finufft``) for
gridding (type-1 NUFFT) and degridding (type-2 NUFFT).  This avoids the
explicit convolution loop used by the standard backend.

Requires the ``finufft`` Python package::

    pip install finufft

or::

    conda install -c conda-forge finufft
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray
from astropy import constants
from astroviper.core.imaging.imaging_utils.gridding_backends.base import GriddingBackend


def _check_finufft():
    """Import finufft and raise a friendly error if not installed."""
    try:
        import finufft

        return finufft
    except ImportError as exc:
        raise ImportError(
            "The FINUFFT backend requires the 'finufft' package.  "
            "Install it with:  pip install finufft  or  "
            "conda install -c conda-forge finufft"
        ) from exc


class FinufftBackend(GriddingBackend):
    """Non-uniform FFT based gridder / degridder using FINUFFT.

    Parameters
    ----------
    eps : float
        Requested relative precision for FINUFFT.  Smaller values are
        more accurate but slower.  Default ``1e-6``.
        This tolerance represents the combined truncation and approximation
        error; FINUFFT automatically tunes its internal parameters to meet
        this value.
    nthreads : int
        Number of OpenMP threads for FINUFFT to use.  ``0`` (the default)
        means use all available cores.  Set to ``1`` for single-threaded
        execution; any positive integer limits parallelism to that many
        threads.
    **kwargs
        Forwarded to :class:`GriddingBackend`.
    """

    def __init__(self, *args, eps: float = 1e-6, nthreads: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.nthreads = int(nthreads)
        # check availability so the user gets a clear error early
        self.finufft = _check_finufft()

    #  FINUFFT wrappers

    def _nufft2d1(self, x, y, strengths, n_modes, eps):
        """Type-1 NUFFT (non-uniform → uniform). CPU version."""
        kwargs = dict(eps=eps)
        if self.nthreads > 0:
            kwargs["nthreads"] = self.nthreads
        return self.finufft.nufft2d1(
            x.astype(np.float64),
            y.astype(np.float64),
            strengths,
            n_modes,
            **kwargs,
        )

    def _nufft2d2(self, x, y, image_plane, eps):
        """Type-2 NUFFT (uniform → non-uniform). CPU version."""
        kwargs = dict(eps=eps)
        if self.nthreads > 0:
            kwargs["nthreads"] = self.nthreads
        return self.finufft.nufft2d2(
            x.astype(np.float64),
            y.astype(np.float64),
            image_plane,
            **kwargs,
        )

    def _uvw_to_finufft_coords(
        self,
        uvw: npt.NDArray[np.floating],
        freq_chan: npt.NDArray[np.floating],
    ):
        """Convert UVW (metres) + frequencies → FINUFFT coordinates in [-pi, pi).

        FINUFFT type-1 computes:

            f[k] = sum_j c_j  exp(+i k . x_j)

        where x_j ∈ [-pi, pi) and k is the integer grid index centered on 0.

        The standard gridder maps ``u`` (metres) → pixel via::

            u_pix = -(freq * cell * N / c) * u

        We need the *angular frequency* version of that mapping so the
        output lands on the same grid.  The relationship is::

            x = 2 * pi * u_pix / N  =  -2 * pi * freq * cell * u / c

        Parameters
        ----------
        uvw : (n_time, n_baseline, 3)
        freq_chan : (n_chan,)

        Returns
        -------
        x_coords : (n_chan, n_points)  – FINUFFT *x* coordinate
        y_coords : (n_chan, n_points)  – FINUFFT *y* coordinate
        """
        c = constants.c.value  # speed of light in m/s

        # Flatten the time × baseline axes
        u_flat = uvw[:, :, 0].ravel()  # (n_points,)
        v_flat = uvw[:, :, 1].ravel()  # (n_points,)

        # Scale per channel → (n_chan, n_points)
        # The negative sign mirrors the standard gridder convention
        scale_x = -2.0 * np.pi * self.cell_size[0] * freq_chan / c  # (n_chan,)
        scale_y = -2.0 * np.pi * self.cell_size[1] * freq_chan / c

        x_coords = scale_x[:, None] * u_flat[None, :]  # (n_chan, n_points)
        y_coords = scale_y[:, None] * v_flat[None, :]

        return x_coords, y_coords

    # --------------------------------------------------------------------- #
    #  grid (type-1 NUFFT)                                                   #
    # --------------------------------------------------------------------- #

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

        # shape in n_time, n_baseline, n_chan, n_vis_pol
        _, _, n_chan, _ = vis_data.shape
        ny, nx = int(self.image_size[0]), int(self.image_size[1])

        # Derive n_pol from weight (matches standard backend convention:
        # weight may have fewer pol planes than vis_data)
        n_wt_pol = weight.shape[3]
        n_pol = n_wt_pol

        # Channel mapping
        if self.chan_mode == "cube":
            n_image_chan = n_chan
            chan_map = np.arange(n_chan, dtype=int)
        else:
            n_image_chan = 1
            chan_map = np.zeros(n_chan, dtype=int)

        # Allocate output arrays
        if grid is None:
            grid = np.zeros((n_image_chan, n_pol, ny, nx), dtype=np.complex128)
        if sum_weight is None:
            sum_weight = np.zeros((n_image_chan, n_pol), dtype=np.float64)

        # Pre-compute FINUFFT coordinates per channel
        x_coords, y_coords = self._uvw_to_finufft_coords(uvw, freq_chan)

        # Mask out NaN UVW points
        nan_mask = np.isnan(uvw[:, :, 0].ravel()) | np.isnan(uvw[:, :, 1].ravel())

        for i_chan in range(n_chan):
            a_chan = chan_map[i_chan]

            xc = x_coords[i_chan]  # (n_points,)
            yc = y_coords[i_chan]

            for i_pol in range(n_pol):
                # weight may have fewer pol planes than vis_data;
                # use min() to broadcast the last weight pol plane
                wt_pol = min(i_pol, n_wt_pol - 1)
                if do_psf:
                    strengths = (
                        weight[:, :, i_chan, wt_pol].ravel().astype(np.complex128)
                    )
                else:
                    strengths = (
                        vis_data[:, :, i_chan, i_pol].ravel()
                        * weight[:, :, i_chan, wt_pol].ravel()
                    ).astype(np.complex128)

                # Zero out NaN entries
                strengths[nan_mask] = 0.0 + 0.0j

                # Also zero any NaN strengths (e.g. NaN vis * 0 weight = NaN)
                nan_strength = np.isnan(strengths)
                strengths[nan_strength] = 0.0 + 0.0j

                # Combined mask: skip points with NaN UVW or NaN strengths
                skip_mask = nan_mask | nan_strength

                # Accumulate per-plane sum of weights
                wt = weight[:, :, i_chan, wt_pol].ravel().copy()
                wt[skip_mask] = 0.0
                sum_weight[a_chan, i_pol] += np.sum(wt)

                # Type-1 NUFFT: non-uniform → uniform
                #
                # FINUFFT type-1 (isign=+1, default) computes:
                #   f[k] = Sum c_j exp(+i k x_j)
                #
                # Our coordinates are x = -2π·cell·u_λ, so:
                #   f[k] = Sum c_j exp(-2πi k·cell·u_λ)
                #
                # Setting l = k·cell, this is the dirty image:
                #   I(l) = Sum c_j exp(-2πi u_λ l)
                #
                # matching the standard gridder's IFFT convention.
                # No coordinate negation or post-IFFT is needed.
                valid = ~skip_mask
                if not np.any(valid):
                    continue

                # Apply the 2D type-1 NUFFT using the precomputed image-plane
                # coordinates (see the construction of x_coords / y_coords
                # earlier in this method). Those coordinates already encode
                # the sign and 2π·cell scaling described above, so no extra
                # coordinate negation or post-processing is required here.
                grid_plane = self._nufft2d1(
                    xc[valid],
                    yc[valid],
                    strengths[valid],
                    (ny, nx),
                    self.eps,
                )
                grid[a_chan, i_pol, :, :] += grid_plane

        return grid, sum_weight

    # --------------------------------------------------------------------- #
    #  degrid (type-2 NUFFT)                                                 #
    # --------------------------------------------------------------------- #

    def degrid(
        self,
        grid: npt.NDArray[np.complexfloating],
        vis: xarray.core.datatree.DataTree,
        *,
        incremental: bool = False,
    ) -> None:

        uvw = vis.UVW.data
        dims = vis.dims
        n_time = dims["time"]
        n_baseline = dims["baseline_id"]
        n_freq = dims["frequency"]
        n_pol = dims["polarization"]
        n_points = n_time * n_baseline

        freq_chan = vis.frequency.data

        # Prepare model visibility output
        if not incremental or "VISIBILITY_MODEL" not in vis:
            modvis = np.zeros((n_time, n_baseline, n_freq, n_pol), dtype=np.complex128)
        else:
            modvis = vis.VISIBILITY_MODEL.data.copy()

        # Flag handling
        flag = vis.FLAG.data.copy()
        uvw_work = uvw.copy()
        uvwmask = np.isnan(uvw_work)
        nan_it, nan_ib = np.where(
            uvwmask.any(axis=-1)
            if uvwmask.ndim == 3
            else np.zeros((n_time, n_baseline), dtype=bool)
        )

        flag[nan_it, nan_ib, :, :] = True
        uvw_work[nan_it, nan_ib, :] = 0.0

        nan_mask = np.isnan(uvw_work[:, :, 0].ravel()) | np.isnan(
            uvw_work[:, :, 1].ravel()
        )

        # Compute FINUFFT coordinates
        x_coords, y_coords = self._uvw_to_finufft_coords(uvw_work, freq_chan)

        nchan_grid = grid.shape[0]
        npol_grid = grid.shape[1]

        # Channel / pol mapping (same logic as standard degrid)
        chanmap = (np.arange(n_freq) / n_freq * nchan_grid).astype(int)
        polmap = np.round(np.arange(n_pol) % npol_grid).astype(int)

        # Pre-compute image planes from the UV grid.
        # Degridding with NUFFT requires evaluating FT(image) at non-uniform
        # (u,v) points, so we first IFFT each UV-grid plane to image space,
        # then use type-2 NUFFT on the *image* (not the UV grid).
        image_cache = {}
        for ac in range(nchan_grid):
            for ap in range(npol_grid):
                grid_plane = np.ascontiguousarray(
                    grid[ac, ap, :, :].astype(np.complex128)
                )
                image_cache[(ac, ap)] = np.ascontiguousarray(
                    np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_plane)))
                )

        for i_chan in range(n_freq):
            a_chan = chanmap[i_chan]
            xc = x_coords[i_chan]
            yc = y_coords[i_chan]

            # Use polarization 0 to form an initial per-channel validity mask.
            # This coarse mask is shared across all polarizations and is then
            # refined per polarization below via `pol_valid`, which additionally
            # applies the flags for each `i_pol`. The use of pol=0 here is
            # intentional (not an off-by-one error) and assumes that the first
            # polarization provides a representative channel-level flag mask.
            #
            # Note: if polarization 0 is heavily flagged while other
            # polarizations have mostly valid data, this choice will cause
            # those samples to be treated as invalid for *all* polarizations,
            # since `valid` already excludes them before the per-pol refinement.
            # This behavior is currently by design (pol 0 acts as a reference
            # quality gate). An alternative approach would be to construct
            # `valid` from a combined flag mask across polarizations (e.g.,
            # requiring that at least one polarization is unflagged), but that
            # would change which samples are included and is therefore not done
            # here.
            valid = ~nan_mask & ~flag[:, :, i_chan, 0].ravel()

            for i_pol in range(n_pol):
                a_pol = polmap[i_pol]

                # Per-pol flag refinement
                pol_valid = valid & ~flag[:, :, i_chan, i_pol].ravel()

                if not np.any(pol_valid):
                    continue

                image_plane = image_cache[(a_chan, a_pol)]

                # Type-2 NUFFT on the *image* to evaluate FT(image) at
                # non-uniform (u,v) points.
                #
                # FINUFFT type-2 (default isign=-1) computes:
                #   c_j = sum_k f_k exp(-i k x_j)
                #
                # We need:
                #   V(u) = sum_n I_n exp(-i 2pi n cell u_lambda)
                #
                # Our coordinates are x = -2pi cell u_lambda, so passing
                # -x = +2pi cell u_lambda gives:
                #   c_j = sum_n I_n exp(-i n (+2pi cell u_lambda)) = V(u)
                vals = self._nufft2d2(
                    xc[pol_valid],
                    yc[pol_valid],
                    image_plane,
                    self.eps,
                )

                # Write back into the (time, baseline) shaped array
                result_flat = np.zeros(n_points, dtype=np.complex128)
                result_flat[pol_valid] = vals
                modvis[:, :, i_chan, i_pol] += result_flat.reshape(n_time, n_baseline)

        dat_dims = vis.VISIBILITY.dims
        dat_coords = vis.VISIBILITY.coords
        vis["VISIBILITY_MODEL"] = xarray.DataArray(
            modvis, coords=dat_coords, dims=dat_dims
        )

    def grid_to_image(self, vis, resid_array, *, do_psf=False, column="VISIBILITY"):
        """Grid visibilities to a dirty image.

        For FINUFFT, the type-1 NUFFT with negated coordinates directly
        produces the dirty image in mode ordering (DC at array center).
        No IFFT or spheroidal correction is needed — just normalize by
        the sum of weights.

        The FINUFFT type-1 already produces an *ifftshift'd* grid that is
        in dirty map form for our purposes, so we skip both the gridding kernel
        correction or any additional FFT and simply normalize FINUFFT output
        by the sum of weights.
        """
        if isinstance(vis, xarray.core.datatree.DataTree):
            list_vis = [vis]
        else:
            list_vis = vis

        grid_accum = None
        sumwt = None

        for ds in list_vis:
            vis_data = ds[column].data
            uvw = ds.UVW.data
            weight = ds.WEIGHT.data.copy()
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

        # The type-1 NUFFT output (using negated UVW coordinates computed in
        # `_uvw_to_finufft_coords`) IS the dirty image — just normalize by
        # sum of weights.
        for chan in range(resid_array.shape[0]):
            for corr in range(resid_array.shape[1]):
                resid_array[chan, corr, :, :] = (
                    np.real(grid_accum[chan, corr, :, :]) / sumwt[chan, corr]
                )
