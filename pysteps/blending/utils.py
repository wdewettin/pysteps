# -*- coding: utf-8 -*-
"""
pysteps.blending.utils

Module with common utilities used by blending methods.

.. autosummary::
    :toctree: ../generated/

    stack_cascades
    recompose_cascade
"""

import numpy as np
from datetime import datetime, timedelta
from pysteps import cascade
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import utils
import os
import netCDF4
from pysteps.io import import_odim_hdf5, import_rmi_nwp_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint
from pysteps.utils.conversion import to_rainrate
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import numpy as np
import xarray as xr


def stack_cascades(R_d, donorm=True):
    """Stack the given cascades into a larger array.

    Parameters
    ----------
    R_d : list
      List of cascades obtained by calling a method implemented in
      pysteps.cascade.decomposition.
    donorm : bool
      If True, normalize the cascade levels before stacking.

    Returns
    -------
    out : tuple
      A three-element tuple containing a four-dimensional array of stacked
      cascade levels and arrays of mean values and standard deviations for each
      cascade level.
    """
    R_c = []
    mu_c = []
    sigma_c = []

    for cascade in R_d:
        R_ = []
        R_i = cascade["cascade_levels"]
        n_levels = R_i.shape[0]
        mu_ = np.ones(n_levels) * 0.0
        sigma_ = np.ones(n_levels) * 1.0
        if donorm:
            mu_ = np.asarray(cascade["means"])
            sigma_ = np.asarray(cascade["stds"])
        for j in range(n_levels):
            R__ = (R_i[j, :, :] - mu_[j]) / sigma_[j]
            R_.append(R__)
        R_c.append(np.stack(R_))
        mu_c.append(mu_)
        sigma_c.append(sigma_)
    return np.stack(R_c), np.stack(mu_c), np.stack(sigma_c)


def blend_optical_flows(flows, weights):
    """Combine advection fields using given weights.

    Parameters
    ----------
    flows : array-like
      A stack of multiple advenction fields having shape
      (S, 2, m, n), where flows[N, :, :, :] contains the motion vectors
      for source N.
      Advection fields for each source can be obtanined by
      calling any of the methods implemented in
      pysteps.motion and then stack all together
    weights : array-like
      An array of shape [number_sources]
      containing the weights to be used to combine
      the advection fields of each source.
      weights are modified to make their sum equal to one.
    Returns
    -------
    out: ndarray_
        Return the blended advection field having shape
        (2, m, n), where out[0, :, :] contains the x-components of
        the blended motion vectors and out[1, :, :] contains the y-components.
        The velocities are in units of pixels / timestep.
    """

    # check inputs
    if isinstance(flows, (list, tuple)):
        flows = np.stack(flows)

    if isinstance(weights, (list, tuple)):
        weights = np.asarray(weights)

    # check weights dimensions match number of sources
    num_sources = flows.shape[0]
    num_weights = weights.shape[0]

    if num_weights != num_sources:
        raise ValueError(
            "dimension mismatch between flows and weights.\n"
            "weights dimension must match the number of flows.\n"
            f"number of flows={num_sources}, number of weights={num_weights}"
        )
    # normalize weigths
    weights = weights / np.sum(weights)

    # flows dimension sources, 2, m, n
    # weights dimension sources
    # move source axis to last to allow broadcasting
    all_c_wn = weights * np.moveaxis(flows, 0, -1)
    # sum uses last axis
    combined_flows = np.sum(all_c_wn, axis=-1)
    # combined_flows [2, m, n]
    return combined_flows


def decompose_NWP(
    R_NWP,
    NWP_output,
    NWP_model,
    analysis_time,
    timestep,
    valid_times,
    num_cascade_levels,
    decomp_method="fft",
    fft_method="numpy",
    domain="spatial",
    normalize=True,
    compute_stats=True,
    compact_output=True,
):
    """Decomposed the NWP forecast data into cascades and saves it in
    a netCDF file

    Parameters
    ----------
    R_NWP: array-like
      Array of dimension (n_timesteps, x, y) containing the precipiation forecast
      from some NWP model.
    NWP_output: str
      The location where to save the file with the NWP cascade
    NWP_model: str
      The name of the NWP model
    analysis_time: numpy.datetime64
      The analysis time of the NWP forecast. The analysis time is assumed to be a
      numpy.datetime64 type as imported by the pysteps importer
    timestep: int
      Timestep in minutes between subsequent NWP forecast fields
    valid_times: array_like
      Array containing the valid times of the NWP forecast fields. The times are
      assumed to be numpy.datetime64 types as imported by the pysteps importer
    num_cascade_levels:
      The number of frequency bands to use. Must be greater than 2.

    Other Parameters
    ----------------
    decomp_method: str
      A string defining the decomposition method to use. Defaults to "fft".
    fft_method: str or tuple
      A string or a (function,kwargs) tuple defining the FFT method to use
      (see :py:func:`pysteps.utils.interface.get_method`).
      Defaults to "numpy". This option is not used if input_domain and
      output_domain are both set to "spectral".
    domain: {"spatial", "spectral"}
      If "spatial", the output cascade levels are transformed back to the
      spatial domain by using the inverse FFT. If "spectral", the cascade is
      kept in the spectral domain. Defaults to "spatial".
    normalize: bool
      If True, normalize the cascade levels to zero mean and unit variance.
      Requires that compute_stats is True. Implies that compute_stats is True.
      Defaults to False.
    compute_stats: bool
      If True, the output dictionary contains the keys "means" and "stds"
      for the mean and standard deviation of each output cascade level.
      Defaults to False.
    compact_output: bool
      Applicable if output_domain is "spectral". If set to True, only the
      parts of the Fourier spectrum with non-negligible filter weights are
      stored. Defaults to False.


    Returns
    -------
    Nothing
    """

    # Make a NetCDF file
    date_string = np.datetime_as_string(analysis_time, "s")
    outfn = os.path.join(
        NWP_output,
        "cascade_"
        + NWP_model
        + "_"
        + date_string[:4]
        + date_string[5:7]
        + date_string[8:10]
        + date_string[11:13]
        + date_string[14:16]
        + date_string[17:19]
        + ".nc",
    )
    ncf = netCDF4.Dataset(outfn, "w", format="NETCDF4")

    # Express times relative to the zero time
    zero_time = np.datetime64("1970-01-01T00:00:00", "ns")
    valid_times = np.array(valid_times) - zero_time
    analysis_time = analysis_time - zero_time

    # Set attributes of decomposition method
    ncf.domain = domain
    ncf.normalized = int(normalize)
    ncf.compact_output = int(compact_output)
    ncf.analysis_time = int(analysis_time)
    ncf.timestep = int(timestep)

    # Create dimensions
    time_dim = ncf.createDimension("time", R_NWP.shape[0])
    casc_dim = ncf.createDimension("cascade_levels", num_cascade_levels)
    x_dim = ncf.createDimension("x", R_NWP.shape[1])
    y_dim = ncf.createDimension("y", R_NWP.shape[2])

    # Create variables (decomposed cascade, means and standard deviations)
    R_d = ncf.createVariable("R_d", np.float64, ("time", "cascade_levels", "x", "y"))
    means = ncf.createVariable("means", np.float64, ("time", "cascade_levels"))
    stds = ncf.createVariable("stds", np.float64, ("time", "cascade_levels"))
    v_times = ncf.createVariable("valid_times", np.float64, ("time",))
    v_times.units = "nanoseconds since 1970-01-01 00:00:00"

    # The valid times are saved as an array of floats, because netCDF files can't handle datetime types
    v_times[:] = np.array([np.float64(valid_times[i]) for i in range(len(valid_times))])

    # Decompose the NWP data
    filter = filter_gaussian(R_NWP.shape[1:], num_cascade_levels)
    fft = utils.get_method(fft_method, shape=R_NWP.shape[1:], n_threads=1)
    decomp_method, _ = cascade.get_method(decomp_method)

    for i in range(R_NWP.shape[0]):
        R_ = decomp_method(
            R_NWP[i, :, :],
            filter,
            fft_method=fft,
            output_domain=domain,
            normalize=normalize,
            compute_stats=compute_stats,
            compact_output=compact_output,
        )

        # Save data to netCDF file
        R_d[i, :, :, :] = R_["cascade_levels"]
        means[i, :] = R_["means"]
        stds[i, :] = R_["stds"]

    # Close the file
    ncf.close()


def load_NWP(NWP_output, start_time, n_timesteps):
    """Loads the decomposed NWP data from the netCDF files

    Parameters
    ----------
    NWP_output: str
      Path to the saved netCDF files containing the decomposed NWP data
    start_time: numpy.datetime64
      The start time of the nowcasting. Assumed to be a numpy.datetime64 type
    n_timesteps: int
      Number of time steps to forecast

    Returns
    -------
    R_d: list
      A list of dictionaries with each element in the list corresponding to
      a different time step. Each dictionary has the same structure as the
      output of the decomposition function
    """

    # Open the file
    ncf = netCDF4.Dataset(NWP_output, "r", format="NETCDF4")

    # Initialise the decomposition dictionary
    decomp_dict = dict()
    decomp_dict["domain"] = ncf.domain
    decomp_dict["normalized"] = bool(ncf.normalized)
    decomp_dict["compact_output"] = bool(ncf.compact_output)

    # Convert the start time and the timestep to datetime64 and timedelta64 type
    zero_time = np.datetime64("1970-01-01T00:00:00", "ns")
    analysis_time = np.timedelta64(int(ncf.analysis_time), "ns") + zero_time

    timestep = ncf.timestep
    timestep = np.timedelta64(timestep, "m")

    valid_times = ncf.variables["valid_times"][:]
    valid_times = np.array(
        [np.timedelta64(int(valid_times[i]), "ns") for i in range(len(valid_times))]
    )
    valid_times = valid_times + zero_time

    # Add the valid times to the output
    decomp_dict["valid_times"] = valid_times

    # Find the indices corresponding with the required start and end time
    start_i = (start_time - analysis_time) // timestep
    assert analysis_time + start_i * timestep == start_time
    end_i = start_i + n_timesteps + 1

    # Initialise the list of dictionaries which will serve as the output (cf: the STEPS function)
    R_d = list()

    for i in range(start_i, end_i):
        decomp_dict_ = decomp_dict.copy()

        cascade_levels = ncf.variables["R_d"][i, :, :, :]

        # In the netcdf file this is saved as a masked array, so we're checking if there is no mask
        assert not cascade_levels.mask

        means = ncf.variables["means"][i, :]
        assert not means.mask

        stds = ncf.variables["stds"][i, :]
        assert not stds.mask

        # Save the values in the dictionary as normal arrays with the filled method
        decomp_dict_["cascade_levels"] = np.ma.filled(cascade_levels)
        decomp_dict_["means"] = np.ma.filled(means)
        decomp_dict_["stds"] = np.ma.filled(stds)

        # Append the output list
        R_d.append(decomp_dict_)

    return R_d


def reprojection(R_src, R_dst):
    src_crs = R_src.attrs["projection"]
    x_src = R_src.x
    y_src = R_src.y
    x1_src = x_src.x1
    y2_src = y_src.y2
    xpixelsize_src = R_src.attrs["xpixelsize"]
    ypixelsize_src = R_src.attrs["ypixelsize"]
    src_transform = A.translation(x1_src, y2_src) * A.scale(
        xpixelsize_src, -ypixelsize_src
    )

    dst_crs = R_dst.attrs["projection"]
    x_dst = R_dst.x
    y_dst = R_dst.y
    x1_dst = x_dst.x1
    y2_dst = y_dst.y2
    xpixelsize_dst = R_dst.attrs["xpixelsize"]
    ypixelsize_dst = R_dst.attrs["ypixelsize"]
    dst_transform = A.translation(x1_dst, y2_dst) * A.scale(
        xpixelsize_dst, -ypixelsize_dst
    )

    R_rprj = np.zeros_like(R_dst[:])

    reproject(
        R_src[:],
        R_rprj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        dst_nodata=np.nan,
    )

    return R_rprj
