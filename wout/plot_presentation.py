# -*- coding: utf-8 -*-

from matplotlib import cm, pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from pprint import pprint
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import io, rcparams, verification
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils import conversion, transformation, dimension
from pysteps.visualization import plot_precip_field
from pysteps.io import import_rmi_nwp_xr
from pysteps.blending.utils import reprojection
from pysteps.nowcasts.linear_blending import forecast
from pysteps.postprocessing import ensemblestats
from pysteps.motion.lucaskanade import dense_lucaskanade


def gaussian(x, max, mean, sigma):
    return max * np.exp(-(x - mean) * (x - mean) / sigma / sigma / 2)


def dummy_nwp(R, n_leadtimes, max=15, mean=3, sigma=0.25, speed=100):
    """Generates dummy NWP data with the same dimension as the input
    precipitation field R. The NWP data is a vertical line with a Gaussian
    profile moving to the left"""

    # R is original radar image
    rows = R.shape[0]
    cols = R.shape[1]

    # Initialise the dummy NWP data
    R_nwp = np.zeros((n_leadtimes, rows, cols))
    x = np.linspace(-5, 5, cols)

    for n in range(n_leadtimes):
        for i in range(rows):
            R_nwp[n, i, :] = gaussian(x, max, mean, sigma)
        mean -= speed / rows

    return R_nwp


def linear_blending(
    R_nowcast,
    timesteps,
    timestep,
    R_nwp=None,
    start_blending=120,
    end_blending=240,
    use_nwp=True,
):
    """This function the same as the forecast function from linear blending.
    The only difference being that it is now a simple post-processing function,
    taking the nowcast fields as input"""

    # Check if NWP data is given as input
    if R_nwp is not None:

        if len(R_nowcast.shape) == 4:
            n_ens_members_nowcast = R_nowcast.shape[0]
            if n_ens_members_nowcast == 1:
                R_nowcast = np.squeeze(R_nowcast)
        else:
            n_ens_members_nowcast = 1

        if len(R_nwp.shape) == 4:
            n_ens_members_nwp = R_nwp.shape[0]
            if n_ens_members_nwp == 1:
                R_nwp = np.squeeze(R_nwp)
        else:
            n_ens_members_nwp = 1

        n_ens_members_max = max(n_ens_members_nowcast, n_ens_members_nwp)
        n_ens_members_min = min(n_ens_members_nowcast, n_ens_members_nwp)

        if n_ens_members_min != n_ens_members_max:
            if n_ens_members_nwp == 1:
                R_nwp = np.repeat(R_nwp[np.newaxis, :, :], n_ens_members_max, axis=0)
            elif n_ens_members_nowcast == 1:
                R_nowcast = np.repeat(
                    R_nowcast[np.newaxis, :, :], n_ens_members_max, axis=0
                )
            else:
                repeats = [
                    (n_ens_members_max + i) // n_ens_members_min
                    for i in range(n_ens_members_min)
                ]

                if n_ens_members_nwp == n_ens_members_min:
                    R_nwp = np.repeat(R_nwp, repeats, axis=0)
                elif n_ens_members_nowcast == n_ens_members_min:
                    R_nowcast = np.repeat(R_nowcast, repeats, axis=0)

        # Check if dimensions are correct
        assert (
            R_nwp.shape == R_nowcast.shape
        ), "The dimensions of R_nowcast and R_nwp need to be identical: dimension of R_nwp = {} and dimension of R_nowcast = {}".format(
            R_nwp.shape, R_nowcast.shape
        )

        # Initialise output
        R_blended = np.zeros_like(R_nowcast)

        # Calculate the weights
        for i in range(timesteps):
            # Calculate what time we are at
            t = (i + 1) * timestep

            # Calculate the weight with a linear relation (weight_nwp at start_blending = 0.0)
            # and (weight_nwp at end_blending = 1.0)
            weight_nwp = (t - start_blending) / (end_blending - start_blending)

            # Set weights at times before start_blending and after end_blending
            if weight_nwp < 0.0:
                weight_nwp = 0.0
            elif weight_nwp > 1.0:
                weight_nwp = 1.0

            # Calculate weight_nowcast
            weight_nowcast = 1.0 - weight_nwp

            # Calculate output by combining R_nwp and R_nowcast,
            # while distinguishing between ensemble and non-ensemble methods
            if n_ens_members_max == 1:
                R_blended[i, :, :] = (
                    weight_nwp * R_nwp[i, :, :] + weight_nowcast * R_nowcast[i, :, :]
                )
            else:
                R_blended[:, i, :, :] = (
                    weight_nwp * R_nwp[:, i, :, :]
                    + weight_nowcast * R_nowcast[:, i, :, :]
                )

        # Find where the NaN values are and replace them with NWP data
        if use_nwp:
            nan_indices = np.isnan(R_blended)
            R_blended[nan_indices] = R_nwp[nan_indices]
    else:
        # If no NWP data is given, the blended field is simply equal to the nowcast field
        R_blended = R_nowcast

    return R_blended


########################################################################
# 0. Input
date_str = "202107141400"  # input("Give date and time (e.g.: 201609281600):    ")
n_timesteps = 96
start_blending = 120
end_blending = 360
n_ens_members = 24
plot_1 = True
plot_2 = True
plot_3 = True
plot_4 = True
plot_5 = True
plot_5_1 = True
plot_6 = True
plot_7 = True
plot_8 = True
plot_9 = True

########################################################################
# 1. Plot radar fields

# Convert date to string
date = datetime.strptime(date_str, "%Y%m%d%H%M")

# Import the data source
data_source = rcparams.data_sources["rmi"]

# Import the radar composite
root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
timestep = data_source["timestep"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]

# Find the radar files in the archive
fns_radar = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=n_timesteps
)

# Read the data from the archive
importer = io.get_method(importer_name, "importer")
R_radar = io.read_timeseries(fns_radar, importer, **importer_kwargs)

# Convert to rain rate
R_radar.attrs["accutime"] = 5.0
R_radar[:], R_radar.attrs = conversion.to_rainrate(R_radar[:], R_radar.attrs)

# Extract geodata to plot
geodata_radar = {
    "projection": R_radar.attrs["projection"],
    "x1": R_radar.x.x1,
    "y1": R_radar.y.y1,
    "x2": R_radar.x.x2,
    "y2": R_radar.y.y2,
    "yorigin": R_radar.attrs["yorigin"],
}

if plot_1:
    for i in range(n_timesteps + 1):
        # Plot the radar rainfall field
        t = date + i * timedelta(minutes=5)
        print(t)
        print(R_radar.t[i])

        plot_precip_field(R_radar[i, :, :], geodata=geodata_radar)
        plt.title("Radar image at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S")))
        plt.savefig(
            "./images/linear_blending/1_radar_image_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

########################################################################
# 2. Plot native NWP fields
# 3. Plot reprojected NWP fields

# Import native NWP data
nwp_date_str = "202107140600"
nwp_date = datetime.strptime(nwp_date_str, "%Y%m%d%H%M")
R_NWP = import_rmi_nwp_xr(
    "./nwp/ao13_{}_native_5min.nc".format(datetime.strftime(nwp_date, "%Y%m%d%H"))
)

# Modify geodata
R_NWP.attrs["projection"] = R_NWP.attrs["projection"] + " lat_0=50.8"

geodata_NWP = {
    "projection": R_NWP.attrs["projection"],
    "x1": R_NWP.x.x1,
    "y1": R_NWP.y.y1,
    "x2": R_NWP.x.x2,
    "y2": R_NWP.y.y2,
    "yorigin": R_NWP.attrs["yorigin"],
}

# Convert to rain rate
R_NWP.attrs["accutime"] = 5.0
R_NWP[:], R_NWP.attrs = conversion.to_rainrate(R_NWP[:], R_NWP.attrs)

# Find start index corresponding with date
t0_str = "202107141005"
t0 = datetime.strptime(t0_str, "%Y%m%d%H%M")
start_i = (date - t0) // timedelta(minutes=5)
print("start_i = {}".format(start_i))

# Initialise reprojection array
R_NWP_rprj = np.zeros((n_timesteps + 1, R_radar.shape[1], R_radar.shape[2]))

for i in range(start_i, start_i + n_timesteps + 1):
    if plot_2:
        # Plot native NWP data
        t = t0 + i * timedelta(minutes=5)
        print(t)
        print(R_NWP.t[i])

        plot_precip_field(R_NWP[i, :, :], geodata=geodata_NWP)
        plt.title("NWP data at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S")))
        plt.savefig(
            "./images/linear_blending/2_nwp_data_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

    # Reproject NWP data onto radar domain
    R_NWP_rprj[i - start_i, :, :] = reprojection(R_NWP[i, :, :], R_radar[0, :, :])

    if plot_3:
        # Plot reprojected NWP data
        plot_precip_field(R_NWP_rprj[i - start_i, :, :], geodata=geodata_radar)
        plt.title(
            "Reprojected NWP data at {}".format(
                datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
            )
        )
        plt.savefig(
            "./images/linear_blending/3_reprojected_nwp_data_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

########################################################################
# 4. Plot dummy NWP data

R_dummy = dummy_nwp(R_radar[0, :, :], n_timesteps + 1)

if plot_4:
    for i in range(n_timesteps + 1):
        t = date + timedelta(minutes=5) * i
        print(t)

        plot_precip_field(R_dummy[i, :, :], geodata=geodata_radar)
        plt.title(
            "Dummy NWP data at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S"))
        )
        plt.savefig(
            "./images/linear_blending/4_dummy_nwp_data_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

########################################################################
# 5. Calculate and plot STEPS nowcast

# Find the radar files in the archive
fns = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the data from the archive
importer = io.get_method(importer_name, "importer")
R_input, _, metadata_input = io.read_timeseries(
    fns, importer, legacy=True, **importer_kwargs
)

# Convert to rain rate
metadata_input["accutime"] = 5.0
R_input, metadata_input = conversion.to_rainrate(R_input, metadata_input)

# Upscale data to 2 km to limit memory usage
print(R_input.shape)
R_input, metadata_input = dimension.aggregate_fields_space(
    R_input, metadata_input, metadata_input["xpixelsize"] * 3
)

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
R_input, metadata_input = transformation.dB_transform(
    R_input, metadata_input, threshold=0.1, zerovalue=-15.0
)

# Nicely print metadata
pprint(metadata_input)

# Estimate the motion field
V = dense_lucaskanade(R_input)

# Define nowcast keyword arguments
nowcast_kwargs = {
    "n_ens_members": n_ens_members,
    "n_cascade_levels": 6,
    "R_thr": -10.0,
    "kmperpixel": metadata_input["xpixelsize"] * 3 / 1000.0,
    "timestep": 5,
    "noise_method": "nonparametric",
    "vel_pert_method": None,  # "bps",
    "mask_method": "incremental",
}

# Calculate the STEPS nowcasts
R_steps = forecast(
    R_input[-3:, :, :],
    V,
    n_timesteps,
    5,
    "steps",
    R_nwp=None,
    start_blending=start_blending,
    end_blending=end_blending,
    use_nwp=True,
    nowcast_kwargs=nowcast_kwargs,
)

# Calculate the mean
R_steps_mean = np.mean(R_steps[:, :, :, :], axis=0)

# Plot the STEPS nowcasts
if plot_5:
    for i in range(n_timesteps):
        t = date + i * timedelta(minutes=5) + timedelta(minutes=5)
        print(t)

        plot_precip_field(R_steps_mean[i, :, :], geodata=geodata_radar)
        plt.title(
            "STEPS forecast at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S"))
        )
        plt.savefig(
            "./images/linear_blending/5_steps_forecast_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

########################################################################
# 5.1 Plot STEPS forecast with NWP data instead of mask

# Downscale the reprojected NWP data
R_NWP_rprj, _ = dimension.aggregate_fields_space(
    R_NWP_rprj, metadata_input, metadata_input["xpixelsize"] * 3
)

# Calculate the blended field
R_steps_1 = linear_blending(
    R_steps,
    n_timesteps,
    5,
    R_nwp=R_NWP_rprj[1:, :, :],
    start_blending=n_timesteps * 5 + 1,
    end_blending=n_timesteps * 5 + 2,
)

# Calculate the mean
R_steps_mean_1 = np.mean(R_steps_1[:, :, :, :], axis=0)

if plot_5_1:
    # Plot the STEPS nowcasts
    for i in range(n_timesteps):
        t = date + i * timedelta(minutes=5) + timedelta(minutes=5)
        print(t)

        plot_precip_field(R_steps_mean_1[i, :, :], geodata=geodata_radar)
        plt.title(
            "STEPS forecast with NWP data at {}".format(
                datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
            )
        )
        plt.savefig(
            "./images/linear_blending/5_1_steps_forecast_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

########################################################################
# 6. Blending with start=0 and end=end and dummy data

# Downscale the dummy NWP data
R_dummy, _ = dimension.aggregate_fields_space(
    R_dummy, metadata_input, metadata_input["xpixelsize"] * 3
)

# Calculate the blended field
R_blended_1 = linear_blending(
    R_steps,
    n_timesteps,
    5,
    R_nwp=R_dummy[1:, :, :],
    start_blending=0,
    end_blending=n_timesteps * 5,
    use_nwp=False,
)

# Calculate the mean
R_blended_mean_1 = np.mean(R_blended_1[:, :, :, :], axis=0)

if plot_6:
    # Plot the blended fields
    for i in range(n_timesteps):
        t = date + i * timedelta(minutes=5) + timedelta(minutes=5)
        print(t)

        plot_precip_field(R_blended_mean_1[i, :, :], geodata=geodata_radar)
        plt.title(
            "Blended forecast at {} (version 1)".format(
                datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
            )
        )
        plt.savefig(
            "./images/linear_blending/6_blended_forecast_version_1_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()

########################################################################
# 7. Blending with start and end and dummy data

# Calculate the blended fields
R_blended_2 = linear_blending(
    R_steps,
    n_timesteps,
    5,
    R_nwp=R_dummy[1:, :, :],
    start_blending=start_blending,
    end_blending=end_blending,
)

# Calculate the mean
R_blended_mean_2 = np.mean(R_blended_2[:, :, :, :], axis=0)


if plot_7:
    # Plot the blended fields
    for i in range(n_timesteps):
        t = date + i * timedelta(minutes=5) + timedelta(minutes=5)
        print(t)

        plot_precip_field(R_blended_mean_2[i, :, :], geodata=geodata_radar)
        plt.title(
            "Blended forecast at {} (version 2)".format(
                datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
            )
        )
        plt.savefig(
            "./images/linear_blending/7_blended_forecast_version_2_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()


########################################################################
# 8. Blending with start and end and real NWP data

# Calculate the blended fields
R_blended_3 = linear_blending(
    R_steps,
    n_timesteps,
    5,
    R_nwp=R_NWP_rprj[1:, :, :],
    start_blending=start_blending,
    end_blending=end_blending,
)

# Calculate the mean
R_blended_mean_3 = np.mean(R_blended_3[:, :, :, :], axis=0)

if plot_8:
    # Plot the blended fields
    for i in range(n_timesteps):
        t = date + i * timedelta(minutes=5) + timedelta(minutes=5)
        print(t)

        plot_precip_field(R_blended_mean_3[i, :, :], geodata=geodata_radar)
        plt.title(
            "Blended forecast at {} (version 3)".format(
                datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
            )
        )
        plt.savefig(
            "./images/linear_blending/8_blended_forecast_version_3_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()


########################################################################
# 9. Blending with start and end and real NWP data

# Calculate the blended fields
R_blended_4 = linear_blending(
    R_steps,
    n_timesteps,
    5,
    R_nwp=R_NWP_rprj[1:, :, :],
    start_blending=start_blending,
    end_blending=end_blending,
    use_nwp=False,
)

# Calculate the mean
R_blended_mean_4 = np.mean(R_blended_4[:, :, :, :], axis=0)

if plot_9:
    # Plot the blended fields
    for i in range(n_timesteps):
        t = date + i * timedelta(minutes=5) + timedelta(minutes=5)
        print(t)

        plot_precip_field(R_blended_mean_4[i, :, :], geodata=geodata_radar)
        plt.title(
            "Blended forecast at {} (version 4)".format(
                datetime.strftime(t, "%Y-%m-%d %H:%M:%S")
            )
        )
        plt.savefig(
            "./images/linear_blending/9_blended_forecast_version_4_{}.png".format(
                datetime.strftime(t, "%Y%m%d%H%M%S")
            ),
            dpi=300,
        )
        plt.close()


########################################################################
# 10. Verification

# Convert radar images to np.ndarray
R_radar = np.array(R_radar[:])

# Downscale radar images
R_radar, _ = dimension.aggregate_fields_space(
    R_radar, metadata_input, metadata_input["xpixelsize"] * 3
)

########################################################################
# BLENDED FORECAST
########################################################################

# compute the exceedance probability of 0.1 mm/h from the ensemble
P_blended_3 = ensemblestats.excprob(R_blended_3[:, -1, :, :], 0.1, ignore_nan=True)

###############################################################################
# ROC curve
# ~~~~~~~~~

roc = verification.ROC_curve_init(0.1, n_prob_thrs=11)
verification.ROC_curve_accum(roc, P_blended_3, R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_ROC(roc, ax, opt_prob_thr=True)
ax.set_title("ROC curve (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/blended_ROC_curve.png", dpi=300)
plt.close()

###############################################################################
# Reliability diagram
# ~~~~~~~~~~~~~~~~~~~

reldiag = verification.reldiag_init(0.1)
verification.reldiag_accum(reldiag, P_blended_3, R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_reldiag(reldiag, ax)
ax.set_title("Reliability diagram (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/blended_Reliability_diagram.png", dpi=300)
plt.close()

###############################################################################
# Rank histogram
# ~~~~~~~~~~~~~~

rankhist = verification.rankhist_init(R_blended_3.shape[0], 0.1)
verification.rankhist_accum(rankhist, R_blended_3[:, -1, :, :], R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_rankhist(rankhist, ax)
ax.set_title("Rank histogram (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/blended_Rank_histogram.png", dpi=300)
plt.close()

########################################################################
# BLENDED FORECAST with use_nwp = False
########################################################################

# compute the exceedance probability of 0.1 mm/h from the ensemble
P_blended_4 = ensemblestats.excprob(R_blended_4[:, -1, :, :], 0.1, ignore_nan=True)

###############################################################################
# ROC curve
# ~~~~~~~~~

roc = verification.ROC_curve_init(0.1, n_prob_thrs=11)
verification.ROC_curve_accum(roc, P_blended_4, R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_ROC(roc, ax, opt_prob_thr=True)
ax.set_title("ROC curve (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/blended_nodata_ROC_curve.png", dpi=300)
plt.close()

###############################################################################
# Reliability diagram
# ~~~~~~~~~~~~~~~~~~~

reldiag = verification.reldiag_init(0.1)
verification.reldiag_accum(reldiag, P_blended_4, R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_reldiag(reldiag, ax)
ax.set_title("Reliability diagram (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/blended__nodata_Reliability_diagram.png", dpi=300)
plt.close()

###############################################################################
# Rank histogram
# ~~~~~~~~~~~~~~

rankhist = verification.rankhist_init(R_blended_4.shape[0], 0.1)
verification.rankhist_accum(rankhist, R_blended_4[:, -1, :, :], R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_rankhist(rankhist, ax)
ax.set_title("Rank histogram (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/blended__nodata_Rank_histogram.png", dpi=300)
plt.close()

########################################################################
# STEPS NOWCAST
########################################################################

# compute the exceedance probability of 0.1 mm/h from the ensemble
P_steps = ensemblestats.excprob(R_steps[:, -1, :, :], 0.1, ignore_nan=True)

###############################################################################
# ROC curve
# ~~~~~~~~~

roc = verification.ROC_curve_init(0.1, n_prob_thrs=11)
verification.ROC_curve_accum(roc, P_steps, R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_ROC(roc, ax, opt_prob_thr=True)
ax.set_title("ROC curve (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/STEPS_ROC_curve.png", dpi=300)
plt.close()

###############################################################################
# Reliability diagram
# ~~~~~~~~~~~~~~~~~~~

reldiag = verification.reldiag_init(0.1)
verification.reldiag_accum(reldiag, P_steps, R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_reldiag(reldiag, ax)
ax.set_title("Reliability diagram (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/STEPS_Reliability_diagram.png", dpi=300)
plt.close()

###############################################################################
# Rank histogram
# ~~~~~~~~~~~~~~

rankhist = verification.rankhist_init(R_steps.shape[0], 0.1)
verification.rankhist_accum(rankhist, R_steps[:, -1, :, :], R_radar[-1, :, :])
fig, ax = plt.subplots()
verification.plot_rankhist(rankhist, ax)
ax.set_title("Rank histogram (+%i min)" % (n_timesteps * 5))
plt.savefig("./images/verification/STEPS_Rank_histogram.png", dpi=300)
plt.close()

################################################################################
# CRPS score
# ~~~~~~~~~~

timesteps = np.arange(5, (n_timesteps + 1) * 5, 5)
CRPS_blended_4 = np.zeros(n_timesteps)
CRPS_steps = np.zeros(n_timesteps)
MAE_NWP_mask = np.zeros(n_timesteps)

R_NWP_rprj_mask = np.copy(R_NWP_rprj[1:, :, :])
nan_indices = np.isnan(R_blended_mean_4)
R_NWP_rprj_mask[nan_indices] = np.nan

# Plot R_NWP_rprj_mask


for i in range(n_timesteps):
    CRPS_steps[i] = verification.CRPS(R_steps[:, i, :, :], R_radar[1 + i, :, :])
    CRPS_blended_4[i] = verification.CRPS(R_blended_4[:, i, :, :], R_radar[1 + i, :, :])
    det_cont_mask = verification.det_cont_fct(
        R_NWP_rprj_mask[i, :, :], R_radar[i + 1, :, :], "MAE"
    )
    MAE_NWP_mask[i] = det_cont_mask["MAE"]

plt.figure()
plt.plot(timesteps, CRPS_blended_4, label="Blended")
plt.plot(timesteps, CRPS_steps, label="STEPS")
plt.plot(timesteps, MAE_NWP_mask, label="NWP")
plt.title("CRPS score")
plt.ylabel("CRPS")
plt.xlabel("time (min)")
plt.legend()
plt.savefig("./images/verification/CRPS.png", dpi=300)
plt.show()
plt.close()
