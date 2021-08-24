# -*- coding: utf-8 -*-

from matplotlib import cm, pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from pprint import pprint
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import io, rcparams
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field
from pysteps.io import import_rmi_nwp_xr
from pysteps.blending.utils import reprojection

# Input
date_str = "202107141000"  # input("Give date and time (e.g.: 201609281600):    ")
n_timesteps = 36
start_blending = 60
end_blending = 120

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
importer_kwargs = data_source["importer_kwargs"]

# for _ in range(168):

filename = os.path.join(
    root_path,
    datetime.strftime(date, path_fmt),
    datetime.strftime(date, fn_pattern) + "." + fn_ext,
)

R_radar = io.import_odim_hdf5(filename, "RATE", **importer_kwargs)

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

"""
# Plot the rainfall field
plot_precip_field(R_radar, geodata=geodata_radar)
plt.title("Radar image at {}".format(datetime.strftime(date, "%Y-%m-%d %H:%M:%S")))
plt.savefig("./images/1_radar_image_{}.png".format(datetime.strftime(date, "%Y%m%d%H%M%S")))
plt.close()
"""

# Plot the rainfall field
plt.figure(1)
plot_precip_field(R_radar, geodata=geodata_radar)
plt.title("Radar image at {}".format(datetime.strftime(date, "%Y-%m-%d %H:%M:%S")))

# date = date + timedelta(minutes=5)

# Import native NWP data
nwp_date_str = "202107140600"
nwp_date = datetime.strptime(nwp_date_str, "%Y%m%d%H%M")
R_NWP = import_rmi_nwp_xr(
    "./nwp/ao13_{}_native_5min.nc".format(datetime.strftime(nwp_date, "%Y%m%d%H"))
)

xpixelsize_NWP = R_NWP.attrs["xpixelsize"]
ypixelsize_NWP = R_NWP.attrs["ypixelsize"]

x_NWP = R_NWP.x
y_NWP = R_NWP.y

R_NWP.attrs["projection"] = R_NWP.attrs["projection"] + " lat_0=50.8"

geodata_NWP = {
    "projection": R_NWP.attrs["projection"],
    "x1": R_NWP.x.x1,
    "y1": R_NWP.y.y1,
    "x2": R_NWP.x.x2,
    "y2": R_NWP.y.y2,
    "yorigin": R_NWP.attrs["yorigin"],
}

R_NWP.attrs["accutime"] = 5.0
R_NWP[:], R_NWP.attrs = conversion.to_rainrate(R_NWP[:], R_NWP.attrs)

t0_str = "202107141005"
t0 = datetime.strptime(t0_str, "%Y%m%d%H%M")

# Plot reprojected NWP data
R_NWP_rprj = reprojection(R_NWP[0, :, :], R_radar)

plt.figure(2)
plot_precip_field(R_NWP[0, :, :], geodata=geodata_NWP)
plt.title("NWP data at {}".format(datetime.strftime(t0, "%Y-%m-%d %H:%M:%S")))

# Plot reprojected NWP data
plt.figure(3)
plot_precip_field(R_NWP_rprj, geodata=geodata_radar)
plt.title(
    "Reprojected NWP data at {}".format(datetime.strftime(t0, "%Y-%m-%d %H:%M:%S"))
)

plt.close()

"""
for i in range(R_NWP.shape[0]):
    # Plot native NWP data
    t = t0 + i * timedelta(minutes=5)
    print(t)

    plot_precip_field(R_NWP[i, :, :], geodata=geodata_NWP)
    plt.title("NWP data at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S")))
    plt.savefig("./images/2_nwp_data_{}.png".format(datetime.strftime(t, "%Y%m%d%H%M%S")))
    plt.close()

    # Reproject NWP data onto radar domain
    R_NWP_rprj = reprojection(R_NWP[i, :, :], R_radar)

    # Plot reprojected NWP data
    plot_precip_field(R_NWP_rprj, geodata=geodata_radar)
    plt.title("Reprojected NWP data at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S")))
    plt.savefig("./images/3_reprojected_nwp_data_{}.png".format(datetime.strftime(t, "%Y%m%d%H%M%S")))
    plt.close()
"""

# Plot dummy NWP data


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


R_dummy = dummy_nwp(R_radar, n_timesteps)

for i in range(n_timesteps):
    t = date + timedelta(minutes=5) * (i + 1)
    print(t)

    plot_precip_field(R_dummy[i, :, :], geodata=geodata_radar)
    plt.title("Dummy NWP data at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S")))
    plt.savefig(
        "./images/4_dummy_nwp_data_{}.png".format(datetime.strftime(t, "%Y%m%d%H%M%S"))
    )
    plt.close()
