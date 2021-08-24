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


# Give date as input
date_str = "202107141000"  # input("Give date and time (e.g.: 201609281600):    ")
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

# date = date + timedelta(minutes=5)

# Import native NWP data
date_str = "202107140600"
date = datetime.strptime(date_str, "%Y%m%d%H%M")
R_NWP = import_rmi_nwp_xr(
    "./nwp/ao13_{}_native_5min.nc".format(datetime.strftime(date, "%Y%m%d%H"))
)

xpixelsize_NWP = R_NWP.attrs["xpixelsize"]
ypixelsize_NWP = R_NWP.attrs["ypixelsize"]

x_NWP = R_NWP.x
y_NWP = R_NWP.y

geodata_NWP = {
    "projection": R_NWP.attrs["projection"] + " lat_0=50.8",
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

"""
# Plot native NWP data
for i in range(R_NWP.shape[0]):
	t = t0 + i * timedelta(minutes=5)
	print(t)
	
	plot_precip_field(R_NWP[i, :, :], geodata=geodata_NWP)
	plt.title("NWP data at {}".format(datetime.strftime(t, "%Y-%m-%d %H:%M:%S")))
	plt.savefig("./images/2_nwp_data_{}.png".format(datetime.strftime(t, "%Y%m%d%H%M%S")))
	plt.close()
"""

# Reproject NWP data onto radar domain

pprint(geodata_radar)
pprint(geodata_NWP)
