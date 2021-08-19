from pysteps.io import import_knmi_hdf5, import_netcdf_pysteps
from pprint import pprint
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
import datetime
import netCDF4
import numpy as np


# 1. Importing and plotting radar data

d = datetime.datetime(2018, 1, 5, 9, 0)

for _ in range(133):

    R = import_knmi_hdf5(
        "/home/wdewettin/Downloads/KNMI/20180105/Uncorrected_Radar/RAD_NL25_RAP_5min_20180105{}.h5".format(
            d.strftime("%H%M")
        )
    )

    geodata = {
        "projection": R.attrs["projection"],
        "x1": R.x.x1,
        "y1": R.y.y1,
        "x2": R.x.x2,
        "y2": R.y.y2,
        "yorigin": R.attrs["yorigin"],
    }

    time = d.strftime("%Y%m%d_%H%M")

    plot_precip_field(R, ptype="depth", units="mm", geodata=geodata)
    plt.savefig("KNMI_radar_{}.png".format(time))
    plt.close()

    print("plotted " + time)

    d = d + datetime.timedelta(minutes=5)

pprint(geodata)

# 2. Importing and plotting NWP data

ncf = netCDF4.Dataset(
    "/home/wdewettin/Downloads/KNMI/20180105/Harmonie/20180105_0600_Pforecast_Harmonie.nc"
)

R = ncf.variables["P_fc"]
time = ncf.variables["time"][:]
x = ncf.variables["x"]
y = ncf.variables["y"]
projection = ncf.variables["crs"].proj4_params

geodata = {
    "projection": projection,
    "x1": min(x),
    "y1": min(y),
    "x2": max(x),
    "y2": max(y),
    "yorigin": "lower",
}

pprint(geodata)

for i in range(R.shape[0]):
    valid_time = datetime.datetime(1970, 1, 1, 0, 0, 0) + datetime.timedelta(
        minutes=time[i]
    )
    valid_time = valid_time.strftime("%Y%m%d_%H%M")

    plot_precip_field(R[i, :, :], ptype="depth", units="mm", geodata=geodata)
    plt.savefig("KNMI_NWP_{}.png".format(valid_time))
    plt.close()

    print("plotted NWP " + valid_time)
