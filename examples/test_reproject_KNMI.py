from pysteps.io import import_knmi_hdf5
from pprint import pprint
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
import datetime
import netCDF4
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling

# 1. Importing and plotting NWP data

ncf = netCDF4.Dataset(
    "/home/wdewettin/Downloads/KNMI/20180105/Harmonie/20180105_0600_Pforecast_Harmonie.nc"
)

pprint(ncf)

R_NWP = ncf.variables["P_fc"]
R_NWP = R_NWP[:, ::-1, :]  # Reverse the order of the rows
time = ncf.variables["time"][:]
x = ncf.variables["x"]
y = ncf.variables["y"]
projection = ncf.variables["crs"].proj4_params

geodata_NWP = {
    "projection": projection,
    "x1": min(x),
    "y1": min(y),
    "x2": max(x),
    "y2": max(y),
    "yorigin": "upper",
}

i = 6

valid_time = datetime.datetime(1970, 1, 1, 0, 0, 0) + datetime.timedelta(
    minutes=time[i]
)
valid_time = valid_time.strftime("%Y%m%d_%H%M")

plot_precip_field(R_NWP[i, :, :], ptype="depth", units="mm", geodata=geodata_NWP)
print("plotted NWP " + valid_time)
plt.title("Original NWP data")
plt.savefig("D.png")
plt.close()

# 1.B Setting NWP data grid info

NWP_transform = A.translation(x[0], y[0]) * A.scale(x[1] - x[0], y[1] - y[0])
print(
    "NWP_transform = A.translation({}, {}) * A.scale({}, {})".format(
        x[0], y[0], x[1] - x[0], y[1] - y[0]
    )
)
NWP_crs = {"init": "EPSG:4326"}
print("NWP_crs = {'init': 'EPSG:4326'}")

# 2. Importing and plotting radar data

d = datetime.datetime(2018, 1, 5, 12, 0)

R, _, metadata = import_knmi_hdf5(
    "/home/wdewettin/Downloads/KNMI/20180105/Uncorrected_Radar/RAD_NL25_RAP_5min_20180105{}.h5".format(
        d.strftime("%H%M")
    ),
    legacy=True,
)

print(R)
print("metadata")
pprint(metadata)

R = import_knmi_hdf5(
    "/home/wdewettin/Downloads/KNMI/20180105/Uncorrected_Radar/RAD_NL25_RAP_5min_20180105{}.h5".format(
        d.strftime("%H%M")
    )
)

"""
radar_crs = rasterio.crs.CRS.to_dict(rasterio.crs.CRS.from_proj4(R.attrs["projection"]))
radar_crs["y_0"] = 300.0
radar_crs["lat_0"] = 89.99999999
proj_str = rasterio.crs.CRS.to_proj4(rasterio.crs.CRS.from_dict(radar_crs))
"""

geodata = {
    "projection": R.attrs["projection"],
    "x1": R.x.x1,
    "y1": R.y.y1,
    "x2": R.x.x2,
    "y2": R.y.y2,
    "yorigin": R.attrs["yorigin"],
}

print("geodata")
pprint(geodata)

time = d.strftime("%Y%m%d_%H%M")

plot_precip_field(R, ptype="depth", units="mm", geodata=geodata)
print("plotted radar " + time)
plt.title("Original radar data")
plt.savefig("E.png")
plt.close()

# 2.B Setting radar data grid info

radar_transform = A.translation(R.x.x1 - 0.5, R.y.y1 - 0.5)  # Moet die 0.5 ???
# ("radar_transform = A.translation({}, {})".format((R.x.x1 + R.x.x2) / 2, (R.y.y1 + R.y.y2) / 2))


radar_crs = geodata["projection"]
radar_crs = rasterio.crs.CRS.to_dict(rasterio.crs.CRS.from_proj4(radar_crs))

for a in range(20):
    radar_crs["lat_0"] = (
        90 - 0.1 ** a
    )  # For some reason, it gives errors if lat_0 = 90.0

    """
    radar_crs["y_0"] = 300.0 # Change false northing
    """

    """
    # Poging met nieuwe projectiestring van Ruben!
    radar_crs = ' +proj=stere +lat_0=90 +lat_ts=60 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +rf=298.252840776255 +units=m +no_defs'
    radar_crs = rasterio.crs.CRS.to_dict(rasterio.crs.CRS.from_proj4(radar_crs))
    radar_crs["lat_0"] = 89.99999999  # For some reason, it gives errors if lat_0 = 90.0
    """

    print("radar_crs = {}".format(radar_crs))

    # 3. Reprojecting radar data

    R_NWP_rprj = np.zeros_like(R)

    reproject(
        R_NWP[i, :, :],
        R_NWP_rprj,
        src_transform=NWP_transform,
        src_crs=NWP_crs,
        dst_transform=radar_transform,
        dst_crs=radar_crs,
        resampling=Resampling.nearest,
        dst_nodata=np.nan,
    )

    plot_precip_field(R_NWP_rprj, ptype="depth", units="mm", geodata=geodata)
    plt.title("Reprojected NWP data with lat_0 = {}".format(radar_crs["lat_0"]))
    plt.savefig("F_{}.png".format(a))

    plt.close()
