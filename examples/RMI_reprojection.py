from pysteps.io import import_odim_hdf5, import_rmi_nwp_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint
from pysteps.utils.conversion import to_rainrate
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import numpy as np
import xarray as xr

"""
# Import radar data

R = import_odim_hdf5(
    "./rainbow/20210704/20210704160500.rad.becomp00.image.rate.beborder00_comp_sri.hdf"
)

geodata = {
    "projection": R.attrs["projection"],
    "x1": R.x.x1,
    "y1": R.y.y1,
    "x2": R.x.x2,
    "y2": R.y.y2,
    "yorigin": R.attrs["yorigin"],
}

pprint(geodata)

# Plot radar data

plt.figure(1)
plot_precip_field(R, geodata=geodata)
plt.title("Radar at 2021-07-04 16:05:00")
"""

# Import native NWP data

R_NWP = import_rmi_nwp_xr("./nwp/ao13_2021070412_native_5min.nc")

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

print(R_NWP.shape)
pprint(geodata_NWP)

R_NWP.attrs["accutime"] = 5.0
R_NWP[:], R_NWP.attrs = to_rainrate(R_NWP[:], R_NWP.attrs)

# Plot native NWP data

plt.figure(2)
plot_precip_field(R_NWP[0, :, :], geodata=geodata_NWP)
plt.title("NWP at 2021-07-04 16:05:00")

# Import reprojected NWP data

R_NWP_radar = import_rmi_nwp_xr("./nwp/ao13_2021070412_radar512_5min.nc")

xpixelsize_radar = R_NWP_radar.attrs["xpixelsize"]
ypixelsize_radar = R_NWP_radar.attrs["ypixelsize"]

x_radar = R_NWP_radar.x + 106329.0
y_radar = R_NWP_radar.y - 64009.0

geodata_NWP_radar = {
    "projection": R_NWP_radar.attrs["projection"],
    "x1": R_NWP_radar.x.x1 + 106329.0,
    "y1": R_NWP_radar.y.y1 - 64009.0,
    "x2": R_NWP_radar.x.x2 + 106329.0,
    "y2": R_NWP_radar.y.y2 - 64009.0,
    "yorigin": R_NWP_radar.attrs["yorigin"],
}

print(R_NWP_radar.shape)
pprint(geodata_NWP_radar)

R_NWP_radar.attrs["accutime"] = 5.0
R_NWP_radar[:], R_NWP_radar.attrs = to_rainrate(R_NWP_radar[:], R_NWP_radar.attrs)

# Plot reprojected NWP data

plt.figure(3)
plot_precip_field(R_NWP_radar[1, :, :], geodata=geodata_NWP_radar)
plt.title("reprojected NWP (by Lesley) at 2021-07-04 16:05:00")

# Reproject native NWP data

NWP_crs = geodata_NWP["projection"]
NWP_transform = A.translation(geodata_NWP["x1"], geodata_NWP["y2"]) * A.scale(
    xpixelsize_NWP, -ypixelsize_NWP
)
radar_crs = geodata_NWP_radar["projection"]
radar_transform = A.translation(
    geodata_NWP_radar["x1"], geodata_NWP_radar["y2"]
) * A.scale(xpixelsize_radar, -ypixelsize_radar)

# Reproject

R_NWP_rprj = np.zeros_like(R_NWP_radar[0, :, :])

reproject(
    R_NWP[0, :, :],
    R_NWP_rprj,
    src_transform=NWP_transform,
    src_crs=NWP_crs,
    dst_transform=radar_transform,
    dst_crs=radar_crs,
    resampling=Resampling.nearest,
    dst_nodata=np.nan,
)

plt.figure(4)
plot_precip_field(R_NWP_rprj, geodata=geodata_NWP_radar)
plt.title("reprojected NWP (by WOUT) at 2021-07-04 16:05:00")

"""
# Plot difference

diff = R_NWP_rprj[:] - R_NWP_radar[0, :, :]
print(float(np.sum(np.abs(diff[:]))))

plt.figure(5)
plot_precip_field(np.abs(diff), geodata=geodata_NWP_radar)
plt.title("Absolute difference between interpolated NWP data sets")

plt.figure(6)
plot_precip_field(diff, geodata=geodata_NWP_radar)
plt.title("Positive difference between interpolated NWP data sets")

plt.figure(7)
plot_precip_field(-diff, geodata=geodata_NWP_radar)
plt.title("Negative difference between interpolated NWP data sets")
"""

# Show all plots

plt.show()
