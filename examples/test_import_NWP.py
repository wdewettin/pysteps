from pysteps.io.importers import import_bom_rf3_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint
import xarray as xr
from pysteps.utils.conversion import to_rainrate
import numpy as np
from pysteps.visualization.utils import reproject_geodata  # ???
import rasterio
import netCDF4

"""
ncf = netCDF4.Dataset("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/20201031_0000_regrid_short.nc")
print(ncf)
print(ncf.variables)
R = ncf.variables["accum_prcp"][:]

for i in range(R.shape[0]):
    plot_precip_field(R[i, :, :])
    plt.show()
    plt.close()
"""

# R, _, metadata = import_bom_rf3_xr("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/20201031_0000_regrid_short.nc", varname="accum_prcp", legacy=True)

R = import_bom_rf3_xr(
    "/home/wdewettin/Downloads/20201031_0000_regrid_short.nc",
    varname="accum_prcp",
    legacy=False,
)
R = xr.DataArray.squeeze(R)

# Removing this makes an error go away
del R.attrs["transform"]

R.isel(time=30).plot()
R.rio.set_crs(R.attrs["projection"])
plt.savefig("Original.png")
plt.close()

R_new = R.rio.reproject("EPSG:2990", nodata=-10)
R_new.where(R_new != -10).isel(time=30).plot()
plt.show()
plt.close()

"""
# Decumulate data
R_dec = xr.DataArray.diff(R, 1, 0)
del R
R = R_dec
"""

"""
pprint(metadata)
# print(reproject_geodata(metadata))

plot_precip_field(R[0, 30, :, :], geodata=metadata)
plt.show()
plt.close()

metadata["projection"] = '+proj=longlat +datum=WGS84 +no_defs'
plot_precip_field(R[0, 30, :, :], geodata=metadata)
plt.show()
plt.close()
"""

"""
metadata["accutime"] = 10

R, metadata = to_rainrate(R, metadata)
pprint(metadata)
"""

"""
R = np.squeeze(R)
print(R.shape)

for i in range(R.shape[0], 10):
    print(i)
    plot_precip_field(R[i, :, :], geodata=metadata)
    plt.savefig("NWP_{}.png".format(i))
    plt.close()
"""

# R = import_bom_rf3_xr("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/radar/bom/prcp-cscn/2/2018/06/16/2_20180616_100000.prcp-cscn.nc")
