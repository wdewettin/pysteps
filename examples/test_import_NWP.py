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
import datetime

# 1. Import NWP data

"""
R_NWP, _, metadata = import_bom_rf3_xr(
    "/home/wdewettin/Downloads/20201031_0000_regrid_short.nc",
    varname="accum_prcp",
    legacy=True,
)
pprint(metadata)
"""

R_NWP = import_bom_rf3_xr(
    "/home/wdewettin/Downloads/20201031_0000_regrid_short.nc",
    varname="accum_prcp",
    legacy=False,
)
R_NWP = xr.DataArray.squeeze(R_NWP)

geodata = {
    "projection": R_NWP.attrs["projection"],
    "x1": R_NWP.x[0],
    "y1": R_NWP.y[-1],
    "x2": R_NWP.x[-1],
    "y2": R_NWP.y[0],
    "yorigin": "upper",
}
# pprint(geodata)

# 2. Decumulate NWP data

R_NWP_dec = R_NWP.diff("time")
R_NWP_dec.attrs = R_NWP.attrs

R_NWP_dec.attrs["accutime"] = 10
R_NWP_dec[:], R_NWP_dec.attrs = to_rainrate(R_NWP_dec[:], R_NWP_dec.attrs)

# 3. Plot NWP data

for i in range(R_NWP_dec.shape[0]):
    plot_precip_field(R_NWP_dec[i, :, :], geodata=geodata)
    print("plotting NWP_{}".format(i))
    plt.savefig("BOM_NWP_{}.png".format(i))
    plt.close()

# 4. Import radar data

d = datetime.datetime(2020, 10, 31, 00, 00, 00)

for i in range(144):
    time = d.time().strftime("%H%M%S")

    """
    R, _, metadata = import_bom_rf3_xr(
        "/home/wdewettin/Downloads/radar/bom/prcp-c10/66/2020/10/31/66_20201031_050000.prcp-c10.nc",
        legacy=True,
    )
    R = np.squeeze(R)
    pprint(metadata)
    """

    R = import_bom_rf3_xr(
        "/home/wdewettin/Downloads/radar/bom/prcp-c10/66/2020/10/31/66_20201031_{}.prcp-c10.nc".format(
            time
        ),
        legacy=False,
    )
    R = xr.DataArray.squeeze(R)

    R.attrs["accutime"] = 10
    R[:], R.attrs = to_rainrate(R[:], R.attrs)

    geodata = {
        "projection": R.attrs["projection"],
        "x1": R.x[0],
        "y1": R.y[-1],
        "x2": R.x[-1],
        "y2": R.y[0],
        "yorigin": "upper",
    }
    # pprint(geodata)

    # 5. Plot radar data

    plot_precip_field(R, geodata=geodata)
    print("plotting radar {}".format(i))
    plt.savefig("BOM_radar_{}.png".format(i))
    plt.close()

    d = d + datetime.timedelta(minutes=10)

########################################################################

"""
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
