import netCDF4
from pysteps.io.importers import import_bom_rf3_xr, import_bom_rf3
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint
from pysteps.utils.conversion import to_rainrate
import numpy as np

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

R, _, metadata = import_bom_rf3_xr("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/20201031_0000_regrid_short.nc", varname="accum_prcp", legacy=True)
R = np.squeeze(R)
R = np.diff(R, 1, 0)

metadata["accutime"] = 10

R, metadata = to_rainrate(R, metadata)

for i in range(R.shape[0]):
    plot_precip_field(R[i, :, :], geodata=metadata)
    plt.savefig("RR_NWP_{}.png".format(i))
    plt.close()

# R = import_bom_rf3_xr("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/radar/bom/
# prcp-cscn/2/2018/06/16/2_20180616_100000.prcp-cscn.nc")
