import netCDF4
from pysteps.io.importers import import_bom_rf3_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt

ncf = netCDF4.Dataset("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/20201031_0000_regrid_short.nc")
R = ncf.variables["accum_prcp"][:]

for i in range(R.shape[0]):
    plot_precip_field(R[i, :, :])
    plt.show()
    plt.close()

"""
R = import_bom_rf3_xr("C:/Users/woutd/Documents/School/UGent/Master/Stage/Pysteps/pysteps_data/20201031_0000_regrid_short.nc")
print(R)
"""