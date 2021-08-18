from pysteps.io import import_knmi_hdf5, import_netcdf_pysteps
from pprint import pprint
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
import datetime
import netCDF4

# 1. Importing and plotting radar data

d = datetime.datetime(2018, 1, 5, 9, 0)

for _ in range(1):
    time = d.time().strftime("%H%M")

    R = import_knmi_hdf5(
        "/home/wdewettin/Downloads/KNMI/20180105/Uncorrected_Radar/RAD_NL25_RAP_5min_20180105{}.h5".format(
            time
        )
    )

    print(R.x)

    geodata = {
        "projection": R.attrs["projection"],
        "x1": R.x.x1,
        "y1": R.y.y1,
        "x2": R.x.x2,
        "y2": R.y.y2,
        "yorigin": R.attrs["yorigin"],
    }

    pprint(geodata)

    plot_precip_field(R, ptype="depth", units="mm", geodata=geodata)
    plt.savefig("KNMI_radar_{}.png".format(time))
    plt.close()

    print("plotted " + time)

    d = d + datetime.timedelta(minutes=5)

# 2. Importing and plotting NWP data

ncf = netCDF4.Dataset(
    "/home/wdewettin/Downloads/KNMI/20180105/Harmonie/20180105_0600_Pforecast_Harmonie.nc"
)
R = ncf.variables["P_fc"]
projection = ncf.variables["crs"].proj4_params
print(projection)

for i in range(1):
    plot_precip_field(R[i, :, :])
    plt.savefig("KNMI_NWP_{}.png".format(i))
    plt.close()
    print("plotted NWP {}".format(i))
