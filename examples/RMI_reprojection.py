from pysteps.io import import_odim_hdf5, import_rmi_nwp_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint
from pysteps.utils.conversion import to_rainrate

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

# Plot radar data

plt.figure(1)
plot_precip_field(R, geodata=geodata)
plt.title("Radar at 2021-07-04 16:05:00")


# Import NWP data

R_NWP = import_rmi_nwp_xr("./nwp/ao13_2021070412_native_5min.nc")

geodata_NWP = {
    "projection": R_NWP.attrs["projection"] + " lat_0=50.8",
    "x1": R_NWP.x.x1,
    "y1": R_NWP.y.y1,
    "x2": R_NWP.x.x2,
    "y2": R_NWP.y.y2,
    "yorigin": R_NWP.attrs["yorigin"],
}

R_NWP.attrs["accutime"] = 5.0
R_NWP[:], R_NWP.attrs = to_rainrate(R_NWP[:], R_NWP.attrs)

pprint(R_NWP.attrs)

plt.figure(2)
plot_precip_field(R_NWP[0, :, :], geodata=geodata_NWP)
plt.title("NWP at 2021-07-04 16:05:00")


# Show all plots

plt.show()
