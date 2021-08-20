from pysteps.io import import_odim_hdf5, import_rmi_nwp_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint

# Import radar data

R = import_odim_hdf5("20210704000000.rad.becomp00.image.rate.beborder00_comp_sri.hdf")

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
plt.title("Radar at 2021-07-04 00:00:00")
plot_precip_field(R, geodata=geodata)

# Import NWP data

R_NWP = import_rmi_nwp_xr("ao13_2021071406_steps_be13_5min.nc")

pprint(R_NWP)

geodata_NWP = {
    "projection": "proj=lcc lon_0=4.55 lat_1=50.8 lat_2=50.8 a=6371229 es=0 +x_0=365950 +y_0=-365950",  # R_NWP.attrs["projection"], # "proj=lcc lon_0=4.55 lat_1=50.8 lat_2=50.8 a=6371229 es=0 +x_0=365950 +y_0=-365950"
    "x1": R_NWP.x.x1 / 1000.0,
    "y1": -R_NWP.y.y2 / 1000.0,
    "x2": R_NWP.x.x2 / 1000.0,
    "y2": -R_NWP.y.y1 / 1000.0,
    "yorigin": R_NWP.attrs["yorigin"],
}

plt.figure(2)
plt.title("NWP at 2021-07-14 10:05:00")
plot_precip_field(R_NWP[0, :, :], geodata=geodata_NWP)

# Show all the figures

plt.show()
