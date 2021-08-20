from pysteps.io import import_odim_hdf5
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt

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

# Show all the figures

plt.show()
