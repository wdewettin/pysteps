from pysteps.io import import_knmi_hdf5
from pprint import pprint
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
import datetime

d = datetime.datetime(2018, 1, 5, 9, 0)

for _ in range(133):
    time = d.time().strftime("%H%M")

    R = import_knmi_hdf5(
        "/home/wdewettin/Downloads/KNMI/20180105/Bias_Corrected_Radar/RAD_NL25_RAP_5min_20180105{}.h5".format(
            time
        )
    )

    geodata = {
        "projection": R.attrs["projection"],
        "x1": R.x.x1,
        "y1": R.y.y1,
        "x2": R.x.x2,
        "y2": R.y.y2,
        "yorigin": R.attrs["yorigin"],
    }

    plot_precip_field(R, ptype="depth", units="mm", geodata=geodata)
    plt.savefig("KNMI_radar_{}.png".format(time))
    plt.close()

    print("plotted " + time)

    d = d + datetime.timedelta(minutes=5)
