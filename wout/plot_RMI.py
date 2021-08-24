# -*- coding: utf-8 -*-

from matplotlib import cm, pyplot as plt
import numpy as np
import os
from datetime import datetime
from pprint import pprint
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import io, rcparams
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field

# Give date as input
date_str = "202106042000"  # input("Give date and time (e.g.: 201609281600):    ")

date = datetime.strptime(date_str, "%Y%m%d%H%M")
data_source = rcparams.data_sources["rmi"]

# Import the radar composite
root_path = data_source["root_path"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
timestep = data_source["timestep"]
importer_kwargs = data_source["importer_kwargs"]

filename = os.path.join(
    root_path,
    datetime.strftime(date, "%Y%m%d"),
    datetime.strftime(date, fn_pattern) + "." + fn_ext,
)

R, _, metadata = io.import_odim_hdf5(filename, "RATE", **importer_kwargs)

# Convert to rain rate
R, metadata = conversion.to_rainrate(R, metadata)

# Nicely print the metadata
pprint(metadata)

# Plot the rainfall field
plot_precip_field(R, geodata=metadata)
plt.show()
