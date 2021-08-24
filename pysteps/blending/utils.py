from pysteps.io import import_odim_hdf5, import_rmi_nwp_xr
from pysteps.visualization import plot_precip_field
import matplotlib.pyplot as plt
from pprint import pprint
from pysteps.utils.conversion import to_rainrate
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import numpy as np
import xarray as xr


def reprojection(R_src, R_dst):
    src_crs = R_src.attrs["projection"]
    x_src = R_src.x
    y_src = R_src.y
    x1_src = x_src.x1
    y2_src = y_src.y2
    xpixelsize_src = R_src.attrs["xpixelsize"]
    ypixelsize_src = R_src.attrs["ypixelsize"]
    src_transform = A.translation(x1_src, y2_src) * A.scale(
        xpixelsize_src, -ypixelsize_src
    )

    dst_crs = R_dst.attrs["projection"]
    x_dst = R_dst.x
    y_dst = R_dst.y
    x1_dst = x_dst.x1
    y2_dst = y_dst.y2
    xpixelsize_dst = R_dst.attrs["xpixelsize"]
    ypixelsize_dst = R_dst.attrs["ypixelsize"]
    dst_transform = A.translation(x1_dst, y2_dst) * A.scale(
        xpixelsize_dst, -ypixelsize_dst
    )

    R_rprj = np.zeros_like(R_dst[:])

    reproject(
        R_src[:],
        R_rprj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        dst_nodata=np.nan,
    )

    return R_rprj
