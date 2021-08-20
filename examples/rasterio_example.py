import numpy as np
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling

with rasterio.Env():

    # As source: a 512 x 512 raster centered on 0 degrees E and 0
    # degrees N, each pixel covering 15".
    rows, cols = src_shape = (512, 512)
    d = 1.0 / 240  # decimal degrees per pixel
    # The following is equivalent to
    # A(d, 0, -cols*d/2, 0, -d, rows*d/2).
    src_transform = A.translation(-cols * d / 2, rows * d / 2) * A.scale(d, -d)
    src_crs = {"init": "EPSG:4326"}
    # src_crs = "proj=lcc lon_0=4.55 lat_1=50.8 lat_2=50.8 a=6371229 es=0 +x_0=365950 +y_0=-365950"
    source = np.ones(src_shape, np.uint8) * 255

    # Destination: a 1024 x 1024 dataset in Web Mercator (EPSG:3857)
    # with origin at 0.0, 0.0.
    dst_shape = (1024, 1024)
    dst_transform = A.translation(-237481.5, 237536.4) * A.scale(425.0, -425.0)
    # dst_crs = {"init": "EPSG:3857"}
    # dst_crs = "+proj=stere +lon_0=4.368 +lat_0=90 +lon_ts=0 +lat_ts=50 +ellps=sphere +x_0=356406 +y_0=3698905"
    dst_crs = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +rf=298.252840776255 +units=m +no_defs"
    destination = np.zeros(dst_shape, np.uint8)

    reproject(
        source,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    # Assert that the destination is only partly filled.
    assert destination.any()
    assert not destination.all()
