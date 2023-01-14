from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from numpy.typing import NDArray

RESOLUTION = 2000


def open_opera_dataset(
    file: Path, projection_cfg: Dict[str, Union[str, NDArray, NDArray]]
) -> xr.Dataset:
    with h5py.File(file, "r") as content:
        data, quality_index = _retrieve_data(content)
        ds = xr.Dataset(
            data_vars=dict(
                data=(["y", "x"], data),
                quality_index=(["y", "x"], quality_index),
            ),
            coords=dict(
                time=pd.Timestamp(file.stem[-12:]),
                x=projection_cfg["x"],
                y=projection_cfg["y"],
                longitude=(["y", "x"], projection_cfg["longitudes"]),
                latitude=(["y", "x"], projection_cfg["latitudes"]),
            ),
            attrs=dict(
                description="OPERA radar composite dataset",
                projection_wkt=projection_cfg["projection_wkt"],
            ),
        )
        return ds


def _retrieve_data(content: h5py.File) -> Tuple[NDArray, NDArray]:
    data = np.ma.masked_less_equal(content["dataset1"]["data1"]["data"][:], -8888000)
    quality_index = content["dataset2"]["data1"]["data"][:]
    return data, quality_index


def get_opera_projection_cfg(
    shape: Tuple[int, int] = (2200, 1900)
) -> Dict[str, Union[str, NDArray, NDArray]]:
    x, y, xx, yy = _get_grid(shape)
    projection_wkt, projection_proj4 = _define_projection()
    longitudes, latitudes = projection_proj4(xx, yy, inverse=True)
    return dict(
        projection_wkt=projection_wkt,
        x=x,
        y=y,
        longitudes=longitudes,
        latitudes=latitudes,
    )


def _get_grid(shape: Tuple[int, int]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    x = np.arange(0, shape[1] * RESOLUTION, RESOLUTION)
    y = np.arange(-shape[0] * RESOLUTION, 0, RESOLUTION)[::-1]
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def _define_projection() -> Tuple[str, pyproj.Proj]:
    projection_wkt = (
        "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 "
        + " +units=m +ellps=WGS84"
    )
    return projection_wkt, pyproj.Proj(projection_wkt)
