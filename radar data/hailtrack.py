from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from pysteps import motion
from scipy.ndimage import map_coordinates

from smash.processing.conversion import get_conversion_rate
from smash.tools.error import SmashError

HAILSIZE_LAWS = [
    "maximum_estimated_size_of_hail_75",
    "maximum_estimated_size_of_hail_95",
]


def compute_tracks(
    track_path: Union[Path, str],
    from_unit: str,
    to_unit: str,
) -> xr.Dataset:
    thresholds = {"cm": 5.00, "in": 2.00}
    threshold = thresholds[to_unit]
    tracks = combine_mesh7595(track_path, from_unit, to_unit, threshold)
    return tracks


def combine_mesh7595(
    file_path: Union[Path, str],
    from_unit: str,
    to_unit: str,
    threshold: float,
) -> xr.Dataset:
    output_name = "maximum_estimated_size_of_hail_75_95"
    track = load_hail_track(file_path, from_unit, to_unit)
    mesh75 = xr.where(track[HAILSIZE_LAWS[0]] <= threshold, track[HAILSIZE_LAWS[0]], 0)
    mesh75.name = output_name

    mesh95 = xr.where(track[HAILSIZE_LAWS[0]] > threshold, track[HAILSIZE_LAWS[1]], 0)
    mesh95.name = output_name

    return xr.merge([track, mesh75 + mesh95])


def load_hail_track(
    file_path: Union[Path, str],
    from_unit: str,
    to_unit: str,
) -> xr.Dataset:
    track = xr.load_dataset(file_path)
    track = track.get(HAILSIZE_LAWS)
    track = convert_hailsize_unit(track, from_unit, to_unit)

    return track


def convert_hailsize_unit(track, from_unit, to_unit) -> xr.Dataset:
    conversion_coef = get_conversion_rate(from_unit, to_unit)
    return conversion_coef * track


def apply_advection_correction(
    ds_radar_scans: xr.Dataset,
    interpolation_timestep: Optional[pd.Timedelta] = pd.Timedelta("1min"),
) -> xr.Dataset:
    """
    Code from:
        https://pysteps.readthedocs.io/en/stable/auto_examples/
        advection_correction.html#sphx-glr-auto-examples-advection-correction-py

    Apply advection correction on radar observation.

    Parameters
    ----------
    ds_radar_scans: xr.Dataset
        Dataset with dims (time, lat, lon).
    interpolation_timestep: pd.Timedelta("Xmin")
        Interpolation timestep in minutes

    Returns
    -------
    ds_advection_correction: xr.Dataset
        Maximum over time with advection corrected radar observation.
        Shape (lat, lon).
    """
    t = interpolation_timestep.total_seconds() / 60
    ds_advection_correction = ds_radar_scans.max("time", keep_attrs=True)
    for var_name in ds_radar_scans.data_vars:
        da_radar_var = ds_radar_scans[var_name]
        advection_correction = ds_advection_correction[var_name].values.copy()
        for timestep in range(da_radar_var.time.size - 1):
            T = (
                pd.Timedelta(
                    da_radar_var.time.values[timestep + 1]
                    - da_radar_var.time.values[timestep]
                ).total_seconds()
                / 60
            )
            advection_correction = np.maximum(
                advection_correction,
                apply_advection_correction_timestep(
                    da_radar_var.values[timestep : (timestep + 2)],
                    T=T,
                    t=t,
                ),
            )
        ds_advection_correction[var_name] = xr.DataArray(
            data=advection_correction,
            dims=da_radar_var.max("time", keep_attrs=True).dims,
            coords=da_radar_var.max("time", keep_attrs=True).coords,
            attrs=da_radar_var.max("time", keep_attrs=True).attrs,
        )
    return ds_advection_correction


def apply_advection_correction_timestep(
    obs: np.ndarray, T: float = 5, t: float = 1
) -> np.ndarray:
    """
    Code from:
        https://pysteps.readthedocs.io/en/stable/auto_examples/
        advection_correction.html#sphx-glr-auto-examples-advection-correction-py

    Apply advection correction on 2 successive radar observation.

    Parameters
    ----------
    obs: np.ndarray
        Array of shape (2, lat, lon) with 2 slices of 2D observations
    T: float
        Time between two observations in minutes
    t: float
        Interpolation timestep in minutes

    Returns
    -------
    obs_advection_corrected: np.ndarray([2D_max_from_previous_to_current])
        Array of shape (lat, lon) of the maximum of "obs", advected on all timesteps
        from slice "previous" to slice "current"
    """
    check_interpolation_timestep(T, t)
    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(np.log(obs), fd_kwargs=fd_kwargs)

    # Perform temporal interpolation
    obs_advection_corrected = np.zeros((obs[0].shape))
    x, y = np.meshgrid(
        np.arange(obs[0].shape[1], dtype=float), np.arange(obs[0].shape[0], dtype=float)
    )
    for timestep in np.arange(t, T + t, t):
        pos1 = (y - timestep / T * V[1], x - timestep / T * V[0])
        obs1 = map_coordinates(obs[0], pos1, order=1)

        pos2 = (y + (T - timestep) / T * V[1], x + (T - timestep) / T * V[0])
        obs2 = map_coordinates(obs[1], pos2, order=1)

        obs_advection_corrected = np.maximum(
            obs_advection_corrected, ((T - timestep) * obs1 + timestep * obs2) * (1 / T)
        )
    return obs_advection_corrected


def check_interpolation_timestep(data_timestep, interpolation_timestep):
    if interpolation_timestep / data_timestep > 1 / 3:
        msg = (
            "Interpolation timestep (t) is too high compared to Data timestep (T), "
            + "and that can create artefacts. "
            + "Please pick t <= 1/3 * T."
        )
        raise SmashError(msg)


__all__ = [compute_tracks]
