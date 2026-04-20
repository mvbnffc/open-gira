"""
Process storm data and write out maximum wind speeds at each grid pixel for
each storm.
"""

import os
import multiprocessing
import logging
import shutil
import tempfile
from typing import Optional
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rioxarray
import xarray as xr

from open_gira.io import bit_pack_dataarray_encoding
from open_gira.wind import (
    estimate_wind_field,
    interpolate_track,
    empty_wind_da,
    WIND_COORDS,
    ENV_PRESSURE,
)
from open_gira.wind_plotting import plot_contours, animate_track


logging.basicConfig(
    format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
)


def cleanup(output_path: str):
    """
    If we don't have a network, or tracks and we can't continue, write empty
    file and quit.
    """
    empty_wind_da().to_netcdf(output_path)
    sys.exit(0)


def write_netcdf_via_local_scratch(
    da: xr.DataArray, output_path: str, encoding: dict
) -> tuple[float, float, str]:
    """
    Temporary workaround for slow direct netCDF writes on shared cluster storage.

    Write netCDF to node-local scratch first, then copy into final location.

    Revisit and simplify this once direct `to_netcdf(output_path, ...)` writes
    are no longer a performance bottleneck on the target environment.

    Returns:
        scratch_write_elapsed: Seconds to serialise netCDF to local scratch
        final_copy_elapsed: Seconds to copy or move result to final output path
        scratch_path: Scratch file path used for the intermediate netCDF
    """
    scratch_parent = (
        os.environ.get("TMPDIR")
        or os.environ.get("TMP")
        or os.environ.get("TEMP")
        or tempfile.gettempdir()
    )
    os.makedirs(scratch_parent, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".nc",
        prefix="open_gira_wind_",
        dir=scratch_parent,
        delete=False,
    ) as handle:
        scratch_path = handle.name

    try:
        scratch_write_start = time.perf_counter()
        da.to_netcdf(scratch_path, encoding=encoding)
        scratch_write_elapsed = time.perf_counter() - scratch_write_start

        final_copy_start = time.perf_counter()
        shutil.copy2(scratch_path, output_path)
        final_copy_elapsed = time.perf_counter() - final_copy_start
    finally:
        try:
            os.remove(scratch_path)
        except OSError:
            pass

    return scratch_write_elapsed, final_copy_elapsed, scratch_path


def process_track(
    track: pd.core.groupby.generic.DataFrameGroupBy,
    longitude: np.ndarray,
    latitude: np.ndarray,
    downscaling_factors: np.ndarray,
    plot_max_wind: bool,
    plot_animation: bool,
    plot_dir: Optional[str],
    grid_coords: tuple[np.ndarray, np.ndarray],
) -> tuple[str, np.ndarray]:
    """
    Interpolate a track, reconstruct the advective and rotational vector wind
    fields, sum them and take the maximum of the wind vector magnitude across
    time. Optionally plot the wind fields and save to disk.

    Args:
        track: Subset of DataFrame describing a track. Must have a temporal
            index and the following fields: `min_pressure_hpa`,
            `max_wind_speed_ms`, `radius_to_max_winds_km`.
        longitude: Longitude values to construct evaluation grid
        latitude: Latitude values to construct evaluation grid
        downscaling_factors: Factors to bring gradient-level winds to surface.
        plot_max_wind: Whether to plot max wind fields
        plot_animation: Whether to plot wind field evolution
        plot_dir: Where to save optional plots.
        grid_coords: Pre-computed meshgrid result

    Returns:
        str: Track identifier
        np.ndarray: 2D array of maximum wind speed experienced at each grid pixel
    """

    (track_id,) = set(track.track_id)

    logging.debug(track_id)

    grid_shape: tuple[int, int] = (len(latitude), len(longitude))

    # we can't calculate the advective component without at least two points
    if len(track) == 1:
        return track_id, np.zeros(grid_shape)

    # basin of first record for storm track (storm genesis for synthetic tracks)
    basin: str = track.iloc[0, track.columns.get_loc("basin_id")]

    # Interpolate the track to avoid the "doughnut effect" that can arise from
    # sparse eye observations. If interpolation fails, fall back to the
    # observed points rather than silently turning the event into a null field.
    used_interpolation = False
    track_to_use = track
    try:
        track_to_use = interpolate_track(track)
        used_interpolation = True
    except (AssertionError, ValueError):
        logging.warning(
            "Could not successfully interpolate %s; falling back to the "
            "original track with %s observed points",
            track_id,
            len(track),
        )

    if len(track_to_use) == 1:
        logging.warning(
            "Track %s has only one usable point after preprocessing; "
            "returning a null wind field",
            track_id,
        )
        return track_id, np.zeros(grid_shape)

    # forward azimuth angle and distances from track eye to next track eye
    geod_wgs84: pyproj.Geod = pyproj.CRS("epsg:4326").get_geod()
    advection_azimuth_deg, _, eye_step_distance_m = geod_wgs84.inv(
        track_to_use.geometry.x.iloc[:-1],
        track_to_use.geometry.y.iloc[:-1],
        track_to_use.geometry.x.iloc[1:],
        track_to_use.geometry.y.iloc[1:],
    )

    # gapfill last period/distance values with penultimate value
    period = track_to_use.index[1:] - track_to_use.index[:-1]
    period = period.append(period[-1:])
    eye_step_distance_m = [*eye_step_distance_m, eye_step_distance_m[-1]]
    track_to_use["advection_azimuth_deg"] = [
        *advection_azimuth_deg,
        advection_azimuth_deg[-1],
    ]

    # calculate eye speed
    track_to_use["eye_speed_ms"] = eye_step_distance_m / period.total_seconds().values

    if not used_interpolation and (period.total_seconds() > 6 * 60 * 60).any():
        logging.warning(
            "Track %s contains gap(s) greater than 6h; coarse-track fallback "
            "may underestimate peak winds",
            track_id,
        )

    # result array
    wind_field: np.ndarray = np.zeros((len(track_to_use), *grid_shape), dtype=complex)

    failed_qa = False
    for track_i, track_point in enumerate(track_to_use.itertuples()):
        try:
            wind_field[track_i, :] = estimate_wind_field(
                longitude,  # degrees
                latitude,  # degrees
                track_point.geometry.x,  # degrees
                track_point.geometry.y,  # degrees
                track_point.radius_to_max_winds_km * 1_000,  # convert to meters
                track_point.max_wind_speed_ms,
                track_point.min_pressure_hpa * 100,  # convert to Pascals
                ENV_PRESSURE[basin] * 100,  # convert to Pascals
                track_point.advection_azimuth_deg,
                track_point.eye_speed_ms,
                grid_coords,
            )
        except AssertionError:
            failed_qa = True

    if failed_qa:
        logging.warning(f"Track {track_id} failed QA")

    # take factors calculated from surface roughness of region and use to downscale speeds
    downscaled_wind_field = downscaling_factors * wind_field

    # find vector magnitude, then take max along timestep axis, giving (y, x)
    # N.B. np.max([np.nan, 1]) = np.nan, so use np.nanmax
    max_wind_speeds: np.ndarray[float] = np.nanmax(
        np.abs(downscaled_wind_field), axis=0
    )

    # any dimensions with a single cell will break the plotting routines
    if 1 not in grid_shape:
        if plot_max_wind:
            plot_contours(
                max_wind_speeds,
                f"{track_id} max wind speed",
                "Wind speed [m/s]",
                os.path.join(plot_dir, f"{track_id}_max_contour.png"),
            )

        if plot_animation:
            animate_track(
                downscaled_wind_field,
                track_to_use,
                os.path.join(plot_dir, f"{track_id}.gif"),
            )

    return track_id, max_wind_speeds


if __name__ == "__main__":
    storm_file_path: str = snakemake.input.storm_file  # noqa: F821
    wind_grid_path: str = snakemake.input.wind_grid  # noqa: F821
    downscale_factors_path: str = snakemake.input.downscaling_factors  # noqa: F821
    storm_set: set[str] = set(snakemake.params.storm_set)  # noqa: F821
    plot_max_wind: bool = snakemake.config["plot_wind"]["max_speed"]  # noqa: F821
    plot_animation: bool = snakemake.config["plot_wind"]["animation"]  # noqa: F821
    n_proc: int = snakemake.threads  # noqa: F821
    plot_dir_path: str = snakemake.output.plot_dir  # noqa: F821
    output_path: str = snakemake.output.wind_speeds  # noqa: F821

    # directory required (snakemake output)
    os.makedirs(plot_dir_path, exist_ok=True)

    logging.info("Reading tracks")
    tracks = gpd.read_parquet(storm_file_path)
    if tracks.empty:
        logging.info("No intersection between network and tracks, writing empty file.")
        cleanup(output_path)

    if storm_set:
        # if we have a storm_set, only keep the matching track_ids
        logging.info("Filtering as per storm set")
        tracks = tracks[tracks.track_id.isin(storm_set)]

    if tracks.empty:
        logging.info("No intersection between network and tracks, writing empty file.")
        cleanup(output_path)

    logging.info(f"\n{tracks}")
    grouped_tracks = tracks.groupby("track_id")

    # grid to evaluate wind speeds on, rioxarray will return midpoints of raster cells as dims
    logging.info("Reading wind evaluation grid")
    grid: xr.DataArray = rioxarray.open_rasterio(wind_grid_path)
    logging.info(f"\n{grid}")

    logging.info("Reading wind downscaling factors")
    downscaling_factors = np.load(downscale_factors_path)
    grid_coords = np.meshgrid(grid.x, grid.y)

    # track is a tuple of track_id and the tracks subset, we only want the latter
    args = (
        (
            track[1],
            grid.x,
            grid.y,
            downscaling_factors,
            plot_max_wind,
            plot_animation,
            plot_dir_path,
            grid_coords,
        )
        for track in grouped_tracks
    )

    logging.info(f"Estimating wind fields for {len(grouped_tracks)} storm tracks")
    max_wind_speeds: list[str, np.ndarray] = []
    if n_proc > 1:
        with multiprocessing.Pool(processes=n_proc) as pool:
            max_wind_speeds = pool.starmap(process_track, args)
    else:
        for arg in args:
            max_wind_speeds.append(process_track(*arg))

    # sort by track_id so we have a reproducible order even after multiprocessing
    max_wind_speeds = sorted(max_wind_speeds, key=lambda pair: pair[0])

    logging.info("Saving maximum wind speeds to disk")
    track_ids, fields = zip(*max_wind_speeds)
    logging.info(
        "Preparing netCDF payload for %s events on a %s x %s grid",
        len(track_ids),
        len(grid.y.values),
        len(grid.x.values),
    )

    stack_start = time.perf_counter()
    stacked_fields = np.stack(fields)
    stack_elapsed = time.perf_counter() - stack_start
    logging.info(
        "Stacked wind fields in %.2fs to shape %s (dtype=%s, %.2f MiB in memory)",
        stack_elapsed,
        stacked_fields.shape,
        stacked_fields.dtype,
        stacked_fields.nbytes / 1024**2,
    )

    da_start = time.perf_counter()
    # write to disk as netCDF with CRS
    da = xr.DataArray(
        data=stacked_fields,
        dims=WIND_COORDS.keys(),
        coords=(
            ("event_id", list(track_ids)),
            ("latitude", grid.y.values),
            ("longitude", grid.x.values),
        ),
        attrs=dict(
            description="Maximum estimated wind speed during event",
            units="m s-1",
        ),
        name="max_wind_speed",
    )

    # TODO: write the appropriate metadata for QGIS to read this successfully
    # as it is, the lat/long values are being ignored
    # you can of course use ncview or xarray to inspect instead...
    # spatial_ref_attrs = pyproj.CRS.from_user_input(4326).to_cf()
    # da["spatial_ref"] = ((), 0, spatial_ref_attrs)
    da = da.rio.write_crs("EPSG:4326")
    da_elapsed = time.perf_counter() - da_start
    logging.info(
        "Constructed DataArray in %.2fs with dims %s",
        da_elapsed,
        dict(da.sizes),
    )

    # pack floating point data as integers on disk
    encoding_start = time.perf_counter()
    encoding = bit_pack_dataarray_encoding(da)
    encoding_elapsed = time.perf_counter() - encoding_start
    logging.info(
        "Calculated netCDF encoding in %.2fs: %s",
        encoding_elapsed,
        encoding,
    )

    # Temporary fix for cluster I/O bottlenecks: write locally first so this
    # can be reverted back to a direct `to_netcdf(output_path, ...)` later.
    logging.info("Writing netCDF via local scratch")
    scratch_write_elapsed, final_copy_elapsed, scratch_path = write_netcdf_via_local_scratch(
        da, output_path, encoding
    )
    try:
        output_size_mib = os.path.getsize(output_path) / 1024**2
    except OSError:
        output_size_mib = float("nan")
    logging.info(
        "Wrote scratch netCDF to %s in %.2fs",
        scratch_path,
        scratch_write_elapsed,
    )
    logging.info(
        "Copied netCDF to %s in %.2fs (%.2f MiB on disk)",
        output_path,
        final_copy_elapsed,
        output_size_mib,
    )

    logging.info("Done estimating wind fields")
