import logging
import os
import shutil
import sys
import tempfile
import time

from dask.delayed import Delayed
import xarray as xr

from open_gira.io import netcdf_packing_parameters
from open_gira.wind import empty_wind_da


def write_netcdf_via_local_scratch(dataset: xr.Dataset, output_path: str, encoding: dict) -> None:
    """
    Temporary workaround for slow direct netCDF writes on shared cluster storage.

    Write netCDF to node-local scratch first, then copy into final location.
    Revisit this once direct writes are no longer a bottleneck.
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
        prefix="open_gira_concat_",
        dir=scratch_parent,
        delete=False,
    ) as handle:
        scratch_path = handle.name

    try:
        scratch_write_start = time.perf_counter()
        serialisation_task: Delayed = dataset.to_netcdf(
            scratch_path,
            encoding=encoding,
            compute=False,
        )
        serialisation_task.compute(scheduler="synchronous")
        scratch_write_elapsed = time.perf_counter() - scratch_write_start

        final_copy_start = time.perf_counter()
        shutil.copy2(scratch_path, output_path)
        final_copy_elapsed = time.perf_counter() - final_copy_start

        output_size_mib = os.path.getsize(output_path) / 1024**2
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
    finally:
        try:
            os.remove(scratch_path)
        except OSError:
            pass


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )

    logging.info("Reading wind fields from each sample")
    # concatenated xarray dataset is chunked by input file
    # N.B. this is lazily loaded
    # cannot use open_mfdataset here, as event_id coordinates are not monotonically increasing
    all_samples = xr.concat(
        [
            xr.open_dataset(path, chunks={"max_wind_speed": 1}).sortby("event_id")
            for path in snakemake.input.sample_paths  # noqa: F821
        ],
        dim="event_id",
    ).sortby("event_id")

    if all_samples.event_id.size == 0:
        logging.info("Input data empty, writing empty file to disk")
        # write empty netcdf (with appropriate schema) and exit
        empty_wind_da().to_netcdf(snakemake.output.concat)  # noqa: F821
        sys.exit(0)

    logging.info("Computing packing factors for all samples")
    # we used dask to allow for chunked calculation of data min/max and to stream to disk
    # use the synchronous scheduler to limit dask to a single process (reducing memory footprint)
    scheduler = "synchronous"

    # compute packing factors for all data, need global min and max
    # the implementation below reads all the data chunks twice, once for min and once for max
    # would be nice to avoid this duplication
    scale_factor, add_offset, fill_value = netcdf_packing_parameters(
        all_samples.max_wind_speed.min().compute(scheduler=scheduler).item(),
        all_samples.max_wind_speed.max().compute(scheduler=scheduler).item(),
        16,
    )

    logging.info("Writing pooled wind fields to disk")
    logging.info(
        "Preparing pooled netCDF for %s events",
        all_samples.event_id.size,
    )
    # Temporary fix for cluster I/O bottlenecks: write locally first so this
    # can be reverted back to a direct `to_netcdf(output_path, ...)` later.
    write_netcdf_via_local_scratch(
        all_samples,
        snakemake.output.concat,  # noqa: F821
        {
            "max_wind_speed": {
                "dtype": "int16",
                "scale_factor": scale_factor,
                "add_offset": add_offset,
                "_FillValue": fill_value,
            }
        },
    )
