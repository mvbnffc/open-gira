# Takes a list of geoparquet files and join corresponding
# geodataframes.

# file1.geoparguet
# id obs1 obs2 geometry
# 0  a    b    geom0
# 1  c    d    geom1

# file2.geoparguet
# id obs1 obs2 geometry
# 0  A    B    GEOM0
# 1  C    D    GEOM1

# joined.geoparguet
# id obs1 obs2 geometry
# 0  a    b    geom0
# 1  c    d    geom1
# 2  A    B    GEOM0
# 3  C    D    GEOM1

# Usage: python join_data [FILE] [output]
# Example: python join_data file1.geoparguet file2.geoparguet joined.geoparguet

import sys
import geopandas as gpd


def append_data(base, slice_files):
    slice_files.pop()
    if len(slice_files) == 0:
        return base
    base = base.append(gpd.read_parquet(slice_files[-1]))
    return append_data(base, slice_files)


if __name__ == "__main__":
    try:
        slice_files = snakemake.input
        output_file = snakemake.output[0]
    except NameError:
        slice_files = sys.argv[1:-1]
        output_file = sys.argv[-1]

    # When getting the input files from snakemake, there is no
    # garantee that they will always in the same order. Sort them for
    # consistency. Makes testing easier.
    slice_files = sorted(slice_files)
    # We're reading the different files as a stack from the top.  Let's
    # reverse the order of files to keep the first file on top.
    slice_files = slice_files[::-1]

    base = gpd.read_parquet(slice_files[-1])
    base = append_data(base, slice_files)
    base.to_parquet(output_file)
