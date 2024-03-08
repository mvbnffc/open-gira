"""
Network creation routines
"""


rule gridfinder_to_geoparquet:
    """
    Store linestrings as geoparquet for faster accessing.
    Store a representative point from each linestring for faster spatial joins.
    """
    input:
        geopackage = rules.download_gridfinder.output.electricity_grid_global,
    output:
        linestring = "{OUTPUT_DIR}/power/gridfinder.geoparquet",
        rep_point = "{OUTPUT_DIR}/power/gridfinder_rep_point.geoparquet",
        plot = "{OUTPUT_DIR}/power/gridfinder.png",
    run:
        import geopandas as gpd
        import pandas as pd
        import datashader as ds
        import spatialpandas
        import spatialpandas.io

        grid = gpd.read_file(input.geopackage)

        # write out linestrings as geoparquet
        grid.to_parquet(output.linestring)

        # cast to spatialpandas for plotting
        grid_sp = spatialpandas.GeoDataFrame(grid)

        # make an integer source category column
        cat = pd.get_dummies(grid_sp.source)
        cat.gridfinder *= 2
        # openstreetmap = 1, gridfinder = 2
        grid_sp["source_id"] = cat.sum(axis=1).astype(int)

        # plot the gridfinder network
        cvs = ds.Canvas(plot_width=1500, plot_height=670)
        agg = cvs.line(grid_sp, geometry='geometry', agg=ds.mean("source_id"))
        img = ds.transfer_functions.shade(agg)
        ds.utils.export_image(img=img, filename=output.plot.split(".")[0], fmt=".png", background="black")

        # pick a point somewhere on each linestring, replace geometry with this
        grid["geometry"] = grid.geometry.representative_point()

        # write out representative points
        spatialpandas.io.to_parquet(spatialpandas.GeoDataFrame(grid), output.rep_point)

"""
To test:
snakemake --cores 1 results/input/gridfinder/gridfinder.geoparquet
"""


rule subset_gridfinder:
    """
    Subset the gridfinder dataset to a country boundary. Can be quite a heavy
    operation depending on the size of the country.

    N.B. Should take around an hour for the USA/gridfinder case. Could be
    reduced by using dask workers from spatialpandas, but probably not worth
    the complexity?
    """
    input:
        gridfinder=rules.gridfinder_to_geoparquet.output.linestring,
        gridfinder_rep_point=rules.gridfinder_to_geoparquet.output.rep_point,
        admin_bounds="{OUTPUT_DIR}/input/admin-boundaries/admin-level-0.geoparquet"
    output:
        gridfinder="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/gridfinder.geoparquet",
    resources:
        mem_mb=8192
    run:
        import geopandas as gpd
        import spatialpandas
        import spatialpandas.io

        import logging

        logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)

        os.makedirs(os.path.dirname(output.gridfinder), exist_ok=True)

        # read admin bounds for country in question
        countries: gpd.GeoDataFrame = gpd.read_parquet(input.admin_bounds)
        country_gp: gpd.GeoDataFrame = countries[countries.GID_0 == wildcards.COUNTRY_ISO_A3]

        grid: gpd.GeoDataFrame = gpd.read_parquet(input.gridfinder)

        if country_gp.geometry.area.sum() > 10:  # square degrees, not a fair measure at high latitudes

            logging.info(f"Using spatialpandas point-in-polygon to subset gridfinder for {wildcards.COUNTRY_ISO_A3}")
            # create a spatialpandas GeoDataFrame for the country
            # it tends to be faster than geopandas for sjoins, about 3x in Mexico test case
            country_sp: spatialpandas.GeoDataFrame = spatialpandas.GeoDataFrame(country_gp)

            # read in representative points of linestrings as spatialpandas geodataframe
            # spatialpandas sjoin can only do point-in-polygon, not linestring-in-polygon
            grid_rep_point: spatialpandas.GeoDataFrame = spatialpandas.io.read_parquet(input.gridfinder_rep_point)

            logging.info(f"Spatially joining gridfinder with {wildcards.COUNTRY_ISO_A3}")
            grid_subset: spatialpandas.GeoDataFrame = spatialpandas.sjoin(grid_rep_point, country_sp, how="inner")

            logging.info(f"Writing gridfinder to {output.gridfinder}")
            # use the joined index to select from the geopandas linestring geodataframe
            grid.loc[grid_subset.index, :].to_parquet(output.gridfinder)

        else:

            logging.info(f"Spatially joining gridfinder with {wildcards.COUNTRY_ISO_A3}")
            grid_subset: gpd.GeoDataFrame = grid.sjoin(country_gp, how="inner")

            logging.info(f"Writing gridfinder to {output.gridfinder}")
            grid_subset[grid.columns].to_parquet(output.gridfinder)

"""
Test with:
snakemake -c1 results/power/by_country/HTI/network/gridfinder.geoparquet
"""


rule subset_targets:
    """
    Subset the targets dataset to a country boundary
    """
    input:
        targets=rules.annotate_targets.output.targets,
        admin_bounds="{OUTPUT_DIR}/input/admin-boundaries/admin-level-0.geoparquet"
    output:
        targets="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/targets.geoparquet",
    run:
        import geopandas as gpd
        import spatialpandas

        import logging

        logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)

        os.makedirs(os.path.dirname(output.targets), exist_ok=True)

        countries: gpd.GeoDataFrame = gpd.read_parquet(input.admin_bounds)
        country_gp: gpd.GeoDataFrame = countries[countries.GID_0 == wildcards.COUNTRY_ISO_A3]

        targets: gpd.GeoDataFrame = gpd.read_parquet(input.targets)

        if country_gp.geometry.area.sum() > 10:  # square degrees, not a fair measure at high latitudes

            logging.info(f"Using spatialpandas point-in-polygon to subset targets for {wildcards.COUNTRY_ISO_A3}")
            # create a spatialpandas GeoDataFrame for the country
            # it tends to be faster than geopandas for sjoins, about 3x in Mexico test case
            country_sp: spatialpandas.GeoDataFrame = spatialpandas.GeoDataFrame(country_gp)

            # spatialpandas sjoin can only do point-in-polygon, not linestring-in-polygon
            # so make some representative points
            targets_rep_point = targets.copy()
            targets_rep_point.geometry = targets_rep_point.geometry.representative_point()
            targets_rep_point = spatialpandas.GeoDataFrame(targets_rep_point)

            logging.info(f"Spatially joining targets with {wildcards.COUNTRY_ISO_A3}")
            targets_subset: spatialpandas.GeoDataFrame = spatialpandas.sjoin(targets_rep_point, country_sp, how="inner")

            logging.info(f"Writing targets to {output.targets}")
            # use the joined index to select from the geopandas linestring geodataframe
            targets.loc[targets_subset.index, :].to_parquet(output.targets)

        else:

            logging.info(f"Spatially joining targets with {wildcards.COUNTRY_ISO_A3}")
            targets_subset: gpd.GeoDataFrame = targets.sjoin(country_gp, how="inner")

            logging.info(f"Writing targets to {output.targets}")
            targets_subset[targets.columns].to_parquet(output.targets)

"""
Test with:
snakemake -c1 results/power/by_country/HTI/network/targets.geoparquet
"""


rule subset_powerplants:
    """
    Subset the powerplants dataset to a country boundary
    """
    input:
        admin_bounds="{OUTPUT_DIR}/input/admin-boundaries/admin-level-0.geoparquet",
        powerplants="{OUTPUT_DIR}/power/powerplants.geoparquet",
    output:
        powerplants="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/powerplants.geoparquet",
    run:
        import geopandas as gpd

        import logging

        logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)

        os.makedirs(os.path.dirname(output.powerplants), exist_ok=True)

        countries: gpd.GeoDataFrame = gpd.read_parquet(input.admin_bounds)
        country_gp: gpd.GeoDataFrame = countries[countries.GID_0 == wildcards.COUNTRY_ISO_A3]

        powerplants: gpd.GeoDataFrame = gpd.read_parquet(input.powerplants)

        logging.info(f"Spatially joining powerplants with {wildcards.COUNTRY_ISO_A3}")
        powerplants_subset: gpd.GeoDataFrame = powerplants.sjoin(country_gp, how="inner")

        logging.info(f"Writing targets to {output.powerplants}")
        powerplants_subset[powerplants.columns].to_parquet(output.powerplants)

"""
Test with:
snakemake -c1 results/power/by_country/HTI/network/powerplants.geoparquet
"""


rule create_power_network:
    """
    Combine power plant, consumer and transmission data for given area
    """
    conda: "../../../environment.yml"
    input:
        plants="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/powerplants.geoparquet",
        targets="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/targets.geoparquet",
        gridfinder="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/gridfinder.geoparquet",
    output:
        edges="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/edges.geoparquet",
        nodes="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/nodes.geoparquet",
        grid_hull="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/convex_hull.json",
    script:
        "../../scripts/preprocess/create_electricity_network.py"

"""
Test with:
snakemake -c1 results/power/by_country/HTI/edges.geoparquet
"""


rule map_network_components:
    """
    Produce a datashader plot of edges within network, coloured by component ID.

    Useful for spotting electricity 'islands'.
    """
    input:
        edges="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/edges.geoparquet",
    output:
        plot="{OUTPUT_DIR}/power/by_country/{COUNTRY_ISO_A3}/network/edges.png",
    run:
        import geopandas as gpd
        import datashader as ds
        import datashader.transfer_functions as tf
        import matplotlib.cm
        import spatialpandas

        edges = gpd.read_parquet(input.edges)

        minx, miny, maxx, maxy = edges.total_bounds
        x = maxx - minx
        y = maxy - miny
        aspect = x / y
        width = 800
        cvs = ds.Canvas(plot_width=width, plot_height=int(width / aspect))

        agg = cvs.line(spatialpandas.GeoDataFrame(edges), geometry='geometry', agg=ds.mean("component_id"))
        img = tf.shade(agg, cmap=matplotlib.cm.Set2_r)

        ds.utils.export_image(img=img, filename=output.plot.split(".")[0], fmt=".png", background="black")

"""
Test with:
snakemake -c1 results/power/by_country/HTI/edges.png
"""


rule create_transport_network:
    """
    Take .geoparquet OSM files and output files of cleaned network nodes and edges
    """
    conda: "../../../environment.yml"
    input:
        nodes="{OUTPUT_DIR}/geoparquet/{DATASET}_{FILTER_SLUG}/raw/{SLICE_SLUG}_nodes.geoparquet",
        edges="{OUTPUT_DIR}/geoparquet/{DATASET}_{FILTER_SLUG}/raw/{SLICE_SLUG}_edges.geoparquet",
        admin="{OUTPUT_DIR}/input/admin-boundaries/gadm36_levels.gpkg",
    output:
        nodes="{OUTPUT_DIR}/geoparquet/{DATASET}_{FILTER_SLUG}/processed/{SLICE_SLUG}_nodes.geoparquet",
        edges="{OUTPUT_DIR}/geoparquet/{DATASET}_{FILTER_SLUG}/processed/{SLICE_SLUG}_edges.geoparquet"
    params:
        # determine the network type from the filter, e.g. road, rail
        network_type=lambda wildcards: wildcards.FILTER_SLUG.replace('filter-', ''),
        # pass in the slice number so we can label edges and nodes with their slice
        # edge and node IDs should be unique across all slices
        slice_number=lambda wildcards: int(wildcards.SLICE_SLUG.replace('slice-', ''))
    script:
        # template the path string with a value from params (can't execute .replace in `script` context)
        "../../scripts/transport/create_{params.network_type}_network.py"

"""
Test with:
snakemake --cores all results/geoparquet/tanzania-mini_filter-road/processed/slice-0_edges.geoparquet
"""