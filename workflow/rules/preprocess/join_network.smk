# Take split .geoparquet files for nodes or edges and output unified .geoparquet
# files for transport networks

rule join_network:
    conda: "../../../environment.yml"
    input:
        nodes = lambda wildcards: expand(
            os.path.join("{OUTPUT_DIR}", "geoparquet", "{DATASET}_{FILTER_SLUG}", "processed", "slice-{i}_nodes.geoparquet"),
            **wildcards,
            i=range(config['slice_count'])
        ),
        # without a slice count in the output (we're aggregating), snakemake can't determine the slice in the input
        # therefore, use expand to generate the inputs from a list of slice numbers
        edges = lambda wildcards: expand(
            os.path.join("{OUTPUT_DIR}", "geoparquet", "{DATASET}_{FILTER_SLUG}", "processed", "slice-{i}_edges.geoparquet"),
            **wildcards,
            i=range(config['slice_count'])
        ),
    output:
        nodes="{OUTPUT_DIR}/{DATASET}_{FILTER_SLUG}/nodes.geoparquet",
        edges="{OUTPUT_DIR}/{DATASET}_{FILTER_SLUG}/edges.geoparquet"
    script:
        "../../scripts/join_network.py"

"""
Test with:
snakemake --cores all results/tanzania-mini_filter-road/nodes.geoparquet
"""