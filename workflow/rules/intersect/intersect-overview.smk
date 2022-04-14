"""Merges the gdp loss statistics.

Provides an overview for all storms. This is useful for further analysis decisions and required to bring together the snakemake wildcards.
"""
import os

stat_csv = os.path.join(
    config['output_dir'], "power_output", "statistics", "combined_storm_statistics.csv"
)
all_indiv_stat_csv = expand(
    os.path.join(
        config['output_dir'],
        "power_output",
        "statistics",
        "{region}",
        "{sample}",
        "combined_storm_statistics_{region}_{sample}.csv",
    ),
    region=REGIONS,
    sample=SAMPLES,
)


def aggregate_input(wildcards):
    checkpoint_output = checkpoints.intersect_winds_indiv.get(**wildcards).output[
        0
    ]  # need this to signify that requires checkpoint data
    print("checkpoint output:", checkpoint_output)  # in this case: "data"

    # "data2/new_{num}.txt" is the last output before the aggregating file
    # os.path.join(checkpoint_output, "{num}.txt") is the output of the rule that makes all the new wildcards
    # glob_wildcards(os.path.join(checkpoint_output, "{num}.txt")).num  # extracts the actual wildcards (here: {num})
    # expand(...,...) to create the output of the last output before the aggregating file

    ret = expand(
        os.path.join(
            config['output_dir'],
            "power_intersection",
            "storm_data",
            "individual_storms",
            "{region}",
            "{sample}",
            "storm_{nh}",
            "storm_r{region}_s{sample}_n{nh}.txt",
        ),
        nh=glob_wildcards(
            os.path.join(checkpoint_output, "TC_r{region}_s{sample}_n{nh}.csv")
        ).nh,
        region=wildcards.region,
        sample=wildcards.sample,
    )
    return ret


rule merge_overview_indiv_stats:
    """Use this rule for the combined stats file for each region sample"""
    input:
        aggregate_input,
    output:
        os.path.join(
            config['output_dir'],
            "power_output",
            "statistics",
            "{region}",
            "{sample}",
            "combined_storm_statistics_{region}_{sample}.csv",
        ),
    script:
        os.path.join("..", "..", "scripts", "intersect", "intersect_overview_individual.py")


rule merge_overview_all_stats:
    """Use this rule for the combined stats file (all)"""
    input:
        all_indiv_stat_csv,
    output:
        stat_csv,
    script:
        os.path.join("..", "..", "scripts", "intersect", "intersect_overview.py")