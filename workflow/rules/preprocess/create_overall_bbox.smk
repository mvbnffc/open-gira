# Create .json file that determines the bounding boxes of the complete data
rule create_overall_bbox:
    input:
        "{OUTPUT_DIR}/input/OSM/{DATASET}.osm.pbf",
    output:
        "{OUTPUT_DIR}/json/{DATASET}.json",
    script:
        "../../scripts/create_overall_bbox.py"


"""
Test with:
snakemake --cores all results/json/tanzania-mini.json
"""
