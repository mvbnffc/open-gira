"""Download Worldpop population counts, constrained individual countries 2020 UN adjusted
(100m resolution)

Reference
---------
https://www.worldpop.org/geodata/listing?id=79
"""

rule download_worldpop:
    params:
        output_dir=config["output_dir"],
        code_country="{country}",
    output:
        os.path.join(
            config["output_dir"],
            "input",
            "population",
            "{country}_ppp_2020_UNadj_constrained.tif",
        ),
    script:
        "../../scripts/download/scrape_url.py"


rule download_worldpop_all:
    input:
        population_raster_by_country = expand(
            os.path.join(
                config["output_dir"],
                "input",
                "population",
                "{country}_ppp_2020_UNadj_constrained.tif",
            ),
            country=COUNTRY_CODES,
        )
