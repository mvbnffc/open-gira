configfile: 'config.yaml'

DATA_DIR = config['data_dir']
OUTPUT_DIR = config['output_dir']
AQUEDUCT_DIR = config['aqueduct_dir']

links = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary_link",
    "secondary",
    "secondary_link"
]
filters = ','.join(links)


rule filter_osm_data:
    input: os.path.join(DATA_DIR, '{pbf_file}.osm.pbf')
    output: os.path.join(DATA_DIR, '{pbf_file}-highway-core.osm.pbf')
    shell: 'osmium tags-filter {input} w/highway={filters} -o {output}'


rule convert_to_geoparquet:
    input:
        cmd='osm_to_pq.py',
        data=os.path.join(DATA_DIR, '{pbf_file}-highway-core.osm.pbf')
    output: os.path.join(DATA_DIR, '{pbf_file}-highway-core.geoparquet')
    shell: 'python {input.cmd} {input.data} {DATA_DIR}'


rule network_hazard_intersection:
    input:
        cmd='network_hazard_intersection.py',
        network=os.path.join(DATA_DIR, '{slug}-highway-core.geoparquet'),
        csv=os.path.join(AQUEDUCT_DIR, 'aqueduct_river.csv')
    output:
        geoparquet=os.path.join(OUTPUT_DIR, '{slug}-highway-core_splits.geoparquet'),
        parquet=os.path.join(OUTPUT_DIR, '{slug}-highway-core_splits.parquet')
    shell: 'python {input.cmd} {input.network} {AQUEDUCT_DIR} {OUTPUT_DIR}'


rule clean:
    shell: 'rm -rf data/*-highway-core.osm.pbf data/*.geoparquet outputs/'
