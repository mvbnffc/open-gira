"""Checks this part of workflow can be proceeded with and provides common variables

Reminder: do not run figures_all until analyse_all has been completed!
"""


import os

# prelim variables
models_future = ['CMCC-CM2-VHR4', 'CNRM-CM6-1-HR', 'EC-Earth3P-HR', 'HadGEM3-GC31-HM']
models_all = ['constant']+models_future  # constant MUST be first
remove_countries = ['USA', 'VEN', 'CYM', 'VCT', 'BHS', 'ATG', 'DMA', 'LCA', 'TTO']
name_cc_constant = 'Constant'
name_cc_future = 'Future'
name_cc_future_diff = 'Future minus current'
name_cc_future_perc_diff = 'Future minus current (change in percent of current)'


# prelim checks
all_folders = [os.path.join(config['output_dir'], f'power_output-{model}') for model in models_all]

for folder in all_folders:  # check exist
    if not os.path.exists(folder):
        raise RuntimeWarning(f"\n----------------\nFolder {folder} does not exist. Can not proceed. Check that the workflow for {os.path.basename(folder)} has been completed and/or correctly named in {config['output_dir']} directory.\n----------------\n")
