import runpy

module = runpy.run_path("two_comp_2d_frequency_sweep_common.py", globals())
module['plot_files']([
    'two_comp_data/frequency_sweep_network_2020_07_24_00_46_53.h5',
    'two_comp_data/frequency_sweep_network_2020_07_28_12_45_51.h5'
])

