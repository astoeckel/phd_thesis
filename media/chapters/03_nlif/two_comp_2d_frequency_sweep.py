import runpy

module = runpy.run_path("two_comp_2d_frequency_sweep_common.py", globals())
module['plot_files'](['two_comp_data/frequency_sweep_2020_07_23_09_26_16.h5'], letter="A", mark_sigmas=[0.2, 0.33, 0.5, 0.8, 1.43])

