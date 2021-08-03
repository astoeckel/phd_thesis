import runpy

module = runpy.run_path("two_comp_2d_frequency_sweep_common.py", globals())
module['plot_files'](['two_comp_2d_frequency_sweep.h5'], title="Static model error $E_\\mathrm{model}$", figsize=(7.45, 3.3)) #, letter="A", mark_sigmas=[0.2, 0.33, 0.5, 0.8, 1.43])

