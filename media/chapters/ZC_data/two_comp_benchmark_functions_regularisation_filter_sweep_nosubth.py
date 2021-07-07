import runpy

module = runpy.run_path("two_comp_benchmark_functions_regularisation_filter_sweep_common.py", globals())
module['do_plot'](subth=False)

