import runpy

module = runpy.run_path("plot_function_common.py", globals())
module['do_plot']('arctan', True)

