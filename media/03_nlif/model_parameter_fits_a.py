import runpy

module = runpy.run_path("model_parameter_fits.py", globals())
module['do_plot'](suffix="(constant inputs)")


