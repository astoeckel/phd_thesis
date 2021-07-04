import runpy

module = runpy.run_path("model_parameter_fits.py", globals())
module['do_plot'](suffix="(with spike noise)", do_use_noise=True, do_relu=True)

