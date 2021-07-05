import runpy

module = runpy.run_path("two_comp_weights_examples_common.py", globals())
module['do_plot'](f=lambda x, y: 0.5 * (x + y))

