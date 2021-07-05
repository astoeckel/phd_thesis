import runpy

module = runpy.run_path("two_comp_weights_examples_common.py", globals())
module['do_plot'](f=lambda x, y: (x * y), gmax=1500, vabs=0.25, tall=True, plot_hyper=True)

