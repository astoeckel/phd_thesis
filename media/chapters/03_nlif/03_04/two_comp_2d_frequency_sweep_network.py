import runpy

module = runpy.run_path("two_comp_2d_frequency_sweep_common.py", globals())
module['plot_files'](['two_comp_2d_frequency_sweep_network.h5'], figsize=(7.45, 3.3), title="Network error $E_\\mathrm{net}$")

