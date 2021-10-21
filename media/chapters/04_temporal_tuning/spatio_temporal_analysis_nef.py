import spatio_temporal_analysis_common

fig, ax = spatio_temporal_analysis_common.plot_analysis(
    utils,
    "spatio_temporal_network_nef.h5",
    "spatio_temporal_network_analysis_nef.h5",
    "esn_decode_delays_1d.h5",
    first_letter="A")

utils.save(fig)
