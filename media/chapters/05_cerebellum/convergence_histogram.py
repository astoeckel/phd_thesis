#   Code for the PhD Thesis
#   "Harnessing Neural Dynamics as a Computational Resource: Building Blocks
#   for Computational Neuroscience and Artificial Agents"
#   Copyright (C) 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import tarfile

#
# WEIGHT HISTOGRAM DATA
#


def load_weight_hist(tarfilename, threshold=1e-15, mode="pcn_to_gr"):
    Ws_hist = {}
    Ws_mag = {}
    with tarfile.open(tarfilename, 'r') as tf:
        for fn in tf.getnames():
            if fn.endswith('.h5'):
                print("Processing", fn, "from", tarfilename)
                with h5py.File(tf.extractfile(fn), 'r') as f:
                    params = json.loads(f['params'].attrs['params'])

                    if 'bias_mode' in params:
                        bias_mode = params['bias_mode']
                    else:
                        bias_mode = 'realistic_pcn_intercepts'

                    if 'decode_bias_granule' in params:
                        if not params['decode_bias_granule']:
                            bias_mode = 'use_jbias'

                    if not bias_mode in Ws_hist:
                        Ws_hist[bias_mode] = {}
                        Ws_mag[bias_mode] = {}

                    if mode == "pcn_to_gr":
                        if 'n_pcn_granule_convergence' in params:
                            conv = params['n_pcn_granule_convergence']
                        else:
                            conv = None
                    elif mode == "go_to_gr":
                        if 'n_golgi_granule_convergence' in params:
                            conv = params['n_golgi_granule_convergence']
                        else:
                            conv = None

                    if mode == "pcn_to_gr":
                        key = 'weights_conn_pcn_go_to_gr_excitatory'
                        W = np.abs(f[key])
                        W_subs = W[:100, :]
                    elif mode == "go_to_gr":
                        key = 'weights_conn_pcn_go_to_gr_inhibitory'
                        W = np.abs(f[key])
                        W_subs = W[100:, :]
                    kind = key.split('_')[-1]
                    if not conv in Ws_hist[bias_mode]:
                        Ws_hist[bias_mode][conv] = []
                        Ws_mag[bias_mode][conv] = []
                    Ws_hist[bias_mode][conv] += list(
                        np.sum(W_subs > threshold, axis=0))
                    Ws_mag[bias_mode][conv] += list(W_subs[W_subs > 0].flatten())
    return Ws_hist, Ws_mag


Ws_hist, Ws_mag = load_weight_hist(utils.datafile('weights.tar'),
                                   mode="go_to_gr")

BIAS_MODE_TO_LABEL = {
    'realistic_pcn_intercepts': 'Realistic PCN Intercepts',
    'jbias_very_realistic_pcn_intercepts': 'Realistic PCN Intercepts',
    'jbias_realistic_pcn_intercepts': 'Realistic PCN Intercepts',
    'uniform_pcn_intercepts': 'Uniform PCN Intercepts',
    'use_jbias': 'Realistic PCN Intercepts, with $J_\\mathrm{bias}$'
}

convs = [1, 3, 5, 9]
fig, axs = plt.subplots(len(convs), 1, figsize=(1.9, 2.475), sharex=True)
for j, bias_mode in enumerate(Ws_hist.keys()):
    i = 0
    for conv, hist in sorted(Ws_hist[bias_mode].items()):
        if not conv in convs:
            continue
        color = cm.get_cmap('tab10')(0)
        axs[i].hist(hist,
                    density=True,
                    bins=np.arange(21),
                    color=color,
                    histtype='step',
                    zorder=1,
                    linewidth=1)
        axs[i].hist(hist,
                    density=True,
                    bins=np.arange(21),
                    color=color,
                    histtype='bar',
                    alpha=0.5,
                    zorder=0,
                    label=BIAS_MODE_TO_LABEL[bias_mode])
        axs[i].axvline(conv + 0.5, linewidth=1, color='k', linestyle='--')
        if conv <= 5:
            offs = 1.1 if conv == 1 else 0.7
            if conv <= 3:
                axs[i].text(
                    conv + offs,
                    0.4,
                    '$\\leftarrow$ Desired\n      convergence = {}'.format(
                        conv),
                    va='center',
                    ha='left',
                    fontsize=7,
                    fontstyle="italic")
            else:
                axs[i].text(
                    conv + offs,
                    0.4,
                    '$\\leftarrow$ Desired\n      conv. = {}'.format(conv),
                    va='center',
                    ha='left',
                    fontsize=7,
                    fontstyle="italic")
        else:
            axs[i].text(
                conv + 0.4,
                0.4,
                'Desired $\\rightarrow$ \nconv. = {}       '.format(conv),
                va='center',
                ha='right',
                fontsize=7,
                fontstyle="italic")
#            else:
#                ax.text(conv, 0.5, 'Desired $\\rightarrow$\nConvergence = {}'.format(conv), va='center', ha='right', fontsize=8)
#            ax.set_xticks(np.arange(20) + 0.5)
#            ax.set_xticklabels(np.arange(20))
#            ax.set_ylabel('Freq.')
#            ax.set_ylim(0, 0.75)
        axs[i].set_ylim(0, 0.75)
        utils.outside_ticks(axs[i])
        if i + 1 < len(convs):
            axs[i].set_xticks([])
            i = i + 1

#ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(1.0, 1.0))
axs[-1].set_xlim(0, 10.0)
axs[-1].set_xlabel('Measured Go $\\to$ Granule Conv.')
axs[-1].set_xticks(np.arange(10) + 0.5)
axs[-1].set_xticklabels(np.arange(10))

axs[0].text(-0.025,
            1.7,
            "$\\textbf{D}$",
            va='top',
            ha='left',
            fontsize=12,
            color='black',
            transform=axs[0].transAxes,
            bbox=dict(facecolor='white', linewidth=0, pad=0.4))

#fig.suptitle('Desired vs. Measured PCN Convergence')
fig.tight_layout(h_pad=-0.2)
utils.save(fig)

