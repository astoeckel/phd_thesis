import nengo_extras.plot_spikes

from two_comp_sim_2d_fun_network import *

def reduce_spike_train(A, ss):
    res = np.zeros((A.shape[0] // ss, A.shape[1]))
    for i in range(ss):
        res += A[i::ss][:res.shape[0]]
    return res

def setup_axs(ax1, ax2, ax3):
    for ax in [ax1, ax2, ax3]:
        ax.set_aspect('auto')
        ax.set_xlim(0, 10)
    ax1.set_xlabel("Time $t$ ($\\mathrm{s}$)")
    ax1.set_title("Target population")
    ax1.set_xlim(0, 10)
    ax1.set_xticks(np.arange(0, 10.1, 0.5), minor=True)

    ax2.set_title("Input population $x_1$")
    ax2.set_xticklabels([])
    ax2.set_xlim(0, 10)
    ax2.set_xticks(np.arange(0, 10.1, 0.5), minor=True)

    ax3.set_title("Input population $x_2$")
    ax3.set_xticklabels([])
    ax3.set_xlim(0, 10)
    ax3.set_xticks(np.arange(0, 10.1, 0.5), minor=True)


f = lambda x, y: x + y
samples, ys, xs = generate_training_data_random_fun(0.8)

rng = np.random.RandomState(4988)
#f = lambda x, y: 0.5 * (x + y)
f = lambda x, y: 0.25 * (x + 1) * (y + 1)
res = run_single_spiking_trial("gc50_noisy",
                               f=f,
                               intermediate=False,
                               decoder_reg=1e-2,
                               reg=0.001,
                               rng=rng,
                               tau_pre_filt=0.001,
                               n_neurons=100,
                               mask_negative=True)


fig = plt.figure(figsize=(6.375, 4.0))
gs1 = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3], hspace=0.5, left=0.05, right=0.46)
gs2 = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.5, left=0.54, right=0.95)

ax2 = fig.add_subplot(gs1[0, 0])
ax3 = fig.add_subplot(gs1[1, 0])
ax1 = fig.add_subplot(gs1[2, 0])

ax1.plot(res["ts"],
         res["tar_dec_filt"],
         color=utils.blues[1],
         label="Decoded signal")
ax1.plot(res["ts"], res["tar_filt"], 'k', label="Filtered target")
#ax1.plot(res["ts"], res["tar_filt"], 'white', linestyle=(1, (1, 1)))
ax1.plot(res["ts"], res["tar"], 'k--', label="Target", linewidth=0.5)
ax1.set_ylabel("Output $x_1 x_2$")
ax1.legend(loc='upper left')

ax2.plot(res["ts"], res["xs"], label="$x_1(t)$", color='k')
ax2.set_ylabel("Input $x_1$")

ax3.plot(res["ts"], res["ys"], label="$x_2(t)$", color='k')
ax3.set_ylabel("Input $x_2$")

ax2.text(-0.16, 1.5, "\\textbf{{{}}}".format("A"), va="bottom", ha="left", size=12, transform=ax2.transAxes)
ax2.text(0.5, 1.5, "\\textbf{Decoded values}", va="bottom", ha="center", transform=ax2.transAxes)


setup_axs(ax1, ax2, ax3)

fig.align_labels([ax1, ax2, ax3])

ax2 = fig.add_subplot(gs2[0, 0])
ax3 = fig.add_subplot(gs2[1, 0])
ax1 = fig.add_subplot(gs2[2, 0])

N = res["Opre"].shape[1] // 2

def pp(A): # pre-process
    return nengo_extras.plot_spikes.cluster(res["ts"], A[:, ::2], filter_width=0.01)[1]

Opre1I = pp(res["Opre"][:, :N][:, res["pre_types"][1, :N]])
Opre1E = pp(res["Opre"][:, :N][:, res["pre_types"][0, :N]])
Opre1 = np.concatenate((Opre1E, Opre1I), axis=1)
n_inh_1 = int(Opre1I.shape[1])

Opre2I = pp(res["Opre"][:, :N][:, res["pre_types"][1, :N]])
Opre2E = pp(res["Opre"][:, :N][:, res["pre_types"][0, :N]])
Opre2 = np.concatenate((Opre2E, Opre2I), axis=1)
n_inh_2 = int(Opre2I.shape[1])

Otar = pp(res["Otar"])

p1 = mpl.patches.Rectangle([0.0, 0.5], 10.0, n_inh_1, color=utils.reds[1], alpha=0.2, linewidth=0)
ax2.imshow(reduce_spike_train(Opre1, 10).T,
           cmap='Greys',
           vmin=0.0,
           vmax=1000.0,
           extent=[0, 10, 0.5, Opre1.shape[1] + 0.5],
           origin='lower',
           interpolation='none',
           zorder=0)
ax2.add_artist(p1)
ax2.set_ylabel("Neuron index")

p2 = mpl.patches.Rectangle([0.0, 0.5], 10.0, n_inh_2, color=utils.reds[1], alpha=0.2, linewidth=0)
ax3.imshow(reduce_spike_train(Opre2, 10).T,
           cmap='Greys',
           vmin=0.0,
           vmax=1000.0,
           extent=[0, 10, 0.5, Opre2.shape[1] + 0.5],
           origin='lower',
           interpolation='none')
ax3.add_artist(p2)
ax3.set_ylabel("Neuron index")

ax1.imshow(reduce_spike_train(Otar, 10).T,
           cmap='Greys',
           vmin=0.0,
           vmax=1000.0,
           extent=[0, 10, 0.5, Otar.shape[1] + 0.5],
           origin='lower',
           interpolation='none')
ax1.set_ylabel("Neuron index")

ax2.text(-0.125, 1.3, "\\textbf{{{}}}".format("B"), va="bottom", ha="left", size=12, transform=ax2.transAxes)
ax2.text(0.5, 1.3, "\\textbf{Spike raster}", va="bottom", ha="center", transform=ax2.transAxes)

setup_axs(ax1, ax2, ax3)

utils.save(fig)

