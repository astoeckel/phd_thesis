import h5py
import os

with h5py.File(utils.datafile('two_comp_weights_examples_statistics.h5'),
               'r') as f:
    E = f['E'][()]
    WE = f['WE'][()]
    WI = f['WI'][()]
    Ns = f['N'][()]


def plot(ax, xs, ys, color):
    obj = ax.plot(xs, np.median(ys, axis=-1), color=color)
    ax.fill_between(xs,
                    np.percentile(ys, 25, axis=-1),
                    np.percentile(ys, 75, axis=-1),
                    color=color,
                    alpha=0.5,
                    linewidth=0.0)
    return obj

def TeXexp(x):
    s = "{:0.2e}".format(x)
    s1, s2 = s.split("e")
    s2 = str(int(s2))
    return f"${s1}\\times 10^{{{s2}}}$"

def plot_ws(ax, ws, color, ay0=0.25, ay1=0.275):
    tot = len(ws)
    ws = ws[ws > 5e-6]
    n_bins = 15
    bins = np.logspace(-6, 1, n_bins)
    ax.hist(ws,
            color=color,
            bins=bins,
            weights=np.ones_like(ws) / tot,
            alpha=0.5,
            zorder=-1)
    ax.hist(ws,
            color=color,
            bins=bins,
            weights=np.ones_like(ws) / tot,
            facecolor='none',
            edgecolor=color,
            linewidth=0.7,
            zorder=0)
    ax.axvline(np.median(ws), color=color, linestyle='--')
    utils.annotate(ax, np.median(ws) * 1.5, ay0, np.median(ws) * 9.0, ay1, TeXexp(np.median(ws)))
    ax.set_xscale('log')
    ax.set_xlim(1e-6, 1e1)
    ax.set_xticks(np.logspace(-6, 0, 3))
    ax.set_xticks(np.logspace(-6, 0, 7), minor=True)
    ax.set_xticklabels([], minor=True)


fig = plt.figure(figsize=(6.4, 1.5))
gs1 = fig.add_gridspec(1, 2, left=0.05, right=0.45)
gs2 = fig.add_gridspec(1, 2, left=0.55, right=0.95)

axs = [
    fig.add_subplot(gs1[0, 0]),
    fig.add_subplot(gs1[0, 1]),
    fig.add_subplot(gs2[0, 0]),
    fig.add_subplot(gs2[0, 1]),
]

obj1 = plot(axs[0], Ns, E[:, 0, 0], utils.blues[0])
obj2 = plot(axs[0], Ns, E[:, 0, 1], utils.oranges[1])
axs[0].set_ylim(1e-3, 2)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title("{Addition}")
axs[0].set_ylabel("NRMSE $E$")
axs[0].set_xlabel("Number of pre-neurons $n$")
axs[0].text(-0.375,
            #-0.16,
            1.05,
            "\\textbf{A}",
            ha="left",
            va="bottom",
            size=12,
            transform=axs[0].transAxes)

axs[0].legend([
    mpl.lines.Line2D([], [], color=utils.blues[0]),
    mpl.lines.Line2D([], [], color=utils.oranges[1])
], ["Current-based LIF", "Two-compartment LIF"],
              loc="upper center",
              bbox_to_anchor=(2.25, 1.45),
              ncol=2)

plot(axs[1], Ns, E[:, 1, 0], utils.blues[0])
plot(axs[1], Ns, E[:, 1, 1], utils.oranges[1])
axs[1].set_ylim(1e-3, 2)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_title("{Multiplication}")
axs[1].set_xlabel("Number of pre-neurons $n$")
#axs[1].set_ylabel("NRMSE $E$")
axs[1].set_yticklabels([])
axs[1].text(#-0.2,
            -0.16,
            1.05,
            "\\textbf{B}",
            ha="left",
            va="bottom",
            size=12,
            transform=axs[1].transAxes)

ws = np.concatenate((WE[-1, 0, 1].flatten(), WI[-1, 0, 1].flatten())) * 1e-3
plot_ws(axs[2], ws, color=utils.oranges[1], ay0=0.21, ay1=0.25)
ws = np.concatenate((WE[-1, 0, 0].flatten(), WI[-1, 0, 0].flatten()))
plot_ws(axs[2], ws, color=utils.blues[1], ay0=0.22, ay1=0.29)

axs[2].set_ylim(0.0, 0.3)
axs[2].set_xlabel("Weight magnitude")
axs[2].set_ylabel("Frequency")
axs[2].text(-0.325,
            1.05,
            "\\textbf{C}",
            ha="left",
            va="bottom",
            size=12,
            transform=axs[2].transAxes)
axs[2].set_title("{Addition}")

#axs[2].legend([
#    mpl.patches.Rectangle((0, 0), 0, 0, color=utils.blues[0], alpha=0.5),
#    mpl.patches.Rectangle((0, 0), 0, 0, color=utils.oranges[0], alpha=0.5),
#], ["LIF", "Two-comp. LIF"],
#              loc="upper center",
#              bbox_to_anchor=(1.0, 1.55),
#              ncol=2)


ws = np.concatenate((WE[-1, 1, 1].flatten(), WI[-1, 1, 1].flatten())) * 1e-3
plot_ws(axs[3], ws, color=utils.oranges[1], ay0=0.19, ay1=0.25)
ws = np.concatenate((WE[-1, 1, 0].flatten(), WI[-1, 1, 0].flatten()))
plot_ws(axs[3], ws, color=utils.blues[1], ay0=0.2, ay1=0.29)
axs[3].set_ylim(0.0, 0.3)
axs[3].set_xlabel("Weight magnitude")
axs[3].text(-0.2,
            1.05,
            "\\textbf{D}",
            ha="left",
            va="bottom",
            size=12,
            transform=axs[3].transAxes)
axs[3].set_yticklabels([])
axs[3].set_title("{Multiplication}")

utils.save(fig)

