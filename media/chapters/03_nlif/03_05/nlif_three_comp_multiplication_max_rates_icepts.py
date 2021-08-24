import numpy as np
import pickle
import nengo_extras.plot_spikes

import lif_utils

with open(
        utils.datafile(
            "nlif_three_comp_multiplication_spike_data_default_icepts.pkl"),
        "rb") as f:
    res = pickle.load(f)

lif_params = {
    'gL': 5e-08,
    'EL': -0.065,
    'v_th': -0.05,
    'v_reset': -0.065,
    'tau_ref': 0.003,
    'Cm': 1e-09
}

E = res["Ptar"]["E"]
gains = res["Ptar"]["gains"]
biases = res["Ptar"]["biases"]

Jtar = res["Jtar"] * 1e9
Jtar_opt = res["Jtar_opt"] * 1e9

i_th = 0.5625

# Account for subthreshold relaxation
J1, J2 = np.copy(Jtar), np.copy(Jtar_opt)
J1[Jtar < i_th] = i_th
J2[np.logical_and(Jtar < i_th, Jtar_opt < i_th)] = 0.0
errs_orig = np.sqrt(np.mean(np.square(J1 - J2), axis=0))
errs = errs_orig / np.max(errs_orig)
errs_nrmse = errs_orig / np.sqrt(np.mean(np.square(J1), axis=0))

fig = plt.figure(figsize=(7.4, 2.0))
gs = fig.add_gridspec(2, 6, wspace=0.8, width_ratios=[1, 1, 1, 1, 1.5, 1.5])
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax2 = fig.add_subplot(gs[0:2, 2:4])

cgs = fig.add_gridspec(1,
                       6,
                       wspace=0.8,
                       top=-0.1,
                       bottom=-0.15,
                       width_ratios=[1, 1, 1, 1, 1.5, 1.5])
cax1 = fig.add_subplot(cgs[0, 0:4])
cax1.spines["left"].set_visible(False)
cax1.spines["bottom"].set_visible(False)
cax1.set_yticks([])
cax2 = fig.add_subplot(cgs[0, 4:6])

xs = np.linspace(-1, 1, 1000)
Js = (xs[:, None] @ E) * gains + biases
As = lif_utils.lif_detailed_rate(Js, **lif_params)

idcs = np.argsort(errs)

for i in range(As.shape[1]):
    ax1.plot(xs,
             As[:, idcs[i]],
             color=mpl.cm.get_cmap('viridis')(errs[idcs[i]]),
             zorder=-1000 + i)
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 100)
ax1.set_xlabel("Represented value $x$")
ax1.set_ylabel("Firing rate $a_i(x)$")
ax1.set_title("\\textbf{Target tuning curves}")

cax1.imshow(np.linspace(0, 1, 100).reshape(1, -1),
            extent=[0, 1, 0, 1],
            cmap='viridis')
cax1.set_aspect('auto')
cax1.set_xlabel("Decoding error (relative to max.)")

ax2.scatter(res["Ptar"]["icpts"],
            res["Ptar"]["maxrs"],
            c=errs,
            cmap='viridis',
            vmin=0.0,
            vmax=1.0)
ax2.set_xlim(-1, 1)
ax2.set_ylim(50, 100)
ax2.set_xlabel("$x$-intercept $\\xi_0$")
ax2.set_ylabel("Maximum rate $a_\\mathrm{max}$")
ax2.set_title("\\textbf{Tuning parameters}")


def do_plot_contour(ax, idx, lbl, cax=None):
    xs = np.linspace(-1, 1, 33)
    ys = np.linspace(-1, 1, 33)
    zs = Jtar[:, idcs[idx]].reshape(33, 33)
    zsp = np.clip(Jtar_opt[:, idcs[idx]].reshape(33, 33), 0.5265, None)
    levels = np.linspace(0.5, 2, 13)

    mpl.rcParams['hatch.linewidth'] = 1.0
    rect = mpl.patches.Rectangle((0, 0),
                                 1,
                                 1,
                                 transform=ax.transAxes,
                                 color=(0.6, 0.6, 0.6),
                                 zorder=-100,
                                 linewidth=0)
    ax.add_artist(rect)
    rect = mpl.patches.Rectangle((-1, -1),
                                 2,
                                 2,
                                 transform=ax.transAxes,
                                 color=(1.0, 1.0, 1.0, 0.5),
                                 fill=False,
                                 hatch="///",
                                 zorder=-99,
                                 linewidth=1)
    ax.add_artist(rect)

    C = ax.contourf(xs, ys, zs, cmap='inferno', levels=levels)
    ax.contour(xs,
               ys,
               zsp,
               linewidths=[1.0],
               linestyles=['--'],
               colors=['white'],
               levels=levels)
    ax.set_aspect(1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    if not cax is None:
        cb = plt.colorbar(C, ax=ax, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.set_ticks(np.arange(0.5, 2.1, 0.5))
        cax.set_xlabel("Somatic current $J$")

    x, y, z = res["Ptar"]["icpts"][idcs[idx]], res["Ptar"]["maxrs"][
        idcs[idx]], errs[idcs[idx]]
    ax2.scatter(x,
                y,
                c=z,
                edgecolor='k',
                linewidth=1.0,
                s=50.0,
                cmap='viridis',
                vmin=0.0,
                vmax=1.0,
                zorder=10)
    ax2.text(x,
             y - 0.25,
             "\\textbf{" + lbl + "}",
             ha="center",
             va="center",
             size=7,
             zorder=100,
             color="k" if z > 0.5 else "white")

    ax.scatter(0.125,
               0.875,
               c=z,
               transform=ax.transAxes,
               edgecolor='k',
               linewidth=1.0,
               s=50.0,
               cmap='viridis',
               vmin=0.0,
               vmax=1.0,
               zorder=200)
    ax.text(0.125,
            0.875 - 0.01,
            "\\textbf{" + lbl + "}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            size=7,
            zorder=201,
            color="k" if z > 0.5 else "white")

    ax.text(0.25, 0.8375, "$E = {:0.1f}\\%$".format(errs_nrmse[idcs[idx]] * 100.0), size=8, transform=ax.transAxes, bbox={
        "color": "white",
        "pad": 0.125,
    })
    rect = mpl.patches.Rectangle((0.03, 0.7875), 0.225, 0.174, color='white', linewidth=0, zorder=99, transform=ax.transAxes)
    ax.add_artist(rect)


ax31 = fig.add_subplot(gs[0, 4])
ax32 = fig.add_subplot(gs[1, 4])
ax33 = fig.add_subplot(gs[0, 5])
ax34 = fig.add_subplot(gs[1, 5])

np.random.seed(7897)
sel = np.linspace(0, 100, 6, dtype=int)[1:-1] + np.random.uniform(
    -10, 10, 4).astype(int)

do_plot_contour(ax31, sel[0], "1", cax=cax2)
do_plot_contour(ax32, sel[1], "2")
do_plot_contour(ax33, sel[2], "3")
do_plot_contour(ax34, sel[3], "4")

ax31.set_xticklabels([])
ax31.set_xlabel("")
ax33.set_xticklabels([])
ax33.set_xlabel("")

ax31.set_title("\\textbf{Current functions example}", x=1.4)

ax1.text(-0.275,
         1.055,
         "\\textbf{A}",
         size=12,
         transform=ax1.transAxes,
         va="baseline",
         ha="left")
ax2.text(-0.275,
         1.055,
         "\\textbf{B}",
         size=12,
         transform=ax2.transAxes,
         va="baseline",
         ha="left")
ax31.text(-0.5,
          1.12,
          "\\textbf{C}",
          size=12,
          transform=ax31.transAxes,
          va="baseline",
          ha="left")

utils.save(fig)

