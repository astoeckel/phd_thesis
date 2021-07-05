import h5py

fig = plt.figure(figsize=(7.05, 2.5))
gs = mpl.gridspec.GridSpec(2,
                           4,
                           height_ratios=[3, 1],
                           width_ratios=[15, 15, 15, 1],
                           hspace=0.15,
                           wspace=0.3)

with h5py.File(utils.datafile("generated/chapters/02_modelling/e24e001049840b77_low_firing_rates_dynamics_sweep.h5"), "r") as f:
    rates = f["rates"][()]
    data = f["results"][()]

    n_trials, n_types, n_rates, n_freqs = data.shape

    for i in range(n_types):
        ax0a = fig.add_subplot(gs[0, i], label=f"ax0a{i}")
        ax0b = fig.add_subplot(gs[0, i], label=f"ax0b{i}")
        ax1 = fig.add_subplot(gs[1, i])

        A = np.abs(data[:, i])
        ss = 100
        avg = np.mean(A / np.percentile(A, 95), axis=0)
        avg = np.mean(avg.reshape(n_rates, -1, ss), axis=-1)
        avg = 10.0 * np.log10(avg)

        fs = f["fs"][()]
        f0 = max(np.abs(fs[0]), np.abs(fs[-1]))
        fs = fs[::ss]

        ax0a.imshow(avg[::-1, :],
                    vmin=-6.0,
                    vmax=0.0,
                    cmap='inferno',
                    origin='lower',
                    extent=[
                        -f0,
                        f0,
                        rates[-1],
                        rates[0],
                    ])
        ax0a.set_aspect("auto")
        utils.remove_frame(ax0a)
        ax0b.set_xlim(-f0, f0)
        ax0b.set_ylim(200, 10)
        ax0b.set_yscale("log")
        #if i != 0:
        #    ax0b.set_yticklabels([])
        utils.add_frame(ax0b)
        ax0b.tick_params(axis="both",
                         direction="in",
                         bottom=True,
                         top=True,
                         left=True,
                         right=True,
                         which="both")

        cmap = mpl.cm.get_cmap("viridis")
        for idx, j in enumerate(np.linspace(0, n_rates - 1, 5, dtype=int)):
            colour = cmap(idx / 4)
            ax0b.plot([-f0, f0], [rates[j], rates[j]],
                      "--k",
                      clip_on=False,
                      linewidth=0.75)
            ax0b.plot([f0], [rates[j]],
                      "s",
                      color=colour,
                      zorder=100,
                      clip_on=False,
                      markersize=5)

            ax1.plot(fs, avg[j], color=colour, linewidth=1.25, zorder=-1)

            if i == 0:
                utils.annotate(
                    ax0b,
                    f0 - 2.5,
                    rates[j],
                    f0 - 20,
                    np.power(
                        10,
                        np.log10(rates[j]) - 0.2 *
                        (np.log10(rates[j]) - np.log10(rates[n_rates // 2]))),
                    "${:0.0f}\\,\\mathrm{{Hz}}$".format(rates[j]),
                    va="center",
                    ha="right",
                    zorder=101)

        ax0b.set_xticks(np.linspace(-f0, f0, 5))
        #ax0b.set_xlabel("Frequency $f$ (Hz)")
        ax0b.set_xticklabels([])
        ax1.set_xticks(np.linspace(-f0, f0, 5))
        ax1.set_xlabel("Frequency $f$ (Hz)")
        if i == 0:
            ax1.set_ylabel("Gain (dB)")
            ax0b.set_ylabel("Spike rate $a_\\mathrm{max}$ ($s^\\mathrm{-1}$)")
        ax1.set_xlim(-f0, f0)
        ax1.set_ylim(-5, 0.1)
        ax1.set_yticks([0, -5])
        ax1.set_yticks(np.arange(0, -6, -1), minor=True)

        ax0b.text(0 if i > 0 else -0.26,
                  1.055,
                  "\\textbf{{{}}}".format(chr(ord('A') + i)),
                  fontsize=12,
                  transform=ax0b.transAxes,
                  ha='left',
                  va='bottom')
        ax0b.set_title("{" + ["Spiking ReLU", "LIF Neuron", "Adaptive LIF"][i] + "}", y=1.0, va="bottom")

    cax = fig.add_subplot(gs[0:2, 3])
    vs = np.linspace(0, -6, 100).reshape((-1, 1))
    cax.imshow(vs, extent=[0, 1, -6, 0], cmap="inferno")
    cax.set_aspect("auto")
    cax.set_xticks([])
    cax.set_ylabel("Gain (dB)")
    utils.add_frame(cax)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

utils.save(fig)

