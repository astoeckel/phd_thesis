np.random.seed(39191)


def plot_result(fig,
                flt_rec,
                flt_pre,
                A,
                B,
                tars,
                high=5.1,
                T=10.0,
                ref_plot=False,
                E_yoffs=[0.0, 0.0],
                gs=None,
                axs=None):

    assert (gs is None) != (axs is None)

    np.random.seed(39191)

    if axs is None:
        axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)]
                        for i in range(2)])

    colors = [utils.blues[0], utils.oranges[1]]

    for j, input_type in enumerate(["step", "noise"]):
        # Simulate the dynamics
        ts, xs, ys = simulate_dynamics(flt_rec,
                                       flt_pre,
                                       A,
                                       B,
                                       input_type,
                                       high=high,
                                       T=T)

        # Generate the target signal
        dt = (ts[-1] - ts[0]) / (len(ts) - 1)
        T = ts[-1]

        # Plot everything
        if not ref_plot:
            ys_tar = np.array([
                scipy.signal.fftconvolve(xs[:, 0], tar(ts, dt),
                                         'full')[:len(ts)] * dt for tar in tars
            ]).T

            for i in range(ys.shape[1]):
                axs[0, j].plot(ts, ys[:, i], color=colors[i], lw=1.5)
                axs[0, j].plot(ts,
                               ys_tar[:, i],
                               color=colors[i],
                               linestyle=(1, (1, 2)),
                               lw=0.7,
                               zorder=10)
                axs[0, j].plot(ts,
                               ys_tar[:, i],
                               'white',
                               linestyle=(0, (1, 2)),
                               lw=0.7,
                               zorder=10)

            axs[1, j].plot(ts, xs, lw=1.0, color='k', clip_on=False)

            # Setup the axes
            axs[0, j].set_xticks([])
            axs[0, j].set_yticks([])
            axs[0, j].spines["bottom"].set_visible(False)
            for i in range(2):
                axs[i, j].set_xlim(0, T)
                axs[i, j].set_yticks([])
                axs[i, j].spines["left"].set_visible(False)
            axs[1, j].set_xticks(np.linspace(0, T, 3))
            axs[1, j].set_xticks(np.linspace(0, T, 5), minor=True)
            axs[1, j].set_xlabel("Time $t$ (s)")
            if j == 0:
                axs[1, j].set_ylim(0, 1)

            rmse = np.sqrt(np.mean(np.square(ys - ys_tar)))
            rms = np.sqrt(np.mean(np.square(ys_tar)))
            axs[0, j].text(1.0,
                           0.6 + E_yoffs[j],
                           "$E = {:0.1f}\\%$".format(
                               (rmse / rms) * 100).format(),
                           ha="right",
                           va="baseline",
                           transform=axs[0, j].transAxes,
                           bbox={"color": "white", "pad": 0.05})

            axs[0, 0].set_title("\\textit{Unit pulse}", y=1.1)
            axs[0, 1].set_title(
                "\\textit{Band-limited noise ($B = 5\\,\\mathrm{Hz}$)}",
                y=1.1)
        else:
            for i in range(ys.shape[1]):
                axs[0, j].plot(ts,
                               ys[:, i],
                               '-',
                               lw=0.7,
                               color=(0.5, 0.5, 0.5))

    return axs

