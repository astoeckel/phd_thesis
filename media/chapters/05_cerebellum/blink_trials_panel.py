import pytry

v_th = 2e-3

def adjust_spines(ax, spines, out=2.5):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', out))
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def load_avg_cr_plot_experimental_data():
    import scipy.interpolate

    days = []
    for day in [1, 2, 3, 4, 5, 6, 8, 14]:
        fn = utils.datafile(f"manual/chapters/05_cerebellum/heiney_et_al_2014_fig3a/cr_day_{day}.txt")
        data = np.loadtxt(fn)

        ts = np.linspace(0, 300, 1000)
        ys = scipy.interpolate.interp1d(data[:, 0], data[:, 1], 'slinear', fill_value='extrapolate')(ts)
        days.append((day, ys))
    return ts * 1e-3, days

def plot_cr_day_summary(ax, ts, days, max_day=1000):
    cmap = mpl.cm.get_cmap("viridis")

    n_days = len(list(filter(lambda x: x[0] <= max_day, days)))

    lines = []
    for i, (day, ys) in enumerate(days):
        if day <= max_day:
            ax.plot(ts[ts < 0.25] * 1e3, ys[ts < 0.25], linewidth=1.5,
                    color=cmap(i / (n_days - 1)),
                    label='Day {}'.format(day))

    ax.legend(
        fontsize=8,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=0.75,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.05),
    )
    ax.axvline(250, ls='--', c=(0.5,0.5, 0.5), linewidth=2, color='#c0c0c0')
    ax.axvline(0.0, ls='--', c=(0.5,0.5, 0.5), linewidth=2, color='#c0c0c0')
    ax.text(0.0, 0.15, '$\\leftarrow$ \\textbf{CS}', ha='left', fontsize=9)
    ax.text(250, 0.8, '\\textbf{US} $\\rightarrow$', ha='right', fontsize=9)
    ax.set_ylim(0, 1)

def compute_per_day_data(data, trials_per_day=100):
    dt = data[-1]['sample_every']
    T  = data[-1]['period']
    t_delay = data[-1]['t_delay']
    n_trials = data[-1]['n_trials']
    n_days = int(n_trials / trials_per_day)
    N = int(T/dt)
    ts = np.arange(0, T, dt)

    days = {}
    for datum in data:
        # Fetch the data per trial
        purk = np.array(datum['purkinje']).reshape(-1, N).T

        # Compute the integral
        reflex_scale = 1.0/25
        v = np.clip(purk[:,:],0,np.inf)*dt/reflex_scale
        v[v < v_th] = 0.0
        pos = np.cumsum(v, axis=0)

        for day in range(n_days):
            trial0 = day * trials_per_day
            trial1 = (day + 1) * trials_per_day            
            avg = np.mean(pos[:, trial0:trial1], axis=1)
            if not (day + 1 in days):
                days[day + 1] = []
            days[day + 1].append(avg)

    res = []
    for day, arrs in days.items():
        avg = np.array(arrs).mean(axis=0)
        res.append((day, avg))

    return ts, res

def plot_cr_for_each_trial(axs, data, trials_per_day=100):
    # https://stackoverflow.com/a/14314054
    def moving_average(a, n=11) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    dt = data[-1]['sample_every']
    T  = data[-1]['period']
    t_puff = data[-1]['t_puff']
    t_delay = data[-1]['t_delay']
    n_delay = int((t_puff + t_delay)/dt)
    n_trials = data[-1]['n_trials']
    n_days = int(n_trials / trials_per_day)
    N = int(T/dt)
    ts = np.arange(0, T, dt)

    for i, datum in enumerate(data):
        # Fetch the data per trial
        purk = np.array(datum['purkinje']).reshape(-1, N).T

        # Compute the integral
        reflex_scale = 1.0/25
        v = np.clip(purk[:,:],0,np.inf)*dt/reflex_scale
        v[v < v_th] = 0.0
        pos = np.cumsum(v, axis=0)

        trials = np.arange(0, n_trials, dtype=int)

        ai, aj = i // 3, i % 3
        ax = axs[ai, aj]

        # Draw the dots into a raster graphics
        from PIL import Image, ImageDraw
        with Image.new("RGB", (185, 90), color='#FFFFFF') as im:
            draw = ImageDraw.Draw(im)
            for j, y in enumerate(pos[n_delay, :]):
                x = trials[j]
                ix, iy = im.width / 500.0 * x, im.height - (im.height / 1.0) * y
                r = 3
                draw.ellipse(((ix - r, iy - r), (ix + r, iy + r)), fill="#808080", outline=None)
            imdata = np.array(im)
            ax.imshow(imdata, extent=[0, 500, 0, 1], interpolation="sinc")
            ax.set_aspect('auto')

        ax.plot(trials[10:], moving_average(pos[n_delay, :]), linewidth=1, c='k', ls='-')
        for i in range(n_days + 1):
            ax.axvline(x=i * trials_per_day, linestyle=(0, (2.5, 1)), linewidth=0.5, color='#2abaf7', clip_on=False)

        adjust_spines(ax, ['left', 'bottom'])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.25)

def plot_cr_for_each_trial_empirical(axs):
    import imageio

    for ax, midx in zip(axs.flat, [14, 15, 17, 18, 19, 20]):
        fn = utils.datafile(f"manual/chapters/05_cerebellum/heiney_et_al_2014_fig3d/cr_m{midx}.png")
        imdata = imageio.imread(fn)
        ax.imshow(imdata, extent=[0, 500, 0, 1], interpolation="sinc")
        ax.set_aspect('auto')


def plot_velocity(ax, data):
    dt = data['sample_every']
    T  = data['period']
    t_puff = data['t_puff']
    t_delay = data['t_delay']
    n_delay = int((t_puff + t_delay)/dt)
    n_trials = data['n_trials']
    N = int(T/dt)
    ts = np.arange(0, T, dt)

    # Fetch the data per trial
    purk = np.array(data['purkinje']).reshape(-1, N).T

    # Compute the integral
    reflex_scale = 1.0/25
    v = np.clip(purk[:,:],0,np.inf)*dt/reflex_scale
    pos = np.cumsum(v, axis=0)

    cmap = mpl.cm.get_cmap('viridis')
    scale = 1000
    ax.plot(v[:,0]*scale, c=cmap(0/pos.shape[1]))
    for i in range(19, v.shape[1], 20):
        label = None
        if (i+1) % 100 == 0:
            label = 'Trial %d' % (i+1)
        ax.plot(v[:,i]*scale, c=cmap(i/pos.shape[1]), lw=1.5, label=label)
    ax.plot(data['reflex']*reflex_scale*scale, ls='--', lw=1.5, c='k', label='Reflex (UR)')
    ax.legend(
        fontsize=8,
        loc='upper left',
        ncol=2,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=0.75,
        bbox_to_anchor=(0.0, 1.05),
        )

def setup_figure():
    fig = plt.figure(figsize=(7.5, 4.0))

    gs1 = fig.add_gridspec(2, 3, left=0.175, right=0.525, top=0.85, bottom=0.575, hspace=0.75, wspace=0.5)
    axs1 = np.array([[fig.add_subplot(gs1[i, j]) for j in range(3)] for i in range(2)])

    gs2 = fig.add_gridspec(2, 3, left=0.175, right=0.525, top=0.375, bottom=0.1, hspace=0.75, wspace=0.5)
    axs2 = np.array([[fig.add_subplot(gs2[i, j]) for j in range(3)] for i in range(2)])

    gs3 = fig.add_gridspec(1, 2, left=0.6, right=0.95, top=0.85, bottom=0.567, hspace=0.75, wspace=0.2)
    axs3 = np.array([fig.add_subplot(gs3[0, j]) for j in range(2)])

    gs4 = fig.add_gridspec(1, 1, left=0.6, right=0.95, top=0.375, bottom=0.092, hspace=0.75, wspace=0.2)
    ax4 = fig.add_subplot(gs4[0, 0])

    for ax in np.array((axs1, axs2)).flat:
        utils.outside_ticks(ax)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 200, 400])
        ax.set_xticks(np.arange(0, 501, 100), minor=True)
        ax.set_yticks(np.linspace(0, 1, 3), minor=True)
        adjust_spines(ax, ["left", "bottom"])

    for ax in np.array((axs1[1], axs2[1])).flat:
        ax.set_xlabel("Trial")

    for i, ax in enumerate(axs3.flat):
        utils.outside_ticks(ax)
        if i != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Eylid closedness (\\%)")
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticks(np.linspace(0, 1, 5), minor=True)
        ax.set_xlim(0, 250)
        ax.set_xlabel("Time from CS onset (ms)")
        ax.set_xticks(np.linspace(0, 250, 6), minor=True)

    utils.outside_ticks(ax4)
    ax4.set_xlabel("Time from CS onset (ms)")
    ax4.set_xlim(0, 400)
    ax4.set_ylim(0, 10)
    ax4.set_yticks([0, 5, 10])
    ax4.set_yticks(np.linspace(0, 10, 5), minor=True)
    ax4.set_xticks(np.linspace(0, 400, 9), minor=True)
    ax4.set_ylabel("Eyelid velocity (mm/s)", labelpad=6.0)

    fig.text(0.3475, 0.88, "\\textbf{CR triggered eyelid control signal (model data)}", size=9, ha="center", va="baseline")
    fig.text(0.125, 0.88, "\\textbf{A}", size=12, ha="left", va="baseline")
    fig.text(0.14, 0.7125, "Fraction of eye closure", ha="right", va="center", rotation=90)


    fig.text(0.3475, 0.405, "\\textbf{CR triggered eyelid closure (empirical data)}", size=9, ha="center", va="baseline")
    fig.text(0.125, 0.405, "\\textbf{B}", size=12, ha="left", va="baseline")
    fig.text(0.14, 0.2375, "Fraction of eye closure", ha="right", va="center", rotation=90)

    fig.text(0.68, 0.88, "\\textbf{Model data}", size=9, ha="center", va="baseline")
    fig.text(0.5425, 0.88, "\\textbf{C}", size=12, ha="left", va="baseline")

    fig.text(0.875, 0.88, "\\textbf{Empirical data}", size=9, ha="center", va="baseline")
    fig.text(0.775, 0.88, "\\textbf{D}", size=12, ha="left", va="baseline")

    fig.text(0.775, 0.405, "\\textbf{Purkinje eyelid velocity decoding (model data)}", size=9, ha="center", va="baseline")
    fig.text(0.5425, 0.405, "\\textbf{E}", size=12, ha="left", va="baseline")

    return fig, axs1, axs2, axs3, ax4



from pytry.read import npz
data = [npz(utils.datafile(f"blink_trial_{i}.npz")) for i in range(6)]

fig, axs1, axs2, axs3, ax4 = setup_figure()

plot_cr_for_each_trial(axs1, data)
plot_cr_for_each_trial_empirical(axs2)

ts, days = compute_per_day_data(data) # Model
plot_cr_day_summary(axs3[0], ts, days, max_day=5)
ts, days = load_avg_cr_plot_experimental_data() # Empirical
plot_cr_day_summary(axs3[1], ts, days, max_day=5)

plot_velocity(ax4, data[0])

utils.save(fig)

