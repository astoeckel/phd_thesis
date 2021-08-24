import tqdm
import scipy.stats
import scipy.interpolate
import h5py

def apply_affine_trafo(xs_src, xs_tar, src, *tar):
    srcp = np.log10(np.median(src, axis=-1))
    tarp = np.log10(np.median(np.concatenate(tar, axis=-1), axis=-1))

    x_max = min(np.max(xs_src), np.max(xs_tar))
    x_min = max(np.min(xs_src), np.min(xs_tar))
    xi = np.log10(np.array((x_min, x_max)))

    srci = scipy.interpolate.interp1d(np.log10(xs_src), srcp)(xi)
    tari = scipy.interpolate.interp1d(np.log10(xs_tar), tarp)(xi)

    A = np.array((
        np.ones(2),
        srci,
    )).T
    b = tari

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    return np.power(10, np.log10(src) * x[1] + x[0])

#def plot_ks(xs, ys):
#    fig, ax = plt.subplots()
#    min_x, max_x = np.min((xs, ys)), np.max((xs, ys))
#    fx = scipy.interpolate.interp1d(sorted(xs), np.linspace(0, 1, len(xs)), kind='nearest', fill_value='extrapolate')
#    fy = scipy.interpolate.interp1d(sorted(ys), np.linspace(0, 1, len(ys)), kind='nearest', fill_value='extrapolate')
#    smpls = np.linspace(min_x, max_x, 1000)
#    ax.plot(smpls, fx(smpls))
#    ax.plot(smpls, fy(smpls))
#    D = np.max(np.abs(fx(smpls) - fy(smpls)))
#    scale = np.sqrt((len(xs) + len(ys)) / (len(xs) * len(ys)))
#    print("p =", 2.0 * np.exp(-2.0 * np.square(D / scale)))
#    print("p* =", scipy.stats.kstest(xs, ys, mode='auto'))

# Load the main data
params_keys = []
errs = None
files = [
    "nlif_frequency_sweep_0.h5",
    "nlif_frequency_sweep_1.h5",
    "nlif_frequency_sweep_2.h5",
    "nlif_frequency_sweep_3.h5",
    "nlif_frequency_sweep_4.h5",
    "nlif_frequency_sweep_5.h5",
    "nlif_frequency_sweep_6.h5",
    "nlif_frequency_sweep_7.h5",
    "nlif_frequency_sweep_8.h5",
    "nlif_frequency_sweep_9.h5",
]
figsize = (7.4, 3.0)
params_keys = [
    "one_comp",
    "two_comp",
    "three_comp",
    "four_comp",
]
for file in files:
    with h5py.File(utils.datafile(file), 'r') as f:
        sigmas = f['sigmas'][()]
        if errs is None:
            errs = np.array(f['errs'][()])
        else:
            errs = np.concatenate((errs, f['errs'][()]), axis=-1)

# Load the reference data from the function sweep experiment
with h5py.File(utils.datafile('dendritic_computation_fourier_example_d9.h5'), 'r') as f:
    sigmas_theo = f["SIGMAS"][()]
    Es_add = f["Es_add"][()]
    Es_mul = f["Es_mul"][()]
    Es_a2d = f["Es_mlp"][()]

def lighten(c, f):
    f = 1.0 + f;
    return np.clip(c * np.array((f, f, f, 1.0)), 0.0, 1.0)

colors = ['#000000', mpl.cm.get_cmap('viridis')(0.3), utils.oranges[1], utils.reds[1]]

styles = {
    "add": {
        "color": "k",
        "linewidth": 1.0,
        "linestyle": ":",
        "zorder": -1,
    },
    "mul": {
        "color": "k",
        "linewidth": 1.0,
        "linestyle": ":",
        "zorder": -1,
    },
    "a2d": {
        "label": "",
        "color": "k",
        "linestyle": ":",
        "linewidth": 1.0,
        "zorder": -1,
    },
    "one_comp": {
        "label": "One-compartment LIF",
        "color": "k",
        "linewidth": 1.5,
        "marker": 'o',
        "markersize": 4,
        "markevery": (0, 5),
        "markeredgewidth": 0.5,
        "zorder": 9,
    },
    "two_comp": {
        "label": "Two-compartment LIF",
        "color": colors[1],
        "linewidth": 1.5,
        "marker": 's',
        "markersize": 4,
        "markevery": (0, 6),
        "markeredgewidth": 0.5,
    },
    "three_comp": {
        "label": "Three-compartment LIF",
        "color": colors[2],
        "linewidth": 1.5,
        "marker": 'd',
        "markersize": 4,
        "markevery": (2, 6),
        "markeredgewidth": 0.5,
    },
    "four_comp": {
        "label": "Four-compartment LIF",
        "color": colors[3],
        "linewidth": 1.5,
        "marker": 'v',
        "markersize": 4,
        "markevery": (4, 6),
        "markeredgewidth": 0.5,
    },
}
lbls = 1 / sigmas
lbls_theo = 1 / sigmas_theo

fig, ax = plt.subplots(figsize=figsize)

for i_param in range(0, len(params_keys)):
    key = params_keys[i_param]
    if key in styles:
        style = dict(**styles[key])
        data = np.median(errs[i_param, :, :], axis=1)
        style_1 = dict(style)
        style_2 = dict(style)
        style_2["label"] = None
        style_2["zorder"] = 10
        style_2["linestyle"] = ''
        ax.plot(lbls, data, markeredgecolor="k", **style_1)
        ax.plot(lbls, data, markeredgecolor="k", **style_2)
        if (key == 'one_comp' or key == 'two_comp' or key == 'three_comp'):
            perc_25 = np.percentile(errs[i_param, :, :], 25, axis=1)
            perc_75 = np.percentile(errs[i_param, :, :], 75, axis=1)
            ax.fill_between(lbls, perc_25, perc_75, color=style["color"], alpha=0.4, lw=0, zorder=-2)

#        for i_smpl in tqdm.tqdm(range(len(lbls))):
#            stats = scipy.stats.bootstrap((errs[i_param, i_smpl, :],), np.median, axis=0)
#            y_low, y_high = stats.confidence_interval.low, stats.confidence_interval.high
#            y = 0.5 * (y_low + y_high)
#            y_pm = 0.5 * (y_high - y_low)
#            ax.errorbar(lbls[i_smpl], y, y_pm, color=style["color"], capsize=2, linewidth=0.5, capthick=0.5)

for idx, (i_param0, i_param1) in enumerate([(2, 3), (2, 1)]):
    key0, key1 = params_keys[i_param0], params_keys[i_param1]
    style_0 = dict(**styles[key0])
    style_1 = dict(**styles[key1])
    for style in [style_0, style_1]:
        del style["markevery"]
        del style["label"]
        style["markeredgecolor"] = 'k'
    y = 1.075 + 0.075 * idx
    ax.plot(-0.0275, y, **style_0, clip_on=False, transform=ax.transAxes)
    ax.plot(-0.0475, y, **style_1, clip_on=False, transform=ax.transAxes)
    ax.text(-0.0425, y, '$p(\\;\\;, \\;\\;)$', va="center", ha="center", transform=ax.transAxes)

    ax.plot([0, 1], [y, y], ':', color='gray', linewidth=0.5, transform=ax.transAxes, clip_on=False, zorder=-10)

    y = np.power(10, 0.22 + 0.175 * idx)
    for i_smpl in range(len(lbls)):
        if lbls[i_smpl] > 10.0:
            continue
#        if idx == 0 and i_smpl == 30:
#            plot_ks(errs[i_param0, i_smpl, :], errs[i_param1, i_smpl, :])
        stats = scipy.stats.kstest(errs[i_param0, i_smpl, :], errs[i_param1, i_smpl, :])
        if stats.pvalue < 0.001:
            ax.plot(lbls[i_smpl], y, "*", markersize=4, markeredgewidth=0.0, color='k', clip_on=False)

red = np.median(1.0 - errs[2] / errs[1], axis=-1)
print("Maximum reduction in error", np.max(red) * 100, lbls[np.argmax(red)])

ax.plot(lbls_theo, np.median(Es_mul, axis=-1), **styles["mul"])
ax.plot(lbls_theo, np.median(Es_add, axis=-1), **styles["add"])
#ax.plot(lbls_theo, np.median(Es_a2d, axis=-1), **styles["a2d"])

ax.set_xlabel('Spatial lowpass filter coefficient $\\rho^{-1}$')
ax.set_ylabel('NRMSE')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-1, 1e1)
ax.set_ylim(5e-3, 1.1)

ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.0))

y_cen = -0.1175 * (3.0 / figsize[1])
arrow_width = 0.015

ax.add_patch(
    mpl.patches.Polygon([(0.0, y_cen), (arrow_width, y_cen - 0.02),
                         (arrow_width, y_cen + 0.02)],
                        facecolor='k',
                        transform=ax.transAxes,
                        clip_on=False))
ax.add_patch(
    mpl.patches.Polygon([(1.0, y_cen), (1.0 - arrow_width, y_cen - 0.02),
                         (1.0 - arrow_width, y_cen + 0.02)],
                        facecolor='k',
                        transform=ax.transAxes,
                        clip_on=False))
ax.text(1.5 * arrow_width,
        y_cen - 0.005,
        'Lower frequencies',
        va='center',
        ha='left',
        transform=ax.transAxes)
ax.text(1.0 - 1.5 * arrow_width,
        y_cen - 0.005,
        'Higher frequencies',
        va='center',
        ha='right',
        transform=ax.transAxes)

ax.set_title("\\textbf{Static model error $E_\\mathrm{model}$}", y=1.2)

utils.annotate(ax, 10.0**(-0.425), 10.0**(-0.85), 10.0**(-0.75), 10.0**(-0.25), 'Additive baseline')
utils.annotate(ax, 10.0**(0.125), 10.0**(-1.0), 10.0**(0.5), 10.0**(-1.0), 'Multiplicative baseline')
#utils.annotate(ax, 10.0**(0.45), 10.0**(-1.25), 10.0**(0.7), 10.0**(-1.25), '2D basis baseline')

utils.save(fig)

