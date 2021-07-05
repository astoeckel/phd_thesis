import scipy.interpolate

# Generate some random-looking monotonic function
np.random.seed(11987)
N = 11

ys1 = np.concatenate(((0, ), np.cumsum(np.abs(np.random.normal(0, 1, N - 1)))))
ys1 /= np.max(ys1)

ys2 = np.concatenate(((0, ), np.cumsum(np.abs(np.random.normal(0, 1, N - 1)))))
ys2 /= np.max(ys2)

xs = np.linspace(0, 1, N)
f1 = scipy.interpolate.interp1d(xs, ys1, 'cubic')
f2 = scipy.interpolate.interp1d(xs, ys2, 'cubic')



def plot_multiplication_traj(ax, x2, N=100):
    x1ss = np.linspace(0, 1, 40)
    cmap = mpl.cm.get_cmap('Blues')
    for i in range(0, len(x1ss) - 1):
        x10, x11 = x1ss[i], x1ss[i + 1]
        x1c = 0.5 * (x10 + x11)
        c = x1c * x2
        ax.plot([x10, x11],
                [f1(x10) + f2(x2), f1(x11) + f2(x2)],
                color=cmap(c),
                linewidth=4, solid_capstyle='round')

    x1s = np.linspace(0, 1, 1000)
    ax.plot(x1s, f1(x1s) + f2(x2), 'k-', linewidth=1)


fig, ax = plt.subplots(1, 1, figsize=(6.0, 1.75))
xs = np.linspace(0, 1, 100)

plot_multiplication_traj(ax, 0)
ax.set_xlim(0, 1)
ax.set_xlabel("$x_1$")
ax.axhline(f1(0.0) + f2(0), color="k", linestyle="--", linewidth=0.5)
ax.axhline(f1(1.0) + f2(0), color="k", linestyle="--", linewidth=0.5)
utils.annotate(ax, 0.625, 0.475, 0.675, 0.375, "$f_1(x_1) + f_2(0)$", ha="left")
utils.vslice(ax, -0.025, f1(0.0) + f2(0), f1(1.0) + f2(0))
ax.text(-0.05,
        f1(1.0) + f2(0),
        "$\\xi_\\mathrm{max} + f_2(0)$",
        va="center",
        ha="right")
ax.text(-0.05,
        f1(0.0) + f2(0),
        "$\\xi_\\mathrm{min} + f_2(0)$",
        va="center",
        ha="right")

delta = 0.6

plot_multiplication_traj(ax, delta)
utils.annotate(ax,
               0.2,
               0.8,
               0.15,
               1.2,
               "$f_1(x_1) + f_2(\\delta)$",
               ha="right")
ax.axhline(f1(0.0) + f2(delta), color="k", linestyle="--", linewidth=0.5)
ax.axhline(f1(1.0) + f2(delta), color="k", linestyle="--", linewidth=0.5)
utils.vslice(ax, 1.025, f1(0.0) + f2(delta), f1(1.0) + f2(delta))
ax.text(1.05,
        f1(1.0) + f2(delta),
        "$\\xi_\\mathrm{max} + f_2(\\delta)$",
        va="center",
        ha="left")
ax.text(1.05,
        f1(0.0) + f2(delta),
        "$\\xi_\\mathrm{min} + f_2(\\delta)$",
        va="center",
        ha="left")

mpl.rcParams['hatch.linewidth'] = 0.5

ax.add_patch(
    mpl.patches.Polygon([(0.0, f1(0.0) + f2(delta)),
                         (0.0, f1(1.0) + f2(0.0)),
                         (1.0, f1(1.0) + f2(0.0)),
                         (1.0, f1(0.0) + f2(delta))],
                        hatch='//', edgecolor='k', linewidth=0.0, facecolor='None'))

ax.set_yticks([])
for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

utils.save(fig)

