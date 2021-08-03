def halton(i, b):
    f = 1
    r = 0
    while i > 0:
        f = f / b
        r = r + f * (i % b)
        i = i // b
    return r


def draw_stepfun(ax, xi, e, **kwargs):
    if e > 0.0:
        ax.plot([-1.0, xi], [0.0, 0.0], **kwargs)
        ax.plot([xi, 1.0], [1.0, 1.0], **kwargs)
        ax.plot([xi], [1.0], marker='o', markersize=3.0, **kwargs)
        ax.plot([xi], [0.0], marker='o', fillstyle="none", markersize=3.0, **kwargs)
    else:
        ax.plot([-1.0, xi], [1.0, 1.0], **kwargs)
        ax.plot([xi, 1.0], [0.0, 0.0], **kwargs)
        ax.plot([xi], [0.0], marker='o', markersize=3.0, **kwargs)
        ax.plot([xi], [1.0], marker='o', fillstyle="none", markersize=3.0, **kwargs)

    yoffs = 0.2 * (np.floor(6.0 * xi) - 6.0 * xi)

    ax.plot([xi, xi], [0.0, 1.0], ':', **kwargs)
    ax.arrow(xi, 0.5 + yoffs, 0.1 * e, 0.0, head_width=0.02, linewidth=0.5, **kwargs)
    ax.scatter(xi, 0.5 + yoffs, marker='o', s=3, **kwargs)


def draw_intercept2d(ax, mx, my, ex, ey, **kwargs):
    if (np.abs(ey) < 1e-6) or (np.abs(ex) < 1e-6):
        xi = 1.0
    else:
        xi = max(
            np.abs((mx + 1.0) / ey),
            np.abs((mx - 1.0) / ey),
            np.abs((my + 1.0) / ex),
            np.abs((my - 1.0) / ex),
        )

    xis = np.linspace(-1, 1, 11) * xi
    ax.plot(mx - ey * xis, my + ex * xis, '-', **kwargs)

    ax.arrow(mx, my, 0.1 * ex, 0.1 * ey, head_width=0.02, linewidth=0.5, **kwargs)
    ax.scatter(mx, my, marker='o', s=3, **kwargs)


fig, axs = plt.subplots(1, 2, figsize=(7.45, 4.0), gridspec_kw={
    "wspace": 0.3,
})

xs = np.linspace(-1, 1, 1000)

seed = 3819
np.random.seed(29901)

for xi in (np.linspace(-1, 1, 11) + np.random.normal(0, 0.05, 11)):
    e = np.random.choice([-1, 1])
    draw_stepfun(axs[0], xi, e, color=np.ones(3) * 0.75, zorder=-1)

axs[0].add_patch(
    mpl.patches.Polygon([(0.3, 0.0), (0.3, 1.0),
                         (0.5, 1.0), (0.5, 0.0)],
                        facecolor=mpl.cm.get_cmap('RdBu')(0.8),
                        zorder=-10,
                        clip_on=True))

draw_stepfun(axs[0], 0.3, 1.0, color='k')
draw_stepfun(axs[0], 0.5, 1.0, color='k')

axs[0].set_xlim(-1, 1)
axs[0].set_ylim(-0.005, 1.3)
axs[0].set_xticks(np.linspace(-1, 1, 9), minor=True)
axs[0].set_yticks(np.linspace(0, 1, 11), minor=True)
axs[0].set_xlabel('Represented value $x$')
axs[0].set_ylabel('Rate $a_i(x)$')

axs[0].set_aspect(1.53)

axs[0].text(0.025, 0.975, '\\textbf{A}', ha="left", va="top", size=12, transform=axs[0].transAxes, bbox={
    "color": "white",
    "pad": 0.5,
})

xs = np.linspace(-1, 1, 256)
xss, yss = np.meshgrid(xs, xs)
As = []

seed, N = 7878, 21

np.random.seed(seed)
for i in range(N):
    mx = 2.0 * halton(i, 2) - 1.0
    my = 2.0 * halton(i, 3) - 1.0
    alpha = np.random.uniform(-np.pi, np.pi)
    ex = np.cos(alpha)
    ey = np.sin(alpha)

    As.append(
        ((xss - mx) * ex + (yss - my) * ey) > 0.0
    )

Ass = np.array(As)
ox, oy = 0.15, -0.1
tar = np.exp(-(np.square(xss - ox) + np.square(yss - oy)) * 10.0) > 0.6
ws = np.linalg.lstsq(Ass.reshape(Ass.shape[0], -1).T, tar.reshape(-1), rcond=None)[0]

axs[1].contourf(xs, xs, Ass.transpose(1, 2, 0) @ ws, zorder=-2, vmin=-0.5, vmax=0.5, cmap='RdBu')
axs[1].plot([ox], [oy], 'k+')

np.random.seed(seed)
for i in range(N):
    mx = 2.0 * halton(i, 2) - 1.0
    my = 2.0 * halton(i, 3) - 1.0
    alpha = np.random.uniform(-np.pi, np.pi)
    ex = np.cos(alpha)
    ey = np.sin(alpha)

    w = np.abs(ws[i])
    if w > np.percentile(np.abs(ws), 80):
        color = 'k'
        zorder = 1
    else:
        color = np.ones(3) * 0.75
        zorder = -1
    draw_intercept2d(axs[1], mx, my, ex, ey, color=color, zorder=zorder)

axs[1].set_xlim(-1, 1)
axs[1].set_ylim(-1, 1)
axs[1].set_xlabel('Represented value $x_1$')
axs[1].set_ylabel('Represented value $x_2$')
axs[1].text(0.025, 0.975, '\\textbf{B}', ha="left", va="top", size=12, transform=axs[1].transAxes, bbox={
    "color": "white",
    "pad": 0.5,
})
axs[1].set_aspect(1)

utils.save(fig)


