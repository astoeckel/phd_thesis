import scipy.interpolate

zs = [
    [2.0, 1.5, 2.0, 2.0, 2.5],
    [1.5, 1.2, 1.5, 0.0, 1.5],
    [0.9, 0.6, 0.3, 0.1, 1.0],
    [1.0, 0.9, 0.8, 0.5, 1.0],
    [1.5, 1.6, 1.0, 1.0, 1.5],
]
xs, ys = np.linspace(-1, 1, 5), np.linspace(1, -1, 5)
f = scipy.interpolate.interp2d(xs, ys, zs, kind='cubic')

def taylor_approx(f, x0, y0, eps=1e-6):
    def dx(f, x, y):
        return (f(x + eps, y) - f(x - eps, y)) / (2.0 * eps)

    def dy(f, x, y):
        return (f(x, y + eps) - f(x, y - eps)) / (2.0 * eps)

    z0 = f(x0, y0)

    dx0, dy0 = dx(f, x0, y0), dy(f, x0, y0)

    return (lambda x, y: z0 + (x - x0) * dx0 + (y - y0) * dy0), dx0[0], dy0[0]

fig, axs = plt.subplots(2, 3, figsize=(8.0, 5.0), gridspec_kw={
    "hspace": 0.1,
    "wspace": 0.4,
})
axs = axs.flatten()

x0, y0 = -0.5, 0.75
it = 0
xs_trace = []

xs, ys = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
xss, yss = np.meshgrid(xs, ys)
zss = f(xs, ys)
print(np.min(zss))

for i in range(6):
    for j in range([1, 1, 2, 5, 20, 1][i]):
        xs_trace.append((x0, y0))

        scale = np.exp(-(it - 10)**2 / (5.0**2))

        fp, dx0, dy0 = taylor_approx(f, x0, y0)
        zssp = np.square(fp(xss, yss) - np.min(zss)) + 3.5 * (np.square(xss - x0) + np.square(yss - y0))

        if j == 0:
            C = axs[i].contourf(xs, ys, zss, cmap='Blues', levels=7)
            axs[i].contour(xs, ys, zss, colors=['white'], levels=C.levels, linestyles=['--'], linewidths=[1.0])
            axs[i].contour(xs, ys, zssp, colors=['black'], levels=C.levels, linestyles=[':'], linewidths=[1.0])
            #axs[i].text(0.0, 1.0, f"Iteration ${it + 1}$", va="top", ha="left", transform=axs[i].transAxes)
            axs[i].set_title(f"\\textbf{{Iteration {it + 1}}}")
            axs[i].set_aspect(1)

            xs_trace_arr = np.array(xs_trace)
            axs[i].plot(xs_trace_arr[:, 0], xs_trace_arr[:, 1], 'k+-')

            axs[i].arrow(x0, y0, -0.2 * dx0, -0.2 * dy0, width=0.03, head_length=0.1, zorder=10, overhang=0.1, clip_on=False, color='k')
            axs[i].plot([x0], [y0], '+', color='white', markersize=13, markeredgewidth=4.5, zorder=11)
            axs[i].plot([x0], [y0], '+', color='black', markersize=10, markeredgewidth=2, zorder=12)

            iy, ix = np.unravel_index(np.argmin(zss, axis=None), zssp.shape)
            x0_min, y0_min = xs[ix], ys[iy]
            axs[i].plot([x0_min], [y0_min], 'o', color=utils.oranges[1], markeredgecolor='white', zorder=13)

            axs[i].set_xlim(-1, 1)
            axs[i].set_ylim(-1, 1)

#            axs[i].set_xticks(np.linspace(-1, 1, 5))
#            axs[i].set_xticks(np.linspace(-1, 1, 9), minor=True)
#            axs[i].set_yticks(np.linspace(-1, 1, 5))
#            axs[i].set_yticks(np.linspace(-1, 1, 9), minor=True)
#            axs[i].set_xticklabels([])
#            axs[i].set_yticklabels([])
            utils.remove_frame(axs[i])

        iy, ix = np.unravel_index(np.argmin(zssp, axis=None), zssp.shape)
        x0, y0 = xs[ix], ys[iy]

        it += 1

utils.save(fig)
