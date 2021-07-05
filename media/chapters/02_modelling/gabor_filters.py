def gabor(x, y, f=2.0, sigma=0.25, theta=np.pi/4, psi=0.0, x0=0.0, y0=0.0):
    xp =  np.cos(theta) * x + np.sin(theta) * y - x0
    yp = -np.sin(theta) * x + np.cos(theta) * y - y0
    E = np.exp(-(np.square(xp) + np.square(yp)) / (2.0 * np.square(sigma)))
    S = np.cos(2.0 * np.pi * f * xp - psi)
    return E * S


np.random.seed(4889)

fig, axs = plt.subplots(2, 8, figsize=(7.8, 2.25), sharey=True, gridspec_kw={
    "wspace": 0.05,
    "hspace": -0.185,
})
xs, ys = np.linspace(-1, 1, 1024), np.linspace(-1, 1, 1024)
xss, yss = np.meshgrid(xs, ys)
for i, ax in enumerate(axs.flat):
    zss = gabor(xss, yss,
        sigma=np.power(10, np.random.uniform(-0.8, -0.25)),
        f=np.power(10, np.random.uniform(-0.25, 0.2)),
        theta=np.random.uniform(-np.pi, np.pi),
        psi=np.random.uniform(-np.pi, np.pi),
        x0=np.random.uniform(-0.125, 0.125),
        y0=np.random.uniform(-0.125, 0.125),
    )
    ax.imshow(zss, extent=[np.min(xs), np.max(xs), np.min(ys), np.max(ys)], cmap='RdBu', vmin=-1, vmax=1)
    ax.set_aspect(1)
#    for spine in ["left", "bottom", "top", "right"]:
#        ax.spines[spine].set_visible(True)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([-1, 0, 1])
#    ax.set_xticks([])
#    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if (i == 0) or (i == 8):
        ax.set_ylabel('$\\xi_2$', labelpad=0.5)
    ax.set_xlabel('$\\xi_1$', labelpad=0.125)
#    utils.outside_ticks(ax)
utils.save(fig)
