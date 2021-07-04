def brighten(c, f):
    return (1.0 - f) * np.array(mpl.colors.to_rgb(c)[:3]) + f


def plot_xor(ax,
             f1=lambda x: x,
             f2=lambda x: x,
             sigma=lambda x: np.sign(x),
             res=200,
             n_levels=3,
             vmax=3,
             show_ylabels=False,
             letter=None,
             title=None,
             title_x=0.5):
    xs = np.linspace(-1.25, 1.25, res)
    xss, yss = np.meshgrid(xs, xs)

    zss = sigma(f1(xss) + f2(yss))
    c1 = mpl.cm.get_cmap('plasma')(0.925)
    c2 = utils.blues[0]  #mpl.cm.get_cmap('viridis')(0.3)

    levels = np.linspace(-vmax, vmax, n_levels)
    levels_cent = np.linspace(-vmax, vmax, n_levels - 1)

    def fac(x):
        x_min = levels_cent[(n_levels - 1) // 2]
        x_max = levels_cent[-1]
        if x_max == x_min:
            x_max += 1
        alpha = (x_max - np.abs(x)) / (x_max - x_min)
        return 0.5 - (0.5 - 0.0) * (1.0 - alpha)

    ax.contourf(xs,
                xs,
                zss,
                vmin=-1,
                vmax=1,
                levels=levels,
                colors=[
                    brighten(c1, fac(x)) if x < 0.0 else brighten(c2, fac(x))
                    for i, x in enumerate(levels_cent)
                ])
    ax.contour(xs,
               xs,
               zss,
               vmin=-1,
               vmax=1,
               levels=levels[1:-1],
               linestyles=['--'],
               colors=['k'],
               linewidths=[0.75])

    obj1 = ax.scatter([-1, 1], [1, -1],
                      marker='s',
                      color=c1,
                      edgecolor='k',
                      s=75,
                      linewidth=0.75,
                      label='Class $-1$',
                      zorder=2)
    obj2 = ax.scatter([-1, 1], [-1, 1],
                      marker='o',
                      color=c2,
                      edgecolor='k',
                      s=75,
                      linewidth=0.75,
                      label='Class $1$',
                      zorder=2)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(xs[0], xs[-1])
    ax.set_xticks([-1, 0, 1])
    ax.set_aspect(1)
    ax.set_xlabel("$x_1$", labelpad=1.0)
    if show_ylabels:
        ax.set_ylabel("$x_2$", labelpad=1.0)
    else:
        ax.set_yticklabels([])

    if not title is None:
        ax.set_title(title, x=title_x)

    if not letter is None:
        ax.text(-0.33 if show_ylabels else 0.0,
                1.195,
                f"\\textbf{{{letter}}}",
                ha="left",
                va="top",
                size=12,
                transform=ax.transAxes)

    utils.outside_ticks(ax)

    return obj1, obj2, ax


def mlp(x):
    xs = np.array((np.real(x), np.imag(x))).T

    N = 100
    Amax = np.random.uniform(0.5, 1, N)
    xi0 = np.random.uniform(-1.0, 1.0, N)
    alpha = Amax / (1.0 - xi0)
    beta = (-xi0 * Amax) / (1.0 - xi0)
    encoder = np.random.normal(0, 1, (N, 2))
    encoder /= np.linalg.norm(encoder, axis=1)[:, None]

    xs_eval = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    ys_eval = np.array([1, -1, -1, 1])

    Js = alpha * (xs_eval @ encoder.T) + beta
    As = np.clip(Js, 0, None)
    D = np.linalg.lstsq(As, ys_eval, rcond=100.0)[0]

    Js = alpha * (xs_eval @ encoder.T) + beta

    return np.clip(alpha * (xs @ encoder.T) + beta, 0, None) @ D.T


np.random.seed(47182)

fig, axs = plt.subplots(1,
                        5,
                        figsize=(7.55, 1.4),
                        gridspec_kw={
                            "wspace": 0.15,
                        })

fig_idx = -1

fig_idx += 1
_, _, ax = plot_xor(axs.flatten()[fig_idx],
                    letter=chr(ord("A") + fig_idx),
                    title="$\\sigma(w_1 x_1 + w_2 x_2 + \\beta)$",
                    show_ylabels=fig_idx == 0)
#for x1, x2 in np.random.uniform(-2, 2, (3, 2)):
#    ax.plot([x1, x2], [-1.25, 1.25], 'k:', linewidth=0.5, zorder=1)
#for y1, y2 in np.random.uniform(-2, 2, (3, 2)):
#    ax.plot([-1.25, 1.25], [y1, y2], 'k:', linewidth=0.5, zorder=1)

#fig_idx += 1
#plot_xor(axs.flatten()[fig_idx],
#         f1=lambda x: 0.5 * x**2 - 0.25,
#         f2=lambda x: 0.5 * x,
#         letter=chr(ord("A") + fig_idx),
#         title="$\\sigma(f_1(x_1) + f_2(x_2))$",
#         show_ylabels=fig_idx==0)

fig_idx += 1
plot_xor(axs.flatten()[fig_idx],
         f1=lambda x: np.cos(2.0 * np.pi * x),
         f2=lambda x: np.sin(2.0 * np.pi * x),
         letter=chr(ord("A") + fig_idx),
         title="$\\sigma(f_1(x_1) + f_2(x_2))$",
         title_x=0.6,
         show_ylabels=fig_idx == 0)

fig_idx += 1
plot_xor(axs.flatten()[fig_idx],
         f1=lambda x: x * 1.0,
         f2=lambda x: x * 1.0j,
         sigma=mlp,
         letter=chr(ord("A") + fig_idx),
         title="MLP $f(x_1, x_2)$",
         show_ylabels=fig_idx == 0)

fig_idx += 1
plot_xor(axs.flatten()[fig_idx],
         sigma=lambda x: np.sign(np.square(x) - 1.0),
         letter=chr(ord("A") + fig_idx),
         title="$\\sigma'(x_1 + x_2)$",
         show_ylabels=fig_idx == 0)

fig_idx += 1
obj1, obj2, _ = plot_xor(axs.flatten()[fig_idx],
                         f1=lambda x: x * 1.0,
                         f2=lambda x: x * 1.0j,
                         sigma=lambda x: np.real(x) * np.imag(x),
                         n_levels=11,
                         vmax=1.6,
                         letter=chr(ord("A") + fig_idx),
                         title="Product $x_1 x_2$",
                         show_ylabels=fig_idx == 0)

fig.legend([obj1, obj2], ["Class $-1$", "Class $1$"],
           ncol=2,
           loc="upper center",
           bbox_to_anchor=(0.5, 1.25))

utils.save(fig)

