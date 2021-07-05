import gen_2d_fun

def rng():
    return np.random.seed(56897)

fig, axs = plt.subplots(3, 7, figsize=(8.0, 3.8), gridspec_kw={
    "hspace": 1.5,
})

for ax in axs.flat:
    ax.set_aspect(1)
    utils.outside_ticks(ax)
for ax in axs[0]:
    utils.remove_frame(ax)

res = 64

cmap = "RdBu"

ic = axs.shape[1] // 2
flt = gen_2d_fun.mk_2d_flt(sigma=1e-3, res=res)
img = gen_2d_fun.gen_2d_fun(flt=flt, res=res, rng=rng())
axs[0, ic].imshow(img, vmin=-2.5, vmax=2.5, interpolation='None', extent=[-1, 1, -1, 1], cmap=cmap)
axs[0, ic].spines["left"].set_visible(True)
axs[0, ic].spines["bottom"].set_visible(True)
axs[0, ic].set_xlim(-1.0, 1.0)
axs[0, ic].set_ylim(-1.0, 1.0)
axs[0, ic].set_xticks([-1, 0, 1])
axs[0, ic].set_yticks([-1, 0, 1])
#axs[0, ic].set_xlabel("$x_1$", labelpad=0.0)
#axs[0, ic].set_ylabel("$x_2$", labelpad=0.0)

for i, sigma in enumerate(np.logspace(-1, 1, axs.shape[1])):
    flt = gen_2d_fun.mk_2d_flt(sigma=1.0 / sigma, res=res)
    axs[1, i].imshow(np.outer(flt, flt), extent=[
        -len(flt) / res,
         len(flt) / res,
        -len(flt) / res,
         len(flt) / res,
    ], interpolation='bilinear', cmap=cmap, vmin=-np.max(np.square(flt)))
    axs[1, i].set_xlim(-1.0, 1.0)
    axs[1, i].set_ylim(-1.0, 1.0)
    axs[1, i].set_xticks([-1, 0, 1])
    axs[1, i].set_yticks([-1, 0, 1])
#    if i == 0:
#        axs[1, i].set_ylabel("$x_2$")
#    else:
#        axs[1, i].set_yticklabels([])
#    axs[1, i].set_xlabel("$x_1$", labelpad=0.0)
#    axs[1, i].set_ylabel("$x_2$", labelpad=0.0)

    axs[1, i].set_title("$\\sigma^{{-1}} = {:0.2f}$".format(sigma))

    img = gen_2d_fun.gen_2d_fun(flt=flt, res=res, rng=rng())
    axs[2, i].imshow(img, vmin=-2.5, vmax=2.5, interpolation='bilinear', extent=[-1, 1, -1, 1], cmap=cmap)
#    axs[2, i].set_xlabel("$x_1$", labelpad=0.0)
#    axs[2, i].set_ylabel("$x_2$", labelpad=0.0)


axs[0, ic].text(0.5, 1.15, "\\textbf{Unfiltered Gaussian Noise}", va="bottom", ha="center", transform=axs[0, ic].transAxes)
axs[1, ic].text(0.5, 1.45, "\\textbf{Gaussian Filter}", va="bottom", ha="center", transform=axs[1, ic].transAxes)
axs[2, ic].text(0.5, 1.25, "\\textbf{Function $\\varphi(x_1, x_2)$}", va="bottom", ha="center", transform=axs[2, ic].transAxes)

axs[0, 0].text(-0.5, 1.15, "\\textbf{A}", va="bottom", ha="left", transform=axs[0, 0].transAxes, size=12)
axs[1, 0].text(-0.5, 1.45, "\\textbf{B}", va="bottom", ha="left", transform=axs[1, 0].transAxes, size=12)
axs[2, 0].text(-0.5, 1.25, "\\textbf{C}", va="bottom", ha="left", transform=axs[2, 0].transAxes, size=12)

utils.save(fig)

