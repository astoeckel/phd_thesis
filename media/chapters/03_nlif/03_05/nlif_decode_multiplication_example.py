data = np.load(utils.datafile("nlif_decode_multiplication_example.npz"),
               allow_pickle=True)

fig, axs = plt.subplots(2, 4, figsize=(7.5, 5.25), gridspec_kw={"hspace": 0.9})

res = data[f"n{0}_res_test"]
xs1, xs2 = data[f"n0_xs1"], data[f"n0_xs2"]
As_pre = data[f"n0_As_test"]
As_pre = As_pre.reshape(res, res, As_pre.shape[-1])


def sort_tuning_curves(As):
    edge = np.abs((1 * (As > 0))[:-1] - (1 * (As > 0))[1:])
    edge_idcs = np.argmax(edge, axis=0)
    print(edge_idcs, np.argsort(edge_idcs))
    return np.argsort(edge_idcs)


#gs = fig.add_gridspec(1, 2, left=0.125, right=0.9, top=1.175, bottom=1.025, wspace=0.3)
gs = fig.add_gridspec(1,
                      2,
                      left=0.25,
                      right=0.775,
                      top=1.175,
                      bottom=1.025,
                      wspace=0.8)
ax1 = fig.add_subplot(gs[0, 0])
for idx, i in enumerate(sort_tuning_curves(As_pre[0, :, :101])):
    ax1.plot(xs1,
             As_pre[0, :, :101][:, i],
             color=mpl.cm.get_cmap('viridis')(idx / 100))
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Represented value $x_1$")
ax1.set_ylabel("Normalised $a_i(x_1)$")
ax1.set_xticks(np.linspace(-1, 1, 9), minor=True)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticks(np.linspace(0, 1, 5), minor=True)

ax2 = fig.add_subplot(gs[0, 1])
for idx, i in enumerate(sort_tuning_curves(As_pre[:, 0, 101:])):
    ax2.plot(xs2,
             As_pre[:, 0, 101:][:, i],
             color=mpl.cm.get_cmap('viridis')(idx / 101))
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Represented value $x_2$")
ax2.set_ylabel("Normalised $a_i(x_2)$")
ax2.set_xticks(np.linspace(-1, 1, 9), minor=True)
ax2.set_yticks([0, 0.5, 1])
ax2.set_yticks(np.linspace(0, 1, 5), minor=True)

cgs = fig.add_gridspec(1, 1, left=0.125, right=0.9, top=0.55, bottom=0.53)
cax = fig.add_subplot(cgs[0, 0])

for i in range(4):
    ax = axs[0, i]

    W = data[f"n{i}_W"]
    As_test = data[f"n{i}_As_test"]
    Js_test_tar, Js_test_dec = data[f"n{i}_Js_test_tar"], data[
        f"n{i}_Js_test_dec"]
    C = ax.contourf(xs1,
                    xs2,
                    Js_test_dec.reshape(res, res) * 1e9,
                    vmin=0.0,
                    vmax=1.0,
                    levels=np.linspace(0.0, 1.0, 11),
                    cmap='inferno')

    if i == 0:
        for c in C.collections:
            c.set_edgecolor("face")
        cbar = fig.colorbar(C, cax=cax, orientation="horizontal")
        cbar.outline.set_visible(False)
        cax.set_xlabel("Somatic current $J$ ($\\mathrm{nA}$)")

    utils.outside_ticks(ax)

    ax.contour(xs1,
               xs2,
               Js_test_tar.reshape(res, res) * 1e9,
               levels=C.levels,
               colors=['white'],
               linewidths=[1.0],
               linestyles=['--'])
    ax.set_aspect(1)

    ax.set_xticks(np.linspace(-1, 1, 5), minor=True)
    ax.set_yticks(np.linspace(-1, 1, 5), minor=True)
    ax.set_xticks(np.linspace(-1, 1, 3))
    ax.set_yticks(np.linspace(-1, 1, 3))
    if i == 0:
        ax.set_ylabel("$x_2$", labelpad=0)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("$x_1$")

    ax.set_title(
        "{" + ["Single comp.", "Two comp.", "Three comp.", "Four comp."][i] +
        "} " + f"($n = {i + 1}$)")

    #    ax.text(-0.15 if i > 0 else -0.3,
    #            1.065,
    #            "\\textbf{{{}}}".format(chr(ord('A') + i)),
    #            size=12,
    #            ha="left",
    #            va="baseline",
    #            transform=ax.transAxes)

    rms = np.sqrt(np.mean(np.square(Js_test_tar)))
    rmse = np.sqrt(np.mean(np.square(Js_test_dec - Js_test_tar)))
    nrmse = rmse / rms
    ax.text(0.05,
            0.95,
            "$E = {:0.2f}\\%$".format(nrmse * 100),
            ha="left",
            va="top",
            bbox={
                "color": "white",
                "pad": 0.1,
            },
            transform=ax.transAxes)

cgs = fig.add_gridspec(1, 4, left=0.125, right=0.9, top=0.0475, bottom=0.03)

for i in range(4):
    xs1, xs2 = data[f"n2_xs1"], data[f"n2_xs2"]
    res = data[f"n2_res_test"]
    W = data[f"n2_W"]
    As_test = data[f"n{i}_As_test"]
    Js_test_tar, Js_test_dec = data[f"n2_Js_test_tar"], data[f"n2_Js_test_dec"]

    ax = axs[1, i]

    g = (As_test @ W[i]).reshape(res, res)
    C = ax.contourf(xs1,
                    xs2,
                    g * 1e6,
                    levels=[
                        np.linspace(0.0, 0.4, 9),
                        np.linspace(0.0, 0.5, 11),
                        np.linspace(0.0, 2.0, 9),
                        np.linspace(0.0, 2.5, 11),
                    ][i],
                    cmap=['Blues', 'Oranges', 'Blues', 'Oranges'][i])
    for c in C.collections:
        c.set_edgecolor("face")

    cax = fig.add_subplot(cgs[0, i])
    cbar = fig.colorbar(C, cax=cax, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.set_ticks(C.levels[::2])
    cbar.set_label("Conductance ($\\si{\\micro\\siemens}$)")

    ax.contour(xs1,
               xs2,
               g.reshape(res, res) * 1e6,
               levels=C.levels,
               colors=['k'],
               linewidths=[0.7],
               linestyles=[':'])
    ax.set_aspect(1)

    ax.set_title([
        "Basal excitatory $g_\\mathrm{E}^1(x_1, x_2)$",
        "Basal inhibitory $g_\\mathrm{I}^1(x_1, x_2)$",
        "Apical excitatory $g_\\mathrm{E}^2(x_1, x_2)$",
        "Apical inhibitory $g_\\mathrm{I}^2(x_1, x_2)$"
    ][i])

    ax.set_xticks(np.linspace(-1, 1, 5), minor=True)
    ax.set_yticks(np.linspace(-1, 1, 5), minor=True)
    ax.set_xticks(np.linspace(-1, 1, 3))
    ax.set_yticks(np.linspace(-1, 1, 3))
    if i == 0:
        ax.set_ylabel("$x_2$")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("$x_1$")

fig.text(0.5,
         1.2125,
         '\\textbf{Pre-population tuning}',
         ha="center",
         va="baseline")
ax.text(
    0.2025,  #0.08,
    1.2125,
    "\\textbf{A}",
    size=12,
    ha="left",
    va="baseline",
    transform=fig.transFigure)

fig.text(0.5,
         0.915,
         '\\textbf{Predicted somatic currents}',
         ha="center",
         va="baseline")
ax.text(0.08,
        0.915,
        "\\textbf{B}",
        size=12,
        ha="left",
        va="baseline",
        transform=fig.transFigure)

fig.text(0.5,
         0.415,
         '\\textbf{Decoded conductances} (three-comp. LIF neuron, $n = 3$)',
         ha="center",
         va="baseline")
ax.text(0.08,
        0.415,
        "\\textbf{C}",
        size=12,
        ha="left",
        va="baseline",
        transform=fig.transFigure)

utils.save(fig)

