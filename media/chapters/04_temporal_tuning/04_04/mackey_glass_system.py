import lmu_utils

np.random.seed(5892)

fig, axs = plt.subplots(1, 4, figsize=(8.1, 1.5))

ys = lmu_utils.mackey_glass(500, 17, 1.0)[100:]
axs[0].axhline(0, linestyle=':', color='grey', lw=0.5)
axs[0].plot(ys, color='k', lw=1.0)

ys = lmu_utils.mackey_glass(5000, 17, 1.0)[100:]
axs[1].plot(ys[:-50], ys[50:], color='k', lw=0.3)

ys = lmu_utils.mackey_glass(500, 30, 1.0)[100:]
axs[2].axhline(0, linestyle=':', color='grey', lw=0.5)
axs[2].plot(ys, color='k', lw=1.0)

ys = lmu_utils.mackey_glass(5000, 30, 1.0)[100:]
axs[3].plot(ys[:-50], ys[50:], color='k', lw=0.3)

for i, ax in enumerate(axs):
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if i % 2 == 1:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    else:
        ax.set_xlabel("Sample index $t$")

fig.text(0.125, 0.9, "\\textbf{A}", size=12, va="baseline")
fig.text(0.31, 0.9, "\\textbf{Mackey-Glass system for} $\\tau = 17$", ha="center", va="baseline")
fig.text(0.525, 0.9, "\\textbf{B}", size=12, va="baseline")
fig.text(0.71, 0.9, "\\textbf{Mackey-Glass system for} $\\tau = 30$", ha="center", va="baseline")

utils.save(fig)
