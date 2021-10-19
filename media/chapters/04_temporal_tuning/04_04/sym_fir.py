fig, ax = plt.subplots()
utils.remove_frame(ax)
ax.plot(0.85,
        1.13,
        '^',
        color=utils.yellows[1],
        markeredgecolor='k',
        markeredgewidth=0.7,
        transform=ax.transAxes,
        clip_on=False)
utils.save(fig)
