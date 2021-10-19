fig, ax = plt.subplots()
utils.remove_frame(ax)
ax.plot(0.85,
        1.13,
        'o',
        color=utils.blues[0],
        markeredgecolor='k',
        markeredgewidth=0.7,
        transform=ax.transAxes,
        clip_on=False)
utils.save(fig)
