fig, ax = plt.subplots()
utils.remove_frame(ax)
ax.plot(0.61,
        1.13,
        'h',
        color=utils.oranges[1],
        markeredgecolor='k',
        markeredgewidth=0.7,
        transform=ax.transAxes,
        clip_on=False)
utils.save(fig)
