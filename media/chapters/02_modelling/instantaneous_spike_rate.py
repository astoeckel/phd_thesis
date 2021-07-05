T = 3.0
dt = 1e-4
ts = np.arange(0, T, dt)
N = len(ts)
rates = 10 + (3 * (1 + np.cos(0.23 * 2.0 * np.pi * ts)) + 1 *
              (1 + np.sin(1.0 + 0.81 * 2.0 * np.pi * ts)) + 2 *
              (1 + np.sin(0.75 + 0.92 * 2.0 * np.pi * ts)))

ts_flt = np.arange(-0.5, 0.5, dt)
tau = 100e-3
flt = (ts_flt >= 0.0) * np.exp(-ts_flt / tau)
flt /= np.sum(flt)

xs = np.zeros_like(rates)
us = np.zeros_like(rates)
for i in range(1, N):
    us[i] = us[i - 1] + rates[i] * dt
    xs[i] = np.floor(us[i]) / dt
    us[i] -= xs[i] * dt

xs_flt = np.convolve(xs, flt, 'same')

fig, (ax, ax2) = plt.subplots(1,
                              2,
                              figsize=(7.45, 1.0),
                              gridspec_kw={"width_ratios": [8, 2]})
for i in np.where(xs > 0)[0]:
    ax.plot([ts[i], ts[i]], [0, 4], 'k-', linewidth=1.0, solid_capstyle='butt')
    ax.plot([ts[i], ts[i]], [0, xs_flt[i]],
            'k:',
            linewidth=0.25,
            solid_capstyle='butt')
ax.plot(ts, xs_flt, 'k-', color=utils.blues[0], linewidth=0.75)
ax.plot(ts, rates, 'k', linewidth=1)
ax.set_xticks(np.arange(0, T + 0.01, 0.5))
ax.set_xticks(np.arange(0, T + 0.01, 0.25), minor=True)
ax.set_ylim(0, 30)
ax.set_xlim(0, T)
ax.set_yticks([0, 10, 20, 30])
ax.set_ylabel('Rate ($\\mathrm{s}^{-1}$)')
ax.set_xlabel('Time $t$ ($\\mathrm{s}$)')
ax.set_title('Filtered spike train')
utils.outside_ticks(ax)
ax.text(-0.095,
        1.06,
        '\\textbf{A}',
        size=12,
        transform=ax.transAxes,
        va='bottom',
        ha='left')

ax2.set_title('Synaptic filter')
ax2.plot(ts_flt, flt / dt, color=utils.oranges[0])
ax2.fill_between(ts_flt,
                 np.zeros_like(ts_flt),
                 flt / dt,
                 color=utils.oranges[2], alpha=0.75)
ax2.set_xlim(-0.1, 0.5)
ax2.set_ylabel('$h(t)$ (a.u.)')
ax2.set_xlabel('Time $t$ ($\\mathrm{s}$)')
ax2.set_xticks(np.arange(0, 0.51, 0.2))
ax2.set_xticks(np.arange(0, 0.51, 0.1), minor=True)
ax2.plot([tau, tau], [0, 1.0 / (np.exp(1) * tau)], 'k:', linewidth=0.5)
ax2.plot([-0.1, tau], [1.0 / (np.exp(1) * tau), 1.0 / (np.exp(1) * tau)],
         'k:',
         linewidth=0.5)
utils.outside_ticks(ax2)
utils.timeslice(ax2, 0, tau, 1.0 / (np.exp(1) * tau))
utils.annotate(ax2,
               tau * 0.5,
               1.1 / (np.exp(1) * tau),
               tau,
               1.7 / (np.exp(1) * tau),
               '$\\tau = 0.1\\,\\mathrm{s}$',
               va='bottom',
               ha='left')
utils.annotate(ax2,
               0.13,
               1.0,
               0.175,
               3.0,
               '$\\int h(t) \\,\\mathrm{d}t = 1$',
               va='bottom',
               ha='left')
ax2.text(-0.375,
         1.06,
         '\\textbf{B}',
         size=12,
         transform=ax2.transAxes,
         va='bottom',
         ha='left')

utils.save(fig)

