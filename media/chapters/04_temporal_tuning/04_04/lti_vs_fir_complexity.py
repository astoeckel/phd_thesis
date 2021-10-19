qs = np.linspace(1, 1000, 1000)
Ns = np.linspace(1, 1000, 1000)

fig, ax = plt.subplots(figsize=(7.2, 1.75))
qs_for_Ns_no_clip = 34 * np.log2(Ns)
qs_for_Ns = np.minimum(Ns, qs_for_Ns_no_clip)
#ax.fill_between(qs, np.zeros_like(qs), qs_for_Ns, alpha=0.5, lw=0.0)
#ax.fill_between(qs, qs_for_Ns, Ns, alpha=0.5, lw=0.0)
ax.plot(qs, qs_for_Ns, 'k')
ax.plot(qs, qs, 'k--', linewidth=0.5)
ax.plot(qs, qs_for_Ns_no_clip, 'k:', linewidth=0.5)
#ax.set_yscale('log')
ax.set_ylim(1, 1000)
ax.set_xlim(1, 1000)
ax.set_xlabel('Sample count $N$')
ax.set_ylabel('State dimensions $q$')
ax.text(500,
        150,
        '$q$-dimensional LTI\nsystem more efficient',
        va='center',
        ha='center')
ax.text(800,
        530,
        'Can use\narbitrary set of $q$\nFIR filters',
        va='center',
        ha='center')
ax.text(300, 700, '$q > N$', va='center', ha='center')

utils.outside_ticks(ax)

utils.save(fig)

