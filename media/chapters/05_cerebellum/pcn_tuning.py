import nengo
from nengo.utils.ensemble import tuning_curves

fig, axs = plt.subplots(1,
                        3,
                        figsize=(7.4, 2.25),
                        gridspec_kw={
                            "wspace": 0.3,
                        })

with open(
        utils.datafile(
            "manual/chapters/05_cerebellum/chadderton_granule_epsc_chart.csv"),
        'r') as f:
    lines = f.readlines()

header = lines[0]
data = np.array([[float(x) for x in line.split(",")] for line in lines[1:]])

ts = data[:, 1]
rates = data[:, 6]
event = data[:, 7]
event_start = ts[event == 1][0]
event_end = ts[event == 1][-1] + 25

event_ext = np.copy(event).astype(bool)
event_ext[np.where(event)[0][-1] + 1] = True

avg_with_event = np.mean(rates[event_ext])
avg_without_event = np.mean(rates[~event_ext])

axs[0].bar(ts + 12.5, rates, width=25, color=utils.blues[0])
axs[0].plot([event_start, event_end], [80, 80],
            'k-',
            linewidth=2,
            solid_capstyle='butt',
            clip_on=False)
axs[0].fill_betweenx([0, 80], [event_start, event_start],
                     [event_end, event_end],
                     color="#C0C0C0",
                     linewidth=0)

axs[0].axhline(avg_without_event, color='k', linestyle='--', linewidth=1.0)
axs[0].axhline(avg_with_event, color='k', linestyle='--', linewidth=1.0)

utils.annotate(axs[0], 1000, avg_with_event + 2, 1350, avg_with_event + 20,
               f"$\\approx \\SI{{{avg_with_event:0.1f}}}{{\\per\\second}}$")

utils.annotate(
    axs[0], 1400, avg_without_event + 2, 1350, avg_without_event + 20,
    f"$\\approx \\SI{{{avg_without_event:0.1f}}}{{\\per\\second}}$")

axs[0].set_xlabel("Time $t$ (\\si{\\milli\\second})")
axs[0].set_ylabel("Granule EPSC rate (\\si{\\per\\second})")
axs[0].set_xlim(0, 2000)
axs[0].set_ylim(0, 80)
axs[0].set_title("Empirical granule cell EPSCs")

with nengo.Network() as model:
    ens_pcn = nengo.Ensemble(n_neurons=100,
                             dimensions=1,
                             max_rates=nengo.dists.Uniform(25, 75),
                             intercepts=nengo.dists.Uniform(-0.15, 0.95))
with nengo.Simulator(model) as sim:
    eval_points, activities = tuning_curves(ens_pcn,
                                            sim,
                                            inputs=np.linspace(-1, 1,
                                                               1001).reshape(
                                                                   -1, 1))

axs[1].plot(eval_points, activities, 'k-', linewidth=0.7)
axs[1].set_xlim(-1, 1)
axs[1].set_ylim(0, 75)
axs[1].set_xlabel("Sensory input $u$")
axs[1].set_ylabel("Firing rate $a_i$ (\\si{\\per\\second})")
axs[1].set_title("PCN Tuning Curves")

data = np.load(utils.datafile("granule_pcn_tuning.npz"))
epscs = data["epscs"]
print(epscs.shape)
ts = data["ts"] * 1e3
xs = data["xs"]
bins_ts = np.arange(0, ts[-1] + 1, 25)
bins_n_epscs = np.array([
    np.sum(epscs[np.logical_and(ts >= t0, ts < t1)])
    for t0, t1 in zip(bins_ts[:-1], bins_ts[1:])
])
axs[2].bar(bins_ts[:-1] + 12.5,
           bins_n_epscs / (0.025 * data["n_granule"]),
           width=25,
           color=utils.blues[0])

t0 = None
for t, x0, x1 in zip(ts, xs[:-1], xs[1:]):
    if t > 2000:
        continue
    if x0 != x1:
        if x1:
            t0 = t
        else:
            axs[2].plot([t0, t], [80, 80],
                        'k-',
                        linewidth=2,
                        solid_capstyle='butt',
                        clip_on=False)
            axs[2].fill_betweenx([0, 80], [t0, t0], [t, t],
                                 color="#C0C0C0",
                                 linewidth=0)

axs[2].set_xlim(0, 2000)
axs[2].set_ylim(0, 80)
axs[2].set_xlabel("Time $t$ (\\si{\\milli\\second})")
axs[2].set_ylabel("Granule EPSC rate (\\si{\\per\second})")
axs[2].set_title("Simulated granule cell EPSCs")

avg_with_event = data["n_epscs_with_input"]
avg_without_event = data["n_epscs_without_input"]
utils.annotate(axs[2], 1000, avg_with_event + 2, 1150, avg_with_event + 20,
               f"$\\approx \\SI{{{avg_with_event:0.1f}}}{{\\per\\second}}$")
utils.annotate(
    axs[2], 1050, avg_without_event + 2, 1150, avg_without_event + 20,
    f"$\\approx \\SI{{{avg_without_event:0.1f}}}{{\\per\\second}}$")
axs[2].axhline(avg_without_event, color='k', linestyle='--', linewidth=1.0)
axs[2].axhline(avg_with_event, color='k', linestyle='--', linewidth=1.0)

axs[0].text(-0.22,
            1.0525,
            "\\textbf{A}",
            size=12,
            ha="left",
            va="baseline",
            transform=axs[0].transAxes)
axs[1].text(-0.22,
            1.0525,
            "\\textbf{B}",
            size=12,
            ha="left",
            va="baseline",
            transform=axs[1].transAxes)
axs[2].text(-0.22,
            1.0525,
            "\\textbf{C}",
            size=12,
            ha="left",
            va="baseline",
            transform=axs[2].transAxes)

utils.save(fig)

