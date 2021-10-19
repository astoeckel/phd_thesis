import scipy.signal
import nengo

def sdft_step(m, N, u, u_out):
    q = len(m)
    assert q > 0 and (q % 2 == 1)
    m[0] = m[0] + u - u_out
    for k in range(1, (q - 1) // 2 + 1):
        f = 2.0 * np.pi * (k + 1) / N
        sf, cf = np.sin(f), np.cos(f)

        i0, i1 = 2 * k - 1, 2 * k
        m0 = m[i0] * cf - m[i1] * sf + u - u_out
        m1 = m[i0] * sf + m[i1] * cf
        m[i0], m[i1] = m0, m1

    return m

dt_outer = 1e-3 # Outer sampling interval
f_smpl_outer = 1.0 / dt_outer # Outer samling rate

q = 3 # Basis order
assert (q > 0) and (q % 2 == 1)
f_max = np.floor(0.5 * (q - 1)) # Frequency that can be reperesented by the basis
f_nyquist = 2.0 * f_max # Nyquist frequency of the basis
f_smpl_inner = 2.0 * f_nyquist # Over-sample twice; better anti-aliasing
dt_inner = np.floor(f_smpl_outer / f_smpl_inner) * dt_outer # Round to a multiple of dt_outer
ss = int(dt_inner / dt_outer) # Subsampling
Nwi = int(1.0 / dt_inner) # Inner window width
Nwo = int(1.0 / dt_outer) # Outer window width
assert(f_max < 0.5 * f_smpl_inner)

print(f"dt_outer = {dt_outer*1e3:0.2f} ms")
print(f"f_smpl_outer = {f_smpl_outer} Hz")
print()
print(f"q = {q}")
print(f"f_max = {f_max}")
print(f"f_nyquist = {f_nyquist}")
print(f"dt_inner = {dt_inner*1e3:0.2f} ms")
print(f"f_smpl_inner = {f_smpl_inner} Hz")
print(f"ss = {ss}")
print(f"Nwi = {Nwi}")
print(f"Nwo = {Nwo}")

# Create some input signal
T = 10.0
ts_outer = np.arange(0, T, dt_outer)
ts_inner = np.arange(0, T, dt_inner)
No, Ni = len(ts_outer), len(ts_inner)

us_outer = nengo.processes.WhiteSignal(period=T, rms=0.4, high=1.0).run(T, dt=dt_outer)[:, 0]
assert len(us_outer) == len(ts_outer)

# Design an IIR low-pass filter that removes all frequencies between f_max and 0.5 * f_smpl_inner
flt = scipy.signal.iirdesign(f_max, 0.5 * f_smpl_inner, 1.0, 60.0, fs=f_smpl_outer, output='sos')
us_outer_flt = scipy.signal.sosfilt(flt, us_outer) * 2.0

# Sub-sample the input
us_inner = us_outer_flt[::ss]

# Run the input through the SDFT
ms_inner = np.zeros((Ni, q))
m = np.zeros(q)
for i in range(Ni):
    m = sdft_step(m, Nwi, us_inner[i], 0.0 if i < Nwi else us_inner[i - Nwi])
    ms_inner[i] = m

ms_outer = np.zeros((No, q))
m = np.zeros(q)
for i in range(No):
    m = sdft_step(m, Nwo, us_outer[i], 0.0 if i < Nwo else us_outer[i - Nwo])
    ms_outer[i] = m

fig, axs = plt.subplots(4, 1, figsize=(7.5, 2.0))
for ax in axs:
    utils.remove_frame(ax)
    ax.axhline(0.0, linestyle=':', lw=0.5, color='grey', clip_on=False)
    #ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, T)

axs[0].plot(ts_outer, us_outer, 'k-', clip_on=False)

axs[1].plot(ts_outer, us_outer_flt, 'k-', clip_on=False)
axs[1].plot(ts_inner, us_inner, 'ko', markersize=4, markeredgecolor='k', markerfacecolor='white', markeredgewidth=0.7, clip_on=False)
for i in range(len(ts_inner)):
    axs[1].plot([ts_inner[i], ts_inner[i]], [0.0, us_inner[i]], 'k-', linewidth=0.5, clip_on=False)


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for j in [0, 1, 2]:
    scale = np.sqrt(np.floor(0.5 * (j + 1)) + 1)
    axs[2].plot(ts_inner, ms_inner[:, j] * scale * ss, 'o', markersize=4, markeredgecolor=colors[j], markerfacecolor='white', markeredgewidth=0.7, clip_on=False)
    axs[2].plot(ts_outer, ms_outer[:, j] * scale, '--', color=colors[j], lw=0.5, zorder=-1)


utils.save(fig)
