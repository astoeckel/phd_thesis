#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

import itertools
import tqdm
import numpy as np
import multiprocessing
import env_guard
import h5py
import pyESN

import nengo


def unpack(As, As_shape):
    return np.unpackbits(
        As, count=np.prod(As_shape)).reshape(*As_shape).astype(np.float64)


def dist_fast(xs, N_wnd=1000, dt=1e-3):
    N = xs.shape[0]
    res = np.zeros(N)
    ss = np.linalg.norm(xs, axis=1) * dt
    iss = np.cumsum(ss)
    return iss - np.concatenate((np.zeros(N_wnd), iss))[:N]


with h5py.File(utils.datafile("spatio_temporal_network.h5"), "r") as f:
    As_shape = f["As_shape"][()]
    xs_train = f["xs_train"][()]
    As_train = unpack(f["As_train"][()], As_shape)
    xs_test = f["xs_test"][()]
    As_test = unpack(f["As_test"][()], As_shape)

xs_train_flt = nengo.Lowpass(100e-3).filtfilt(xs_train)
As_train_flt = nengo.Lowpass(100e-3).filtfilt(As_train)
ss_train = dist_fast(xs_train_flt)
D = np.linalg.lstsq(As_train_flt, ss_train, rcond=1e-3)[0]

xs_test_flt = nengo.Lowpass(100e-3).filtfilt(xs_test)
As_test_flt = nengo.Lowpass(100e-3).filtfilt(As_test)
ss = dist_fast(xs_test_flt)


def esn_experiment(xs_train, ys_train, xs_test, ys_test):
    np.random.seed(5781)
    esn = pyESN.ESN(n_inputs=2,
                    n_outputs=1,
                    n_reservoir=1000,
                    teacher_forcing=True,
                    spectral_radius=0.8)
    esn.fit(xs_train, ys_train)
    return esn.predict(xs_test)[:, 0]


xs, ys = np.cumsum(xs_test_flt, axis=0).T * 1e-2

fig = plt.figure(figsize=(7.375, 2.0))
gs = fig.add_gridspec(3,
                      2,
                      width_ratios=[1, 2],
                      height_ratios=[1, 1, 2],
                      hspace=0.5)

ax_map = fig.add_subplot(gs[0:3, 0])
ax_vx = fig.add_subplot(gs[0, 1])
ax_vy = fig.add_subplot(gs[1, 1])
ax_ss = fig.add_subplot(gs[2, 1])

N_tot = xs.shape[0]
N0 = 20000
N1 = 1
dt = 1e-3
ts = np.arange(0, N0 - N1) * dt

ax_map.plot(xs, ys, lw=0.5, color='grey', linestyle=':')
ax_map.plot(xs[-N0:-N1],
            ys[-N0:-N1],
            lw=1.0,
            color='k',
            linestyle='-',
            solid_capstyle='butt')
ax_map.set_xlabel("Location $x_1$")
ax_map.set_ylabel("Location $x_2$")
ax_map.set_xlim(-35, 15)
ax_map.set_ylim(-25, 25)

for i, idx in enumerate(np.arange(-N0, 0, 999)):
    c = mpl.cm.get_cmap('viridis')(i / 19)
    ax_map.plot(xs[idx],
                ys[idx],
                'o',
                color=c,
                markersize=4,
                markeredgewidth=0.7,
                markeredgecolor='k')
    ax_ss.plot(i / 20.0,
               0.0,
               'o',
               color=c,
               markersize=4,
               markeredgecolor='k',
               markeredgewidth=0.7,
               zorder=100,
               clip_on=False,
               transform=ax_ss.transAxes)

ax_vx.spines["bottom"].set_visible(False)
ax_vx.set_xticks([])
ax_vx.set_ylim(-2, 2)
ax_vx.set_yticks(np.linspace(-2, 2, 5), minor=True)

ax_vy.spines["bottom"].set_visible(False)
ax_vy.set_xticks([])
ax_vy.set_ylim(-2, 2)
ax_vy.set_yticks(np.linspace(-2, 2, 5), minor=True)

ax_vx.plot(ts, xs_test_flt[-N0:-N1, 0], 'k-', lw=0.7)
ax_vx.set_ylabel("$\\dot x_1(t)$")
ax_vy.plot(ts, xs_test_flt[-N0:-N1, 1], 'k-', lw=0.7)
ax_vy.set_ylabel("$\\dot x_2(t)$")

ax_ss.plot(ts, ss[-N0:-N1], 'k-', lw=0.5)
ax_ss.plot(ts, As_test_flt[-N0:-N1] @ D, color='k', lw=1.25)
ax_ss.plot(ts, ss[-N0:-N1], 'k:', lw=0.5, color='white')

esn_ss_pred = utils.run_with_cache(esn_experiment, xs_train_flt, ss_train, xs_test_flt, ss)
ax_ss.plot(ts, esn_ss_pred[-N0:-N1], 'k--', lw=0.5, zorder=-1, color='grey')

ax_ss.set_xlabel("Time $t$ (s)")
ax_ss.set_ylabel("$d_{[t - \\theta, t]}$")

for ax in [ax_vx, ax_vy, ax_ss]:
    ax.set_xlim(0, 20)
ax_ss.set_xticks(np.linspace(0, 20, 9))
ax_ss.set_xticks(np.linspace(0, 20, 9))
ax_ss.set_ylim(0.0, 2.0)
ax_ss.set_yticks(np.linspace(0.0, 2.0, 5), minor=True)

E = np.sqrt(np.mean(np.square(ss - As_test_flt @ D))) / np.sqrt(
    np.mean(np.square(ss - np.mean(ss))))
ax_ss.text(1.0,
           1.0,
           "$E = {:0.1f}\\%$".format(E * 100.0),
           va="top",
           ha="right",
           transform=ax_ss.transAxes)
utils.annotate(ax_ss, 7.5, 1.3, 9.0, 1.75, "ESN")

E_ESN = np.sqrt(np.mean(np.square(ss - esn_ss_pred))) / np.sqrt(
    np.mean(np.square(ss - np.mean(ss))))
print("E_ESN =", E_ESN)

fig.align_labels([ax_ss, ax_map])
fig.align_labels([ax_vx, ax_vy, ax_ss])

fig.text(0.071, 0.9225, "\\textbf{A}", va="baseline", ha="left", size=12)
ax_map.set_title("\\textbf{Trajectory} $(x_1(t), x_2(t))$")

fig.text(0.3825, 0.9225, "\\textbf{B}", va="baseline", ha="left", size=12)
ax_vx.set_title("\\textbf{Velocities and decoded windowed distance}")

utils.save(fig)

