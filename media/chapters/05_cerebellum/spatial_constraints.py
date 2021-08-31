#   Code for the PhD Thesis
#   "Harnessing Neural Dynamics as a Computational Resource: Building Blocks
#   for Computational Neuroscience and Artificial Agents"
#   Copyright (C) 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.


def min_radius(p_th=0.9, sigma=0.25, n=1000):
    max_x = 2
    rs = np.linspace(0, max_x, n)
    ps = np.exp(-rs**2 / sigma**2)
    ps = np.cumsum(ps / np.sum(ps))

    valid_idcs = np.arange(n, dtype=int)[(1 - ps) > p_th]
    r = rs[np.max(valid_idcs)]
    return r


def plot_spatial_data(spatial_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.35, 2.25), gridspec_kw={
        "wspace": 0.3
    })

    x_golgi = spatial_data["golgi_locations"]
    x_granule = spatial_data["granule_locations"]
    ps = spatial_data["ps"]

    I = ax1.imshow(ps,
                   extent=[1, 10000, 1, 100],
                   interpolation='none',
                   vmin=0.0,
                   vmax=1.0,
                   cmap='Blues')
    ax1.set_aspect('auto')
    ax1.set_xticks([1, 2000, 4000, 6000, 8000])
    ax1.set_xlabel('Granule cell index $i$')
    ax1.set_ylabel('Golgi cell index $j$')
    utils.outside_ticks(ax1)

    rect = plt.Rectangle((6725, 75), 2750, 21, facecolor='white')
    ax1.add_artist(rect)

    cax = fig.add_axes([0.36, 0.79, 0.077, 0.03])
    cb = plt.colorbar(I, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.text(0.5,
             -2.5,
             '$p_{ij}$',
             ha='center',
             va='bottom',
             transform=cax.transAxes)
    utils.outside_ticks(cax)

    cmap = cm.get_cmap('Blues')
    for j, p_th in enumerate(np.linspace(0.25, 0.9, 5)):
        r = min_radius(p_th)
        for i in range(x_golgi.shape[0]):
            circle = plt.Circle(x_golgi[i],
                                r,
                                fill=True,
                                linewidth=1,
                                color=cmap(p_th),
                                zorder=100 * j + i)
            ax2.add_artist(circle)

    ax2.scatter(x_granule[:, 0],
                x_granule[:, 1],
                marker='o',
                color='black',
                s=2,
                label='Granule',
                zorder=1000)
    ax2.scatter(x_golgi[:, 0],
                x_golgi[:, 1],
                marker='+',
                color='#f57900',
                s=35,
                label='Golgi',
                zorder=2000)

    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-0.275, 0.275)
    ax2.set_xlabel('Spatial location $x_1$')
    ax2.set_ylabel('Spatial location $x_2$')
    utils.outside_ticks(ax2)

    ax2.legend(loc='upper right',
               fontsize=8,
               prop={
                   "style": "italic"
               },
               facecolor="white",
               edgecolor="none",
               frameon=True,
               fancybox=False,
               framealpha=1.0,
               borderpad=0.3,
               handlelength=1.25,
               handletextpad=0.2).set_zorder(10000)

    ax1.text(-0.135,
             0.925,
             '$\\textbf{A}$',
             ha='right',
             va='bottom',
             fontsize=12,
             transform=ax1.transAxes)

    ax2.text(-0.14,
             0.925,
             '$\\textbf{B}$',
             ha='right',
             va='bottom',
             fontsize=12,
             transform=ax2.transAxes)

    return fig


with h5py.File(utils.datafile("cerebellum_detailed_neurons_example.h5")) as f:
    utils.save(
        plot_spatial_data({
            "golgi_locations": f["spatial"]["golgi_locations"][()],
            "granule_locations": f["spatial"]["granule_locations"][()],
            "ps": f["spatial"]["ps"][()],
        }))

