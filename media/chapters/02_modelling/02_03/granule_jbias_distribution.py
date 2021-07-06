import lif_utils
import scipy.optimize
import scipy.stats

np.random.seed(58729)

# Load the tuning curves
data = np.load(utils.datafile("granule_cell_response_curves.npy"),
               allow_pickle=True).item()
Js_lst = data["Js_lst"]
Gs_lst = data["Gs_lst"]
E_rev_leaks = data["E_rev_leaks"]
n_E_rev_leak = len(E_rev_leaks)

# Chadderton, 2004, Figure 1b
orig_bins = np.linspace(-90, -30, 16)
orig_qty = np.array([0, 2, 4, 4, 6, 15, 12, 12, 8, 3, 5, 1, 3, 0, 0])
orig_qty_density = orig_qty / (np.sum(orig_qty) * 4)

# Chadderton, 2004, Figure 1g (control)
orig_Js, orig_Gs = np.array([
    5.7789,
    -0.0357,
    8.392,
    1.6794,
    10.9045,
    5.0251,
    13.4171,
    9.3899,
    15.8543,
    13.302,
    18.392,
    16.9873,
    20.9799,
    19.4045,
]).reshape(-1, 2).T
orig_Gs /= 0.2  # spike count measured over a 250ms interval

# Fit a Gaussian to the resting potential histogram
p_mu, p_sigma = scipy.optimize.curve_fit(scipy.stats.norm.pdf,
                                         orig_bins[:-1] + 2,
                                         orig_qty_density,
                                         p0=(-64, 10))[0]
ps = np.linspace(-90, -30, 100)
ps_density = scipy.stats.norm.pdf(ps, p_mu, p_sigma)
ps_density_coarse = scipy.stats.norm.pdf(orig_bins[:-1] + 2, p_mu, p_sigma)

# Fit ReLUs to the measured data
gains, biases = np.zeros((2, n_E_rev_leak))
valid = [False] * n_E_rev_leak
for i in range(n_E_rev_leak):
    Js, Gs = Js_lst[i], Gs_lst[i]
    Gs_valid = Gs > 5
    if np.sum(Gs_valid) >= 2:
        gains[i], biases[i] = np.polyfit(Js[Gs_valid], Gs[Gs_valid], 1)
        if biases[i] < 150:
            valid[i] = True

# Generate comparable data for NEF-style intercept distributions
lif_params = dict(
    Cm=5e-12,
    gL=3.5e-10,
    v_th=-35e-3,
    tau_ref=1e-3,
    EL=-64e-3,
    v_reset=-64e-3,
)
N_nef = 1024
max_rates = np.random.uniform(100, 200, N_nef)
encoders = np.random.choice([-1, 1], N_nef)
x_intercepts = np.random.uniform(-0.9, 0.9, N_nef)
idcs = np.argsort(x_intercepts * encoders)
encoders = encoders[idcs]
x_intercepts = x_intercepts[idcs]
J0s = lif_utils.lif_detailed_rate_inv(1e-3, **lif_params)
Jmaxs = lif_utils.lif_detailed_rate_inv(max_rates, **lif_params)
nef_gains = (Jmaxs - J0s) / (1.0 - x_intercepts) 
nef_biases = (J0s - x_intercepts * Jmaxs) / (1.0 - x_intercepts)

xs = np.linspace(-1, 1, 1001)
Js = (encoders * nef_gains)[None, :] * xs[:, None] + nef_biases[None, :]
As = lif_utils.lif_detailed_rate(Js, **lif_params)

fig, ((axA, axB, axC), (axD, axE, axF)) = plt.subplots(2, 3, figsize=(7.4, 3.0), gridspec_kw={
    "wspace": 0.4,
    "hspace": 0.7,
})

Blue = utils.blues[1]
Orange = utils.oranges[1]


for i in range(N_nef - 1, 0, -30):
    colour = 'k'#mpl.cm.get_cmap('plasma')((i / (N_nef - 1)))
    #colour = mpl.cm.get_cmap('plasma')((i / (N_nef - 1)))
    axA.plot(xs, As[:, i], color=colour, linewidth=0.75)
axA.set_xlabel('Represented $x$')
axA.set_ylabel('Rate $a_i(x)$ ($\\mathrm{s}^{-1}$)')
axA.text(-0.275, 1.0325, '\\textbf{A}', va="bottom", ha="left", fontsize=12, transform=axA.transAxes)
axA.set_title('Desired tuning curves')

axB.hist(nef_biases * 1e12,
            bins=np.linspace(-70, 30, 16),
            fill=False,
            edgecolor=Orange,
            density=True)
axB.hist(nef_biases * 1e12,
            bins=np.linspace(-70, 30, 16),
            fill=True,
            color=Orange,
            alpha=0.5,
            density=True,
            label='ReLU bias')
axB.set_xlim(-70, 30)
axB.set_title('Required bias histogram')
axB.set_ylabel('Density')
axB.set_xlabel('Bias current $\\beta_i$ (pA)')
axB.text(-0.275, 1.0325, '\\textbf{B}', va="bottom", ha="left", fontsize=12, transform=axB.transAxes)

xs = np.linspace(-70, 20, 1000)
axB.plot(xs,
         0.05 * np.exp(0.05 * (xs - 19)),
         'k--',
         linewidth=2)
#utils.annotate(axB, -2.5, 0.021, -10, 0.03, '$\\frac{1}{20}\\exp\\left( \\frac{\\beta - 20}{20} \\right)$', va='bottom', ha='right')
utils.annotate(axB, -2.5, 0.021, -10, 0.03, 'Exponential fit', va='bottom', ha='right', fontdict={"size": 8})

cmap = cm.ScalarMappable(
    mpl.colors.Normalize(vmin=-90,
                         vmax=-35), 'viridis')

for i, v in enumerate(orig_bins[:-1]):
    axC.plot(v + 2, 0.0, 'o', markersize=4, color=cmap.to_rgba(v), zorder=100, clip_on=False)
    axC.set_xlim()
axC.set_axisbelow(True)

axC.bar(orig_bins[:-1] + 2,
           orig_qty_density,
           width=4,
           fill=True,
           color=Blue,
           alpha=0.5,
           label='Empirical')
axC.bar(orig_bins[:-1] + 2,
           orig_qty_density,
           width=4,
           fill=False,
           edgecolor=Blue)
axC.plot(ps,
            ps_density,
            color='k',
            linestyle='--',
            linewidth=2,
            label='Gaussian fit')
axC.set_xlim(-90, -30)
axC.set_ylim(0, 0.055)
axC.set_xticks([-90, -70, -50, -30])
axC.set_ylabel("Density")
axC.set_xlabel("Resting potential $v_\\mathrm{rest}$ (mV)")
axC.text(-0.275, 1.0325, '\\textbf{C}', va="bottom", ha="left", fontsize=12, transform=axC.transAxes)
axC.set_title('Empirical $v_\\mathrm{rest}$ histogram')
utils.annotate(axC, -54, 0.03, -50, 0.0325, 'Gaussian\nfit', va='center', ha='left', fontdict={"size": 8})


idcs = np.argsort(E_rev_leaks)
E_rev_last = None
i_last = 0
for idx, i in enumerate(idcs):
    if idx > 0 and E_rev_leaks[i] - E_rev_leaks[i_last] < 2e-3:
        continue
    i_last = i
    if not valid[i]:
        continue
    color = cmap.to_rgba(E_rev_leaks[i] *
                         1e3)  #* np.array((0.9, 0.9, 0.9, 1.0))
    Js, Gs = Js_lst[i], Gs_lst[i]
    axD.plot(Js * 1e12, Gs, color=color, zorder=1, linewidth=1)
#    label = 'ReLU fit' if idx == 0 else None
#    axD.plot(Js * 1e12,
#                np.maximum(0, gains[i] * Js + biases[i]),
#                color='gray',
#                linestyle='--',
#                linewidth=1,
#                zorder=0,
#                label=label)
axD.plot(Js_lst[0] * 1e12,
            Gs_lst[0],
            linestyle='-',
            color='k',
            linewidth=2,
            label='LIF model')
axD.scatter(orig_Js,
               orig_Gs,
               color='white',
               marker='+',
               zorder=3,
               s=50,
               linewidth=4,
               label='Empirical')
axD.scatter(orig_Js,
               orig_Gs,
               color='k',
               marker='+',
               zorder=3,
               s=30,
               linewidth=2,
               label='Empirical')
axD.set_xlim(-10, 30)
axD.set_ylim(0, 100)
axD.set_ylabel('Spike rate ($\\mathrm{s}^{-1}$)')
axD.set_xlabel('Mean input current $J$ (pA)')
#axD.legend(loc='upper center',
#              bbox_to_anchor=(0.5, 1.35),
#              ncol=3,
#              borderaxespad=0,
#              handlelength=1.5,
#              handletextpad=0.2,
#              columnspacing=0.8,
#              labelspacing=0.4)

axD.text(-0.275, 1.0325, '\\textbf{D}', va="bottom", ha="left", fontsize=12, transform=axD.transAxes)
axD.set_title('Sampled response curves')


#cax = fig.add_axes([0.75, 0.725, 0.075, 0.075])
#plt.colorbar(cmap, cax=cax, orientation='horizontal')
#cax.set_xlabel('$v_\\mathrm{rest}$', labelpad=-14)
#cax.set_xticklabels(['$-75\\mathrm{mV}$', '$-50\\mathrm{mV}$'])
#cax.text(0.5,
#         0.5,
#         '$v_\\mathrm{rest}$',
#         va='center',
#         ha='center',
#         color='white',
#         transform=cax.transAxes)
#utils.outside_ticks(cax)

#rect = mpl.patches.Rectangle((0.1, 0.675),
#                             0.4,
#                             0.325,
#                             facecolor='white',
#                             transform=axD.transAxes,
#                             zorder=2,
#                             alpha=0.9)
#axD.add_patch(rect)

#α, β = np.polyfit(biases[biases < 50], gains[biases < 50] * 1e-12, 1)
#axE.plot(biases, gains * 1e-12, '+', color=Blue, alpha=0.5, markersize=5, markeredgecolor=None)
#axE.plot(xs, α * xs + β, 'k--', linewidth=2)
#axE.set_xlim(-50, 50)
#axE.set_xlabel('Bias ($\\mathrm{s}^{-1}$)')
#axE.set_ylabel('Gain ($\\mathrm{s}^{-1}/pA$)')

xs = np.linspace(-10, 10, 100)
biases = biases / gains
biases -= biases[0] # Biases are relative to the mean vRest trace

#p = scipy.stats.skewnorm.fit(biases * 1e12, 0.1, loc=0, scale=1)
p = scipy.stats.norm.fit(biases * 1e12, loc=0, scale=1)

axE.hist(biases * 1e12,
            bins=np.linspace(-10, 10, 16),
            fill=False,
            edgecolor=Orange,
            density=True)
axE.hist(biases * 1e12,
            bins=np.linspace(-10, 10, 16),
            fill=True,
            color=Orange,
            alpha=0.5,
            density=True,
            label='ReLU bias')

axE.plot(xs,
#            scipy.stats.skewnorm.pdf(xs, *p),
            scipy.stats.norm.pdf(xs, *p),
            'k--',
            linewidth=2,
            label='Skewed Gaussian')
axE.set_xlabel('Bias current $\\beta_i$ (pA)')
axE.set_ylabel('Density', labelpad=8.1)
axE.set_xlim(-10, 10)
#axE.set_ylim(0, 0.065)
axE.set_title('Equivalent biases')
axE.text(-0.275, 1.0325, '\\textbf{E}', va="bottom", ha="left", fontsize=12, transform=axE.transAxes)
utils.annotate(axE, -2.5, 0.1, -4, 0.15, 'Gaussian\nfit', va='center', ha='right', fontdict={"size": 8})


# Compute the tuning curves that are achievable with the actual bias current
# distribution
idcs_sort = np.argsort(nef_biases)
nef_biases_sampled = (np.sort(scipy.stats.norm(*p).rvs(N_nef) * 1e-12) + biases[0])[idcs_sort]

nef_gains = Jmaxs - nef_biases_sampled

xs = np.linspace(-1, 1, 1001)
Js = (encoders * nef_gains)[None, :] * xs[:, None] + nef_biases_sampled[None, :]
As = lif_utils.lif_detailed_rate(Js, **lif_params)

for i in range(N_nef - 1, 0, -30):
    colour = 'k'
    #colour = mpl.cm.get_cmap('plasma')((i / (N_nef - 1)))
    axF.plot(xs, As[:, i], color=colour, linewidth=0.75)
axF.text(-0.275, 1.0325, '\\textbf{F}', va="bottom", ha="left", fontsize=12, transform=axF.transAxes)
axF.set_ylabel('Rate $a_i(x)$ ($\\mathrm{s}^{-1}$)')
axF.set_xlabel('Represented $x$')
axF.set_title('Realisable tuning curves')

utils.save(fig)

