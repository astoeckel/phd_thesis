import nlif
from nlif.parameter_optimisation import optimise_sgd

neuron = nlif.ThreeCompLIFCond(v_th=-50e-3,
                               v_spike=20e-3,
                               tau_ref=2e-3,
                               tau_spike=1e-3,
                               v_reset=-65e-3,
                               C_m=1e-9,
                               E_E=20e-3,
                               E_I=-75e-3,
                               E_L=-65e-3,
                               g_c1=50e-9,
                               g_c2=200e-9)
#neuron = nlif.ThreeCompLIFCond(g_c2=200e-9)
assm = neuron.assemble()

sys_orig = assm.reduced_system()
print("Original system")
print("a =", sys_orig.a_const)
print("A =", sys_orig.A)
print("b =", sys_orig.b_const)
print("B =", sys_orig.B)
print("L =", sys_orig.L)
print("c =", sys_orig.c)
print()

sys_cond = assm.reduced_system().condition()
print("Conditioned system")
print("a =", sys_cond.a_const)
print("A =", sys_cond.A)
print("b =", sys_cond.b_const)
print("B =", sys_cond.B)
print("L =", sys_cond.L)
print("c =", sys_cond.c)

# Generate some training data
np.random.seed(4897)
gs_train = np.random.uniform(0, 1000e-9, (1000, assm.n_inputs))
Js_train = assm.isom_empirical_from_rate(gs_train)
gs_test = np.random.uniform(0, 1000e-9, (1001, assm.n_inputs))
Js_test = assm.isom_empirical_from_rate(gs_test)

def optimise(sys, alpha):
    valid_train = Js_train > 1.5e-9
    valid_test = Js_test > 1.5e-9
    print(np.sum(valid_train), np.sum(valid_test))
    _, errs_train, errs_test = optimise_sgd(sys,
                           gs_train[valid_train],
                           Js_train[valid_train],
                           gs_test[valid_test],
                           Js_test[valid_test],
                           N_epochs=200,
                           alpha=alpha,
                           N_batch=10,
                           normalise_error=False)
    return errs_train / sys.out_scale, errs_test / sys.out_scale

errs_train_orig, errs_test_orig = optimise(sys_orig, alpha=3e-7)
errs_train_cond, errs_test_cond = optimise(sys_cond, alpha=1e-2)

fig, axs = plt.subplots(1, 2, figsize=(7.45, 1.75), gridspec_kw={
    "hspace": 0.4
})

def setup_ax(ax, letter):
    ax.set_ylim(5e-3, 1e-1)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("RMSE (nA)")
    ax.text(-0.16, 1.0275, f"\\textbf{{{letter}}}", size=12, transform=ax.transAxes, ha="left", va="bottom")

axs[0].semilogy(errs_test_orig * 1e9, color='k', label='Validation')
axs[0].semilogy(errs_train_orig * 1e9, color='k', linestyle='--', linewidth=0.7, label="Training")
axs[0].legend(loc='best')
axs[1].semilogy(errs_test_cond * 1e9, color='k', label='Validation')
axs[1].semilogy(errs_train_cond * 1e9, color='k', linestyle='--', linewidth=0.7, label="Training")
axs[1].legend(loc='best')

setup_ax(axs[0], "A")
axs[0].set_title("\\textbf{Without conditioning}")

setup_ax(axs[1], "B")
axs[1].set_title("\\textbf{With conditioning}")

utils.save(fig)


