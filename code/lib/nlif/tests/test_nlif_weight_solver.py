import numpy as np
import nlif
import nlif.solver
import copy
import matplotlib.pyplot as plt
import tqdm as tqdm

from nlif.tests.utils import nef


def main():
    # Generate some target neuron type
    neuron = nlif.ThreeCompLIFCond()
    assm = neuron.assemble()
    sys = assm.reduced_system().condition()

    # Generate some pre-activities
    ens1 = nef.Ensemble(101, 1)
    ens2 = nef.Ensemble(102, 1)
    xs1 = np.linspace(-1, 1, 10)
    xs2 = np.linspace(-1, 1, 10)

    xss1, xss2 = np.meshgrid(xs1, xs2)
    xss1 = xss1.flatten()
    xss2 = xss2.flatten()

    As1 = ens1(xss1.reshape(1, -1)).T
    As2 = ens2(xss2.reshape(1, -1)).T
    As_pre = np.concatenate((As1, As2), axis=1)

    # Generate some target currents
    Js_tar = np.array(((xss1 + xss2) * 0.5e-9,
                       #(xss1 * xss2) * 1e-9,
                       )) * sys.out_scale

    # Generate some initial weights
    W = np.random.uniform(0.0, 1.0, (
        Js_tar.shape[0],
        sys.n_inputs,
        As_pre.shape[1],
    )) / sys.in_scale
    #W *= 0;

    # Try to run the solver!
    solver = nlif.solver.Solver(debug=False)
    W_new = np.copy(W)
    for i in tqdm.tqdm(range(100)):
        W_new += np.random.uniform(0.0, 1.0, (
            Js_tar.shape[0],
            sys.n_inputs,
            As_pre.shape[1],
        )) / sys.in_scale
        W_new = solver.nlif_solve_weights_iter(
            sys,
            As_pre,
            Js_tar,
            W_new,
            J_th=0.0,
            reg1=1e-2,
            reg2=1e-6,
            alpha1=1e0,
            alpha2=1e3,
            alpha3=1e-3,
            n_threads=1)

#    fig, ax = plt.subplots()
#    C = ax.contourf(xs1,
#                    xs2,
#                    assm.i_som(As_pre @ W_new[0].T,
#                               reduced_system=sys).reshape(xs1.size, xs2.size),
#                    vmin=-1.5e-9,
#                    vmax=1.5e-9,
#                    levels=np.linspace(-1.5e-9, 1.5e-9, 11))
#    ax.contour(xs1,
#               xs2,
#               Js_tar[0].reshape(xs1.size, xs2.size) / sys.out_scale,
#               levels=C.levels,
#               colors=["white"],
#               linestyles=['--'])

#    plt.show()

if __name__ == "__main__":
    main()

