import numpy as np
import nlif
import nlif.solver
import copy

def main():
    neuron = nlif.TwoCompLIFCond()
    assm = neuron.assemble()

    gs_train = np.random.uniform(0, 1000e-9, (1000, assm.n_inputs))
    Js_train = assm.isom_empirical_from_rate(gs_train)

    gs_test = np.random.uniform(0, 1000e-9, (1001, assm.n_inputs))
    Js_test = assm.isom_empirical_from_rate(gs_test)

    valid_train = Js_train > 1e-9
    valid_test = Js_test > 1e-9

    sys = assm.reduced_system().condition()

    sys_new = copy.deepcopy(sys)
    solver = nlif.solver.Solver(debug=True)
    for i in range(10):
        sys_new = solver.nlif_solve_parameters_iter(
            sys_new,
            gs_train[valid_train] * sys.in_scale,
            Js_train[valid_train] * sys.out_scale,
            reg1=0.0,
            reg2=1e-3,
            alpha1=1.0,
            alpha2=1.0,
            alpha3=100.0)

if __name__ == "__main__":
    main()

