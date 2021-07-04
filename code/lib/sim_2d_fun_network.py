#    Code for the "Nonlinear Synaptic Interaction" Paper
#    Copyright (C) 2017-2020   Andreas Stöckel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import griddata
import nengo

import os, sys
import bioneuronqp

from nef_synaptic_computation.lif_utils import *
from nef_synaptic_computation.tuning_curves import *

from gen_2d_fun import *
from parameters import *

# Note: The following function was copied from an old version
# nengo.utils.functions
class HilbertCurve(object):
    """Hilbert curve function.
    Pre-calculates the Hilbert space filling curve with a given number
    of iterations. The curve will lie in the square delimited by the
    points (0, 0) and (1, 1).
    Arguments
    ---------
    n : int
        Iterations.
    """
    # Implementation based on
    # https://en.wikipedia.org/w/index.php?title=Hilbert_curve&oldid=633637210

    def __init__(self, n):
        self.n = n
        self.n_corners = (2 ** n) ** 2
        self.corners = np.zeros((self.n_corners, 2))
        self.steps = np.arange(self.n_corners)

        steps = np.arange(self.n_corners)
        for s in 2 ** np.arange(n):
            r = np.empty_like(self.corners, dtype='int')
            r[:, 0] = 1 & (steps // 2)
            r[:, 1] = 1 & (steps ^ r[:, 0])
            self._rot(s, r)
            self.corners += s * r
            steps //= 4

        self.corners /= (2 ** n) - 1

    def _rot(self, s, r):
        swap = r[:, 1] == 0
        flip = np.all(r == np.array([1, 0]), axis=1)

        self.corners[flip] = (s - 1 - self.corners[flip])
        self.corners[swap] = self.corners[swap, ::-1]

    def __call__(self, u):
        """Evaluate pre-calculated Hilbert curve.
        Arguments
        ---------
        u : ndarray (M,)
            Positions to evaluate on the curve in the range [0, 1].
        Returns
        -------
        ndarray (M, 2)
            Two-dimensional curve coordinates.
        """
        step = np.asarray(u * len(self.steps))
        return np.vstack((
            np.interp(step, self.steps, self.corners[:, 0]),
            np.interp(step, self.steps, self.corners[:, 1]))).T

################################################################################
# Population setup                                                             #
################################################################################


def generate_tuning_curves(n_neurons=N_NEURONS,
                           n_dimensions=1,
                           neuron_model=None,
                           relu_params=None,
                           pinh=PINH,
                           rng=None,
                           decoder_reg=DECODER_REG,
                           max_rates=MAX_RATES,
                           n_samples=N_SAMPLES):
    # Make sure either "neuron_model" or "relu_params" is set
    assert (neuron_model is None) != (relu_params is None)

    # Fetch the default random number generate if none is given
    if rng is None:
        rng = np.random

    # Spawn a new random number generate based on the current one. This ensures
    # that the next random state does not depend on the number of neurons or
    # samples
    rng = np.random.RandomState(rng.randint(1 << 31))

    # Choose the tuning curve intercepts, max_rates, and encoders
    icpts = rng.uniform(-0.95, 0.95, n_neurons)
    maxrs = rng.uniform(*max_rates, n_neurons)

    # Select the encoders
    encoders = rng.normal(0, 1, (n_dimensions, n_neurons))
    encoders = (encoders / np.linalg.norm(encoders, axis=0))

    # Define the current translation function J and the response curve G
    def J(ξs):
        return ξs * gain + bias

    if relu_params is None:
        G = neuron_model.estimate_lif_rate_from_current
        GInv = neuron_model.estimate_lif_current_from_rate
    else:
        G = lambda J: np.maximum(0, relu_params["α"] * J + relu_params["β"])
        GInv = lambda a: (a - relu_params["β"]) / relu_params["α"]

    # Current required to generate a small rate
    J0, J1 = GInv(1e-3), GInv(maxrs)

    # Compute the gain and bias
    radius = np.sqrt(n_dimensions)
    gain = ((J1 - J0) / (radius - icpts))
    bias = (J0 - gain * icpts)

    # Compute the activities for the given samples
    samples = rng.uniform(-1, 1, (n_samples, n_dimensions))
    As = G(J(samples @ encoders))

    # Compute the identity decoders
    lambda_ = np.square(decoder_reg * n_samples * max_rates[-1])
    decoders = np.linalg.lstsq(As.T @ As + lambda_ * np.eye(n_neurons),
                               As.T @ samples,
                               rcond=None)[0]

    # Assign an "inhibitory"/"excitatory" flag to each neuron
    inhibitory = rng.uniform(0, 1, n_neurons) < pinh
    neuron_types = np.array((~inhibitory, inhibitory), dtype=np.bool)

    # Return the functions J, G, and the encoders
    return {
        "J": J,
        "G": G,
        "GInv": GInv,
        "E": encoders,
        "D": decoders,
        "types": neuron_types,
        "n_dimensions": n_dimensions,
        "n_neurons": n_neurons,
        "A": lambda x: G(J(x @ encoders))
    }


################################################################################
# Data generation                                                              #
################################################################################


def generate_grid(res=RES):
    xs = np.linspace(-1, 1, res)
    xss, yss = np.meshgrid(xs, xs)
    return xs, np.array((xss.flatten(), yss.flatten())).T


def generate_training_data_random_fun(sigma, res=RES, rms=0.5, rng=None):
    if rng is None:
        rng = np.random

    # Generate the underlying grid
    xs, data_in = generate_grid(res)

    # Generate a random function with the same resolution
    flt = mk_2d_flt(sigma, res)
    data_tar = gen_2d_fun(flt, res, rng).flatten().reshape(-1, 1) * rms

    return data_in, data_tar, xs


def generate_training_data(f, res=RES):
    xs, data_in = generate_grid(res)
    data_tar = f(data_in[:, 0], data_in[:, 1]).flatten().reshape(-1, 1)
    return data_in, data_tar, xs


def eval(w, gEs, gIs):
    return (w[0] + w[1] * gEs + w[2] * gIs) / (w[3] + w[4] * gEs + w[5] * gIs)


def RMSE_norm(actual, tar, rms=None):
    rms = np.sqrt(np.mean(tar**2)) if rms is None else rms
    rmse = np.sqrt(np.mean((actual - tar)**2))
    return rmse / rms


def solve_qp(Apre, Jpost, ws, pre_types, iTh, reg, tol=TOL, max_iter=MAX_ITER):
    # Assemble the connection matrix given the pre-neuron types
    n_samples, n_pre = Apre.shape
    n_samples, n_post = Jpost.shape
    connection_matrix = None
#    if not pre_types is None:
#        connection_matrix = np.zeros((2, n_pre, n_post))
#        connection_matrix[0] = np.tile(pre_types[0], (n_post, 1)).T
#        connection_matrix[1] = np.tile(pre_types[1], (n_post, 1)).T

    # Call the actual weight solver
    return bioneuronqp.solve(Apre=Apre,
                             Jpost=Jpost,
                             ws=ws,
                             connection_matrix=connection_matrix,
                             iTh=iTh,
                             nonneg=True,
                             renormalise=True,
                             tol=tol,
                             reg=reg,
                             n_threads=N_SOLVER_THREADS,
                             progress_callback=None,
                             max_iter=max_iter)


def run_single_spiking_trial(model_name,
                             f,
                             intermediate=False,
                             mask_negative=True,
                             dt=SIM_DT,
                             ss=SIM_SS,
                             T=SIM_T,
                             silent=False,
                             rng=None,
                             reg=None,
                             decoder_reg=DECODER_REG,
                             split_exc_inh=True,
                             tau_pre_filt=None,
                             tau_filt=100e-3,
                             compute_err_model=True,
                             compute_err_net=True,
                             compute_pre_spike_times=False,
                             compute_post_spike_times=False,
                             apply_syn_flt_to_tar=True,
                             fit_parameters=False,
                             res=RES,
                             pinh=PINH,
                             max_rates=MAX_RATES,
                             n_neurons=N_NEURONS,
                             n_samples=N_SAMPLES,
                             tau_syn_e=TAU_SYN_E,
                             tau_syn_i=TAU_SYN_I,
                             weights=None):

    # Variable declarations. Looks weird. Because... Python.
    model, p, ws, Px, Py, Pint, Ptar, Jtar, Jint, Apre, Aint, data_in, data_tar,\
    pre_types, int_types, iTh, WE, WI, WE_int, WI_int, Emodel, Emodel_int, Enet, ts,\
    xs, ys, Opre, Jpre_x, Jpre_y, Oint, Otar, tar, tar_filt, tar_shift, tar_dec, tar_dec_filt,\
    Jtar_opt, Jint_opt, Atar_opt, Aint_opt,\
    Tpre, Tpost, w_per_neuron, w_per_neuron_int = [None] * 44

    #
    # Setup
    #

    # Fetch the default random number generate if none is given
    if rng is None:
        rng = np.random

    # Derive a few new RNGs from the given RNG to make results more
    # reproducable (i.e., not depend on the number of samples)
    rng1 = np.random.RandomState(rng.randint(1 << 31))
    rng2 = np.random.RandomState(rng.randint(1 << 31))
    rng3 = np.random.RandomState(rng.randint(1 << 31))

    #
    # Fetch some parameters
    #

    # Fetch the neuron model
    model = NEURON_MODELS[model_name.split("_")[0]]
    model_lin = NEURON_MODELS["linear"]

    # Fetch the solver parameters ws
    p = SOLVER_PARAMS[model_name]
    ws = mkws(model_name)

    # If alpha and beta are present in the solver parameters, we'll pretend that
    # the neuron is a linear rectified unit
    relu_params = p if ("α" in p) and ("β" in p) else None

    # Fetch the threshold potential if "mask_negative" is specified
    iTh = ITH if mask_negative else None

    # Alias for the number of neurons
    N = n_neurons
    # N0, N1, N2, N3 = 0, 1, 2, 3  # Debug offsets (to avoid ambiguous mat muls)
    N0, N1, N2, N3 = 0, 0, 0, 0

    #
    # Neuron population setup
    #

    # Generate a random set of tuning curves for each population
    if not silent:
        print("Generating population tuning curves and training data...")

    kwargs = {
        "rng": rng2,
        "pinh": pinh,
        "max_rates": max_rates,
        "n_samples": n_samples,
        "decoder_reg": decoder_reg,
    }

    # Create the two pre-populations
    Px = generate_tuning_curves(N + N0, 1, model_lin, **kwargs)
    Py = generate_tuning_curves(N + N1, 1, model_lin, **kwargs)

    # Create the target population
    if relu_params is None:
        Ptar = generate_tuning_curves(N + N2, 1, model, **kwargs)
    else:
        Ptar = generate_tuning_curves(N + N2, 1, relu_params=p, **kwargs)

    # Create the intermediate population
    if intermediate:
        Pint = generate_tuning_curves(N * 2 + N3, 2, model, **kwargs)
        if split_exc_inh:
            int_types = Pint["types"]

    #
    # Weight computation
    #

    # Generate the input training data for the pre-populations
    if callable(f):
        data_in, data_tar, data_xs = generate_training_data(f, res)
    else:
        data_in, data_tar, data_xs = generate_training_data_random_fun(f, res, rng=rng1)

    # Sub-sample both data_in and data_tar
    idx_train = rng3.randint(0, res * res, n_samples)

    # Fetch the target currents
    Jtar = Ptar["J"](data_tar @ Ptar["E"])
    Jint = Pint["J"](data_in @ Pint["E"]) if intermediate else None

    # Fetch the neuron types
    if split_exc_inh:
        pre_types = np.concatenate((Px["types"], Py["types"]), axis=1)

    # Compute the activities of the pre- and intermediate populations
    Apre_x = Px["A"](data_in[:, 0].reshape(-1, 1))
    Apre_y = Py["A"](data_in[:, 1].reshape(-1, 1))
    Apre = np.concatenate((Apre_x, Apre_y), axis=1)
    Aint = Pint["A"](data_in) if intermediate else None
    Apre_int = Aint if intermediate else Apre
    pre_int_types = int_types if intermediate else pre_types

    # Compute all weights if no weight array was given
    if weights is None:
        # Compute the connection weights between the pre- and intermediate
        # population
        if intermediate:
            if not silent:
                print("Computing intermediate connection weights...")
            WE_int, WI_int = solve_qp(Apre=Apre[idx_train],
                                      Jpost=Jint[idx_train],
                                      ws=ws,
                                      pre_types=pre_types,
                                      iTh=iTh,
                                      reg=reg)

        # Compute the connection weights between the pre-/intermediate and the
        # target population
        if not silent:
            print("Computing target connection weights...")
        WE, WI = solve_qp(Apre=Apre_int[idx_train],
                          Jpost=Jtar[idx_train],
                          ws=ws,
                          pre_types=pre_int_types,
                          iTh=iTh,
                          reg=reg)
    else:
        WE_int = weights["WE_int"]
        WI_int = weights["WI_int"]
        WE = weights["WE"]
        WI = weights["WI"]

    #
    # Static error computation
    #
    if compute_err_model:
        if not silent:
            print("Computing static errors...")
        if intermediate:
            gEs_int, gIs_int = Apre @ WE_int, Apre @ WI_int
            Jint_opt = eval(ws, gEs_int, gIs_int)
            Aint_opt = Pint["G"](Jint_opt)
            Emodel_int = RMSE_norm(Aint_opt @ Pint["D"], data_in)

        gEs, gIs = Apre_int @ WE, Apre_int @ WI
        Jtar_opt = eval(ws, gEs, gIs)
        Atar_opt = Ptar["G"](Jtar_opt)
        Emodel = RMSE_norm(Atar_opt @ Ptar["D"], data_tar)

    #
    # Network simulation
    #

    if compute_err_net or compute_pre_spike_times or compute_post_spike_times:
        # Compile the simulators
        sim = model.simulator(dt=dt,
                              ss=ss,
                              record_out=compute_err_net,
                              record_spike_times=compute_post_spike_times)
        sim_lin = model_lin.simulator(
            dt=dt,
            ss=ss,
            record_out=compute_err_net,
            record_spike_times=compute_pre_spike_times)

        # Generate the input spike trains
        if not silent:
            print("Computing input spike trains...")
        ts = np.arange(0, T, dt * ss)
        n_samples = len(ts)
        xs, ys = 2.0 * HilbertCurve(4)(np.linspace(0, 1, n_samples)).T - 1
        Jpre_x = Px["J"](xs.reshape(-1, 1) @ Px["E"])
        Jpre_y = Py["J"](ys.reshape(-1, 1) @ Py["E"])
        if compute_err_net or compute_post_spike_times:
            Opre = np.zeros((n_samples, Px["n_neurons"] + Py["n_neurons"]))
        if compute_pre_spike_times:
            Tpre = []
        for i in range(Px["n_neurons"]):
            res = sim_lin.simulate((Jpre_x[:, i], np.zeros(n_samples)))
            if compute_err_net or compute_post_spike_times:
                Opre[:, i] = res.out
            if compute_pre_spike_times:
                Tpre.append(res.times)
        for i in range(Py["n_neurons"]):
            res = sim_lin.simulate((Jpre_y[:, i], np.zeros(n_samples)))
            if compute_err_net or compute_post_spike_times:
                Opre[:, Px["n_neurons"] + i] = res.out
            if compute_pre_spike_times:
                Tpre.append(res.times)

        # Only continue if we're supposed to compute the actual network error and
        # not just the spike times
        if compute_err_net or compute_post_spike_times:
            if not tau_pre_filt is None:
                Opre = nengo.Lowpass(tau_pre_filt).filt(Opre, dt=dt * ss)

            # Generate the intermediate population spike trains
            if intermediate:
                if not silent:
                    print("Computing intermediate spike trains...")
                Oint = np.zeros((n_samples, Pint["n_neurons"]))
                for i in range(Pint["n_neurons"]):
                    Oint[:, i] = sim.simulate_filtered(
                        ((Opre @ WE_int[:, i]).T, (Opre @ WI_int[:, i]).T),
                        (tau_syn_e, tau_syn_i)).out

            # Compute the initial population spike trains
            if not silent:
                print("Computing target spike trains...")
            Opre_int = Oint if intermediate else Opre
            Otar = np.zeros((n_samples, Ptar["n_neurons"]))
            if compute_post_spike_times:
                Tpost = []

            for i in range(Ptar["n_neurons"]):
                res = sim.simulate_filtered(
                    ((Opre_int @ WE[:, i]).T, (Opre_int @ WI[:, i]).T),
                    (tau_syn_e, tau_syn_i))
                if compute_err_net:
                    Otar[:, i] = res.out
                if compute_post_spike_times:
                    Tpost.append(res.times)

            if compute_err_net:
                #
                # Dynamic error
                #

                if not silent:
                    print("Computing network errors...")

                tar = griddata(data_in, data_tar, (xs.flatten(), ys.flatten())).flatten()
                tau_syn_avg = 0.5 * (tau_syn_e + tau_syn_i)

                # Filter according ot the synaptic time constant
                if apply_syn_flt_to_tar:
                    tar_filt = nengo.Lowpass(tau_syn_avg).filt(tar, dt=dt * ss)
                    if intermediate:
                        tar_filt = nengo.Lowpass(tau_syn_avg).filt(tar_filt,
                                                                   dt=dt * ss)
                else:
                    n_phase = int(
                        (tau_syn_avg * (2 if intermediate else 1)) / (dt * ss))
                    tar_shift = np.concatenate(
                        (np.zeros(n_phase), tar[:-n_phase]))
                    tar_filt = tar_shift

                # Final filter
                tar_filt = nengo.Lowpass(tau_filt).filt(tar_filt, dt=dt * ss)

                # Compute the error
                tar_dec = (Otar @ Ptar["D"])[:, 0]
                tar_dec_filt = nengo.Lowpass(tau_filt).filt(tar_dec,
                                                            dt=dt * ss)

                Enet = RMSE_norm(tar_dec_filt, tar_filt)

    return {
        "model": model,
        "p": p,
        "ws": ws,
        "Px": Px,
        "Py": Py,
        "Pint": Pint,
        "Ptar": Ptar,
        "Jtar": Jtar,
        "Jint": Jint,
        "Apre": Apre,
        "Aint": Aint,
        "data_in": data_in,
        "data_tar": data_tar,
        "pre_types": pre_types,
        "int_types": int_types,
        "iTh": iTh,
        "weights": {
            "WE": WE,
            "WI": WI,
            "WE_int": WE_int,
            "WI_int": WI_int,
        },
        "errors": {
            "Emodel": Emodel,
            "Emodel_int": Emodel_int,
            "Enet": Enet,
        },
        "ts": ts,
        "xs": xs,
        "ys": ys,
        "Opre": Opre,
        "Oint": Oint,
        "Otar": Otar,
        "Tpre": Tpre,
        "Tpost": Tpost,
        "tar": tar,
        "tar_filt": tar_filt,
        "tar_shift": tar_shift,
        "tar_dec": tar_dec,
        "tar_dec_filt": tar_dec_filt,
        "Jpre_x": Jpre_x,
        "Jpre_y": Jpre_y,
        "Atar_opt": Atar_opt,
        "Aint_opt": Aint_opt,
        "Jtar_opt": Jtar_opt,
        "Jint_opt": Jint_opt,
        "w_per_neuron": w_per_neuron,
        "w_per_neuron_int": w_per_neuron_int
    }

