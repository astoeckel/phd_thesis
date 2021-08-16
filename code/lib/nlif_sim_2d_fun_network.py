import numpy as np
from scipy.interpolate import griddata
import nengo

import os, sys
import nlif
import nlif.solver

from nef_synaptic_computation.lif_utils import *
from nef_synaptic_computation.tuning_curves import *

from gen_2d_fun import *
from nlif_parameters import *


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
        self.n_corners = (2**n)**2
        self.corners = np.zeros((self.n_corners, 2))
        self.steps = np.arange(self.n_corners)

        steps = np.arange(self.n_corners)
        for s in 2**np.arange(n):
            r = np.empty_like(self.corners, dtype='int')
            r[:, 0] = 1 & (steps // 2)
            r[:, 1] = 1 & (steps ^ r[:, 0])
            self._rot(s, r)
            self.corners += s * r
            steps //= 4

        self.corners /= (2**n) - 1

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
        return np.vstack((np.interp(step, self.steps, self.corners[:, 0]),
                          np.interp(step, self.steps, self.corners[:, 1]))).T


################################################################################
# Population setup                                                             #
################################################################################


def generate_tuning_curves(n_neurons=N_NEURONS,
                           n_dimensions=1,
                           assm=None,
                           pinh=PINH,
                           rng=None,
                           decoder_reg=DECODER_REG,
                           max_rates=MAX_RATES,
                           intercepts=INTERCEPTS,
                           n_samples=N_SAMPLES):
    # Fetch the default random number generate if none is given
    if rng is None:
        rng = np.random

    # Spawn a new random number generate based on the current one. This ensures
    # that the next random state does not depend on the number of neurons or
    # samples
    rng = np.random.RandomState(rng.randint(1 << 31))

    # Choose the tuning curve intercepts, max_rates, and encoders
    icpts = rng.uniform(*intercepts, n_neurons)
    maxrs = rng.uniform(*max_rates, n_neurons)

    # Select the encoders
    encoders = rng.normal(0, 1, (n_dimensions, n_neurons))
    encoders = (encoders / np.linalg.norm(encoders, axis=0))

    # Define the current translation function J and the response curve G
    def J(ξs):
        return ξs * gain + bias

    G = assm.lif_rate
    GInv = assm.lif_rate_inv

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
    if not pinh is None:
        inhibitory = rng.uniform(0, 1, n_neurons) < pinh
        neuron_types = np.array((~inhibitory, inhibitory), dtype=bool)
    else:
        neuron_types = np.ones((2, n_neurons), dtype=bool)

    # Return the functions J, G, and the encoders
    return {
        "gains": gain,
        "biases": bias,
        "icpts": icpts,
        "maxrs": maxrs,
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


def eval(assm, sys, A, W):
    return assm.i_som(A @ W.T, reduced_system=sys)


def RMSE_norm(actual, tar, rms=None):
    rms = np.sqrt(np.mean(tar**2)) if rms is None else rms
    rmse = np.sqrt(np.mean((actual - tar)**2))
    return rmse / rms


def solve(Apre,
          Jpost,
          assm,
          reduced_system,
          pre_types,
          iTh,
          tol=TOL,
          max_iter=MAX_ITER,
          N_epochs=N_EPOCHS,
          alpha1=1e0,
          alpha2=1e0,
          alpha3=1e-3,
          reg1=1e-3,
          reg2=1e-6,
          gamma=0.95):
    # Scale iTh
    iTh = (None if iTh is None else (iTh * reduced_system.out_scale))

    # Assemble the connection matrix given the pre-neuron types
    n_samples, n_pre = Apre.shape
    n_samples, n_post = Jpost.shape

    # Determine which input channels are excitatory and which inhibitory
    channel_types = []
    v_th = assm.lif_parameters()["v_th"]
    for i in range(assm.n_compartments):
        for chan in assm.channels[i]:
            if not chan.is_static:
                channel_types.append(1 if chan.is_inhibitory(v_th) else 0)
    assert len(channel_types) == reduced_system.n_inputs

    # Assemble W_mask
    W_mask = np.zeros((n_post, reduced_system.n_inputs, n_pre), dtype=bool)
    for i in range(reduced_system.n_inputs):
        W_mask[:, i] = pre_types[channel_types[i]]

    # Compute the initial weights; this is the same for every post-neuron
    Apre_scale = 1.0 / np.max(Apre)
    W = SOLVER[0].nlif_solve_weights_iter(reduced_system,
                                          Apre * Apre_scale,
                                          np.zeros((1, n_samples)),
                                          np.zeros((1, reduced_system.n_inputs,
                                                    n_pre)),
                                          W_mask[0:1],
                                          reg1=1e-1,
                                          progress_callback=None,
                                          n_threads=1)
    W = np.repeat(W, n_post, axis=0)

    # Compute the actual weights
    epoch = [0]

    def progress(n, n_total):
        n_it = epoch[0] * n_post + n
        n_it_total = N_epochs * n_post
        sys.stderr.write(f"\r{n_it}/{n_it_total} iterations...")
        return True

    for i in range(N_epochs):
        scale = np.power(gamma, -i)
        epoch[0] = i
        W[np.abs(W) < np.percentile(np.abs(W), 95) * 1e-3] = 0.0 # Slightly improves runtime by making P more sparse
        W = SOLVER[0].nlif_solve_weights_iter(reduced_system,
                                              Apre * Apre_scale,
                                              Jpost.T *
                                              reduced_system.out_scale,
                                              W,
                                              W_mask,
                                              reg1=reg1,
                                              reg2=reg2,
                                              alpha1=alpha1,
                                              alpha2=alpha2,
                                              alpha3=alpha3 * scale,
                                              J_th=iTh,
                                              progress_callback=progress,
                                              n_threads=N_SOLVER_THREADS)

    return W * Apre_scale / reduced_system.in_scale


def run_single_spiking_trial(model_name,
                             f,
                             intermediate=False,
                             mask_negative=True,
                             dt=SIM_DT,
                             ss=SIM_SS,
                             T=SIM_T,
                             silent=False,
                             rng=None,
                             reg=SOLVER_REG,
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
                             intercepts=INTERCEPTS,
                             max_rates_tar=MAX_RATES_TAR,
                             intercepts_tar=INTERCEPTS_TAR,
                             n_neurons=N_NEURONS,
                             n_samples=N_SAMPLES,
                             tau_syn_e=TAU_SYN_E,
                             tau_syn_i=TAU_SYN_I,
                             weights=None,
                             tar_filt=None,
                             N_epochs=N_EPOCHS):

    # Variable declarations. Looks weird. Because... Python.
    assm, sys, model, p, ws, Px, Py, Pint, Ptar, Jtar, Jint, Apre, Aint, \
    data_in, data_tar, pre_types, int_types, iTh, W, W_int, Emodel, \
    Emodel_int, Enet, ts, xs, ys, Opre, Jpre_x, Jpre_y, Oint, Otar, tar, \
    tar_shift, tar_dec, tar_dec_filt, Jtar_opt, Jint_opt, Atar_opt, Aint_opt,\
    Tpre, Tpost, w_per_neuron, w_per_neuron_int = [None] * 43

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

    # Fetch the neuron model and the corresponding model parameters
    assm_lin, sys_lin = get_neuron_sys(NEURON_MODELS["lif"])
    assm, sys = get_neuron_sys(NEURON_MODELS[model_name])

    # Fetch the threshold potential if "mask_negative" is specified
    iTh = (assm.i_th() * ITH) if mask_negative else None

    # Alias for the number of neurons
    N = n_neurons
    N0, N1, N2, N3 = 0, 1, 2, 3  # Debug offsets (to avoid ambiguous mat muls)
    #N0, N1, N2, N3 = 0, 0, 0, 0

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
        "intercepts": intercepts,
        "n_samples": n_samples,
        "decoder_reg": decoder_reg,
    }

    # Create the two pre-populations
    Px = generate_tuning_curves(N + N0, 1, assm_lin, **kwargs)
    Py = generate_tuning_curves(N + N1, 1, assm_lin, **kwargs)

    # Create the intermediate population
    if intermediate:
        Pint = generate_tuning_curves(N * 2 + N3, 2, assm_lin, **kwargs)
        if split_exc_inh:
            int_types = Pint["types"]

    # Create the target population
    kwargs["max_rates"] = max_rates_tar
    kwargs["intercepts"] = intercepts_tar
    Ptar = generate_tuning_curves(N + N2, 1, assm, **kwargs)

    #
    # Weight computation
    #

    # Generate the input training data for the pre-populations
    if callable(f):
        data_in, data_tar, data_xs = generate_training_data(f, res)
    else:
        data_in, data_tar, data_xs = generate_training_data_random_fun(
            f, res, rng=rng1)

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
            W_int = solve(
                Apre=Apre[idx_train],
                Jpost=Jint[idx_train],
                assm=assm_lin,
                reduced_system=sys_lin,
                pre_types=pre_types,
                iTh=iTh,
                N_epochs=1,
                reg1=reg,
            )  # We don't need many iterations for the linear neurons

        # Compute the connection weights between the pre-/intermediate and the
        # target population
        if not silent:
            print("Computing target connection weights...")
        W = solve(Apre=Apre_int[idx_train],
                  Jpost=Jtar[idx_train],
                  assm=assm,
                  reduced_system=sys,
                  pre_types=pre_int_types,
                  iTh=iTh,
                  N_epochs=N_epochs,
                  reg1=reg)
    else:
        W_int = weights["W_int"]
        W = weights["W"]

    #
    # Static error computation
    #
    if compute_err_model:
        if not silent:
            print("Computing static errors...")
        if intermediate:
            gs = np.einsum('Nm,nkm->Nnk', Apre, W_int)
            Jint_opt = assm.i_som(gs, reduced_system=sys_lin)
            Aint_opt = Pint["G"](Jint_opt)
            Emodel_int = RMSE_norm(Aint_opt @ Pint["D"], data_in)

        gs = np.einsum('Nm,nkm->Nnk', Apre_int, W)
        Jtar_opt = assm.i_som(gs, reduced_system=sys)
        Atar_opt = Ptar["G"](Jtar_opt)
        Emodel = RMSE_norm(Atar_opt @ Ptar["D"], data_tar)


    #
    # Network simulation
    #
    if compute_err_net or compute_pre_spike_times or compute_post_spike_times:
        with nlif.Simulator(assm_lin, dt=dt, ss=ss, record_out=compute_err_net, record_spike_times=compute_pre_spike_times) as sim_lin, \
             nlif.Simulator(assm, dt=dt, ss=ss, record_out=compute_err_net, record_spike_times=compute_post_spike_times) as sim:
            # Generate the input spike trains
            if not silent:
                print("Computing input spike trains...")
            ts = np.arange(0, T, dt * ss)
            n_samples = len(ts)
            xs, ys = 2.0 * HilbertCurve(4)(np.linspace(0, 1, n_samples)).T - 1
            Jpre_x = Px["J"](xs.reshape(-1, 1) @ Px["E"]) / 5e-3 # mul_E / mul_I
            Jpre_y = Py["J"](ys.reshape(-1, 1) @ Py["E"]) / 5e-3 # mul_E / mul_I
            if compute_err_net or compute_post_spike_times:
                Opre = np.zeros((n_samples, Px["n_neurons"] + Py["n_neurons"]))
            if compute_pre_spike_times:
                Tpre = []
            for i in range(Px["n_neurons"]):
                gX = np.array((Jpre_x[:, i], np.zeros(n_samples))).T
                res = sim_lin.simulate(gX)
                if compute_err_net or compute_post_spike_times:
                    Opre[:, i] = res.out
                if compute_pre_spike_times:
                    Tpre.append(res.times)
            for i in range(Py["n_neurons"]):
                gY = np.array((Jpre_y[:, i], np.zeros(n_samples))).T
                res = sim_lin.simulate(gY)
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
                        Oint[:, i] = sim_lin.simulate_filtered(
                            Opre @ W_int[i].T,
                            (tau_syn_e, tau_syn_i)).out

                # Compute the initial population spike trains
                if not silent:
                    print("Computing target spike trains...")
                Opre_int = Oint if intermediate else Opre
                Otar = np.zeros((n_samples, Ptar["n_neurons"]))
                if compute_post_spike_times:
                    Tpost = []

                # Compute the actual neuron population spike trains
                for i in range(Ptar["n_neurons"]):
                    res = sim.simulate_filtered(Opre_int @ W[i].T,
                        (tau_syn_e, tau_syn_i) * (sys.n_inputs // 2))
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
                    if tar_filt is None:
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
        "assm": assm,
        "sys": sys,
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
            "W": W,
            "W_int": W_int,
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

