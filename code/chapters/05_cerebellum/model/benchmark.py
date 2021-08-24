#!/usr/bin/env python3

# Nengo Cerebellum Test Code; Andreas Stöckel, Terry Stewart; 2020

import numpy as np
import nengo


def create_random_conv(theta, dt=1e-3, high=1.0, rng=None):
    """
    This function creates a random linear filter (not used at the moment).

    theta: Size of the window represented by the Delay Network in seconds.
    dt:    Length of a time-step in seconds.
    high:  Maximum frequency that should be used to generate the filter.
    rng:   Random number generator instance that should be used to generate the
           filter. If None, will use the global numpy random number generator.
    """
    # Default to the global random number generator
    if rng is None:
        rng = np.random

    # Use a WhiteSignal instance to create a random filter
    sig = nengo.processes.WhiteSignal(period=max(theta, 1.1 / high),
                                      high=high,
                                      rms=0.5)
    flt = sig.run(t=theta, d=1, dt=dt, rng=rng)[:, 0]
    return flt


def benchmark_delay_network_lstsq(xs_tar, As, sigma=0.1):
    """
    Code used to determine the RMSE resulting from trying to decode the target
    signal xs_tar from the (filtered) neural activities As.

    xs_tar: Contains the target signal that should be decoded from the neural
            activity matrix.
    As:     Filtered neural activities over time.
    sigma:  Used to determine the regularisation factor; this should be equal
            to the amount of noise in the filtered activities relative to the
            maximum firing rate.
    """
    # Check some dimensions
    assert xs_tar.ndim == 1
    assert As.ndim == 2
    assert xs_tar.shape[0] == As.shape[0]

    # Compute this for a random subset of neurons
    if As.shape[1] > 1000:
        all_idcs = np.arange(As.shape[1], dtype=int)
        idcs = np.random.RandomState(58791).choice(all_idcs,
                                                   1000,
                                                   replace=False)
        As = As[:, idcs]

    # Try to compute a decoding matrix
    reg = As.shape[0] * np.square(sigma * np.max(As))
    D = np.linalg.lstsq(As.T @ As + reg * np.eye(As.shape[1]),
                        As.T @ xs_tar,
                        rcond=None)[0]

    # Compute the RMS of the signal -- this can be used to compute the
    # NRMSE when evaluating the results
    rms = np.sqrt(np.mean(np.square(xs_tar)))

    # Compute the RMSE between xs_tar and the decoded output
    rmse = np.sqrt(np.mean(np.square(xs_tar - As @ D)))

    return rmse, rms


def benchmark_delay_network_random_conv_single(xs,
                                               As,
                                               theta,
                                               high=1.0,
                                               dt=1e-3,
                                               sigma=0.1,
                                               rng=None):
    """
    Estimates how well the delay network can decode a be used to compute a
    (random) function of the input signal.

    ts: Contains the time in seconds for each sample.
    xs: Is the original one-dimensional signal that is being fed into the delay
        network.
    As: Is the recorded spike-output of the neurons in the network. This should
        already be filtered with a suitable 
    theta: Size of the window represented by the Delay Network in seconds.
    high:  Maximum frequency that should be used to generate the filter.
    dt:    Length of a time-step in seconds.
    sigma: Used to determine the regularisation factor; this should be equal
           to the amount of noise in the filtered activities relative to the
           maximum firing rate.
    rng:   Random number generator instance that should be used to generate the
           filter. If None, will use the global numpy random number generator.
    """

    # Create a random filter
    flt = create_random_conv(theta=theta, dt=dt, high=high, rng=rng)

    # Convolve the input signal with the random filter
    xs_tar = np.convolve(xs, flt, 'same')

    # Compute how well the delayed signal can be approximated
    return benchmark_delay_network_lstsq(xs_tar=xs_tar, As=As, sigma=sigma)


def benchmark_delay_network_random_conv_average(n_repeat,
                                                xs,
                                                As,
                                                theta,
                                                high=1.0,
                                                dt=1e-3,
                                                sigma=0.1,
                                                rng=None):
    """
    Averages over n_repeat runs of "benchmark_delay_network_random_conv_single".
    See that function for a description.
    """
    E = np.zeros((n_repeat, 2))
    for i in range(n_repeat):
        E[i] = benchmark_delay_network_random_conv_single(xs=xs,
                                                          As=As,
                                                          theta=theta,
                                                          high=high,
                                                          dt=dt,
                                                          sigma=sigma,
                                                          rng=rng)
    return np.mean(E, axis=0)


def benchmark_delay_network_delay(xs,
                                  As,
                                  theta,
                                  delays=[1.0],
                                  dt=1e-3,
                                  sigma=0.1):
    """
    Estimates how well the delay network can decode a be used to compute a
    delay.

    xs: Original input signal.
    As: Filtered neural activities.
    theta: Width of the delay network window in seconds.
    delays: Array of delays to benchmark
    dt: Simulation timestep in seconds.
    sigma: Regularisation factor.
    """

    # Compute how well the delayed signal can be approximated
    Es = np.zeros((len(delays), 2))
    for i, delay in enumerate(delays):
        # Shift the input signal by n samples
        n = int(theta * delay / dt)
        n0, n1 = 0, max(0, len(xs) - n)
        xs_tar = np.concatenate((np.zeros(n), xs[n0:n1]))

        Es[i] = benchmark_delay_network_lstsq(xs_tar=xs_tar,
                                              As=As,
                                              sigma=sigma)
    return Es


def white_noise_input(high=2.0):
    """
    White noise bandlimit in Hz.
    """
    return ("white_noise", high)

def pulse_input(t_off=0.9, t_on=0.1):
    """
    Pule on and off times as percentage of theta.
    """
    return ("pulse", t_off, t_on)


def build_test_network(input_descr,
                       T=10.0,
                       rng=None,
                       bias_mode="jbias_realistic_pcn_intercepts",
                       golgi_inh_bias_mode="none",
                       pcn_xi_lower_bound=None,
                       pcn_max_rates=None,
                       n_pcn=100,
                       probe_granule_decoded=False,
                       probe_pcn_spikes=False,
                       **kwargs):
    """
    Builds the test network. Injects a bandlimited white noise signal into the
    Granule-Golgi circuit and records the neural activities with a filter of
    100ms.
    """
    import nengo_bio as bio
    from granule_golgi_circuit import GranuleGolgiCircuit

    # Use the global random number generator if none is given
    if rng is None:
        rng = np.random

    # Make sure that "bias_mode" is valid
    assert bias_mode in {
        None,
        "uniform_pcn_intercepts", "realistic_pcn_intercepts", "very_realistic_pcn_intercepts",
        "jbias_uniform_pcn_intercepts", "jbias_realistic_pcn_intercepts", "jbias_very_realistic_pcn_intercepts",
        "exc_jbias_uniform_pcn_intercepts", "exc_jbias_realistic_pcn_intercepts", "exc_jbias_very_realistic_pcn_intercepts",
        "inh_jbias_uniform_pcn_intercepts", "inh_jbias_realistic_pcn_intercepts", "inh_jbias_very_realistic_pcn_intercepts",
    }

    # Either bias_mode OR pcn_xis must be set, not both
    assert (bias_mode is None) != (pcn_xi_lower_bound is None)

    # Make sue golgi_inh_bias_mode is valid
    assert golgi_inh_bias_mode in {
        "none", "recurrent", "lugaro", "recurrent_and_lugaro"
    }

    # Create the network with a seed depending on the given rng
    with nengo.Network(seed=rng.randint(1 << 31)) as model:
        # Select the PCN tuning curves
        pcn_encs = rng.choice([-1, 1], n_pcn)
        if not bias_mode is None:
            if "very_realistic_pcn_intercepts" in bias_mode:
                pcn_xis = rng.uniform(-0.35, 0.95, n_pcn)
            elif "realistic_pcn_intercepts" in bias_mode:
                pcn_xis = rng.uniform(-0.15, 0.95, n_pcn)
            else:
                pcn_xis = rng.uniform(-0.95, 0.95, n_pcn)
        else:
            pcn_xis = rng.uniform(pcn_xi_lower_bound, 0.95, n_pcn)

        # Enable/disable lugaro cells based on the bias mode

        # Parse golgi_inh_bias_mode
        kwargs["use_lugaro"] = ("lugaro" in golgi_inh_bias_mode)
        kwargs["use_golgi_recurrence"] = ("recurrent" in golgi_inh_bias_mode)

        bio_bias_mode = None
        if not bias_mode is None:
            if "exc_jbias_" in bias_mode:
                bio_bias_mode = bio.ExcJBias
            elif "inh_jbias_" in bias_mode:
                bio_bias_mode = bio.InhJBias
            elif "jbias_" in bias_mode:
                bio_bias_mode = bio.JBias
            else:
                bio_bias_mode = bio.Decode

        if pcn_max_rates is None:
            pcn_max_rates = nengo.dists.Uniform(50, 100)
        else:
            pcn_max_rates = nengo.dists.Uniform(*pcn_max_rates)

        # Only vary the granule cell bias mode
        kwargs["bias_mode_granule"] = bio_bias_mode

        # Pre-cerebellum nuclei (Input)
        ens_pcn = bio.Ensemble(n_neurons=n_pcn,
                               dimensions=1,
                               p_exc=1.0,
                               label="ens_pcn",
                               max_rates=pcn_max_rates,
                               encoders=pcn_encs.reshape(-1, 1),
                               intercepts=pcn_xis)

        # The actual network
        net_granule_golgi = GranuleGolgiCircuit(ens_pcn, **kwargs)

        # Depending on the given input type, either feed a white noise signal or
        # a pulse signal into the network
        input_type = input_descr[0]
        if input_type == "white_noise":
            _, high = input_descr
            nd_in = nengo.Node(nengo.processes.WhiteSignal(high=high,
                               period=T, rms=0.5))
        elif input_type == "pulse":
            theta = net_granule_golgi.theta
            _, t_off, t_on = input_descr
            t_off, t_on = theta * t_off, theta * t_on
            nd_in = nengo.Node(
                lambda t: 1.0 if (t % (t_on + t_off)) > t_off else 0.0)

        # Connect the input node to the pre-cerebellum nuclei
        nengo.Connection(nd_in, ens_pcn, synapse=None)

        # Probe both the input, and the granule cells
        p_in = nengo.Probe(nd_in, synapse=None)
        p_granule = nengo.Probe(net_granule_golgi.ens_granule.neurons,
                                synapse=100e-3)
        if probe_granule_decoded:
            p_granule_decoded  = nengo.Probe(net_granule_golgi.ens_granule,
                                synapse=100e-3)
        else:
            p_granule_decoded = None
        if probe_pcn_spikes:
            p_pcn = nengo.Probe(ens_pcn.neurons, synapse=None)
        else:
            p_pcn = None

    return model, net_granule_golgi, p_in, p_granule, p_granule_decoded, p_pcn


def build_and_run_test_network(input_descr,
                               T=10.0,
                               record_weights=False,
                               probe_granule_decoded=False,
                               probe_spatial_data=False,
                               probe_pcn_spikes=False,
                               probe_pcn_go_to_gr_connection=False,
                               **kwargs):
    """
    Builds a Granule Golgi Circuit instance with a test input and runs it for
    the specified amount of time.
    """

    # Build the model
    model, net_granule_golgi, p_in, p_granule, p_granule_decoded, p_pcn = \
        build_test_network(
            input_descr, T,
            probe_granule_decoded=probe_granule_decoded, probe_pcn_spikes=probe_pcn_spikes, **kwargs)

    # Run the simulation for a while
    with nengo.Simulator(model, progress_bar=None) as sim:
        sim.run(T)

    # Record the weights if desired
    if record_weights:
        import os, json, h5py, hashlib

        # Copy the parameters
        params = {
            "input_descr": input_descr,
            "random_seed": kwargs["rng"].randint(1 << 32),
            **kwargs
        }
        params["rng"] = None

        # Generate a unique filename for these parameters
        m = hashlib.sha256()
        m.update(json.dumps(params).encode('utf-8'))
        fn = "out/weights/weights_" + m.hexdigest()[:8] + ".h5"

        # Create the target directory
        os.makedirs(os.path.dirname(fn), exist_ok=True)

        # Open the target h5 file
        with h5py.File(fn, 'w') as f:
            # Create a dummy dataset containing the parameters
            dset_params = f.create_dataset("params", (1,), dtype='i8')
            dset_params.attrs["params"] = json.dumps(params)

            # Generic function for storing the weight matrices of a connection
            def store_weights_for_conn(conn_name):
                # Do nothing if these weights do not exist
                if not hasattr(net_granule_golgi, conn_name):
                    return

                # Try to fetch the weights
                conn = getattr(net_granule_golgi, conn_name)
                W = sim.model.params[conn].weights
                if not isinstance(W, dict):
                    W = {"combined": W}

                # Create a dataset for each weight matrix
                for key, value in W.items():
                    dset_W = f.create_dataset(
                        "weights_" + conn_name + "_" + repr(key),
                        data=value,
                        chunks=True,
                        compression="lzf")

            # Store the individual weight matrices
            store_weights_for_conn("conn_pcn_gr_go_lg_to_go")
            store_weights_for_conn("conn_pcn_go_to_gr")
            store_weights_for_conn("conn_pcn_to_gr")
            store_weights_for_conn("conn_pcn_to_go")
            store_weights_for_conn("conn_gr_to_go")
            store_weights_for_conn("conn_go_to_gr")

    # Return the measured data
    ts = sim.trange()
    xs = sim.data[p_in][:, 0]
    As = sim.data[p_granule]

    #      0   1   2     3   4                        5     6     7
    res = [ts, xs, None, As, net_granule_golgi.theta, None, None, None]

    if probe_spatial_data:
        net = net_granule_golgi
        golgi = net.ens_golgi
        granule = net.ens_granule
        x_golgi = sim.data[golgi].locations
        x_granule = sim.data[granule].locations
        cvt_go_gr = net.conn_pcn_go_to_gr.connectivity[(golgi, granule)]
        spatial_data = {
            "golgi_locations": x_golgi,
            "granule_locations": x_granule,
            "ps": cvt_go_gr.get_probabilities(golgi.n_neurons, granule.n_neurons, golgi, granule, sim.data),
            "W": sim.model.params[net.conn_pcn_go_to_gr].weights,
        }
        res[5] = spatial_data

    if probe_pcn_spikes:
        res[6] = sim.data[p_pcn]

    if probe_pcn_go_to_gr_connection:
        res[7] = sim.data[net_granule_golgi.conn_pcn_go_to_gr]

    if probe_granule_decoded:
        ys = sim.data[p_granule_decoded]
        res[2] = ys

    return tuple(filter(lambda x: not x is None, res))


def _run_benchmark_callback(p):
    """
    Runs a single benchmark. This is the entry point for multiprocessing.
    """
    import warnings

    # Fetch the experiment parameters
    idx, seed, delays, kwargs = p

    # Create a RandomState instance
    rng = np.random.RandomState(seed)

    # For good measure, also set the global random seed to something that is
    # derrived from the given seed
    np.random.seed((seed * 319 + 457) % (1 << 31))

    # Run and evaluate the experiment
    with warnings.catch_warnings():  # Silence the decoder cache UserWarning
        warnings.filterwarnings("ignore", category=UserWarning)
        ts, xs, As, theta = build_and_run_test_network(rng=rng,
                                                       **kwargs)
        E = benchmark_delay_network_delay(xs=xs,
                                          As=As,
                                          delays=delays,
                                          theta=theta)

    # Return the resulting error values as well as the index describing the
    # location of the error values within the error matrix.
    return idx, E


def slin(vmin, vmax):
    """
    Creates a callback function for a linear sweep from vmin to vmax.
    """
    return lambda i, n: vmin + (i / (n - 1)) * (vmax - vmin)


def slog(vmin, vmax, base=10):
    """
    Creates a callback function for a logarithmic sweep from vmin to vmax.
    """
    vmin = np.log(vmin) / np.log(base)
    vmax = np.log(vmax) / np.log(base)
    return lambda i, n: np.power(base, vmin + (i / (n - 1)) * (vmax - vmin))


def sweep(name, n, cback, sweep=None):
    """
    This function is used to generate a (nested) set of sweeps.

    name:  Name of the parameter that is to be swept.
    n:     Number of points along the sweep.
    cback: A callback function that is called for each element in the sweep with
           the paremters i (current index), n (number of elements).
           Such a callback can be created by a call to slin or slog.
    sweep: May be set to the result of another call to sweep, facilitating
           the creation of multi-dimensional sweeps.
    """
    if sweep is None:
        sweep = [{}]

    # Combine the specified sweep with the given sweep
    res = []
    for i in range(n):
        for args in sweep:
            params = {**args} # Copy the arguments
            params[name] = cback(i, n)
            res.append(params)
    return res


def run_benchmark(sweep,
                  n_repeat,
                  n_delays,
                  filename_prefix="nengo_cerebellum",
                  seed=45781,
                  randomize_all=False,
                  concurrency=None):
    """
    Runs the benchmark and writes the results to an h5 file with an
    automatically generated name. Returns the name of the h5 file.

    sweep:    The result of a call to the "sweep" function in this module.
    n_repeat: The number of repetitions with different random seeds.
    seed:     The random seed that should be used to construct each individual
              network. The same seed is used for the same repetition, i.e.,
              the first execution of the network for each parameter set will
              use the same seed, the second execution of the network with the
              same parameter set will use a different seed.
    randomize_all:
              If set to true, a different seed is used for all executions of the
              network.
    """
    import os, json, datetime, multiprocessing
    import h5py # Use h5 for storing files
    from nengo_bio.internal.env_guard import EnvGuard
    from tqdm import tqdm

    # Create the sweep over the delays and the random inputs
    delays = list(np.linspace(0, 1, n_delays))

    # Create n_repeat instances of the parameters with different seeds
    params = []
    for i in range(len(sweep)):
        for j in range(n_repeat):
            local_seed = seed + j
            if randomize_all:
                local_seed += i * n_repeat
            params.append([(i, j), local_seed, delays, sweep[i]])

    # Create the target h5 file that will store both the execution results and
    # the parameters.
    os.makedirs('out', exist_ok=True)
    filename = ("out/" + filename_prefix + "_" + 
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".h5")
    with h5py.File(filename, "w") as f:
        # Create the errors dataset
        dims = (len(sweep), n_delays, 2, n_repeat)
        Es = f.create_dataset("errors", dims, dtype='f8')

        # Store the parameters as a JSON-encoded string
        Es.attrs["params"] = json.dumps(params)

        # Disable multithreading; we're already launching multiple processes
        # for inter-task parallelism, which is far more efficient than
        # intra-task parallelism
        with EnvGuard({
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
        }) as env:
            # Run the simulation with "concurrency" threads
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(concurrency) as pool:
                for idcs, err in tqdm(pool.imap_unordered(_run_benchmark_callback,
                                                          params),
                                      total=len(params)):
                    Es[idcs[0], :, :, idcs[1]] = err

    # Return the name of the file the results were saved to
    return filename



if __name__ == "__main__":
    run_benchmark()

#    print(ts.shape, xs.shape, As.shape)

