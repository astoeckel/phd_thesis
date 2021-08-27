# Nengo Cerebellum Test Code; Andreas Stöckel, Terry Stewart; 2020

import nengo
import nengo_bio as bio
import numpy as np
import pytry
import matplotlib

import sys, os
sys.path.append(os.path.dirname(__file__))
from granule_golgi_circuit import GranuleGolgiCircuit

import scipy.stats

# Run using
# pytry model/blink_trial.py --mode two_populations_dales_principle --use_spatial_constraints True --n_pcn_golgi_convergence 100 --n_pcn_granule_convergence 5 --n_granule_golgi_convergence 100 --n_golgi_granule_convergence 5 --n_golgi_golgi_convergence 100 --n_granule 10000 --n_golgi 100 --data_dir=exp --data_filename two_populations_dales_principle_detailed --data_format npz

def 𐌈(**kwargs):
    return kwargs

class EyeblinkReflex(nengo.Process):
    """
    This class implements the trajectory generation for the agonist and
    antagonist muscles generating an eyeblink.
    """

    P_AGONIST    = [ 0.29104462, -0.5677302,  0.0125624,  81.2206584]
    T_AGONIST    = 250e-3
    P_ANTAGONIST = [ 0.33981309, 10.7624779,  0.3168123,   3.4498503]
    T_ANTAGONIST = 275e-3

    def __init__(self):
        super().__init__(default_size_in=1, default_size_out=2)

    @staticmethod
    def skewnorm(x, mu, a, std, s):
        return (s * std) * scipy.stats.skewnorm.pdf(x, a, mu, std)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        # Compute the convolutions
        def mkconv(P, T0, T1=1.0):
            ts = np.arange(T0, T1, dt)
            return EyeblinkReflex.skewnorm(ts, *P)

        conv_ag = mkconv(
            EyeblinkReflex.P_AGONIST, EyeblinkReflex.T_AGONIST)
        conv_an = mkconv(
            EyeblinkReflex.P_ANTAGONIST, EyeblinkReflex.T_ANTAGONIST)

        state=np.zeros(1 + len(conv_ag) + len(conv_an))
        I_lx = slice(0, 1)                                 # last x
        I_ag = slice(I_lx.stop, I_lx.stop + len(conv_ag))  # agonist
        I_an = slice(I_ag.stop, I_ag.stop + len(conv_an))  # antagonist

        def step(t, x, state=state):
            # Compute the differential, compute a positive and negative branch
            # of the differential
            lx, ag, an = state[I_lx], state[I_ag], state[I_an]
            dx = (x - lx[0]) / dt
            dxp, dxn = np.maximum(0, dx), np.maximum(0, -dx)

            # Store the last x value
            lx[0] = x

            # Shift the history by one element and insert the new
            # agonist/antagonist
            ag[1:] = ag[:-1]
            an[1:] = an[:-1]
            ag[0] = dxp
            an[0] = dxn

            # Return the current state as output
            return np.array((
                np.sum(conv_ag * ag) * dt,
                np.sum(conv_an * an) * dt))

        return step



class Eyelid(nengo.Process):
    """
    The Eyelid class models the eylid. It turns an agonist and antagonist input
    into an eyelid "closedness" between zero (full closed) and one (full open).
    """

    def __init__(self):
        super().__init__(default_size_in=2, default_size_out=1)
        self.step = self.make_step(2, 1, 1e-3, None, None)


    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state=np.zeros(1)
        state[0] = 0.0

        def step(t, x, state=state):
            # Update the eyelid location
            state[0] = np.clip(state[0] + (np.clip(x[0], 0, None) -
                                          (np.clip(x[1], 0, None))) * dt, 0, 1)

            # Return the state as output
            return np.array((state[0],))

        setattr(step, 'state', state);

        return step

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    @property
    def _nengo_html_(self):
        return """
<svg width="200" height="100" viewBox="-1.9 -1.5 3.5 2.75" version="1.1">
	<defs>
		<clipPath clipPathUnits="userSpaceOnUse" id="clipPath973">
			<path style="fill:#ffffff" d="M -2.3,0 C -2.3,0 -0.9,{o} 0,{o} 1,{o} 2.3,0 2.3,0 2.3,0 1,-{o} 0,-{o} -1,-{o} -2.3,0 -2.3,0 Z" />
		</clipPath>
	</defs>
	<path
		style="fill:#ffffff;stroke:#000000;stroke-width:0.04px"
		d="M -2.3,0 C -2.3,0 -0.9,{o} 0,{o} 1,{o} 2.3,0 2.3,0 2.3,0 1,-{o} 0,-{o} -1,-{o} -2.3,0 -2.3,0 Z" />
	<path
		style="fill:#000000"  clip-path="url(#clipPath973)"
		d="m 0.00568441,-0.83354084 a 0.80501685,0.80501685 0 0 0 -0.80511881,0.8051188 0.80501685,0.80501685 0 0 0 0.80511881,0.80511881 0.80501685,0.80501685 0 0 0 0.8051188,-0.80511881 0.80501685,0.80501685 0 0 0 -0.8051188,-0.8051188 z m 0,0.57877603 a 0.22641097,0.22641097 0 0 1 0.22634277,0.22634277 0.22641097,0.22641097 0 0 1 -0.22634277,0.22634277 0.22641097,0.22641097 0 0 1 -0.22634277,-0.22634277 0.22641097,0.22641097 0 0 1 0.22634277,-0.22634277 z" />
</svg>""".format(o=1.0 - self.step.state[0])



def make_lmu(q=6, theta=1.0):
    # Do Aaron's math to generate the matrices
    #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
    Q = np.arange(q, dtype=np.float64)
    R = (2*Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B

def add_labels(model, locals):
    for k, v in locals.items():
        if isinstance(v, (nengo.Node, nengo.Ensemble)) and v.label is None:
            v.label = k


class BlinkTrial(pytry.PlotTrial):
    def params(self):
        # Build a temporary Golgi-Granule circuit instance to get the default parameters
        with nengo.Network() as model:
            pre = bio.Ensemble(n_neurons=100, dimensions=1)
            net = GranuleGolgiCircuit(pre)

        # Experiment setup
        self.param('time between trials', period=0.8)
        self.param('time between tone and puff', t_delay=0.15)
        self.param('tone length', t_tone=0.1)
        self.param('puff length', t_puff=0.1)
        self.param('number of trials', n_trials=4)
        self.param('only run minimal model', do_minimal=True)

        # Recording
        self.param('save data from plots', save_plot_data=True)
        self.param('probe resolution', sample_every=0.001)

        # Motor model
        self.param('eyelid opening constant', eye_bias=4)

        # Non-granule-golgi time constants
        self.param('learning rate', learning_rate=1e-4)
        self.param('tau for learning rule', tau_pre=0.2)
        self.param('tau for error feedback', tau_error=0.2)
        self.param('tau for purkinje output', tau_purkinje=0.01)

        # Granule Golgi Circuit
        self.param('q', q=net.q)
        self.param('theta', theta=net.theta)
        self.param('tau for granule', tau=net.tau)
        self.param('use cosine intercept distribution', use_cosine=True)
        self.param('granule golgi mode', mode=net.mode)
        self.param('use_spatial_constraints', use_spatial_constraints=net.use_spatial_constraints)
        self.param('n_pcn_golgi_convergence', n_pcn_golgi_convergence=-1)
        self.param('n_pcn_granule_convergence', n_pcn_granule_convergence=-1)
        self.param('n_granule_golgi_convergence', n_granule_golgi_convergence=-1)
        self.param('n_golgi_granule_convergence', n_golgi_granule_convergence=-1)
        self.param('n_golgi_golgi_convergence', n_golgi_golgi_convergence=-1)
        self.param('n_granule', n_granule=net.n_granule)
        self.param('n_golgi', n_golgi=net.n_golgi)
        self.param('n_pcn', n_pcn=100)
        self.param('pcn_max_rates_lower', pcn_max_rates_lower=25)
        self.param('pcn_max_rates_upper', pcn_max_rates_upper=50)
        self.param('bias_mode', bias_mode="realistic_pcn_intercepts")


    def evaluate(self, p, plt):
        # Make sure the convergence numbers are either non-negative or None
        if p.n_pcn_golgi_convergence < 0:
            p.n_pcn_golgi_convergence = None
        if p.n_pcn_granule_convergence < 0:
            p.n_pcn_granule_convergence = None
        if p.n_granule_golgi_convergence < 0:
            p.n_granule_golgi_convergence = None
        if p.n_golgi_granule_convergence < 0:
            p.n_golgi_granule_convergence = None
        if p.n_golgi_golgi_convergence < 0:
            p.n_golgi_golgi_convergence = None

        t_tone_start = 0.0
        t_tone_end = t_tone_start + p.t_tone
        t_puff_start = t_tone_end + p.t_delay
        t_puff_end = t_puff_start + p.t_puff

        def puff_func(t):
            if t_puff_start < t % p.period < t_puff_end:
                return 1
            else:
                return 0

        def tone_func(t):
            if t_tone_start < t % p.period < t_tone_end:
                return 1
            else:
                return 0

        model = nengo.Network()
        with model:

            ###########################################################################
            # Setup the conditioned stimulus (i.e., a tone) and the unconditioned     #
            # stimulus (i.e., a puff)                                                 #
            ###########################################################################
            nd_tone = nengo.Node(tone_func)
            nd_puff = nengo.Node(puff_func)


            ###########################################################################
            # Setup the reflex generator and the eye-motor system                     #
            ###########################################################################

            # The reflex pathway is across the Trigeminal nucleus in the brainstem;
            # we don't model this in this particular model

            # Scaling factor that has to be applied to the reflex trajectory to scale
            # it to a range from 0 to 1
            reflex_scale = 1.0 / 25.0

            # The reflex system takes an input and produces the reflex trajectory on
            # the rising edge (convolves the differential of the input with the
            # trajectory)
            nd_reflex = nengo.Node(EyeblinkReflex()) # Unscaled output
            nd_reflex_out = nengo.Node(size_in=1)    # Normalised output
            nengo.Connection(nd_reflex[0], nd_reflex_out, transform=reflex_scale,
                             synapse=None)

            if not p.do_minimal:
                # The eyelid component represents the state of the eye in the world.
                # It receives two inputs, an agonist input (closing the eye, dim 0) and an
                # antagonist input (opening the eye, dim 1).
                nd_eyelid = nengo.Node(Eyelid())         # Unscaled input
                eyelid_in = nengo.Ensemble(n_neurons=100, dimensions=2)

                nengo.Connection(eyelid_in, nd_eyelid[0], transform=1.0 / reflex_scale,
                                 function=lambda x: max(x[0], x[1]),
                                 synapse=0.005)

                # Constantly open the eye a little bit
                nd_eye_bias = nengo.Node(p.eye_bias)
                nengo.Connection(nd_eye_bias, nd_eyelid[1])

                # We can't detect the puff if the eye is closed, multiply the output from
                # nd_puff with the amount the eye is opened. This is our unconditioned
                # stimulus
                # NOTE: Currently disabled by commenting out the line below
                c0, c1 = nengo.Node(size_in=1), nengo.Node(size_in=1) # Only for GUI
                nengo.Connection(nd_eyelid, c0, synapse=None)
                nengo.Connection(c0, c1, synapse=None)
            #    nengo.Connection(c1, nd_us[1], synapse=None)

            # Connect the unconditioned stimulus to the reflex generator
            nd_us = nengo.Node(lambda t, x: x[0] * (1 - x[1]), size_in=2, size_out=1)
            nengo.Connection(nd_puff, nd_us[0], synapse=None)
            nengo.Connection(nd_us, nd_reflex)
            if not p.do_minimal:
                nengo.Connection(nd_reflex_out, eyelid_in[0])

            ###########################################################################
            # Generate a neural representation of the conditioned stimulus            #
            ###########################################################################

            # Make sure that "bias_mode" is valid
            assert p.bias_mode in {
                None,
                "uniform_pcn_intercepts", "realistic_pcn_intercepts", "very_realistic_pcn_intercepts",
                "jbias_uniform_pcn_intercepts", "jbias_realistic_pcn_intercepts", "jbias_very_realistic_pcn_intercepts",
                "exc_jbias_uniform_pcn_intercepts", "exc_jbias_realistic_pcn_intercepts", "exc_jbias_very_realistic_pcn_intercepts",
                "inh_jbias_uniform_pcn_intercepts", "inh_jbias_realistic_pcn_intercepts", "inh_jbias_very_realistic_pcn_intercepts",
            }

            rng = np.random
            pcn_encs = rng.choice([-1, 1], p.n_pcn)
            if not p.bias_mode is None:
                if "very_realistic_pcn_intercepts" in p.bias_mode:
                    pcn_xis = rng.uniform(-0.35, 0.95, p.n_pcn)
                elif "realistic_pcn_intercepts" in p.bias_mode:
                    pcn_xis = rng.uniform(-0.15, 0.95, p.n_pcn)
                else:
                    pcn_xis = rng.uniform(-0.95, 0.95, p.n_pcn)

            pcn_max_rates = nengo.dists.Uniform(p.pcn_max_rates_lower, p.pcn_max_rates_upper)

            nd_cs = nengo.Node(size_in=1)
            ens_pcn = bio.Ensemble(n_neurons=p.n_pcn,
                                   dimensions=1,
                                   p_exc=1.0,
                                   label="ens_pcn",
                                   max_rates=pcn_max_rates,
                                   encoders=pcn_encs.reshape(-1, 1),
                                   intercepts=pcn_xis,)
            nengo.Connection(nd_tone, nd_cs, synapse=None)
            nengo.Connection(nd_cs, ens_pcn)

            ###########################################################################
            # Generate a LMU representation of the conditioned stimulus               #
            ###########################################################################

            bio_bias_mode = None
            if not p.bias_mode is None:
                if "exc_jbias_" in p.bias_mode:
                    bio_bias_mode = bio.ExcJBias
                elif "inh_jbias_" in p.bias_mode:
                    bio_bias_mode = bio.InhJBias
                elif "jbias_" in p.bias_mode:
                    bio_bias_mode = bio.JBias
                else:
                    bio_bias_mode = bio.Decode

            kwargs = 𐌈(
                mode=p.mode,
                use_spatial_constraints=p.use_spatial_constraints,
                n_pcn_golgi_convergence=p.n_pcn_golgi_convergence,
                n_pcn_granule_convergence=p.n_pcn_granule_convergence,
                n_granule_golgi_convergence=p.n_granule_golgi_convergence,
                n_golgi_granule_convergence=p.n_golgi_granule_convergence,
                n_golgi_golgi_convergence=p.n_golgi_golgi_convergence,
                n_granule=p.n_granule,
                n_golgi=p.n_golgi,
                qp_solver_extra_args={'max_iter': 200},
                tau=p.tau,
                q=p.q,
                theta=p.theta,
                golgi_intercepts=nengo.dists.CosineSimilarity(p.q+2) if p.use_cosine else nengo.dists.Uniform(-1,1),
                granule_intercepts=nengo.dists.CosineSimilarity(p.q+2) if p.use_cosine else nengo.dists.Uniform(-1,1),
                bias_mode_granule=bio_bias_mode,
            )
            net_granule_golgi = GranuleGolgiCircuit(
                ens_pcn,
                **kwargs
            )

            ###########################################################################
            # Learn the connection from the Granule cells to the Purkinje cells via   #
            # input from the Interior Olive                                           #
            ###########################################################################

            # This is the US pathway; the data is relayed from the Trigeminal nucleus
            # to the Interior Olive.

            ens_cn = nengo.Ensemble(n_neurons=100, dimensions=1)
            ens_prn = nengo.Ensemble(n_neurons=100, dimensions=1)
            ens_io = nengo.Ensemble(n_neurons=100, dimensions=1)
            ens_purkinje = nengo.Ensemble(n_neurons=100, dimensions=1)

            # Represent the error signal in ens_io
            nengo.Connection(nd_reflex_out[0], ens_io, transform=-1)
            nengo.Connection(ens_cn, ens_prn, transform=1, synapse=p.tau_error)
            nengo.Connection(ens_prn, ens_io, transform=1, synapse=p.tau_error)

            # Project from the granule onto the Purkinje cells
            c_learn = nengo.Connection(
                net_granule_golgi.ens_granule.neurons, ens_purkinje,
                transform=np.zeros((ens_purkinje.dimensions, net_granule_golgi.ens_granule.n_neurons)),
                learning_rule_type=nengo.learning_rules.PES(learning_rate=p.learning_rate, pre_synapse=p.tau_pre))
            nengo.Connection(ens_io, c_learn.learning_rule)

            ###########################################################################
            # Project from CN onto the motor system
            ###########################################################################

            nengo.Connection(ens_purkinje, ens_cn)
            if not p.do_minimal:
                nengo.Connection(ens_cn, eyelid_in[1])

            p_nd_reflex_out = nengo.Probe(nd_reflex_out, sample_every=p.sample_every)
            if not p.do_minimal:
                p_eyelid = nengo.Probe(nd_eyelid, sample_every=p.sample_every)
            p_purkinje = nengo.Probe(ens_purkinje, synapse=p.tau_purkinje, sample_every=p.sample_every)
            p_granule = nengo.Probe(net_granule_golgi.ens_granule, synapse=0.03, sample_every=p.sample_every)

        add_labels(model, locals=locals())

        sim = nengo.Simulator(model)
        with sim:
            sim.run(p.period*p.n_trials)

        dt = p.sample_every
        steps = int(p.period/dt)

        purk = sim.data[p_purkinje].reshape(-1, steps).T
        v = np.clip(purk[:,:],0,np.inf)*dt/reflex_scale
        pos = np.cumsum(v, axis=0)

        if plt:
            t = np.arange(steps)*dt

            ax1 = plt.subplot(4, 1, 1)
            ax1.set_ylabel('granule')
            ax2 = plt.subplot(4, 1, 2)
            ax2.set_ylabel('purkinje')
            ax3 = plt.subplot(4, 1, 3)
            ax3.set_ylabel('eye position\n(due to reflex)')
            ax4 = plt.subplot(4, 1, 4)
            ax4.set_ylabel('eye position\n at puff start')
            ax4.set_xlabel('trial')

            n_steps = len(sim.data[p_purkinje])
            cmap = matplotlib.cm.get_cmap("viridis")
            for i in range(0, n_steps, steps):
                color = cmap(i / n_steps)
                #if not p.do_minimal:
                #    ax1.plot(t, sim.data[p_eyelid][i:i+steps], label='eyelid %d'%(i//steps), ls='--')
                ax2.plot(t, sim.data[p_purkinje][i:i+steps], color=color)
                ax3.plot(t, np.cumsum(np.abs(sim.data[p_purkinje][i:i+steps]))*dt/reflex_scale, color=color)
            ax1.plot(t, sim.data[p_granule][:steps])
            ax2b = ax2.twinx()
            ax2b.plot(t, sim.data[p_nd_reflex_out][:steps], c='k', ls='--')
            ax4.plot(pos[int(t_puff_start/dt)])

        r = dict(final_pos=pos[int(t_puff_start/dt),-1],
                 pos_at_puff_start=pos[int(t_puff_start/dt)],
                 )
        if p.save_plot_data:
            r['purkinje']=sim.data[p_purkinje]
            r['granule']=sim.data[p_granule][:steps]
            r['reflex']=sim.data[p_nd_reflex_out][:steps]
        return r
