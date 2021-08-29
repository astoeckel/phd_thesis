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

import numpy as np
import nengo


def _make_delay_network(q, theta):
    Q = np.arange(q, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i - j + 1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B


def _discretize_lti(dt, A, B):
    import scipy.linalg
    Ad = scipy.linalg.expm(A * dt)
    Bd = np.dot(np.dot(np.linalg.inv(A), (Ad - np.eye(A.shape[0]))), B)
    return Ad, Bd


def _make_nef_lti(tau, A, B):
    AH = tau * A + np.eye(A.shape[0])
    BH = tau * B
    return AH, BH


def _make_control_lti(tau, q):
    # Do not alter the random state
    print("(1a)")
#    rs = np.random.get_state()
#    try:
    print("(1b)")
    rng = np.random.RandomState(np.random.randint(2 << 31))
    taus = np.logspace(-2, -0.5, q)

    print("(1c)")
    done = False
    while not done:
        try:
            C = np.eye(q) #rng.randn(q, q)
            print("(1c1)")
            L, V = np.linalg.eig(C)
            print("(1c2)")
            L = -5.0 * (np.abs(np.real(L)) + np.imag(L) * 1.0j)
            print("(1c3)")
            print("A", V)
            print("B", L)
#            import pdb; pdb.set_trace()
            print("B1", np.diag(L))
            print("C", V @ np.diag(L))
            print("D", np.linalg.inv(V))
            A = V @ np.diag(L) @ np.linalg.inv(V)
            print("(1c4)")
            B = (-1) ** np.arange(q)
            print("(1c5)")

            print("(1d)")
            S = np.diag(np.minimum(1.0, np.abs(1.0 / np.linalg.solve(-A, B))))
            A = S @ A @ np.linalg.inv(S)
            B = S @ B

            done = True
        except Exception as e:
            print("Error:", str(err))

    print("(1e)")
    AH, BH = _make_nef_lti(tau, A, B)
    BH = BH.reshape(-1, 1)
#    finally:
#        np.random.set_state(rs)

    return AH, BH


class Legendre(nengo.Process):
    def __init__(self, theta, q):
        self.q = q
        self.theta = theta
        self.A, self.B = _make_delay_network(self.q, self.theta)
        super().__init__(default_size_in=1, default_size_out=q)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros(self.q)
        Ad, Bd = _discretize_lti(dt, self.A, self.B)

        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x)
            return state

        return step_legendre


class GranuleGolgiCircuit(nengo.Network):
    def _kwargs_golgi(self):
        return {
            'n_neurons': self.n_golgi,
            'dimensions': self.q,
            'intercepts': self.golgi_intercepts,
            'max_rates': self.golgi_max_rates,
            'label': 'ens_golgi',
        }

    def _kwargs_granule(self):
        return {
            'n_neurons': self.n_granule,
            'dimensions': self.q,
            'intercepts': self.granule_intercepts,
            'max_rates': self.granule_max_rates,
            'label': 'ens_granule',
        }

    def _build_direct_mode_network(self):
        with self:
            # Create a direct implementation of the delay network
            self.nd_delay_network = nengo.Node(Legendre(self.theta, self.q))

            # Create a "ens_granule" ensemble as the intermediate layer we're
            # learning from
            kwargs_granule = self._kwargs_granule()
            self.ens_granule = nengo.Ensemble(**kwargs_granule)

            # Connect the network up
            nengo.Connection(self.ens_pcn, self.nd_delay_network)
            nengo.Connection(self.nd_delay_network, self.ens_granule)

    def _build_single_population_network(self):
        use_esn = self.mode == "echo_state"
        with self:
            # Create the "granule" ensemble
            kwargs_granule = self._kwargs_granule()
            self.ens_granule = nengo.Ensemble(**kwargs_granule)

            if self.use_control_lti:
                AH, BH = _make_control_lti(self.q, self.tau)
            elif not use_esn:
                # Compute the Delay Network coefficients
                AH, BH = _make_nef_lti(
                    self.tau, *_make_delay_network(q=self.q, theta=self.theta))
            else:
                # Do not alter the random state
                rs = np.random.get_state()
                try:
                    import scipy.sparse
                    # Adapted from "generate_internal_weights.m" in
                    # http://minds.jacobs-university.de/uploads/papers/freqGen.zip
                    A = scipy.sparse.random(self.q, self.q,
                                            min(5.0 / self.q, 1)).toarray()
                    A[A != 0] = A[A != 0] - 0.5
                    maxEigVal = maxVal = np.max(np.abs(np.linalg.eigvals(A)))
                    A = A / (maxEigVal)

                    B = np.random.uniform(-1, 1, (self.q, 1))

                    AH, BH = _make_nef_lti(self.tau, 25 * A, B)
                finally:
                    np.random.set_state(rs)

            # Build the recurrent connection
            nengo.Connection(self.ens_granule,
                             self.ens_granule,
                             transform=AH,
                             synapse=self.tau)

            # Build the input connection
            nengo.Connection(
                self.ens_pcn,
                self.ens_granule,
                transform=BH,
                synapse=self.tau,
            )

    def _build_two_population_network(self):
        # Decide whether or not nengo bio should be used
        use_nengo_bio = self.mode == "two_populations_dales_principle"
        if use_nengo_bio:
            import nengo_bio as bio
            Ensemble = bio.Ensemble
            Connection = bio.Connection
        else:
            Ensemble = nengo.Ensemble
            Connection = nengo.Connection

        with self:
            # Build the golgi cell ensemble
            kwargs_golgi = self._kwargs_golgi()
            if use_nengo_bio:
                kwargs_golgi['p_inh'] = 1.0
                if self.use_spatial_constraints:
                    kwargs_golgi['locations'] = bio.NeuralSheetDist()
            self.ens_golgi = Ensemble(**kwargs_golgi)

            # Build the granule cell ensemble
            kwargs_granule = self._kwargs_granule()
            if use_nengo_bio:
                kwargs_granule['p_exc'] = 1.0
                if self.use_spatial_constraints:
                    kwargs_granule['locations'] = bio.NeuralSheetDist()
            self.ens_granule = Ensemble(**kwargs_granule)

            # Compute the Delay Network coefficients
            print("(1)")
            if self.use_control_lti:
                AH, BH = _make_control_lti(self.tau, self.q)
            else:
                AH, BH = _make_nef_lti(
                    self.tau, *_make_delay_network(q=self.q, theta=self.theta))
            print("(2)")

            # Make the recurrent connections
            if use_nengo_bio:
                # Assemble the argments that are being passed to the solver
                kwargs_solver = {
                    'relax': self.solver_relax,
                    'reg': self.solver_reg,
                    'extra_args': {**{
                        'tol': self.solver_tol,
                        'renormalise': self.solver_renormalise,
                    }, **self.qp_solver_extra_args}
                }

                # Create the Lugaro cell ensemble if the corresponding flag is
                # set
                if self.use_lugaro:
                    self.ens_lugaro = Ensemble(
                        n_neurons=self.n_lugaro,
                        dimensions=1,
                        p_inh=1.0,
                        intercepts=nengo.dists.Uniform(-0.9, -0.1),
                        max_rates=nengo.dists.Uniform(50, 100)
                    )
                else:
                    self.ens_lugaro = None

                # Input, Granule, Golgi, (Lugaro) -> Golgi
                self.conn_pcn_gr_go_lg_to_go = Connection(
                    (self.ens_pcn, self.ens_granule,)
                        + ((self.ens_golgi,) if self.use_golgi_recurrence else ())
                        + ((self.ens_lugaro,) if self.use_lugaro else ()),
                    self.ens_golgi,
                    transform=np.concatenate((
                            BH.reshape(-1, 1),
                            AH,)
                        + ((np.zeros((AH.shape[0], AH.shape[0])),) if self.use_golgi_recurrence else ())
                        + ((np.zeros((AH.shape[0], 1)),) if self.use_lugaro else ()), axis=1),
                    connectivity={
                        (self.ens_pcn, self.ens_golgi): # PCN -> Golgi
                        bio.ConstrainedConnectivity(
                            convergence=self.n_pcn_golgi_convergence,
                        ),
                        (self.ens_granule, self.ens_golgi):
                        bio.SpatiallyConstrainedConnectivity( # Granule -> Golgi
                            convergence=self.n_granule_golgi_convergence,
                            sigma=self.spatial_sigma,
                        ),
                        (self.ens_golgi, self.ens_golgi): # Golgi -> Golgi
                        bio.SpatiallyConstrainedConnectivity(
                            convergence=self.n_golgi_golgi_convergence,
                            sigma=self.spatial_sigma,
                        ),
                        (self.ens_lugaro, self.ens_golgi): # Lugaro -> Golgi
                        bio.DefaultConnectivity(),
                    },
                    synapse_exc=self.tau,
                    synapse_inh=self.tau,
                    n_eval_points=self.n_eval_points,
                    solver=bio.solvers.QPSolver(**kwargs_solver),
                    bias_mode=self.bias_mode_golgi)

                # Input, Golgi -> Granule
                self.conn_pcn_go_to_gr = Connection(
                    (self.ens_pcn, self.ens_golgi),
                    self.ens_granule,
                    transform=np.concatenate((
                        BH.reshape(-1, 1),
                        AH,
                    ), axis=1),
                    connectivity={
                        (self.ens_pcn, self.ens_granule): # PCN -> Granule
                        bio.ConstrainedConnectivity(
                            convergence=self.n_pcn_granule_convergence,
                        ),
                        (self.ens_golgi, self.ens_granule): # Golgi -> Granule
                        bio.SpatiallyConstrainedConnectivity(
                            convergence=self.n_golgi_granule_convergence,
                            divergence=self.n_golgi_granule_divergence,
                            sigma=self.spatial_sigma,
                        ),
                    },
                    synapse_exc=self.tau,
                    synapse_inh=self.tau,
                    n_eval_points=self.n_eval_points,
                    solver=bio.solvers.QPSolver(**kwargs_solver),
                    bias_mode=self.bias_mode_granule)
            else:
                # Input -> Granule
                self.conn_pcn_to_gr = Connection(self.ens_pcn,
                           self.ens_granule,
                           transform=BH,
                           synapse=self.tau)

                # Input -> Golgi
                self.conn_pcn_to_go = Connection(self.ens_pcn,
                           self.ens_golgi,
                           transform=BH,
                           synapse=self.tau)

                # Granule -> Golgi
                self.conn_gr_to_go = Connection(self.ens_granule,
                           self.ens_golgi,
                           transform=AH,
                           synapse=self.tau)

                # Golgi -> Granule
                self.conn_go_to_gr = Connection(self.ens_golgi,
                           self.ens_granule,
                           transform=AH,
                           synapse=self.tau)

    def __init__(self,
                 ens_pcn,
                 n_golgi=20,
                 n_granule=200,
                 n_lugaro=20,
                 n_go_gr_total=None,
                 go_gr_ratio=None,
                 n_eval_points=1000,
                 n_pcn_golgi_convergence=None,
                 n_pcn_granule_convergence=None,
                 n_golgi_golgi_convergence=None,
                 n_golgi_granule_convergence=None,
                 n_golgi_granule_divergence=None,
                 n_granule_golgi_convergence=None,
                 spatial_sigma=0.25,
                 q=6,
                 theta=0.4,
                 tau=60e-3,
                 solver_relax=False,
                 solver_reg=1e-4,
                 solver_tol=1e-2,
                 solver_renormalise=False,
                 golgi_max_rates=None,
                 granule_max_rates=None,
                 golgi_intercepts=None,
                 granule_intercepts=None,
                 mode="two_populations",
                 use_lugaro=False,
                 use_golgi_recurrence=False,
                 use_spatial_constraints=False,
                 use_control_lti=False,
                 bias_mode_golgi=None,
                 bias_mode_granule=None,
                 qp_solver_extra_args={}):

        # Make sure the give mode is valid
        valid_modes = {
            "direct",
            "echo_state",
            "single_population",
            "two_populations",
            "two_populations_dales_principle",
        }
        if not mode in valid_modes:
            raise ValueError("\"mode\" must be one of {}".format(
                str(valid_modes)))

        # Convert neural ratios to absolute numbers
        assert (n_golgi is None) == (n_granule is None)
        assert (n_go_gr_total is None) == (go_gr_ratio is None)
        assert ((n_golgi is None) or (n_granule is None)) != \
               ((n_go_gr_total is None) or (go_gr_ratio is None))
        if not n_go_gr_total is None:
            n_golgi = int(n_go_gr_total * go_gr_ratio)
            n_granule = int(n_go_gr_total * (1.0 - go_gr_ratio))

        # Determine whether or not to use nengo bio
        use_nengo_bio = mode == "two_populations_dales_principle"

        # Use biologically plausible maximum rates
        if golgi_max_rates is None:
            golgi_max_rates = nengo.dists.Uniform(50, 100)
        if granule_max_rates is None:
            granule_max_rates = nengo.dists.Uniform(50, 100)

        # Use cosine similarity for intercepts
        if golgi_intercepts is None:
            golgi_intercepts = nengo.dists.CosineSimilarity(q + 2)
        if granule_intercepts is None:
            granule_intercepts = nengo.dists.CosineSimilarity(q + 2)

        # Copy the given parameters
        self.ens_pcn = ens_pcn
        self.n_golgi = n_golgi
        self.n_granule = n_granule
        self.n_lugaro = n_lugaro
        self.n_eval_points = n_eval_points
        self.n_pcn_golgi_convergence = n_pcn_golgi_convergence
        self.n_pcn_granule_convergence = n_pcn_granule_convergence
        self.n_golgi_golgi_convergence = n_golgi_golgi_convergence
        self.n_golgi_granule_convergence = n_golgi_granule_convergence
        self.n_golgi_granule_divergence = n_golgi_granule_divergence
        self.n_granule_golgi_convergence = n_granule_golgi_convergence
        self.spatial_sigma = spatial_sigma
        self.q = q
        self.theta = theta
        self.tau = tau
        self.solver_relax = solver_relax
        self.solver_reg = solver_reg
        self.solver_tol = solver_tol
        self.solver_renormalise = solver_renormalise
        self.golgi_max_rates = golgi_max_rates
        self.granule_max_rates = granule_max_rates
        self.golgi_intercepts = golgi_intercepts
        self.granule_intercepts = granule_intercepts
        self.mode = mode
        self.use_lugaro = use_lugaro
        self.use_golgi_recurrence = use_golgi_recurrence
        self.use_spatial_constraints = use_spatial_constraints
        self.use_control_lti = use_control_lti
        self.bias_mode_golgi = bias_mode_golgi
        self.bias_mode_granule = bias_mode_granule
        self.qp_solver_extra_args = qp_solver_extra_args

        # Call the inherited network constructor
        super().__init__(label="Granule/Golgi Layer")

        # Instantiate different circuits depending on the mode we're using
        if mode in {
                "direct",
        }:
            self._build_direct_mode_network()
        elif mode in {"echo_state", "single_population"}:
            self._build_single_population_network()
        elif mode in {"two_populations", "two_populations_dales_principle"}:
            self._build_two_population_network()
        else:
            assert False

