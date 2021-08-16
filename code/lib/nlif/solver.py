#!/usr/bin/env python3

#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2019-2021  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os, sys
import json
import signal
import copy

import numpy as np

import logging

logger = logging.getLogger(__name__)

###############################################################################
# C API Data Types                                                            #
###############################################################################

import ctypes
from ctypes import c_int, c_ubyte, c_double, c_char_p

c_double_p = ctypes.POINTER(ctypes.c_double)
c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)

#
# From common.h
#

NlifProgressCallback = ctypes.CFUNCTYPE(ctypes.c_ubyte, c_int, c_int)

NlifWarningCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, c_int)

#
# From nlif_solver.h
#


class NlifParameterProblem(ctypes.Structure):
    _fields_ = [
        ("n_compartments", c_int),
        ("n_inputs", c_int),
        ("n_samples", c_int),
        ("L", c_double_p),
        ("c", c_double_p),
        ("a_const", c_double_p),
        ("A", c_double_p),
        ("b_const", c_double_p),
        ("B", c_double_p),
        ("a_const_mask", c_ubyte_p),
        ("A_mask", c_ubyte_p),
        ("b_const_mask", c_ubyte_p),
        ("B_mask", c_ubyte_p),
        ("g_in", c_double_p),
        ("J_tar", c_double_p),
        ("alpha1", c_double),
        ("alpha2", c_double),
        ("alpha3", c_double),
        ("reg1", c_double),
        ("reg2", c_double),
        ("j_threshold", c_double),
        ("relax_subthreshold", c_ubyte),
        ("objective_val", c_double_p),
    ]


PNlifParameterProblem = ctypes.POINTER(NlifParameterProblem)


class NlifWeightProblem(ctypes.Structure):
    _fields_ = [
        ("n_pre", c_int),
        ("n_post", c_int),
        ("n_compartments", c_int),
        ("n_inputs", c_int),
        ("n_samples", c_int),
        ("L", c_double_p),
        ("c", c_double_p),
        ("a_const", c_double_p),
        ("A", c_double_p),
        ("b_const", c_double_p),
        ("B", c_double_p),
        ("A_in", c_double_p),
        ("J_tar", c_double_p),
        ("W", c_double_p),
        ("W_mask", c_ubyte_p),
        ("alpha1", c_double),
        ("alpha2", c_double),
        ("alpha3", c_double),
        ("reg1", c_double),
        ("reg2", c_double),
        ("j_threshold", c_double),
        ("relax_subthreshold", c_ubyte),
        ("objective_vals", c_double_p),
    ]


PNlifWeightProblem = ctypes.POINTER(NlifWeightProblem)


class NlifSolverParameters(ctypes.Structure):
    _fields_ = [
        ("use_sanathanan_koerner", c_ubyte),
        ("tolerance", c_double),
        ("max_iter", c_int),
        ("progress", NlifProgressCallback),
        ("warn", NlifWarningCallback),
        ("n_threads", c_int),
    ]


PNlifSolverParameters = ctypes.POINTER(NlifSolverParameters)

#
# From two_comp_solver.h
#


class TwoCompWeightProblem(ctypes.Structure):
    _fields_ = [
        ("n_pre", c_int),
        ("n_post", c_int),
        ("n_samples", c_int),
        ("a_pre", c_double_p),
        ("j_post", c_double_p),
        ("model_weights", c_double_p),
        ("connection_matrix_exc", c_ubyte_p),
        ("connection_matrix_inh", c_ubyte_p),
        ("regularisation", c_double),
        ("j_threshold", c_double),
        ("relax_subthreshold", c_ubyte),
        ("non_negative", c_ubyte),
        ("synaptic_weights_exc", c_double_p),
        ("synaptic_weights_inh", c_double_p),
        ("objective_vals", c_double_p),
    ]


PTwoCompWeightProblem = ctypes.POINTER(TwoCompWeightProblem)


class TwoCompSolverParameters(ctypes.Structure):
    _fields_ = [
        ("renormalise", ctypes.c_ubyte),
        ("tolerance", c_double),
        ("max_iter", c_int),
        ("progress", NlifProgressCallback),
        ("warn", NlifWarningCallback),
        ("n_threads", c_int),
    ]


PTwoCompSolverParameters = ctypes.POINTER(TwoCompSolverParameters)

###############################################################################
# PYTHON WRAPPER CLASS                                                        #
###############################################################################


def default_progress_callback(n_done, n_total):
    sys.stderr.write("\rSolved {}/{} neuron weights".format(n_done, n_total))
    sys.stderr.flush()
    return True


def default_warning_callback(msg, idx):
    print("WARN: " + str(msg, "utf-8"))


class SigIntHandler:
    """
    Class responsible for canceling the neuron weight solving process whenever
    the SIGINT event is received.
    """
    def __init__(self):
        self._old_handler = None
        self._args = None
        self.triggered = False

    def __enter__(self):
        def handler(*args):
            self._args = args
            self.triggered = True

        try:
            self._old_handler = signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Ignore errors -- this is most likely triggered when
            # signal.signal is not called from the main thread. This is e.g.
            # what happens in Nengo GUI.
            pass
        return self

    def __exit__(self, type, value, traceback):
        if self._old_handler:
            signal.signal(signal.SIGINT, self._old_handler)
            if self.triggered:
                self._old_handler(*self._args)
            self._old_handler = None


class Solver:
    @staticmethod
    def _cpp_dir(*f):
        return os.path.join(os.path.realpath(os.path.dirname(__file__)), 'cpp',
                            *f)

    @staticmethod
    def _compile_library(debug=False, parallel_compile=True):
        from .internal import cmodule

        # List all dependencies
        D = lambda *s: Solver._cpp_dir(*s)
        deps = [
            D("solver", "common.cpp"),
            D("solver", "two_comp_solver.cpp"),
            D("solver", "nlif_solver_parameters.cpp"),
            D("solver", "nlif_solver_weights.cpp"),
            D("solver", "qp.cpp"),
            D("solver", "threadpool.cpp"),
            D("extern", "osqp", "amd_1.c"),
            D("extern", "osqp", "amd_2.c"),
            D("extern", "osqp", "amd_aat.c"),
            D("extern", "osqp", "amd_control.c"),
            D("extern", "osqp", "amd_defaults.c"),
            D("extern", "osqp", "amd_info.c"),
            D("extern", "osqp", "amd_order.c"),
            D("extern", "osqp", "amd_postorder.c"),
            D("extern", "osqp", "amd_post_tree.c"),
            D("extern", "osqp", "amd_preprocess.c"),
            D("extern", "osqp", "amd_valid.c"),
            D("extern", "osqp", "auxil.c"),
            D("extern", "osqp", "cs.c"),
            D("extern", "osqp", "error.c"),
            D("extern", "osqp", "kkt.c"),
            D("extern", "osqp", "lin_alg.c"),
            D("extern", "osqp", "lin_sys.c"),
            D("extern", "osqp", "osqp.c"),
            D("extern", "osqp", "polish.c"),
            D("extern", "osqp", "proj.c"),
            D("extern", "osqp", "qdldl.c"),
            D("extern", "osqp", "qdldl_interface.c"),
            D("extern", "osqp", "scaling.c"),
            D("extern", "osqp", "SuiteSparse_config.c"),
            D("extern", "osqp", "util.c"),
        ]

        # Compile the library
        soname = cmodule.compile_cpp_library(
            deps=deps,
            include_dirs=[D("extern", "eigen"),
                          D("extern"), D()],
            debug=debug,
            parallel=parallel_compile,
            name="solver")

        # Load the shared object and return the handle
        logger.info(f"Loading solver library from {soname}...")
        lib = cmodule.SharedLibrary(soname)
        lib.nlif_strerr.restype = c_char_p
        return lib

    def __init__(self, debug=False, parallel_compile=True):
        self._dll = Solver._compile_library(debug=debug,
                                            parallel_compile=parallel_compile)

    @staticmethod
    def _to_c_bool_mat(m, copy=False):
        return np.clip(0, 1, m.astype(dtype=np.uint8, order='C', copy=copy))

    def nlif_solve_parameters_iter(self,
                                   reduced_system,
                                   gs,
                                   Js,
                                   alpha1=1.0,
                                   alpha2=1.0,
                                   alpha3=1.0,
                                   reg1=1e-9,
                                   reg2=1e-3,
                                   J_th=None,
                                   tol=1e-6,
                                   return_objective_val=False,
                                   use_sanathanan_koerner=False,
                                   progress_callback=default_progress_callback,
                                   warning_callback=default_warning_callback,
                                   n_threads=0,
                                   max_iter=0):
        # Copy the given system
        sys = copy.deepcopy(reduced_system)

        # Check the input parameters
        gs, Js = np.asarray(gs), np.asarray(Js)
        assert gs.ndim == 2
        assert Js.ndim == 1
        assert gs.shape[0] == Js.shape[0]
        assert gs.shape[1] == sys.n_inputs

        # Fetch some handy aliases for parameters
        n, k, N = sys.n_compartments, sys.n_inputs, gs.shape[0]

        # Make sure gs and Js have the right format
        c_gs = gs.astype(dtype=np.float64, order='C', copy=False)
        assert c_gs.shape == (N, k)
        c_Js = Js.astype(dtype=np.float64, order='C', copy=False)
        assert c_Js.shape == (N, )

        # Copy the matrices from the given system and check their shape
        c_L = sys.L.astype(dtype=np.float64, order='C', copy=False)
        assert c_L.shape == (n, n)
        c_c = sys.c.astype(dtype=np.float64, order='C', copy=False)
        assert c_c.shape == (n, )

        c_a_const = sys.a_const.astype(dtype=np.float64, order='C', copy=True)
        assert c_a_const.shape == (n, )
        c_A = sys.A.astype(dtype=np.float64, order='C', copy=True)
        assert c_A.shape == (n, k)
        c_b_const = sys.b_const.astype(dtype=np.float64, order='C', copy=True)
        assert c_b_const.shape == (n, )
        c_B = sys.B.astype(dtype=np.float64, order='C', copy=True)
        assert c_B.shape == (n, k)

        c_a_const_mask = Solver._to_c_bool_mat(sys.a_const_mask)
        assert c_a_const_mask.shape == (n, )
        c_A_mask = Solver._to_c_bool_mat(sys.A_mask)
        assert c_A_mask.shape == (n, k)
        c_b_const_mask = Solver._to_c_bool_mat(sys.b_const_mask)
        assert c_b_const_mask.shape == (n, )
        c_B_mask = Solver._to_c_bool_mat(sys.B_mask)
        assert c_B_mask.shape == (n, k)

        # Reserve memory for the objective value
        if return_objective_val:
            c_objective_val = np.zeros(tuple(), dtype=np.float64, order='C')

        # Matrix conversion helper functions
        def PDouble(mat):
            return mat.ctypes.data_as(c_double_p)

        def PBool(mat):
            return mat.ctypes.data_as(c_ubyte_p)

        # Assemble the parameter problem
        problem = NlifParameterProblem()
        problem.n_compartments = n
        problem.n_inputs = k
        problem.n_samples = N

        problem.L = PDouble(c_L)
        problem.c = PDouble(c_c)

        problem.a_const = PDouble(c_a_const)
        problem.A = PDouble(c_A)
        problem.b_const = PDouble(c_b_const)
        problem.B = PDouble(c_B)

        problem.a_const_mask = PBool(c_a_const_mask)
        problem.A_mask = PBool(c_A_mask)
        problem.b_const_mask = PBool(c_b_const_mask)
        problem.B_mask = PBool(c_B_mask)

        problem.g_in = PDouble(c_gs)
        problem.J_tar = PDouble(c_Js)

        problem.alpha1 = alpha1
        problem.alpha2 = alpha2
        problem.alpha3 = alpha3
        problem.reg1 = reg1
        problem.reg2 = reg2

        problem.j_threshold = -np.inf if J_th is None else J_th
        problem.relax_subthreshold = not (J_th is None)

        if return_objective_val:
            problem.objective_val = PDouble(c_objective_val)

        # Progress wrapper
        with SigIntHandler() as sig_int_handler:

            def progress_callback_wrapper(n_done, n_total):
                if sig_int_handler.triggered:
                    return False
                if progress_callback:
                    return progress_callback(n_done, n_total)
                return True

            params = NlifSolverParameters()
            params.use_sanathanan_koerner = use_sanathanan_koerner
            params.tolerance = tol
            params.max_iter = max_iter
            params.progress = NlifProgressCallback(progress_callback_wrapper)
            params.warn = NlifWarningCallback(
                0 if warning_callback is None else warning_callback)
            params.n_threads = n_threads

            # Actually run the solver
            err = self._dll.nlif_solve_parameters_iter(ctypes.pointer(problem),
                                                       ctypes.pointer(params))
            if err != 0:
                raise RuntimeError(self._dll.nlif_strerr(err))

        # Write the updated matrices to the system
        sys.a_const[...] = c_a_const
        sys.A[...] = c_A
        sys.b_const[...] = c_b_const
        sys.B[...] = c_B

        # Return the updated system
        return sys

    ITER = [0]

    def nlif_solve_weights_iter(self,
                                reduced_system,
                                As,
                                Js,
                                W,
                                W_mask=None,
                                alpha1=1.0,
                                alpha2=1.0,
                                alpha3=1e-3,
                                reg1=1e-3,
                                reg2=1e-6,
                                J_th=None,
                                tol=1e-6,
                                return_objective_vals=False,
                                use_sanathanan_koerner=False,
                                progress_callback=default_progress_callback,
                                warning_callback=default_warning_callback,
                                n_threads=0,
                                max_iter=0):
        # Make some handy aliases
        sys = reduced_system
        As, Js = np.asarray(As), np.asarray(Js)
        assert As.ndim == 2
        assert Js.ndim == 2
        assert As.shape[0] == Js.shape[1]
        N = N_samples = As.shape[0]
        m = N_pre = As.shape[1]
        M = N_post = Js.shape[0]
        k = N_input = sys.n_inputs
        n = sys.n_compartments

        # Use an all-to-all connection if no mask is given
        if W_mask is None:
            W_mask = np.ones((N_post, N_input, N_pre),
                             order='C',
                             dtype=np.uint8)

        # Check some more dimensions
        assert W.ndim == W_mask.ndim == 3
        assert W.shape == W_mask.shape
        assert W.shape == (N_post, N_input, N_pre)

        # Make sure As and Js have the right format
        c_As = As.astype(dtype=np.float64, order='C', copy=False)
        assert c_As.shape == (N, m)
        c_Js = Js.astype(dtype=np.float64, order='C', copy=False)
        assert c_Js.shape == (M, N)

        # Copy the matrices from the given system and check their shape
        c_L = sys.L.astype(dtype=np.float64, order='C', copy=False)
        assert c_L.shape == (n, n)
        c_c = sys.c.astype(dtype=np.float64, order='C', copy=False)
        assert c_c.shape == (n, )

        c_a_const = sys.a_const.astype(dtype=np.float64, order='C', copy=False)
        assert c_a_const.shape == (n, )
        c_A = sys.A.astype(dtype=np.float64, order='C', copy=False)
        assert c_A.shape == (n, k)
        c_b_const = sys.b_const.astype(dtype=np.float64, order='C', copy=False)
        assert c_b_const.shape == (n, )
        c_B = sys.B.astype(dtype=np.float64, order='C', copy=False)
        assert c_B.shape == (n, k)

        c_W = W.astype(dtype=np.float64, order='C', copy=True)
        assert c_W.shape == (M, k, m)

        c_W_mask = Solver._to_c_bool_mat(W_mask)
        assert c_W_mask.shape == (M, k, m)

        # Reserve memory for the objective value
        if return_objective_vals:
            c_objective_vals = np.zeros(n_post, dtype=np.float64, order='C')

        # Matrix conversion helper functions
        def PDouble(mat):
            return mat.ctypes.data_as(c_double_p)

        def PBool(mat):
            return mat.ctypes.data_as(c_ubyte_p)

        # Assemble the parameter problem
        problem = NlifWeightProblem()
        problem.n_pre = m
        problem.n_post = M
        problem.n_compartments = n
        problem.n_inputs = k
        problem.n_samples = N

        problem.L = PDouble(c_L)
        problem.c = PDouble(c_c)

        problem.a_const = PDouble(c_a_const)
        problem.A = PDouble(c_A)
        problem.b_const = PDouble(c_b_const)
        problem.B = PDouble(c_B)

        problem.A_in = PDouble(c_As)
        problem.J_tar = PDouble(c_Js)

        problem.W = PDouble(c_W)
        problem.W_mask = PBool(c_W_mask)

        problem.alpha1 = alpha1
        problem.alpha2 = alpha2
        problem.alpha3 = alpha3
        problem.reg1 = reg1
        problem.reg2 = reg2

        problem.j_threshold = -np.inf if J_th is None else J_th
        problem.relax_subthreshold = not (J_th is None)

        if return_objective_vals:
            problem.objective_vals = PDouble(c_objective_val)

        # Progress wrapper
        with SigIntHandler() as sig_int_handler:

            def progress_callback_wrapper(n_done, n_total):
                if sig_int_handler.triggered:
                    return False
                if progress_callback:
                    return progress_callback(n_done, n_total)
                return True

            params = NlifSolverParameters()
            params.use_sanathanan_koerner = use_sanathanan_koerner
            params.tolerance = tol
            params.max_iter = max_iter
            params.progress = NlifProgressCallback(progress_callback_wrapper)
            params.warn = NlifWarningCallback(
                0 if warning_callback is None else warning_callback)
            params.n_threads = n_threads

            # Actually run the solver
            err = self._dll.nlif_solve_weights_iter(ctypes.pointer(problem),
                                                    ctypes.pointer(params))
            if err != 0:
                raise RuntimeError(self._dll.nlif_strerr(err))

        # Return the updated weight matrix
        return c_W

    def two_comp_solve(self,
                       Apre,
                       Jpost,
                       ws,
                       connection_matrix=None,
                       iTh=None,
                       nonneg=True,
                       renormalise=True,
                       tol=None,
                       reg=None,
                       use_lstsq=False,
                       return_objective_vals=False,
                       progress_callback=default_progress_callback,
                       warning_callback=default_warning_callback,
                       n_threads=0,
                       max_iter=0):
        """
        Solves a synaptic weight qp problem.


        Parameters
        ==========

        Apre:
            Pre-synaptic activities. Must be a n_samples x n_pre matrix.
        Jpost:
            Desired post-synaptic currents. Must be a n_samples x n_post matrix.
        ws:
            Neuron model parameters of the form
                        b0 + b1 * gExc + b2 * gInh
                Jequiv = --------------------------
                        a0 + a1 * gExc + a2 * gInh
            where ws = [b0, b1, b2, a0, a1, a2]. Use [0, 1, -1, 1, 0, 0] for a
            standard current-based LIF neuron.
        connection_matrix:
            A binary 2 x n_pre x n_post matrix determining which neurons are
            excitatorily and which are inhibitorily connected.
        iTh:
            Threshold current. The optimization problem is relaxed for currents
            below the given value. If "None", relaxation is deactivated.
        nonneg:
            Whether synaptic weights should be nonnegative
        renormalise:
            Whether the problem should be renormalised. Only set this to true if the
            target currents are in plausible biological scales (i.e. currents in pA
            to nA).
        tol:
            Solver tolerance. Default is 1e-6
        reg:
            Regularisation. Default is 1e-1
        use_lstsq:
            Ignored. For compatibility with the nengo_bio internal solver.
        return_objective_vals:
            If true, returns the achieved objective values.
        progress_callback:
            Function that is regularly being called with the current progress. May
            return "False" to cancel the solving process, must return "True"
            otherwise. Set to "None" to use no progress callback.
        warning_callback:
            Function that is being called whenever a warning is triggered. Set to
            "None" to use no progress callback.
        n_threads:
            Maximum number of threads to use when solving for weights. Uses all
            available CPU cores if set to zero.
        max_iter:
            Maximum number of iterations to take. The default (zero) means that no
            such limit exists.


        Returns
        =======

        Two matrices, WE, WI containing the weights for all post-neurons. If
        "return_objective_vals" is set to true, a third return value is returned,
        a vector of all objective values for the post-neurons.
        """

        # Set some default values
        if tol is None:
            tol = 1e-6
        if reg is None:
            reg = 1e-1

        # Fetch some counts
        assert Apre.shape[0] == Jpost.shape[0]
        Nsamples = Apre.shape[0]
        Npre = Apre.shape[1]
        Npost = Jpost.shape[1]

        # Use an all-to-all connection if connection_matrix is set to None
        if connection_matrix is None:
            connection_matrix = np.ones((2, Npre, Npost), dtype=np.bool)

        # Create a neuron model parameter vector for each neuron, if the parameters
        # are not already in this format
        assert ws.size == 6 or ws.ndim == 2, "Model weight vector must either be 6-element one-dimensional or a 2D matrix"
        if (ws.size == 6):
            ws = np.repeat(ws.reshape(1, -1), Npost, axis=0)
        else:
            assert ws.shape[0] == Npost and ws.shape[1] == 6, \
                "Invalid model weight matrix shape"

        # Make sure the connection matrix has the correct size
        assert connection_matrix.shape[0] == 2
        assert connection_matrix.shape[1] == Npre
        assert connection_matrix.shape[2] == Npost

        # Make sure all matrices are in the correct format
        c_a_pre = Apre.astype(dtype=np.float64, order='C', copy=False)
        c_j_post = Jpost.astype(dtype=np.float64, order='C', copy=False)
        c_connection_matrix_exc = Solver._to_c_bool_mat(connection_matrix[0])
        c_connection_matrix_inh = Solver._to_c_bool_mat(connection_matrix[1])
        c_model_weights = ws.astype(dtype=np.float64, order='C', copy=False)
        c_we = np.zeros((Npre, Npost), dtype=np.float, order='C')
        c_wi = np.zeros((Npre, Npost), dtype=np.float, order='C')
        if return_objective_vals:
            c_objective_vals = np.zeros((Npost, ), dtype=np.float, order='C')

        # Matrix conversion helper functions
        def PDouble(mat):
            return mat.ctypes.data_as(c_double_p)

        def PBool(mat):
            return mat.ctypes.data_as(c_ubyte_p)

        # Assemble the weight solver problem
        problem = TwoCompWeightProblem()
        problem.n_pre = Npre
        problem.n_post = Npost
        problem.n_samples = Nsamples
        problem.a_pre = PDouble(c_a_pre)
        problem.j_post = PDouble(c_j_post)
        problem.model_weights = PDouble(c_model_weights)
        problem.connection_matrix_exc = PBool(c_connection_matrix_exc)
        problem.connection_matrix_inh = PBool(c_connection_matrix_inh)
        problem.regularisation = reg
        problem.j_threshold = 0.0 if iTh is None else iTh
        problem.relax_subthreshold = not (iTh is None)
        problem.non_negative = nonneg
        problem.synaptic_weights_exc = PDouble(c_we)
        problem.synaptic_weights_inh = PDouble(c_wi)
        problem.objective_vals = PDouble(
            c_objective_vals) if return_objective_vals else None

        # Progress wrapper
        with SigIntHandler() as sig_int_handler:

            def progress_callback_wrapper(n_done, n_total):
                if sig_int_handler.triggered:
                    return False
                if progress_callback:
                    return progress_callback(n_done, n_total)
                return True

            params = TwoCompSolverParameters()
            params.renormalise = renormalise
            params.tolerance = tol
            params.max_iter = max_iter
            params.progress = NlifProgressCallback(progress_callback_wrapper)
            params.warn = NlifWarningCallback(
                0 if warning_callback is None else warning_callback)
            params.n_threads = n_threads

            # Actually run the solver
            err = self._dll.two_comp_solve(ctypes.pointer(problem),
                                           ctypes.pointer(params))

        if err != 0:
            raise RuntimeError(self._dll.nlif_strerr(err))

        if return_objective_vals:
            return c_we, c_wi, c_objective_vals
        return c_we, c_wi


###############################################################################
# MAIN PROGRAM -- RUNS THE ABOVE CODE                                         #
###############################################################################

if __name__ == "__main__":
    ###########################################################################
    # Imports                                                                 #
    ###########################################################################

    # Enable logging for the compiler
    logging.basicConfig(level="INFO")

    # Import some code used in the test classes
    from .tests.utils.nef import Ensemble

    # Used for measuring the ellapsed time
    import time

    # Load the solver
    solve = Solver().two_comp_solve

    # Well-tested reference implementation from nengo_bio
    try:
        from nengo_bio.internal.qp_solver import solve as solve_ref
    except ImportError:
        solve_ref = None

    # Plot the results if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    ###########################################################################
    # Actual test code                                                        #
    ###########################################################################

    def compute_error(J_tar, J_dec, i_th):
        if i_th is None:
            e_invalid = 0
            e_valid = np.sum(np.square(J_tar - J_dec))
        else:
            valid = J_tar > i_th
            invalid_violated = np.logical_and(J_tar < i_th, J_dec > i_th)
            e_invalid = np.sum(np.square(i_th - J_dec[invalid_violated]))
            e_valid = np.sum(np.square(J_tar[valid] - J_dec[valid]))

        return np.sqrt((e_valid + e_invalid) / J_tar.size)

    def E(Apre, Jpost, WE, WI, iTh):
        return compute_error(Jpost.T, Apre.T @ WE - Apre.T @ WI, iTh)

    np.random.seed(34812)

    ens1 = Ensemble(101, 1)
    ens2 = Ensemble(102, 1)

    xs = np.linspace(-1, 1, 100).reshape(1, -1)
    Apre = ens1(xs)
    Jpost = ens2.J(xs)

    ws = np.array([0, 1, -1, 1, 0, 0], dtype=np.float64)

    kwargs = {
        "Apre": Apre.T,
        "Jpost": Jpost.T,
        "ws": ws,
        "iTh": 0.0,
        "tol": 1e-3,
        "reg": 1e-2,
        "renormalise": True,
    }

    print("Solving weights using libbioneuronqp...")
    t1 = time.perf_counter()
    WE, WI = solve(**kwargs)
    sys.stderr.write('\n')
    t2 = time.perf_counter()
    print("Time : ", t2 - t1)
    print("Error: ", E(Apre, Jpost, WE, WI, kwargs['iTh']))
    print()

    if not solve_ref is None:
        print("Solving weights using nengobio.qp_solver (cvxopt)")
        t1 = time.perf_counter()
        WE_ref, WI_ref = solve_ref(**kwargs)
        t2 = time.perf_counter()
        print("Time : ", t2 - t1)
        print("Error: ", E(Apre, Jpost, WE_ref, WI_ref, kwargs['iTh']))
        print()

    ###########################################################################
    # Plotting                                                                #
    ###########################################################################

    if not plt is None:
        if solve_ref is None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 18))

        ax1.plot(xs[0], Apre.T)
        ax1.set_xlabel("Stimulus $x$")
        ax1.set_ylabel("Response rate")
        ax1.set_title("Ens 1 Tuning curves")

        ax2.plot(xs[0], Jpost.T)
        ax2.set_ylim(-1, 4)
        ax2.set_xlabel("Stimulus $x$")
        ax2.set_ylabel("Input current $J$")
        ax2.set_title("Ens 2 Desired Input Currents")

        ax3.plot(Apre.T @ WE - Apre.T @ WI)
        ax3.set_ylim(-1, 4)
        ax3.set_xlabel("Stimulus $x$")
        ax3.set_ylabel("Input current $J$")
        ax3.set_title("Ens 2 Desired Input Currents (libbioneuronqp)")

        if not solve_ref is None:
            ax4.plot(Apre.T @ WE_ref - Apre.T @ WI_ref)
            ax4.set_ylim(-1, 4)
            ax4.set_xlabel("Stimulus $x$")
            ax4.set_ylabel("Input current $J$")
            ax4.set_title("Ens 2 Desired Input Currents (nengo_bio.qp_solver)")

        plt.tight_layout()
        plt.show()

