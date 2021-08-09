/*
 *  libnlif -- Multi-compartment LIF simulator and weight solver
 *  Copyright (C) 2017-2021  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @file nlif_solver.h
 *
 * C API for the trust-region based n-LIF solver.
 *
 * @author Andreas Stöckel
 */

#ifndef NLIF_NLIF_SOLVER_H
#define NLIF_NLIF_SOLVER_H

#include "common.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The NlifParameterProblem structure describes the optimization problem for
 * the Nlif parameters.
 */
typedef struct {
	/**
	 * Number of compartments.
	 */
	int n_compartments;

	/**
	 * Number of inputs in the neuron.
	 */
	int n_inputs;

	/**
	 * Number of samples in the optimisation problem.
	 */
	int n_samples;

	/**
	 * Graph Laplacian describing the neuron's internal connectivity.
	 */
	double *L;

	/**
	 * Vector of length "n_compartments" describing how the individual
	 * compartments are coupled to the soma.
	 */
	double *c;

	/**
	 * Vector of length "n_compartments" of input-independent conductance
	 * values.
	 */
	double *a_const;

	/**
	 * Matrix of size "n_compartments x n_inputs" of input-dependent
	 * conductance values.
	 */
	double *A;

	/**
	 * Vector of length "n_compartments" of input-independent bias currents.
	 */
	double *b_const;

	/**
	 * Matrix of size "n_compartments x n_inputs" of input-dependent
	 * currents.
	 */
	double *B;

	/**
	 * Mask for a_const determining which parameters should be optimised.
	 */
	NlifBool *a_const_mask;

	/**
	 * Mask for A determining which parameters should be optimised.
	 */
	NlifBool *A_mask;

	/**
	 * Mask for b_const determining which parameters should be optimised.
	 */
	NlifBool *b_const_mask;

	/**
	 * Mask for B determining which parameters should be optimised.
	 */
	NlifBool *B_mask;

	/**
	 * Inputs. Must be a n_samples x n_inputs matrix.
	 */
	double *g_in;

	/**
	 * Desired target currents. Must be a n_samples matrix.
	 */
	double *J_tar;

	/**
	 * Weighting factors for the individual sub-problems.
	 *
	 * alpha1 : Weighting of the <v_eq, c> - J problem
	 * alpha2 : Weighting of the A v_eq = b problem
	 * alpha3 : Size of the trust-region.
	 */
	double alpha1, alpha2, alpha3;

	/**
	 * L2 regularisation factor for the parameters and the voltages. The
	 * parameters should only be very loosely regularised.
	 */
	double reg1, reg2;

	/**
	 * Subthreshold current, only relevant if relax_subthreshold is True.
	 */
	double j_threshold;

	/**
	 * If true, enables subthreshold relaxation.
	 */
	bool relax_subthreshold;

	/**
	 * Pointer at a storage location for the objective value.
	 */
	double *objective_val;
} NlifParameterProblem;

/**
 * The NlifParameterProblem structure describes the optimization problem for
 * the n-LIF parameters.
 */
typedef struct {
	/**
	 * Number of pre-neurons.
	 */
	int n_pre;

	/**
	 * Number of post-neurons.
	 */
	int n_post;

	/**
	 * Number of compartments.
	 */
	int n_compartments;

	/**
	 * Number of inputs in the neuron.
	 */
	int n_inputs;

	/**
	 * Number of samples in the optimisation problem.
	 */
	int n_samples;

	/**
	 * Graph Laplacian describing the neuron's internal connectivity.
	 */
	double *L;

	/**
	 * Vector of length "n_compartments" describing how the individual
	 * compartments are coupled to the soma.
	 */
	double *c;

	/**
	 * Vector of length "n_compartments" of input-independent conductance
	 * values.
	 */
	double *a_const;

	/**
	 * Matrix of size "n_compartments x n_inputs" of input-dependent
	 * conductance values.
	 */
	double *A;

	/**
	 * Vector of length "n_compartments" of input-independent bias currents.
	 */
	double *b_const;

	/**
	 * Matrix of size "n_compartments x n_inputs" of input-dependent
	 * currents.
	 */
	double *B;

	/**
	 * Inputs. Must be a n_samples x n_pre matrix.
	 */
	double *A_in;

	/**
	 * Desired target currents. Must be a n_post x n_samples matrix.
	 */
	double *J_tar;

	/**
	 * Initial weights. Must be a n_post x n_input x n_pre tensor.
	 */
	double *W;

	/**
	 * Connectivity matrix. Must be a n_post x n_input x n_pre tensor.
	 */
	NlifBool *W_mask;

	/**
	 * Weighting factors for the individual sub-problems.
	 *
	 * alpha1 : Weighting of the <v_eq, c> - J problem
	 * alpha2 : Weighting of the A v_eq = b problem
	 * alpha3 : Size of the trust-region.
	 */
	double alpha1, alpha2, alpha3;

	/**
	 * L2 regularisation factor for the parameters and the voltages. The
	 * parameters should only be very loosely regularised.
	 */
	double reg1, reg2;

	/**
	 * Subthreshold current, only relevant if relax_subthreshold is True.
	 */
	double j_threshold;

	/**
	 * If true, enables subthreshold relaxation.
	 */
	bool relax_subthreshold;

	/**
	 * Pointer at a storage location for the objective values.
	 */
	double *objective_vals;
} NlifWeightProblem;

/**
 * Parameters used when solving for weights.
 */
typedef struct {
	/**
	 * If true, uses the Sananthan-Koerner iteration to re-weight the individual
	 * samples.
	 */
	bool use_sanathanan_koerner;

	/**
	 * Sets the absolute and relative abortion criteria (eps_rel and eps_abs
	 * in osqp).
	 */
	double tolerance;

	/**
	 * Maximum number of iterations. Zero disables any limitations.
	 */
	int max_iter;

	/**
	 * Callback being called to indicate progress. May be NULL if no such
	 * callback function should be called. This function is called in
	 * approximately 100ms intervals and may return "false" if the operation
	 * is to be cancelled. Otherwise, the callback should return "true".
	 * */
	NlifProgressCallback progress;

	/**
	 * Callback called to issue warning messages.
	 */
	NlifWarningCallback warn;

	/**
	 * Number of threads to use. A value smaller or equal to zero indicates that
	 * all available processor cores should be used.
	 */
	int n_threads;
} NlifSolverParameters;

/**
 * Performs a single iteration of the trust-region algorithm for solving for
 * weights. Updates the given matrices in-place.
 */
DLL_PUBLIC NlifError nlif_solve_parameters_iter(NlifParameterProblem *problem,
                                                NlifSolverParameters *params);

/**
 * Performs a single iteration of the trust-region algorithm for solving for
 * weights. Updates the given matrices in-place.
 */
DLL_PUBLIC NlifError nlif_solve_weights_iter(NlifWeightProblem *problem,
                                             NlifSolverParameters *params);

#ifdef __cplusplus
}
#endif

#endif /* NLIF_NLIF_SOLVER_H */
