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
 * @file two_comp_solver.h
 *
 * Main header for the two-compartment LIF solver
 *
 * @author Andreas Stöckel
 */

#ifndef NLIF_TWO_COMP_SOLVER_H
#define NLIF_TWO_COMP_SOLVER_H

#include "common.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The TwoCompWeightProblem structure describes the optimization problem for
 * computing the weights between two neuron populations.
 */
typedef struct {
	/**
	 * Number of neurons in the pre-population.
	 */
	int n_pre;

	/**
	 * Number of neurons in the post-population.
	 */
	int n_post;

	/**
	 * Number of samples.
	 */
	int n_samples;

	/**
	 * Pre-population activites for each sample as a n_samples x n_pre matrix.
	 * Should be in row-major order (i.e. the activtivities for all neurons for
	 * one sample are stored consecutively in memory).
	 */
	double *a_pre;

	/**
	 * Desired input currents for each sample as a n_samples x n_post matrix.
	 * Should be in row-major order (i.e. the desired input currents for all
	 * neurons for one sample are stored consecutively in memory).
	 */
	double *j_post;

	/**
	 * Neuron model weights as a n_post x 6 matrix (should be irow-major
	 * format).
	 */
	double *model_weights;

	/**
	 * Matrix determining the connectivity between pre- and post-neurons. This
	 * is a n_pre x n_post matrix.
	 */
	NlifBool *connection_matrix_exc;

	/**
	 * Matrix determining the connectivity between pre- and post-neurons. This
	 * is a n_pre x n_post matrix.
	 */
	NlifBool *connection_matrix_inh;

	/**
	 * Regularisation factor.
	 */
	double regularisation;

	/**
	 * Target neuron threshold current. Only valid if relax_sub_threshold is
	 * true.
	 */
	double j_threshold;

	/**
	 * Relax the optimization problem for subthreshold neurons.
	 */
	NlifBool relax_subthreshold;

	/**
	 * Ensure that synaptic weights are non-negative.
	 */
	NlifBool non_negative;

	/**
	 * Output memory region in which the resulting excitatory neural weights
	 * should be stored. This is a n_pre x n_post matrix.
	 */
	double *synaptic_weights_exc;

	/**
	 * Output memory region in which the resulting inhibitory neural weights
	 * should be stored. This is a n_pre x n_post matrix.
	 */
	double *synaptic_weights_inh;

	/**
	 * Array in which the objective function values for each neuron are stored.
	 * May be null if the caller is not interested in these values.
	 */
	double *objective_vals;
} TwoCompWeightProblem;

/**
 * Parameters used when solving for weights.
 */
typedef struct {
	/**
	 * If set to true, rescales the given problem assuming the input uses
	 * biologically plausible parameters.
	 */
	NlifBool renormalise;

	/**
	 * Sets the absolute and relative abortion criteria. (eps_rel and eps_abs
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
} TwoCompSolverParameters;

DLL_PUBLIC NlifError two_comp_solve(TwoCompWeightProblem *problem,
                                    TwoCompSolverParameters *params);

#ifdef __cplusplus
}
#endif

#endif /* NLIF_TWO_COMP_SOLVER_H */
