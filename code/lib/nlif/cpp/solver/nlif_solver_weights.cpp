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
 * @file nlif_solver.cpp
 *
 * Implements the trust-region based n-LIF solver.
 *
 * @author Andreas Stöckel
 */

//#define NLIF_DEBUG

#ifdef NLIF_DEBUG
#include <iostream>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "matrix_types.hpp"
#include "nlif_solver.h"
#include "qp.hpp"
#include "threadpool.hpp"

using namespace Eigen;

/******************************************************************************
 * INTERNAL C++ CODE                                                          *
 ******************************************************************************/

#ifdef NLIF_DEBUG
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
#endif

namespace {
class NlifWeightSolver {
private:
	const NlifWeightProblem problem;
	const NlifSolverParameters params;

	int m, n, k, N;

	const MatrixMap L;
	const VectorMap c;

	const VectorMap a_const;
	const MatrixMap A;
	const VectorMap b_const;
	const MatrixMap B;

	const MatrixMap A_in;
	const VectorMap J_tar;

	MatrixMap W;
	const BoolMatrixMap W_mask;

	MatrixXd W_static;

	const size_t i_post;
	const size_t n_W;

	/**
	 * Distributes the individual weights and computes the original equilibrium potentials.
	 */
	VectorXd compute_theta0()
	{
		// Initialise the result vector
		VectorXd theta0 = VectorXd::Zero(n_W + N * n);

		// Write the model parameters to the parameter vector
		int idx = 0;
		for (int i = 0; i < k * m; i++) {
			if (W_mask(i)) {
				theta0[idx++] = W(i);
			}
		}

		// Solve for the equilibrium potential for each parameter set and write
		// that to the parameter vector
		for (int i = 0; i < N; i++) {
			const VectorXd g = W * A_in.row(i).transpose();
			MatrixXd Ai = (a_const + A * g).asDiagonal();
			Ai += L;

			VectorXd bi = b_const + B * g;

			VectorXd veqi = Ai.llt().solve(bi);
			for (int j = 0; j < n; j++) {
				theta0[idx++] = veqi[j];
			}
		}

		return theta0;
	}

	/**
	 * Distributes the weights pack into the weight matrix
	 */
	void unravel_theta(const VectorXd &theta)
	{
		int idx = 0;
		for (int i = 0; i < k * m; i++) {
			if (W_mask(i)) {
				W(i) = theta[idx++];
			}
		}
	}

	/**
	 * Splits the static parts of the weight matrix out into a separate matrix.
	 */
	void make_static_weight_matrix()
	{
		for (int i = 0; i < k * m; i++) {
			W_static(i) = W_mask(i) ? 0.0 : W(i);
		}
	}

	MatrixXd W_dyn(const VectorXd &a) {
		MatrixXd res = MatrixXd::Zero(k, n_W);
		int idx = 0;
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < m; j++) {
				if (W_mask(i, j)) {
					res(i, idx) = a[j];
					idx++;
				}
			}
		}
		return res;
	}

	/**
	 * Computes that weights for the Sananthan-Koerner iteration by computing
	 * the determinant of each A[g_k] matrix.
	 */
	VectorXd compute_sanathanan_koerner_weights()
	{
		// Initialise the result vector
		VectorXd weights = VectorXd::Zero(N);

		// Solve for the equilibrium potential for each parameter set and write
		// that to the parameter vector
		for (int i = 0; i < N; i++) {
			const VectorXd g = W * A_in.row(i).transpose();
			MatrixXd Ai = (a_const + A * g).asDiagonal();
			Ai += L;

			weights[i] = 1.0 / Ai.determinant();
		}

		return double(N) * (weights / weights.sum());
	}

	bool is_subthreshold(int i)
	{
		return problem.relax_subthreshold && (J_tar[i] <= problem.j_threshold);
	}

	size_t count_subthreshold()
	{
		size_t res = 0;
		for (int i = 0; i < N; i++) {
			if (is_subthreshold(i)) {
				res++;
			}
		}
		return res;
	}

#ifdef NLIF_DEBUG
	void print_debug_info()
	{
		std::cout << "m = " << m << std::endl;
		std::cout << "n = " << n << std::endl;
		std::cout << "k = " << k << std::endl;
		std::cout << "N = " << N << std::endl;
		std::cout << "L = \n" << L.format(CleanFmt) << std::endl;
		std::cout << "c = \n" << c.format(CleanFmt) << std::endl;

		std::cout << "A = \n" << A.format(CleanFmt) << std::endl;
		std::cout << "a_const = \n" << a_const.format(CleanFmt) << std::endl;
		std::cout << "B = \n" << B.format(CleanFmt) << std::endl;
		std::cout << "b_const = \n" << b_const.format(CleanFmt) << std::endl;

//		std::cout << "A_in = \n" << A_in.format(CleanFmt) << std::endl;
//		std::cout << "J_tar = \n" << J_tar.format(CleanFmt) << std::endl;

		std::cout << "W = \n" << W.format(CleanFmt) << std::endl;
		std::cout << "W_mask = \n" << W_mask.format(CleanFmt) << std::endl;
		std::cout << "n_W = " << n_W << std::endl;

		std::cout << "alpha1 = " << problem.alpha1 << std::endl;
		std::cout << "alpha2 = " << problem.alpha2 << std::endl;
		std::cout << "alpha3 = " << problem.alpha3 << std::endl;
		std::cout << "reg1 = " << problem.reg1 << std::endl;
		std::cout << "reg2 = " << problem.reg2 << std::endl;
		std::cout << "j_threshold = " << problem.j_threshold << std::endl;
		std::cout << "relax_subthreshold = " << problem.relax_subthreshold
		          << std::endl;

		std::cout << "use_sanathanan_koerner = "
		          << params.use_sanathanan_koerner << std::endl;
		std::cout << "tol = " << params.tolerance << std::endl;
	}
#endif

public:
	NlifWeightSolver(NlifWeightProblem *problem, NlifSolverParameters *params,
	                 size_t i_post)
	    : problem(*problem),
	      params(*params),

	      m(problem->n_pre),
	      n(problem->n_compartments),
	      k(problem->n_inputs),
	      N(problem->n_samples),

	      L(problem->L, n, n),
	      c(problem->c, n, 1),

	      a_const(problem->a_const, n),
	      A(problem->A, n, k),
	      b_const(problem->b_const, n),
	      B(problem->B, n, k),

	      A_in(problem->A_in, N, m),
	      J_tar(problem->J_tar + i_post * N, N),
	      W(problem->W + i_post * k * m, k, m),
	      W_mask(problem->W_mask + i_post * k * m, k, m),
	      W_static(k, m),

	      i_post(i_post),
	      n_W(W_mask.cast<int>().sum())
	{
		// Extract the static portions from the weight matrix
		make_static_weight_matrix();

#ifdef NLIF_DEBUG
		print_debug_info();
#endif
	}

	NlifError run() {
		//
		// Step 1: Count stuff and setup indices used to partition the matrices.
		//
		const size_t n_invalid = count_subthreshold();
		const size_t n_valid = N - n_invalid;

		const size_t n_vars = n_W + N * n;

		const size_t v0 = 0;
		const size_t v1 = v0 + n_W;       // Weights
		const size_t v2 = v1 + N * n;     // Auxiliary variables
		const size_t v3 = v2 + n_invalid; // Slack variables

		const size_t p1 = 0;
		const size_t p2 = p1 + n_valid;   // Current error
		const size_t p3 = p2 + N * n;     // Voltage error
		const size_t p4 = p3 + n_vars;    // Trust region
		const size_t p5 = p4 + n_vars;    // Regularisation
		const size_t p6 = p5 + n_invalid; // Penalize slack variables

		const size_t g1 = 0;
		const size_t g2 = g1 + n_W; // Non-negative weights
		const size_t g3 = g2 + n_invalid; // Subthreshold constraints
		const size_t g4 = g3 + n_invalid; // Slack non-negativity

		const double alpha1 = std::sqrt(problem.alpha1);
		const double alpha2 = std::sqrt(problem.alpha2);
		const double alpha3 = std::sqrt(problem.alpha3);

#ifdef NLIF_DEBUG
		std::cout << "n_invalid = " << n_invalid << std::endl;
		std::cout << "n_valid = " << n_valid << std::endl;
#endif

		//
		// Step 2: Compute the initial parameter vector for the trust region
		//
		VectorXd theta0 = compute_theta0();
		const VectorXd weights = params.use_sanathanan_koerner
		                       ? compute_sanathanan_koerner_weights()
		                       : VectorXd::Constant(N, 1.0);

		//
		// Step 3: Calculate the sparsity pattern of the P matrix
		//
		VectorXi Pssp = VectorXi::Zero(v3);
		for (size_t i = v0; i < v1; i++) {
			Pssp[i] += N * n;  // For Part 2: Voltage errors
			Pssp[i] += 1;  // For Part 3: Trust region
			Pssp[i] += 1;  // For Part 4: Regularisation
		}
		for (size_t i = v1; i < v2; i++) {
			Pssp[i] += 1;  // For Part 1: Current errors
			Pssp[i] += n;  // For Part 2: Voltage errors
			Pssp[i] += 1;  // For Part 3: Trust region
			Pssp[i] += 1;  // For Part 4: Regularisation
		}
		for (size_t i = v2; i < v3; i++) {
			Pssp[i] += 1;  // For Part 1: Current estimate, subthreshold cstr.
		}

		//
		// Step 4: Assemble the quadratic terms
		//
		VectorXd q = VectorXd::Zero(p6);
		SpMatrixXd P(p6, v3);
		P.reserve(Pssp);

		// Step 4.1: Current errors
		{
			size_t idx = 0;
			for (int i = 0; i < N; i++) {
				if (!is_subthreshold(i)) {
					for (int j = 0; j < n; j++) {
						P.insert(p1 + idx, v1 + i * n + j) = alpha1 * c[j];
					}
					q[idx] = alpha1 * J_tar[i];
					idx++;
				}
			}
		}

		// Step 4.2: Voltage errors
		for (int i = 0; i < N; i++) {
			// Fetch the input and voltages at the current expansion point
			const VectorXd a_in = A_in.row(i).transpose();
			const VectorMap v0s(theta0.data() + v1 + i * n, n);

			// Assemble the individual sub-matrices for A, b
			MatrixXd Pw = MatrixXd::Zero(n, v1); // Terms linear in w
			MatrixXd Pv = MatrixXd::Zero(n, n);  // Terms linear in v
			VectorXd qi = VectorXd::Zero(n);     // Constant terms

			// Constant terms
			qi -= b_const;
			qi -= B * W_static * a_in;
			qi += ((A * W_static * a_in).array() * v0s.array()).matrix();
			qi -= ((A * W * a_in).array() * v0s.array()).matrix();

			// Voltage-dependent terms
			Pv += L;
			Pv += a_const.asDiagonal();
			Pv += (A * W * a_in).asDiagonal();

			// Weight-dependent terms
			Pw -= B * W_dyn(a_in);
			Pw += v0s.asDiagonal() * A * W_dyn(a_in);

			// Copy the sub-matrices into the sparse matrix
			for (size_t i0 = 0; i0 < size_t(n); i0++) {
				const size_t i_global = p2 + i * n + i0;
				for (size_t i1 = 0; i1 < size_t(n_W); i1++) {
					P.insert(i_global, v0 + i1) =
						alpha2 * weights[i] * Pw(i0, i1);
				}
				for (size_t i1 = 0; i1 < size_t(n); i1++) {
					P.insert(i_global, v1 + i * n + i1) =
					    alpha2 * weights[i] * Pv(i0, i1);
				}
				q[i_global] =
				    -alpha2 * weights[i] *
				    qi[i0];  // Minus here because we compute -PTq below
			}
		}

		// Step 4.3: Trust region
		const double alpha3p = alpha3 * std::sqrt(double(N));
		for (size_t i = v0; i < v1; i++) {
			P.insert(p3 + i - v0, i) = alpha3p;
			q[p3 + i - v0] = alpha3p * theta0[i];
		}
		for (size_t i = v1; i < v2; i++) {
			P.insert(p3 + i - v0, i) = alpha3;
			q[p3 + i - v0] = alpha3 * theta0[i];
		}

		// Step 4.4: Regularisation
		const double lambda1 = std::sqrt(N * problem.reg1);
		const double lambda2 = std::sqrt(problem.reg2);
		for (size_t i = v0; i < v1; i++) {
			P.insert(p4 + i - v0, i) = lambda1;
		}
		for (size_t i = v1; i < v2; i++) {
			P.insert(p4 + i - v0, i) = lambda2;
		}

		// Step 4.5: Penalize subthreshold constraint violations
		for (size_t i = v2; i < v3; i++) {
			P.insert(p5 + i - v2, i) = alpha1;
		}

		// Compute PTP and PTb
		P.prune(1e-9);
		SpMatrixXd PTP(v3, v3);
		PTP.selfadjointView<Upper>().rankUpdate(P.transpose());
		VectorXd PTq = -P.transpose() * q;

		/*std::cout << P.col(0).toDense().format(CleanFmt) << std::endl;
		auto ptr = P.outerIndexPtr();
		for (size_t i = 0; i < v3; i++) {
			std::cout << (ptr[i + 1] - ptr[i]) << ", " << Pssp[i] << std::endl;
		}*/

		//
		// Step 5: Assemble the inequality constraints
		//

		// Compute the sparsity pattern of G
		VectorXi Gssp = VectorXi::Zero(v3);
		for (size_t i = v0; i < v1; i++) {
			Gssp[i] = 2;  // Regularisation, trust region
		}
		for (size_t i = v1; i < v2; i++) {
			Gssp[i] = 3;  // Subthreshold constraints, regularisation, trust region
		}
		for (size_t i = v2; i < v3; i++) {
			Gssp[i] = 2;  // Each slack variable is used twice
		}

		SpMatrixXd G(g4, v3);
		G.reserve(Gssp);
		VectorXd h = VectorXd::Zero(g4);

		// Step 5.1: Make sure that the weights are non-negative
		for (size_t i = v0; i < v1; i++) {
			G.insert(g1 + i - v0, i) = -1.0;
			h[g1 + i - v0] = 0.0;
		}

		// Step 5.2: Subthreshold constraints
		{
			size_t idx = 0;
			for (int i = 0; i < N; i++) {
				if (is_subthreshold(i)) {
					for (int j = 0; j < n; j++) {
						G.insert(g2 + idx, v1 + i * n + j) = c[j];
					}
					G.insert(g2 + idx, v2 + idx) = -1.0;
					h[g2 + idx] = problem.j_threshold;
					idx++;
				}
			}
		}

		// Step 5.3: Slack variable non-negativity
		for (size_t i = v2; i < v3; i++) {
			G.insert(g3 + i - v2, i) = -1.0;
			h[g3 + i - v2] = 0.0;
		}

		// Remove superfluous entries from G
		G.prune(1e-9);

/*		auto ptr = G.outerIndexPtr();
		for (size_t i = 0; i < v3; i++) {
			std::cout << (ptr[i + 1] - ptr[i]) << ", " << Gssp[i] << std::endl;
		}*/

		//
		// Step 6: Solve the problem
		//

		// Solve the actual problem
		QPResult res =
		    solve_qp(PTP, PTq, G, h, params.tolerance, params.max_iter);
		if (res.status != 0) {
			std::stringstream ss;
			ss << "Error while computing nlif parameters: ";
			ss << res.status_to_str();
			if (params.warn) {
				params.warn(ss.str().c_str(), -1);
			}
		}
		if (!res.has_solution()) {
			return NL_ERR_QP;
		}

		// Strictly enforce weight non-negativity (some entries in the parameter
		// vector may have very small negative entries, i.e., -1e-30).
		for (size_t i = v0; i < v1; i++) {
			res.x[i] = std::max(0.0, res.x[i]);
		}

		// Distribute the updated weights back into the weight matrix
		unravel_theta(res.x);

		// Store the objective values
		if (problem.objective_vals) {
			problem.objective_vals[i_post] = res.objective_val;
		}

		return NL_ERR_OK;
	}
};
}  // namespace

/******************************************************************************
 * EXTERNAL C API                                                             *
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

NlifError nlif_solve_weights_iter(NlifWeightProblem *problem,
                                  NlifSolverParameters *params)
{
	// Construct the kernal that is being executed -- here, we're solving the
	// weights for a single post-neuron.
	auto kernel = [&](size_t idx) {
		NlifWeightSolver(problem, params, idx).run();
	};

	// Construct the progress callback
	bool did_cancel = false;
	auto progress = [&](size_t cur, size_t max) {
		if (params->progress) {
			if (!params->progress(cur, max)) {
				did_cancel = true;
			}
			return !did_cancel;
		}
		return true;
	};

	// Create a threadpool and solve the weights for all neurons. Do not create
	// a threadpool if there is only one set of weights to solve for, or the
	// number of threads has explicitly been set to one.
	int n_threads = std::min(params->n_threads, problem->n_post);
#ifdef NLIF_DEBUG
	std::cout << "n_threads = " << n_threads << std::endl;
#endif
	if ((n_threads != 1) && (problem->n_post > 1)) {
		Threadpool pool(params->n_threads);
		pool.run(problem->n_post, kernel, progress);
	} else {
		for (int i = 0; i < problem->n_post; i++) {
			kernel(i);
			if (params->progress) {
				if (!params->progress(i + 1, problem->n_post)) {
					return NL_ERR_CANCEL;
				}
			}
		}
	}
	return did_cancel ? NL_ERR_CANCEL : NL_ERR_OK;
}

#ifdef __cplusplus
}
#endif

