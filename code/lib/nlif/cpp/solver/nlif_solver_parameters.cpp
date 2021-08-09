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
class NlifParameterSolver {
private:
	const NlifParameterProblem problem;
	const NlifSolverParameters params;

	int n, k, N;

	const MatrixMap L;
	const VectorMap c;

	VectorMap a_const;
	MatrixMap A;
	VectorMap b_const;
	MatrixMap B;

	const BoolVectorMap a_const_mask;
	const BoolMatrixMap A_mask;
	const BoolVectorMap b_const_mask;
	const BoolMatrixMap B_mask;

	VectorXd a_const_static;
	MatrixXd A_static;
	VectorXd b_const_static;
	MatrixXd B_static;

	MatrixXd a_const_dyn;
	MatrixXd b_const_dyn;

	const MatrixMap g_in;
	const VectorMap J_tar;

	const size_t n_a_const;
	const size_t n_A;
	const size_t n_b_const;
	const size_t n_B;

	/**
	 * Distributes the individual entries of the reduced system matrices into
	 * the parameter vector and computes the original equilibrium potentials.
	 */
	VectorXd compute_theta0()
	{
		// Initialise the result vector
		VectorXd theta0 =
		    VectorXd::Zero(n_a_const + n_A + n_b_const + n_B + N * n);

		// Write the model parameters to the parameter vector
		int idx = 0;
		for (int i = 0; i < n; i++) {
			if (a_const_mask(i))
				theta0[idx++] = a_const(i);
		}
		for (int i = 0; i < n * k; i++) {
			if (A_mask(i))
				theta0[idx++] = A(i);
		}
		for (int i = 0; i < n; i++) {
			if (b_const_mask(i))
				theta0[idx++] = b_const(i);
		}
		for (int i = 0; i < n * k; i++) {
			if (B_mask(i))
				theta0[idx++] = B(i);
		}

		// Solve for the equilibrium potential for each parameter set and write
		// that to the parameter vector
		for (int i = 0; i < N; i++) {
			const VectorXd g = g_in.row(i);
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
			const VectorXd g = g_in.row(i);
			MatrixXd Ai = (a_const + A * g).asDiagonal();
			Ai += L;

			weights[i] = 1.0 / Ai.determinant();
		}

		return double(N) * (weights / weights.sum());
	}

	/**
	 * Distributes the parameter dimensions in theta back into the reduced
	 * system matrices.
	 */
	void unravel_theta(const VectorXd &theta)
	{
		int idx = 0;
		for (int i = 0; i < n; i++) {
			if (a_const_mask(i))
				a_const(i) = theta[idx++];
		}
		for (int i = 0; i < n * k; i++) {
			if (A_mask(i))
				A(i) = theta[idx++];
		}
		for (int i = 0; i < n; i++) {
			if (b_const_mask(i))
				b_const(i) = theta[idx++];
		}
		for (int i = 0; i < n * k; i++) {
			if (B_mask(i))
				B(i) = theta[idx++];
		}
	}

	void make_static_matrices()
	{
		for (int i = 0; i < n; i++) {
			a_const_static(i) = a_const_mask(i) ? 0.0 : a_const(i);
		}
		for (int i = 0; i < n * k; i++) {
			A_static(i) = A_mask(i) ? 0.0 : A(i);
		}
		for (int i = 0; i < n; i++) {
			b_const_static(i) = b_const_mask(i) ? 0.0 : b_const(i);
		}
		for (int i = 0; i < n * k; i++) {
			B_static(i) = B_mask(i) ? 0.0 : B(i);
		}
	}

	MatrixXd A_dyn_at(const VectorXd &g)
	{
		const size_t n_total = n_a_const + n_A + n_b_const + n_B;
		MatrixXd res = MatrixXd::Zero(n, n_total);
		int idx = n_a_const;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				if (A_mask(i, j))
					res(i, idx++) = g[j];
			}
		}
		return res;
	}

	MatrixXd B_dyn_at(const VectorXd &g)
	{
		const size_t n_total = n_a_const + n_A + n_b_const + n_B;
		MatrixXd res = MatrixXd::Zero(n, n_total);
		int idx = n_a_const + n_A + n_b_const;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				if (B_mask(i, j))
					res(i, idx++) = g[j];
			}
		}
		return res;
	}

	void make_dynamic_matrices()
	{
		const size_t n_total = n_a_const + n_A + n_b_const + n_B;

		{
			a_const_dyn = MatrixXd::Zero(n, n_total);
			int idx = 0;
			for (int i = 0; i < n; i++) {
				if (a_const_mask[i])
					a_const_dyn(i, idx++) = 1.0;
			}
		}

		{
			b_const_dyn = MatrixXd::Zero(n, n_total);
			int idx = n_a_const + n_A;
			for (int i = 0; i < n; i++) {
				if (b_const_mask[i])
					b_const_dyn(i, idx++) = 1.0;
			}
		}
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
		std::cout << "n = " << n << std::endl;
		std::cout << "k = " << k << std::endl;
		std::cout << "N = " << N << std::endl;
		std::cout << "L = \n" << L.format(CleanFmt) << std::endl;
		std::cout << "c = \n" << c.format(CleanFmt) << std::endl;

		std::cout << "A = \n" << A.format(CleanFmt) << std::endl;
		std::cout << "a_const = \n" << a_const.format(CleanFmt) << std::endl;
		std::cout << "B = \n" << B.format(CleanFmt) << std::endl;
		std::cout << "b_const = \n" << b_const.format(CleanFmt) << std::endl;

		std::cout << "A_mask = \n" << A_mask.format(CleanFmt) << std::endl;
		std::cout << "a_const_mask = \n"
		          << a_const_mask.format(CleanFmt) << std::endl;
		std::cout << "B_mask = \n" << B_mask.format(CleanFmt) << std::endl;
		std::cout << "b_const_mask = \n"
		          << b_const_mask.format(CleanFmt) << std::endl;

		std::cout << "A_static = \n" << A_static.format(CleanFmt) << std::endl;
		std::cout << "a_const_static = \n"
		          << a_const_static.format(CleanFmt) << std::endl;
		std::cout << "B_static = \n" << B_static.format(CleanFmt) << std::endl;
		std::cout << "b_const_static = \n"
		          << b_const_static.format(CleanFmt) << std::endl;

		std::cout << "a_const_dyn = \n"
		          << a_const_dyn.format(CleanFmt) << std::endl;
		std::cout << "b_const_dyn = \n"
		          << b_const_dyn.format(CleanFmt) << std::endl;

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
	NlifParameterSolver(NlifParameterProblem *problem,
	                    NlifSolverParameters *params)
	    : problem(*problem),
	      params(*params),

	      n(problem->n_compartments),
	      k(problem->n_inputs),
	      N(problem->n_samples),

	      L(problem->L, n, n),
	      c(problem->c, n, 1),

	      a_const(problem->a_const, n),
	      A(problem->A, n, k),
	      b_const(problem->b_const, n),
	      B(problem->B, n, k),

	      a_const_mask(problem->a_const_mask, n),
	      A_mask(problem->A_mask, n, k),
	      b_const_mask(problem->b_const_mask, n),
	      B_mask(problem->B_mask, n, k),

	      a_const_static(n),
	      A_static(n, k),
	      b_const_static(n),
	      B_static(n, k),

	      g_in(problem->g_in, N, k),
	      J_tar(problem->J_tar, N),

	      n_a_const(a_const_mask.cast<int>().sum()),
	      n_A(A_mask.cast<int>().sum()),
	      n_b_const(b_const_mask.cast<int>().sum()),
	      n_B(B_mask.cast<int>().sum())
	{
		// Extract the static (masked) portions of the parameter matrices
		make_static_matrices();

		// Prepare the dynamic matrices
		make_dynamic_matrices();

#ifdef NLIF_DEBUG
		print_debug_info();
#endif
	}

	NlifError run()
	{
		//
		// Step 1: Count stuff and setup indices used to partition the matrices
		//         into smaller parts.
		//
		const size_t n_invalid = count_subthreshold();
		const size_t n_valid = N - n_invalid;

		const size_t n_vars = n_a_const + n_A + n_b_const + n_B + N * n;

		const size_t v0 = 0;
		const size_t v1 = v0 + n_a_const + n_A;
		const size_t v2 = v1 + n_b_const + n_B;
		const size_t v3 = v2 + N * n;      // Auxiliary variables
		const size_t v4 = v3 + n_invalid;  // Slack variables constraints

		const size_t p1 = 0;
		const size_t p2 = p1 + n_valid;    // Current error
		const size_t p3 = p2 + N * n;      // Voltage error
		const size_t p4 = p3 + n_vars;     // Trust region
		const size_t p5 = p4 + n_vars;     // Regularisation
		const size_t p6 = p5 + n_invalid;  //  Penalize slack variables

		const size_t g1 = 0;
		const size_t g2 = g1 + n_a_const + n_A;  // Non-negative parameters
		const size_t g3 = g2 + n_invalid;        // Subthreshold constraints
		const size_t g4 = g3 + n_invalid;        // Slack non-negativity

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
		VectorXi Pssp = VectorXi::Zero(v4);
		for (size_t i = v0; i < v2; i++) {
			Pssp[i] += n;  // For Part 2: Voltage constraints
			Pssp[i] += 1;  // For Part 3: Trust region
			Pssp[i] += 1;  // For Part 4: Regularisation
		}
		for (size_t i = v2; i < v3; i++) {
			Pssp[i] += 1;  // For Part 1: Current estimate
			Pssp[i] += n;  // For Part 2: Voltage constraints
			Pssp[i] += 1;  // For Part 3: Trust region
			Pssp[i] += 1;  // For Part 4: Regularisation
		}
		for (size_t i = v3; i < v4; i++) {
			Pssp[i] += 1;  // For Part 1: Slack variables
		}

		//
		// Step 4: Assemble the quadratic terms
		//
		VectorXd q = VectorXd::Zero(p6);
		SpMatrixXd P(p6, v4);
		P.reserve(Pssp);

		// Step 4.1: Current errors
		{
			size_t idx = 0;
			for (int i = 0; i < N; i++) {
				if (!is_subthreshold(i)) {
					for (int j = 0; j < n; j++) {
						P.insert(p1 + idx, v2 + i * n + j) = alpha1 * c[j];
					}
					q[idx] = alpha1 * J_tar[i];
					idx++;
				}
			}
		}

		// Step 4.2: Voltage errors

		for (int i = 0; i < N; i++) {
			// Fetch the current input sample, and the corresponding slices of
			// theta
			const VectorXd g = g_in.row(i);
			const VectorMap v0s(theta0.data() + v2 + i * n, n);

			// Assemble the individual sub-matrices for A, b
			MatrixXd Pg = MatrixXd::Zero(n, v2);
			MatrixXd Pv = MatrixXd::Zero(n, n);
			VectorXd qi = VectorXd::Zero(n);

			// Constant parts for the static sections of the system matrices
			qi -= b_const_static;
			qi -= B_static * g;
			qi += (a_const_static.array() * v0s.array()).matrix();
			qi += ((A_static * g).array() * v0s.array()).matrix();

			// Constant parts for terms depending both on theta0 and v0
			qi -= (a_const.array() * v0s.array()).matrix();
			qi -= ((A * g).array() * v0s.array()).matrix();

			// Voltage-dependent parts
			Pv += L;
			Pv += a_const.asDiagonal();
			Pv += (A * g).asDiagonal();

			// Parameter-dependent parts
			Pg -= b_const_dyn;
			Pg -= B_dyn_at(g);
			Pg += v0s.asDiagonal() * a_const_dyn;
			Pg += v0s.asDiagonal() * A_dyn_at(g);

			// Copy the sub-matrices into the sparse matrix
			for (size_t i0 = 0; i0 < size_t(n); i0++) {
				const size_t i_global = p2 + i * n + i0;
				for (size_t i1 = v0; i1 < v2; i1++) {
					P.insert(i_global, i1) =
					    alpha2 * weights[i] * Pg(i0, i1 - v0);
				}
				for (size_t i1 = 0; i1 < size_t(n); i1++) {
					P.insert(i_global, v2 + i * n + i1) =
					    alpha2 * weights[i] * Pv(i0, i1);
				}
				q[i_global] =
				    -alpha2 * weights[i] *
				    qi[i0];  // Minus here because we compute -PTq below
			}
		}

		// Step 4.3: Trust region
		for (size_t i = v0; i < v2; i++) {
			P.insert(p3 + i - v0, i) = alpha3 * std::sqrt(N);
			q[p3 + i - v0] = alpha3 * std::sqrt(N) * theta0[i];
		}
		for (size_t i = v2; i < v3; i++) {
			P.insert(p3 + i - v0, i) = alpha3;
			q[p3 + i - v0] = alpha3 * theta0[i];
		}

		// Step 4.4: Regularisation
		const double lambda1 = std::sqrt(N * problem.reg1);
		const double lambda2 = std::sqrt(problem.reg2);
		for (size_t i = v0; i < v2; i++) {
			P.insert(p4 + i - v0, i) = lambda1;
		}
		for (size_t i = v2; i < v3; i++) {
			P.insert(p4 + i - v0, i) = lambda2;
		}

		// Step 4.5: Penalize subthreshold constraint violations
		for (size_t i = v3; i < v4; i++) {
			P.insert(p5 + i - v3, i) = alpha1;
		}

		// Compute PTP and PTb
		SpMatrixXd PTP(v4, v4);
		PTP.selfadjointView<Upper>().rankUpdate(P.transpose());
		VectorXd PTq = -P.transpose() * q;

		//
		// Step 5: Assemble the inequality constraints
		//

		// Compute the sparsity pattern of G
		VectorXi Gssp = VectorXi::Zero(v4);
		for (size_t i = v0; i < v1; i++) {
			Gssp[i] = 1;  // Each parameter in A and a_const is used once
		}
		for (size_t i = v2; i < v3; i++) {
			Gssp[i] = 1;  // Each voltage is used once
		}
		for (size_t i = v3; i < v4; i++) {
			Gssp[i] = 2;  // Each slack variable is used twice
		}

		SpMatrixXd G(g4, v4);
		G.reserve(Gssp);
		VectorXd h = VectorXd::Zero(g4);

		// Step 5.1: Make sure that the A and a_const parameters are
		// non-negative
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
						G.insert(g2 + idx, v2 + i * n + j) = c[j];
					}
					G.insert(g2 + idx, v3 + idx) = -1.0;
					h[g2 + idx] = problem.j_threshold;
					idx++;
				}
			}
		}

		// Step 5.3: Slack variable non-negativity
		for (size_t i = v3; i < v4; i++) {
			G.insert(g3 + i - v3, i) = -1.0;
			h[g3 + i - v3] = 0.0;
		}

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

		// Distribute the updated parameters back into the individual system
		// matrices
		unravel_theta(res.x);

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

NlifError nlif_solve_parameters_iter(NlifParameterProblem *problem,
                                     NlifSolverParameters *params)
{
	return NlifParameterSolver(problem, params).run();
}

#ifdef __cplusplus
}
#endif

