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

#ifndef NLIF_COMMON_H
#define NLIF_COMMON_H

#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enum describing possible return values of bioneuronqp_solve.
 */
typedef enum {
	NL_ERR_OK = 0,
	NL_ERR_INVALID_N_PRE = -1,
	NL_ERR_INVALID_N_POST = -2,
	NL_ERR_INVALID_N_SAMPLES = -3,
	NL_ERR_INVALID_A_PRE = -4,
	NL_ERR_INVALID_J_POST = -5,
	NL_ERR_INVALID_MODEL_WEIGHTS = -6,
	NL_ERR_INVALID_CONNECTION_MATRIX_EXC = -7,
	NL_ERR_INVALID_CONNECTION_MATRIX_INH = -8,
	NL_ERR_INVALID_REGULARISATION = -9,
	NL_ERR_INVALID_SYNAPTIC_WEIGHTS_EXC = -10,
	NL_ERR_INVALID_SYNAPTIC_WEIGHTS_INH = -11,
	NL_ERR_INVALID_TOLERANCE = -12,
	NL_ERR_CANCEL = -13,
	NL_ERR_QP = -14,
} NlifError;

/**
 * Converts the given error code into a string.
 */
DLL_PUBLIC const char *nlif_strerr(NlifError err);

/**
 * Internally used Boolean type; this should be compatible with the C++ "bool"
 * type.
 */
typedef unsigned char NlifBool;

/**
 * Callback function type used to inform the caller about the current progress
 * of the weight solving process. Must return "true" if the computation should
 * continue; returning "false" cancels the computation and two_comp_solve
 * will return NL_ERR_CANCEL.
 */
typedef bool (*NlifProgressCallback)(int n_done, int n_total);

/**
 * Callback function type used to inform the caller about any warnings that
 * pop up during the weight solving process.
 */
typedef void (*NlifWarningCallback)(const char *msg, int i_post);

#ifdef __cplusplus
}
#endif

#endif /* NLIF_COMMON_H */
