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

#include "visibility.h"
#include "common.h"

/******************************************************************************
 * EXTERNAL C API                                                             *
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Enum NlifError                                                             *
 ******************************************************************************/

const char *nlif_strerr(NlifError err)
{
	switch (err) {
		case NL_ERR_OK:
			return "no error";
		case NL_ERR_INVALID_N_PRE:
			return "n_pre is invalid";
		case NL_ERR_INVALID_N_POST:
			return "n_post is invalid";
		case NL_ERR_INVALID_N_SAMPLES:
			return "n_samples is invalid";
		case NL_ERR_INVALID_A_PRE:
			return "a_pre is invalid";
		case NL_ERR_INVALID_J_POST:
			return "j_post is invalid";
		case NL_ERR_INVALID_MODEL_WEIGHTS:
			return "model_weights is invalid";
		case NL_ERR_INVALID_CONNECTION_MATRIX_EXC:
			return "connection_matrix_exc is invalid";
		case NL_ERR_INVALID_CONNECTION_MATRIX_INH:
			return "connection_matrix_inh is invalid";
		case NL_ERR_INVALID_REGULARISATION:
			return "regularisation is invalid";
		case NL_ERR_INVALID_SYNAPTIC_WEIGHTS_EXC:
			return "synaptic_weights_exc is invalid";
		case NL_ERR_INVALID_SYNAPTIC_WEIGHTS_INH:
			return "synaptic_weights_inh is invalid";
		case NL_ERR_INVALID_TOLERANCE:
			return "tolerance is invalid";
		case NL_ERR_QP:
			return "error while solving the quadratic program";
		case NL_ERR_CANCEL:
			return "canceled by user";
	}
	return "unknown error code";
}

#ifdef __cplusplus
}
#endif
