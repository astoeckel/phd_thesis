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

#ifndef NLIF_MATRIX_TYPES_HPP
#define NLIF_MATRIX_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace {
using SpMatrixXd = Eigen::SparseMatrix<double>;
using MatrixMap = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using VectorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
using BoolMatrixMap =
    Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>>;
using BoolVectorMap =
    Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>>;
using BoolVector = Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>;

}  // namespace

#endif /* NLIF_MATRIX_TYPES_HPP */
