#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2017-2021  Andreas Stöckel
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


# Implementation of
# Diederik P. Kingma, Jimmy Ba Adam: A Method for Stochastic Optimization
# ICLR 2015
# (Source code comments are directly copied from the paper)

import numpy as np


class Adam:
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        assert alpha > 0.0
        assert (beta1 > 0.0) and (beta1 < 1.0)
        assert (beta2 > 0.0) and (beta2 < 1.0)
        assert epsilon > 0.0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, p, dp):
        # Increase the iteration number
        self.t += 1

        for key in range(len(dp)):
            # Initialize the first and second moment estimates for
            # not-yet-encountered parameter vectors
            if not key in self.m:
                self.m[key] = np.zeros(p[key].shape)
                self.v[key] = np.zeros(p[key].shape)

            # Update the biased first and second moment estimate
            β1, β2 = self.beta1, self.beta2
            self.m[key] = β1 * self.m[key] + (1.0 - β1) * dp[key]
            self.v[key] = β2 * self.v[key] + (1.0 - β2) * (dp[key]**2)

            # Compute the bias-corrected first and second moment estimate
            mhat = self.m[key] / (1.0 - (β1**self.t))
            vhat = self.v[key] / (1.0 - (β2**self.t))

            # Update the parameters
            p[key][...] = p[key] - self.alpha * mhat / (np.sqrt(vhat) +
                                                        self.epsilon)

