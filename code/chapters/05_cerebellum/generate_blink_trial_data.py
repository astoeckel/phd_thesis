#!/usr/bin/env python3

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

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model')) # blinktrial
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib')) # pytry

from blink_trial import BlinkTrial

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')

for i in range(6):
    BlinkTrial().run(
         n_trials=500
         seed=3924 + i,
         learning_rate=0.00018,
         tau=60e-3,
         tau_error=0.1,
         tau_pre=75e-3,
         t_delay=0.15,
         mode='two_populations_dales_principle',
         use_spatial_constraints=True,
         n_pcn_golgi_convergence=100,
         n_pcn_granule_convergence=5,
         n_granule_golgi_convergence=100,
         n_golgi_granule_convergence=5,
         n_golgi_golgi_convergence=100,
         n_granule=10000
         n_golgi=100,
         q=6,
         theta=0.4,
         data_filename=f'blink_trial_{i}',
         data_dir=data_dir, verbose=False, data_format='npz'
    )

