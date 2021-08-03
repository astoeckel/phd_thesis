/*
 *  libbioneuronqp -- Library solving for synaptic weights
 *  Copyright (C) 2020  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef NLIF_THREADPOOL_HPP
#define NLIF_THREADPOOL_HPP

#include <functional>
#include <memory>

class Threadpool {
private:
	class Impl;
	std::unique_ptr<Impl> m_impl;

public:
	using Kernel = std::function<void(size_t)>;
	using Progress = std::function<bool(size_t cur, size_t max)>;

	Threadpool(unsigned int n_threads = 0);
	~Threadpool();

	void run(unsigned int n_work_items, const Kernel &kernel,
	         Progress progress = Progress());
};

#endif /* NLIF_THREADPOOL_HPP */
