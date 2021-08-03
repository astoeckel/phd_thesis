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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "threadpool.hpp"

/******************************************************************************
 * Class Threadpool::Impl                                                     *
 ******************************************************************************/

class Threadpool::Impl {
private:
	using Kernel = std::function<void(size_t)>;
	std::atomic<uint64_t> m_generation;
	std::atomic<bool> m_done;
	std::atomic<unsigned int> m_max_work_idx;
	std::atomic<unsigned int> m_cur_work_idx;
	std::atomic<unsigned int> m_workers_done;
	std::condition_variable m_pool_cond;
	std::condition_variable m_main_cond;
	std::mutex m_pool_mtx;
	std::mutex m_main_mtx;
	std::vector<std::thread> m_pool;
	Kernel m_kernel;

	static void thread_main(Impl *self)
	{
		uint64_t generation = 0;
		while (!self->m_done) {
			// Wait for a new work order, but at most 100ms
			{
				std::unique_lock<std::mutex> lock(self->m_pool_mtx);
				self->m_pool_cond.wait_for(lock,
				                           std::chrono::milliseconds(100));
			}

			// Make sure we see any update to the variables accessed below
			std::atomic_thread_fence(std::memory_order_acquire);

			// Prevent each thread from doing the same work twice (because
			// m_pool_cond may wake up spuriously)
			const uint64_t current_generation = self->m_generation;
			if (self->m_done || (current_generation <= generation)) {
				continue;
			}
			generation = current_generation;

			// Work on each work item
			const unsigned int max_work_idx = self->m_max_work_idx;
			unsigned int cur_work_idx = self->m_cur_work_idx;
			while (true) {
				while (!self->m_cur_work_idx.compare_exchange_weak(
				    cur_work_idx, cur_work_idx + 1))
					;
				if (cur_work_idx >= max_work_idx) {
					break;
				}
				self->m_kernel(cur_work_idx);
			}

			// This worker is done, increment the counter
			self->m_workers_done++;
			std::atomic_thread_fence(std::memory_order_release);

			// Notify the main thread
			self->m_main_cond.notify_one();
		}
	}

public:
	Impl(unsigned int n_threads)
	    : m_generation(0U),
	      m_done(false),
	      m_max_work_idx(0U),
	      m_cur_work_idx(0U)
	{
		if (n_threads == 0) {
			n_threads = std::max(std::thread::hardware_concurrency(), 1U);
		}
		for (unsigned int i = 0U; i < n_threads; i++) {
			m_pool.emplace_back(thread_main, this);
		}
	}

	~Impl()
	{
		// Set m_done to true and make sure that all threads can see this update
		m_done = true;
		std::atomic_thread_fence(std::memory_order_release);

		// Notify all threads about the state update
		m_pool_cond.notify_all();

		// Wait for all threads to finish
		for (size_t i = 0; i < m_pool.size(); i++) {
			m_pool[i].join();
		}
	}

	void run(unsigned int n_work_items, const Kernel &kernel, Progress progress)
	{
		// Set the current kernel
		m_kernel = kernel;
		m_max_work_idx = n_work_items;
		m_cur_work_idx = 0U;
		m_workers_done = 0U;

		// Make sure that the above variables are set before the generation
		// number is increased.
		std::atomic_thread_fence(std::memory_order_release);

		// Increment the generation. This must be the last operation, since a
		// spuriously waking thread will start working as soon as the generation
		// number is incremented.
		m_generation++;

		// Make sure all memory updates are visible to the thread
		std::atomic_thread_fence(std::memory_order_release);

		// Notify all threads
		m_pool_cond.notify_all();

		// Wait for all work items to be completed
		while (m_workers_done < m_pool.size()) {
			// Wait for a notification from a child thread about being done,
			// but at most 100 milliseconds
			std::unique_lock<std::mutex> lock(m_main_mtx);
			m_main_cond.wait_for(lock, std::chrono::milliseconds(100));

			// Notify the caller about any progress
			if (progress) {
				const unsigned int cur_work_idx =
				    std::min<unsigned int>(m_cur_work_idx, n_work_items);
				if (!progress(cur_work_idx, n_work_items)) {
					m_cur_work_idx = n_work_items;
					progress = nullptr;  // No longer call the callback
				}
			}

			// Make sure we see any update to the m_workers_done variable
			std::atomic_thread_fence(std::memory_order_acquire);
		}

		// Call the progress callback one last time to ensure that something
		// along the lines of "100%" done can be displayed.
		if (progress) {
			progress(n_work_items, n_work_items);
		}
	}
};

/******************************************************************************
 * Class Threadpool                                                           *
 ******************************************************************************/

Threadpool::Threadpool(unsigned int n_threads) : m_impl(new Impl(n_threads)) {}
Threadpool::~Threadpool() {}

void Threadpool::run(unsigned int n_work_items, const Kernel &kernel,
                     Progress progress)
{
	m_impl->run(n_work_items, kernel, progress);
}
