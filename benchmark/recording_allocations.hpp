#ifndef VKEXEC_BENCHMARK_RECORDING_ALLOCATIONS_HPP
#define VKEXEC_BENCHMARK_RECORDING_ALLOCATIONS_HPP

#include <cstddef>

namespace recording_allocations {
auto begin() noexcept -> void;
auto end() noexcept -> std::size_t;
}// namespace recording_allocations

#endif
