#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP

//! \file
//! Algorithm naming over backend-neutral compute pipelines.

#include <vkexec/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <cstdint>

namespace vkexec {

//! Backend-neutral owning compute algorithm.
using algorithm = compute_pipeline;

template<typename Params>
[[nodiscard]] auto dispatch_compute(
  descriptor_heap_t strategy, algorithm const &algo, Params const &params, std::uint32_t work_count)
  -> descriptor_compute_pass_closure
{ return compute_pass(strategy, algo, params, work_count); }

[[nodiscard]] inline auto dispatch_compute(
  descriptor_heap_t strategy, algorithm const &algo, std::uint32_t work_count) -> descriptor_compute_pass_closure
{ return compute_pass(strategy, algo, work_count); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP
