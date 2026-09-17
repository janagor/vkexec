#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP

//! \file
//! Thin Algorithm naming over heap compute pipelines (vkgsplat-oriented).

#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include <cstdint>

namespace vkexec {

/**
 * Alias for an owning bindless heap compute pipeline (Algorithm naming).
 *
 * Prefer this name when mirroring vkgsplat-style `Algorithm` usage; the
 * underlying type remains `heap_compute_pipeline`.
 *
 * @see heap_compute_pipeline, dispatch_heap, compute_heap_pass
 */
using heap_algorithm = heap_compute_pipeline;

/**
 * Builds a bindless heap compute pass for `algo` with push-data `params`.
 *
 * Equivalent to `compute_heap_pass(algo, params, work_count)`. Named
 * `dispatch_heap` (not `dispatch`) so it does not shadow `struct dispatch`.
 *
 * @see bind_heap, compute_heap_pass
 */
template<typename Params>
[[nodiscard]] auto dispatch_heap(heap_algorithm const &algo, Params const &params, std::uint32_t work_count)
  -> heap_compute_pass_closure
{ return compute_heap_pass(algo, params, work_count); }

//! Builds a bindless heap compute pass for `algo` without push-data.
[[nodiscard]] inline auto dispatch_heap(heap_algorithm const &algo, std::uint32_t work_count)
  -> heap_compute_pass_closure
{ return compute_heap_pass(algo, work_count); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP
