#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP

//! \file
//! Thin Kompute/vkgsplat-style Algorithm naming over heap compute pipelines.

#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include <cstdint>

namespace vkexec {

/**
 * Alias for an owning bindless heap compute pipeline (Algorithm naming).
 *
 * Prefer this name when mirroring Kompute/vkgsplat `Algorithm` usage; the
 * underlying type remains `heap_compute_pipeline`.
 *
 * @see heap_compute_pipeline, dispatch, compute_heap_pass
 */
using heap_algorithm = heap_compute_pipeline;

/**
 * Builds a bindless dispatch pass for `algo` with push-data `params`.
 *
 * Equivalent to `compute_heap_pass(algo, params, work_count)`.
 */
template<typename Params>
[[nodiscard]] auto dispatch(heap_algorithm const &algo, Params const &params, std::uint32_t work_count)
  -> heap_compute_pass_closure
{ return compute_heap_pass(algo, params, work_count); }

//! Builds a bindless dispatch pass for `algo` without push-data.
[[nodiscard]] inline auto dispatch(heap_algorithm const &algo, std::uint32_t work_count) -> heap_compute_pass_closure
{ return compute_heap_pass(algo, work_count); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_ALGORITHM_HPP
