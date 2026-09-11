#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! `compute_heap_pass` factories for bindless / descriptor-heap compute.

#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>

#include <type_traits>

namespace vkexec {

/**
 * Builds a bindless compute pass with push-data `params` and direct dispatch.
 *
 * @param bind Bindless compute bind (typically from `heap_compute_pipeline::bind`).
 * @param params Trivially copyable push-data blob.
 * @param groups Workgroup counts.
 */
template<typename Params>
auto compute_heap_pass(compute_bind bind, Params const &params, dispatch groups) -> heap_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  return heap_compute_pass_closure{ .inner = compute_pass(bind, params, groups) };
}

/**
 * Builds a bindless compute pass with push-data `params` and indirect dispatch.
 */
template<typename Params>
auto compute_heap_pass(compute_bind bind, Params const &params, indirect_dispatch groups) -> heap_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  return heap_compute_pass_closure{ .inner = compute_pass(bind, params, groups) };
}

//! Builds a bindless compute pass without push-data (direct dispatch).
auto compute_heap_pass(compute_bind bind, dispatch groups) -> heap_compute_pass_closure;

//! Builds a bindless compute pass without push-data (indirect dispatch).
auto compute_heap_pass(compute_bind bind, indirect_dispatch groups) -> heap_compute_pass_closure;

/**
 * Bindless / descriptor-heap path: no descriptor set; push payload uses `cmd_push_data`.
 *
 * @param pipe Heap compute pipeline.
 * @param params Trivially copyable push-data blob.
 * @param work_count Invocation count along X (converted via `groups_for`).
 */
template<typename Params>
auto compute_heap_pass(heap_compute_pipeline const &pipe, Params const &params, std::uint32_t work_count)
  -> heap_compute_pass_closure
{ return compute_heap_pass(pipe.bind(), params, pipe.groups_for(work_count)); }

//! Bindless compute pass without push-data for `work_count` invocations.
auto compute_heap_pass(heap_compute_pipeline const &pipe, std::uint32_t work_count) -> heap_compute_pass_closure;

//! Bindless compute pass with push-data and indirect dispatch groups.
template<typename Params>
auto compute_heap_pass(heap_compute_pipeline const &pipe, Params const &params, indirect_dispatch groups)
  -> heap_compute_pass_closure
{ return compute_heap_pass(pipe.bind(), params, groups); }

//! Bindless compute pass without push-data and indirect dispatch groups.
auto compute_heap_pass(heap_compute_pipeline const &pipe, indirect_dispatch groups) -> heap_compute_pass_closure;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
