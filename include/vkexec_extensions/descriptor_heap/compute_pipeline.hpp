#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP

#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>

#include <type_traits>

namespace vkexec {

template<typename Params>
auto compute_heap_pass(compute_bind bind, Params const &params, dispatch groups) -> heap_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  return heap_compute_pass_closure{ .inner = compute_pass(bind, params, groups) };
}

template<typename Params>
auto compute_heap_pass(compute_bind bind, Params const &params, indirect_dispatch groups)
  -> heap_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  return heap_compute_pass_closure{ .inner = compute_pass(bind, params, groups) };
}

auto compute_heap_pass(compute_bind bind, dispatch groups) -> heap_compute_pass_closure;

auto compute_heap_pass(compute_bind bind, indirect_dispatch groups) -> heap_compute_pass_closure;

/// Bindless / descriptor-heap path: no descriptor set; push payload uses `cmd_push_data`.
template<typename Params>
auto compute_heap_pass(heap_compute_pipeline const &pipe, Params const &params, std::uint32_t work_count)
  -> heap_compute_pass_closure
{ return compute_heap_pass(pipe.bind(), params, pipe.groups_for(work_count)); }

auto compute_heap_pass(heap_compute_pipeline const &pipe, std::uint32_t work_count) -> heap_compute_pass_closure;

template<typename Params>
auto compute_heap_pass(heap_compute_pipeline const &pipe, Params const &params, indirect_dispatch groups)
  -> heap_compute_pass_closure
{ return compute_heap_pass(pipe.bind(), params, groups); }

auto compute_heap_pass(heap_compute_pipeline const &pipe, indirect_dispatch groups) -> heap_compute_pass_closure;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
