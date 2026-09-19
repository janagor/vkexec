#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! Descriptor-strategy overloads for `compute_pass`.

#include <vkexec/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <cstdint>
#include <type_traits>

namespace vkexec {

template<typename Params>
auto compute_pass(descriptor_heap_t /*strategy*/, compute_bind bind, Params const &params, dispatch groups)
  -> descriptor_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  return descriptor_compute_pass_closure{ .inner = compute_pass(bind, params, groups) };
}

template<typename Params>
auto compute_pass(descriptor_heap_t /*strategy*/, compute_bind bind, Params const &params, indirect_dispatch groups)
  -> descriptor_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  return descriptor_compute_pass_closure{ .inner = compute_pass(bind, params, groups) };
}

auto compute_pass(descriptor_heap_t strategy, compute_bind bind, dispatch groups) -> descriptor_compute_pass_closure;
auto compute_pass(descriptor_heap_t strategy, compute_bind bind, indirect_dispatch groups)
  -> descriptor_compute_pass_closure;

template<typename Params>
auto compute_pass(descriptor_heap_t strategy,
  compute_pipeline const &pipe,
  Params const &params,
  std::uint32_t work_count) -> descriptor_compute_pass_closure
{ return compute_pass(strategy, pipe.bind(), params, pipe.groups_for(work_count)); }

inline auto compute_pass(descriptor_heap_t strategy, compute_pipeline const &pipe, std::uint32_t work_count)
  -> descriptor_compute_pass_closure
{ return compute_pass(strategy, pipe.bind(), pipe.groups_for(work_count)); }

template<typename Params>
auto compute_pass(descriptor_heap_t strategy,
  compute_pipeline const &pipe,
  Params const &params,
  indirect_dispatch groups) -> descriptor_compute_pass_closure
{ return compute_pass(strategy, pipe.bind(), params, groups); }

inline auto compute_pass(descriptor_heap_t strategy, compute_pipeline const &pipe, indirect_dispatch groups)
  -> descriptor_compute_pass_closure
{ return compute_pass(strategy, pipe.bind(), groups); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
