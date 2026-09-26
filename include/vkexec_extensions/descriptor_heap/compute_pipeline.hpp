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
#include <utility>

namespace vkexec {

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind bind,
  Params const &params,
  dispatch groups)
{
  auto step = descriptor_compute_pass_step<Params, dispatch>{
    .inner = detail::make_compute_pass_step(bind, params, groups) };
  return make_pass_adaptor(std::move(step));
}

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind bind,
  Params const &params,
  indirect_dispatch groups)
{
  auto step = descriptor_compute_pass_step<Params, indirect_dispatch>{
    .inner = detail::make_compute_pass_step(bind, params, groups) };
  return make_pass_adaptor(std::move(step));
}

[[nodiscard]] inline auto compute_pass_custom(
  compute_pass_t const & /*cpo*/, descriptor_heap_t /*strategy*/, compute_bind bind, dispatch groups)
{
  auto step = descriptor_compute_pass_step<detail::no_push_constants, dispatch>{
    .inner = detail::make_compute_pass_step(bind, detail::no_push_constants{}, groups) };
  return make_pass_adaptor(step);
}

[[nodiscard]] inline auto compute_pass_custom(
  compute_pass_t const & /*cpo*/, descriptor_heap_t /*strategy*/, compute_bind bind, indirect_dispatch groups)
{
  auto step = descriptor_compute_pass_step<detail::no_push_constants, indirect_dispatch>{
    .inner = detail::make_compute_pass_step(bind, detail::no_push_constants{}, groups) };
  return make_pass_adaptor(step);
}

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  Params const &params,
  std::uint32_t work_count)
{ return cpo(strategy, pipe.bind(), params, pipe.groups_for(work_count)); }

inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::uint32_t work_count)
{ return cpo(strategy, pipe.bind(), pipe.groups_for(work_count)); }

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  Params const &params,
  indirect_dispatch groups)
{ return cpo(strategy, pipe.bind(), params, groups); }

inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  indirect_dispatch groups)
{ return cpo(strategy, pipe.bind(), groups); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
