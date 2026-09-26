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

namespace detail {

  template<push_constant_type Push, dispatch_kind Dispatch> struct descriptor_compute_pass_data
  {
    compute_bind bind{};
    [[no_unique_address]] Push push{};
    Dispatch dispatch_info{};
  };

}// namespace detail

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind bind,
  Params const &params,
  dispatch groups) -> detail::expr_closure<compute_pass_t, detail::descriptor_compute_pass_data<Params, dispatch>>
{
  return detail::make_expr_closure(compute_pass_t{},
    detail::descriptor_compute_pass_data<Params, dispatch>{ .bind = bind, .push = params, .dispatch_info = groups });
}

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind bind,
  Params const &params,
  indirect_dispatch groups)
  -> detail::expr_closure<compute_pass_t, detail::descriptor_compute_pass_data<Params, indirect_dispatch>>
{
  return detail::make_expr_closure(compute_pass_t{},
    detail::descriptor_compute_pass_data<Params, indirect_dispatch>{
      .bind = bind, .push = params, .dispatch_info = groups });
}

[[nodiscard]] inline auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind bind,
  dispatch groups)
  -> detail::expr_closure<compute_pass_t, detail::descriptor_compute_pass_data<detail::no_push_constants, dispatch>>
{
  return detail::make_expr_closure(compute_pass_t{},
    detail::descriptor_compute_pass_data<detail::no_push_constants, dispatch>{
      .bind = bind, .push = {}, .dispatch_info = groups });
}

[[nodiscard]] inline auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind bind,
  indirect_dispatch groups) -> detail::expr_closure<compute_pass_t,
  detail::descriptor_compute_pass_data<detail::no_push_constants, indirect_dispatch>>
{
  return detail::make_expr_closure(compute_pass_t{},
    detail::descriptor_compute_pass_data<detail::no_push_constants, indirect_dispatch>{
      .bind = bind, .push = {}, .dispatch_info = groups });
}

namespace detail {

  template<push_constant_type Push, dispatch_kind Dispatch, class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(compute_pass_t /*tag*/,
    descriptor_compute_pass_data<Push, Dispatch> data,
    Env const & /*env*/) -> descriptor_compute_pass_step<Push, Dispatch>
  {
    return { .inner = compute_pass_step<Push, Dispatch>{
               .bind = data.bind, .push = std::move(data.push), .dispatch_info = data.dispatch_info } };
  }

}// namespace detail

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  Params const &params,
  std::uint32_t work_count) -> decltype(cpo(strategy, pipe.bind(), params, pipe.groups_for(work_count)))
{ return cpo(strategy, pipe.bind(), params, pipe.groups_for(work_count)); }

inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::uint32_t work_count) -> decltype(cpo(strategy, pipe.bind(), pipe.groups_for(work_count)))
{ return cpo(strategy, pipe.bind(), pipe.groups_for(work_count)); }

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  Params const &params,
  indirect_dispatch groups) -> decltype(cpo(strategy, pipe.bind(), params, groups))
{ return cpo(strategy, pipe.bind(), params, groups); }

inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  indirect_dispatch groups) -> decltype(cpo(strategy, pipe.bind(), groups))
{ return cpo(strategy, pipe.bind(), groups); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
