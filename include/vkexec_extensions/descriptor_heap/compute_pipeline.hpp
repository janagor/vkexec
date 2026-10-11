#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! Descriptor-strategy overloads for `compute_pass`.
//! Heap shaders require complete `image_use` and `buffer_use` declarations for
//! every potentially indexed descriptor before graph planning. The overloads
//! taking vectors make this explicit; empty vectors assert no resource access.

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/detail/attributes.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace detail {

  template<push_constant_type Push, dispatch_kind Dispatch> struct descriptor_compute_pass_data
  {
    compute_bind bind{};
    VKEXEC_NO_UNIQUE_ADDRESS Push push{};
    Dispatch dispatch_info{};
  };

}// namespace detail

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind const &bind,
  Params const &params,
  dispatch groups) -> detail::expr_closure<compute_pass_t, detail::descriptor_compute_pass_data<Params, dispatch>>
{
  return detail::make_expr_closure(compute_pass_t{},
    detail::descriptor_compute_pass_data<Params, dispatch>{ .bind = bind, .push = params, .dispatch_info = groups });
}

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind const &bind,
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
  compute_bind const &bind,
  dispatch groups)
  -> detail::expr_closure<compute_pass_t, detail::descriptor_compute_pass_data<detail::no_push_constants, dispatch>>
{
  return detail::make_expr_closure(compute_pass_t{},
    detail::descriptor_compute_pass_data<detail::no_push_constants, dispatch>{
      .bind = bind, .push = {}, .dispatch_info = groups });
}

[[nodiscard]] inline auto compute_pass_custom(compute_pass_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  compute_bind const &bind,
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
               .bind = std::move(data.bind), .push = std::move(data.push), .dispatch_info = data.dispatch_info } };
  }

}// namespace detail

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  Params const &params,
  std::uint32_t work_count) -> decltype(cpo(strategy, pipe.bind(), params, pipe.groups_for(work_count)))
{ return cpo(strategy, pipe.bind(), params, pipe.groups_for(work_count)); }

template<detail::push_constant_type Params>
[[nodiscard]] auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::vector<image_use> images,
  std::vector<buffer_use> buffers,
  Params const &params,
  std::uint32_t work_count) -> decltype(cpo(strategy,
  pipe.bind().with_resources(std::move(images), std::move(buffers)),
  params,
  pipe.groups_for(work_count)))
{
  return cpo(
    strategy, pipe.bind().with_resources(std::move(images), std::move(buffers)), params, pipe.groups_for(work_count));
}

inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::uint32_t work_count) -> decltype(cpo(strategy, pipe.bind(), pipe.groups_for(work_count)))
{ return cpo(strategy, pipe.bind(), pipe.groups_for(work_count)); }

[[nodiscard]] inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::vector<image_use> images,
  std::vector<buffer_use> buffers,
  std::uint32_t work_count) -> decltype(cpo(strategy,
  pipe.bind().with_resources(std::move(images), std::move(buffers)),
  pipe.groups_for(work_count)))
{
  return cpo(strategy, pipe.bind().with_resources(std::move(images), std::move(buffers)), pipe.groups_for(work_count));
}

template<detail::push_constant_type Params>
auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  Params const &params,
  indirect_dispatch groups) -> decltype(cpo(strategy, pipe.bind(), params, groups))
{ return cpo(strategy, pipe.bind(), params, groups); }

template<detail::push_constant_type Params>
[[nodiscard]] auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::vector<image_use> images,
  std::vector<buffer_use> buffers,
  Params const &params,
  indirect_dispatch groups)
  -> decltype(cpo(strategy, pipe.bind().with_resources(std::move(images), std::move(buffers)), params, groups))
{ return cpo(strategy, pipe.bind().with_resources(std::move(images), std::move(buffers)), params, groups); }

inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  indirect_dispatch groups) -> decltype(cpo(strategy, pipe.bind(), groups))
{ return cpo(strategy, pipe.bind(), groups); }

[[nodiscard]] inline auto compute_pass_custom(compute_pass_t const &cpo,
  descriptor_heap_t strategy,
  owned::compute_pipeline const &pipe,
  std::vector<image_use> images,
  std::vector<buffer_use> buffers,
  indirect_dispatch groups)
  -> decltype(cpo(strategy, pipe.bind().with_resources(std::move(images), std::move(buffers)), groups))
{ return cpo(strategy, pipe.bind().with_resources(std::move(images), std::move(buffers)), groups); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_COMPUTE_PIPELINE_HPP
