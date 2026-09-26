#ifndef VKEXEC_BIND_RESOURCES_HPP
#define VKEXEC_BIND_RESOURCES_HPP

//! \file
//! Pipeable resource-table lowering and descriptor binding.

#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/scheduler.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace detail {
  struct bind_resources_step_state;

  [[nodiscard]] auto record_bind_resources_step(context &ctx,
    VkCommandBuffer cmd,
    handles::compute_pipeline const *pipe,
    resource_table const &table,
    std::span<std::byte const> push,
    std::shared_ptr<bind_resources_step_state> &state) -> status;

  auto release_bind_resources_step(std::shared_ptr<bind_resources_step_state> const &state) -> void;
}// namespace detail

//! Descriptor-set resource binding step for a pass graph.
struct bind_resources_closure
{
  handles::compute_pipeline const *pipe{ nullptr };
  resource_table table;
  std::vector<std::byte> push;
  std::shared_ptr<detail::bind_resources_step_state> state;

  auto record(context &ctx, VkCommandBuffer cmd, detail::pass_cleanup &cleanup) -> status;
  auto after_gpu() const -> void;
};

template<detail::push_constant_type Push> struct bind_resources_step
{
  handles::compute_pipeline const *pipe{ nullptr };
  resource_table table;
  [[no_unique_address]] Push push{};
  std::shared_ptr<detail::bind_resources_step_state> state;

  auto record(context &ctx, VkCommandBuffer cmd, detail::pass_cleanup & /*cleanup*/) -> status
  { return detail::record_bind_resources_step(ctx, cmd, pipe, table, detail::push_bytes(push), state); }

  auto after_gpu() const -> void { detail::release_bind_resources_step(state); }
};

//! Builds a descriptor-set-backed resource-table graph step with runtime-sized push data.
[[nodiscard]] auto bind_resources(handles::compute_pipeline const &pipe,
  resource_table const &table,
  std::span<std::byte const> push) -> bind_resources_closure;

template<std::size_t Extent>
[[nodiscard]] auto bind_resources(handles::compute_pipeline const &pipe,
  resource_table const &table,
  std::span<std::byte const, Extent> push) -> bind_resources_closure
{ return bind_resources(pipe, table, std::span<std::byte const>{ push }); }

template<std::size_t Extent>
[[nodiscard]] auto bind_resources(handles::compute_pipeline const &pipe,
  resource_table const &table,
  std::span<std::byte, Extent> push) -> bind_resources_closure
{ return bind_resources(pipe, table, std::span<std::byte const>{ push }); }

//! Builds a descriptor-set-backed resource-table graph step without push data.
[[nodiscard]] inline auto bind_resources(handles::compute_pipeline const &pipe, resource_table const &table)
{ return bind_resources_step<detail::no_push_constants>{ .pipe = &pipe, .table = table, .push = {}, .state = {} }; }

//! Typed push-data overload for descriptor-set resource binding.
template<class Params>
  requires std::is_trivially_copyable_v<Params> && (!detail::is_byte_span_v<Params>)
[[nodiscard]] auto
  bind_resources(handles::compute_pipeline const &pipe, resource_table const &table, Params const &params)
{ return bind_resources_step<Params>{ .pipe = &pipe, .table = table, .push = params, .state = {} }; }

}// namespace vkexec

#endif// VKEXEC_BIND_RESOURCES_HPP
