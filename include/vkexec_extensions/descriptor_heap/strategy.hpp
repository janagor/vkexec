#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP

#include <vkexec/bind_resources.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_extensions/descriptor_heap/resource_table.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace detail {
  struct descriptor_heap_bind_resources_step_state;

  struct descriptor_heap_dynamic_bind_resources_data
  {
    handles::compute_pipeline const *pipe{ nullptr };
    resource_table table;
    heap_table_lower_env env;
    std::vector<std::byte> push;
  };

  template<push_constant_type Push> struct descriptor_heap_bind_resources_data
  {
    handles::compute_pipeline const *pipe{ nullptr };
    resource_table table;
    heap_table_lower_env env;
    [[no_unique_address]] Push push{};
  };

  [[nodiscard]] auto record_descriptor_heap_bind_resources_step(context &ctx,
    VkCommandBuffer cmd,
    handles::compute_pipeline const *pipe,
    resource_table const &table,
    heap_table_lower_env env,
    std::span<std::byte const> push,
    std::shared_ptr<descriptor_heap_bind_resources_step_state> &state) -> status;

  auto release_descriptor_heap_bind_resources_step(
    std::shared_ptr<descriptor_heap_bind_resources_step_state> const &state) -> void;
}// namespace detail

struct descriptor_heap_t
{
};
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr descriptor_heap_t descriptor_heap{};

//! Descriptor-heap resource binding step for a pass graph.
struct descriptor_heap_runtime_bind_resources_step
{
  handles::compute_pipeline const *pipe{ nullptr };
  resource_table table;
  heap_table_lower_env env;
  std::vector<std::byte> push;
  std::shared_ptr<detail::descriptor_heap_bind_resources_step_state> state;

  auto record(context &ctx, VkCommandBuffer cmd, detail::pass_cleanup &cleanup) -> status;
  auto after_gpu() const -> void;
};

template<detail::push_constant_type Push> struct descriptor_heap_bind_resources_step
{
  handles::compute_pipeline const *pipe{ nullptr };
  resource_table table;
  heap_table_lower_env env;
  [[no_unique_address]] Push push{};
  std::shared_ptr<detail::descriptor_heap_bind_resources_step_state> state;

  auto record(context &ctx, VkCommandBuffer cmd, detail::pass_cleanup & /*cleanup*/) -> status
  {
    return detail::record_descriptor_heap_bind_resources_step(
      ctx, cmd, pipe, table, env, detail::push_bytes(push), state);
  }

  auto after_gpu() const -> void { detail::release_descriptor_heap_bind_resources_step(state); }
};

[[nodiscard]] auto make_descriptor_heap_bind_resources_step(descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const> push) -> descriptor_heap_runtime_bind_resources_step;

template<std::size_t Extent>
[[nodiscard]] auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const, Extent> push)
  -> detail::expr_closure<bind_resources_t, detail::descriptor_heap_dynamic_bind_resources_data>
{
  return detail::make_expr_closure(bind_resources_t{},
    detail::descriptor_heap_dynamic_bind_resources_data{
      .pipe = &pipe, .table = table, .env = env, .push = { push.begin(), push.end() } });
}

template<std::size_t Extent>
[[nodiscard]] auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t strategy,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte, Extent> push)
  -> detail::expr_closure<bind_resources_t, detail::descriptor_heap_dynamic_bind_resources_data>
{ return bind_resources_custom(bind_resources_t{}, strategy, pipe, table, env, std::span<std::byte const>{ push }); }

[[nodiscard]] inline auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const> push)
  -> detail::expr_closure<bind_resources_t, detail::descriptor_heap_dynamic_bind_resources_data>
{
  return detail::make_expr_closure(bind_resources_t{},
    detail::descriptor_heap_dynamic_bind_resources_data{
      .pipe = &pipe, .table = table, .env = env, .push = { push.begin(), push.end() } });
}

[[nodiscard]] inline auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env)
  -> detail::expr_closure<bind_resources_t, detail::descriptor_heap_bind_resources_data<detail::no_push_constants>>
{
  return detail::make_expr_closure(bind_resources_t{},
    detail::descriptor_heap_bind_resources_data<detail::no_push_constants>{
      .pipe = &pipe, .table = table, .env = env, .push = {} });
}

template<class Params>
  requires std::is_trivially_copyable_v<Params> && (!detail::is_byte_span_v<Params>)
[[nodiscard]] auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  Params const &params) -> detail::expr_closure<bind_resources_t, detail::descriptor_heap_bind_resources_data<Params>>
{
  return detail::make_expr_closure(bind_resources_t{},
    detail::descriptor_heap_bind_resources_data<Params>{ .pipe = &pipe, .table = table, .env = env, .push = params });
}

namespace detail {

  template<class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(bind_resources_t /*tag*/,
    descriptor_heap_dynamic_bind_resources_data data,
    Env const & /*env*/) -> descriptor_heap_runtime_bind_resources_step
  {
    return {
      .pipe = data.pipe, .table = std::move(data.table), .env = data.env, .push = std::move(data.push), .state = {}
    };
  }

  template<push_constant_type Push, class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(bind_resources_t /*tag*/,
    descriptor_heap_bind_resources_data<Push> data,
    Env const & /*env*/) -> descriptor_heap_bind_resources_step<Push>
  {
    return {
      .pipe = data.pipe, .table = std::move(data.table), .env = data.env, .push = std::move(data.push), .state = {}
    };
  }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
