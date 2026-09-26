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
  descriptor_heap_t strategy,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const, Extent> push)
{
  return make_pass_adaptor(
    make_descriptor_heap_bind_resources_step(strategy, pipe, table, env, std::span<std::byte const>{ push }));
}

template<std::size_t Extent>
[[nodiscard]] auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t strategy,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte, Extent> push)
{
  return make_pass_adaptor(
    make_descriptor_heap_bind_resources_step(strategy, pipe, table, env, std::span<std::byte const>{ push }));
}

[[nodiscard]] inline auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t strategy,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const> push)
{ return make_pass_adaptor(make_descriptor_heap_bind_resources_step(strategy, pipe, table, env, push)); }

[[nodiscard]] inline auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env)
{
  return make_pass_adaptor(descriptor_heap_bind_resources_step<detail::no_push_constants>{
    .pipe = &pipe, .table = table, .env = env, .push = {}, .state = {} });
}

template<class Params>
  requires std::is_trivially_copyable_v<Params> && (!detail::is_byte_span_v<Params>)
[[nodiscard]] auto bind_resources_custom(bind_resources_t const & /*cpo*/,
  descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  Params const &params)
{
  return make_pass_adaptor(descriptor_heap_bind_resources_step<Params>{
    .pipe = &pipe, .table = table, .env = env, .push = params, .state = {} });
}

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
