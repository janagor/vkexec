#ifndef VKEXEC_BIND_RESOURCES_HPP
#define VKEXEC_BIND_RESOURCES_HPP

//! \file
//! Pipeable resource-table lowering and descriptor binding.

#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/detail/lower_and_bind_push.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>

#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

//! A graph step that lowers a table and records backend-specific bind/push commands.
template<class Backend>
  requires detail::descriptor_backend<Backend> && detail::descriptor_table_backend<Backend>
struct bind_resources_closure
{
  pipeline_resources const *pipe{ nullptr };
  resource_table table;
  Backend::lower_env env;
  std::vector<std::byte> push;
};

/**
 * Builds a pipeable table-lowering step. The lowered binding is released after GPU completion.
 *
 * The default backend uses descriptor sets. Extension users select another backend explicitly
 * or use a strategy-tag overload supplied by that extension.
 */
template<class Backend = detail::set_descriptor_backend>
  requires detail::descriptor_backend<Backend> && detail::descriptor_table_backend<Backend>
[[nodiscard]] auto bind_resources(pipeline_resources const &pipe,
  resource_table const &table,
  typename Backend::lower_env env,
  std::span<std::byte const> push = {}) -> bind_resources_closure<Backend>
{
  return bind_resources_closure<Backend>{
    .pipe = &pipe, .table = table, .env = env, .push = { push.begin(), push.end() }
  };
}

//! Typed push-data overload for `bind_resources`.
template<class Backend = detail::set_descriptor_backend, class Params>
  requires detail::descriptor_backend<Backend> && detail::descriptor_table_backend<Backend>
           && std::is_trivially_copyable_v<Params>
           && (!std::same_as<std::remove_cvref_t<Params>, std::span<std::byte const>>)
[[nodiscard]] auto bind_resources(pipeline_resources const &pipe,
  resource_table const &table,
  typename Backend::lower_env env,
  Params const &params) -> bind_resources_closure<Backend>
{
  auto const bytes = std::as_bytes(std::span{ &params, 1 });
  return bind_resources<Backend>(pipe, table, env, bytes);
}

namespace detail {

  template<class Backend> struct bound_release_state
  {
    bound_release_state() = default;
    bound_release_state(bound_release_state const &) = delete;
    auto operator=(bound_release_state const &) -> bound_release_state & = delete;
    bound_release_state(bound_release_state &&) = delete;
    auto operator=(bound_release_state &&) -> bound_release_state & = delete;

    context *ctx{ nullptr };
    pipeline_resources const *pipe{ nullptr };
    std::optional<typename Backend::bound_type> bound;

    auto release() noexcept -> void
    {
      if (ctx != nullptr && pipe != nullptr && bound.has_value()) {
        Backend::release(*ctx, *pipe, *bound);
        bound.reset();
      }
    }

    ~bound_release_state() { release(); }
  };

  template<class Backend>
    requires descriptor_backend<Backend> && descriptor_table_backend<Backend>
  [[nodiscard]] auto make_bind_resources_step(bind_resources_closure<Backend> closure) -> pass_step
  {
    auto state = std::make_shared<bound_release_state<Backend>>();
    return pass_step{
      .record = [closure = std::move(closure), state](
                  context &ctx, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) mutable -> status {
        if (closure.pipe == nullptr) { return fail(errc::invalid_argument, "bind_resources requires a pipeline"); }
        state->release();
        auto lowered = lower_and_bind_push<Backend>(
          ctx, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *closure.pipe, closure.table, closure.env, closure.push);
        if (!lowered) { return fail(lowered); }
        state->ctx = &ctx;
        state->pipe = closure.pipe;
        state->bound.emplace(expected_take(lowered));
        return {};
      },
      .after_gpu = [state]() -> void { state->release(); },
    };
  }

}// namespace detail

//! Starts a pass graph with one resource-table lowering step.
template<class Backend>
[[nodiscard]] auto operator|(schedule_sender snd, bind_resources_closure<Backend> closure) -> pass_graph_sender
{
  return pass_graph_sender{
    .ctx = snd.ctx,
    .steps = { detail::make_bind_resources_step(std::move(closure)) },
  };
}

//! Appends a resource-table lowering step to an existing pass graph.
template<class Backend>
[[nodiscard]] auto operator|(pass_graph_sender graph, bind_resources_closure<Backend> closure) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_bind_resources_step(std::move(closure))); }

}// namespace vkexec

#endif// VKEXEC_BIND_RESOURCES_HPP
