#include <vkexec/bind_resources.hpp>

#include <vkexec/detail/bind_resources.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/detail/lower_and_bind_push.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace vkexec {

namespace detail {
  struct bind_resources_step_state : bound_release_state<set_descriptor_backend>
  {
  };
}// namespace detail

auto bind_resources(handles::compute_pipeline const &pipe, resource_table const &table, std::span<std::byte const> push)
  -> bind_resources_closure
{
  return bind_resources_closure{
    .pipe = &pipe,
    .table = table,
    .push = std::vector<std::byte>(push.begin(), push.end()),
    .state = {},
  };
}

auto detail::record_bind_resources_step(context &ctx,
  VkCommandBuffer cmd,
  handles::compute_pipeline const *pipe,
  resource_table const &table,
  std::span<std::byte const> push,
  std::shared_ptr<bind_resources_step_state> &state) -> status
{
  using backend = set_descriptor_backend;
  if (pipe == nullptr) { return fail(errc::invalid_argument, "bind_resources requires a pipeline"); }
  if (!state) {
    state = std::make_shared<bind_resources_step_state>();
  } else {
    state->release();
  }
  auto lowered =
    lower_and_bind_push<backend>(ctx, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *pipe, table, empty_table_lower_env{}, push);
  if (!lowered) { return fail(lowered); }
  state->ctx = &ctx;
  state->pipe = pipe;
  state->bound.emplace(expected_take(lowered));
  return {};
}

auto detail::release_bind_resources_step(std::shared_ptr<bind_resources_step_state> const &state) -> void
{
  if (state) { state->release(); }
}

auto bind_resources_closure::record(
  context &ctx, VkCommandBuffer cmd, [[maybe_unused]] detail::pass_cleanup &cleanup) -> status
{ return detail::record_bind_resources_step(ctx, cmd, pipe, table, std::span<std::byte const>{ push }, state); }

auto bind_resources_closure::after_gpu() const -> void { detail::release_bind_resources_step(state); }

}// namespace vkexec
