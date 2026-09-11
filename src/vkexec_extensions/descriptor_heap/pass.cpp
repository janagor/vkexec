#include <vkexec_extensions/descriptor_heap/pass.hpp>
#include <vkexec_extensions/descriptor_heap/push_data.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>
#include <utility>

namespace vkexec {

auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.pipeline);
  if (!push.empty()) {
    if (auto pushed = cmd_push_data(ctx, cmd, push); !pushed) { return fail(pushed); }
  }
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
  return {};
}

auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.pipeline);
  if (!push.empty()) {
    if (auto pushed = cmd_push_data(ctx, cmd, push); !pushed) { return fail(pushed); }
  }
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
  return {};
}

namespace detail {

  auto make_heap_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step
  {
    return pass_step{ .record = [closure = std::move(closure)](
                                  context &record_ctx, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
      std::span<std::byte const> const push_bytes{ closure.push };
      if (closure.is_indirect) { return record_heap_pass(record_ctx, cmd, closure.bind, push_bytes, closure.indirect); }
      return record_heap_pass(record_ctx, cmd, closure.bind, push_bytes, closure.groups);
    } };
  }

  namespace {

    auto append_heap_step(pass_graph_sender graph, pass_step step) -> pass_graph_sender
    {
      graph.steps.push_back(std::move(step));
      return graph;
    }

  }// namespace

}// namespace detail

auto operator|(schedule_sender snd, heap_compute_pass_closure closure) -> pass_graph_sender
{
  return pass_graph_sender{
    .ctx = snd.ctx,
    .steps = { detail::make_heap_prebuilt_step(std::move(closure.inner)) },
  };
}

auto operator|(pass_graph_sender graph, heap_compute_pass_closure closure) -> pass_graph_sender
{ return detail::append_heap_step(std::move(graph), detail::make_heap_prebuilt_step(std::move(closure.inner))); }

}// namespace vkexec
