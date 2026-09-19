#include <vkexec/pass.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/record_with_binding.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vkexec {

auto bind_compute(pipeline_resources const &pipe, VkDescriptorSet set) -> compute_bind
{ return compute_bind{ .pipeline = pipe.pipeline, .layout = pipe.pipeline_layout, .set = set }; }

auto record_pass(VkCommandBuffer cmd, compute_bind bind, void const *push, std::uint32_t push_bytes, dispatch groups)
  -> void
{
  auto const *const bytes = static_cast<std::byte const *>(push);
  std::span<std::byte const> const push_data{ bytes, push_bytes };
  (void)detail::bind_and_push<detail::set_descriptor_backend>(
    nullptr, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push_data);
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
}

auto record_pass(VkCommandBuffer cmd,
  compute_bind bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void
{
  auto const *const bytes = static_cast<std::byte const *>(push);
  std::span<std::byte const> const push_data{ bytes, push_bytes };
  (void)detail::bind_and_push<detail::set_descriptor_backend>(
    nullptr, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push_data);
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
}

auto record_pass(VkCommandBuffer cmd,
  pipeline_resources const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void
{ record_pass(cmd, bind_compute(pipe, set), push, push_bytes, groups); }

auto compute_pass(compute_bind bind, dispatch groups) -> prebuilt_compute_pass_closure
{
  prebuilt_compute_pass_closure closure{};
  closure.bind = bind;
  closure.groups = groups;
  return closure;
}

auto compute_pass(compute_bind bind, indirect_dispatch groups) -> prebuilt_compute_pass_closure
{
  prebuilt_compute_pass_closure closure{};
  closure.bind = bind;
  closure.indirect = groups;
  closure.is_indirect = true;
  return closure;
}

namespace detail {

  auto record_pass_steps(submit_scope &scope, std::span<pass_step const> steps) -> detail::status
  {
    for (pass_step const &step : steps) {
      if (auto recorded = step.record(*scope.ctx, scope.cmd, scope.cleanup); !recorded) {
        return detail::fail(recorded);
      }
    }
    return scope.end_recording();
  }

  auto open_and_record_pass(context *ctx, std::span<pass_step const> steps) -> detail::result<submit_scope>
  {
    auto opened = submit_scope::open(*ctx);
    if (!opened) { return detail::fail(opened); }
    submit_scope scope = detail::expected_take(opened);
    // Release loans on record failure so callers never see a half-open scope.
    if (auto recorded = record_pass_steps(scope, steps); !recorded) {
      scope.release();
      return detail::fail(recorded);
    }
    return scope;
  }

  auto make_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step
  {
    return pass_step{ .record = [closure = std::move(closure)](
                                  context & /*record_ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
                       void const *push_ptr =
                         closure.push.empty() ? nullptr : static_cast<void const *>(closure.push.data());
                       auto const push_size = static_cast<std::uint32_t>(closure.push.size());
                       if (closure.is_indirect) {
                         record_pass(cmd, closure.bind, push_ptr, push_size, closure.indirect);
                       } else {
                         record_pass(cmd, closure.bind, push_ptr, push_size, closure.groups);
                       }
                       return {};
                     },
      .after_gpu = {} };
  }

  auto append_step(pass_graph_sender graph, pass_step step) -> pass_graph_sender
  {
    graph.steps.push_back(std::move(step));
    return graph;
  }

}// namespace detail

auto operator|(schedule_sender snd, prebuilt_compute_pass_closure closure) -> pass_graph_sender
{ return pass_graph_sender{ .ctx = snd.ctx, .steps = { detail::make_prebuilt_step(std::move(closure)) } }; }

auto operator|(pass_graph_sender graph, prebuilt_compute_pass_closure closure) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_prebuilt_step(std::move(closure))); }

auto operator|(pass_graph_sender graph, barrier::transfer_to_compute_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

auto operator|(pass_graph_sender graph, barrier::compute_to_compute_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

auto operator|(pass_graph_sender graph, barrier::compute_to_graphics_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

auto operator|(pass_graph_sender graph, barrier::graphics_to_compute_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

auto operator|(pass_graph_sender graph, barrier::compute_read_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

auto operator|(pass_graph_sender &&snd, submit_t /*tag*/) -> pass_graph_async_sender
{ return pass_graph_async_sender{ std::move(snd) }; }

auto operator|(pass_graph_sender &snd, submit_t /*tag*/) -> pass_graph_async_sender
{ return pass_graph_async_sender{ snd.ctx, snd.steps }; }

}// namespace vkexec
