#ifndef VKEXEC_PASS_HPP
#define VKEXEC_PASS_HPP

#include <vkexec/barrier.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/detail/submit_scope.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

struct dispatch
{
  std::uint32_t x{ 1 };
  std::uint32_t y{ 1 };
  std::uint32_t z{ 1 };
};

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr auto dispatch_groups_for(std::uint32_t work_count, std::uint32_t local_x) noexcept -> dispatch
{
  std::uint32_t const group_size = local_x == 0U ? 1U : local_x;
  return dispatch{ .x = (work_count + group_size - 1U) / group_size };
}
// NOLINTEND(bugprone-easily-swappable-parameters)

struct indirect_dispatch
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{ 0 };
};

struct compute_bind
{
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkPipelineLayout layout{ VK_NULL_HANDLE };
  VkDescriptorSet set{ VK_NULL_HANDLE };
};

[[nodiscard]] inline auto bind_compute(pipeline_resources const &pipe, VkDescriptorSet set = VK_NULL_HANDLE)
  -> compute_bind
{ return compute_bind{ .pipeline = pipe.pipeline, .layout = pipe.pipeline_layout, .set = set }; }

inline auto
  record_pass(VkCommandBuffer cmd, compute_bind bind, void const *push, std::uint32_t push_bytes, dispatch groups)
    -> void
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.pipeline);
  if (bind.set != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.layout, 0, 1, &bind.set, 0, nullptr);
  }
  upload_push_constants(cmd, bind.layout, push, push_bytes);
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
}

inline auto record_pass(VkCommandBuffer cmd,
  compute_bind bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.pipeline);
  if (bind.set != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.layout, 0, 1, &bind.set, 0, nullptr);
  }
  upload_push_constants(cmd, bind.layout, push, push_bytes);
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
}

inline auto record_pass(VkCommandBuffer cmd,
  pipeline_resources const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void
{ record_pass(cmd, bind_compute(pipe, set), push, push_bytes, groups); }

struct pass_step
{
  std::function<void(context &, VkCommandBuffer, detail::pass_cleanup &)> record;
};

struct pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

  context *ctx{ nullptr };
  std::vector<pass_step> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    context *ctx{};
    std::vector<pass_step> steps;
    Receiver receiver;

    auto start() noexcept -> void
    {
      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        run();
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
    }

    auto run() -> void
    {
      detail::submit_scope scope = detail::submit_scope::open(*ctx);
      for (pass_step const &step : steps) { step.record(*ctx, scope.cmd, scope.cleanup); }
      scope.end_recording();
      ctx->submit_and_wait(scope.cmd);
      scope.release();
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  { return op_state<Receiver>{ ctx, steps, std::move(receiver) }; }
};

struct pass_graph_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::vector<pass_step> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  explicit pass_graph_async_sender(pass_graph_sender graph) : ctx(graph.ctx), steps(std::move(graph.steps)) {}

  pass_graph_async_sender(context *host, std::vector<pass_step> graph_steps) : ctx(host), steps(std::move(graph_steps))
  {}

  template<class Receiver> struct op_state
  {
    context *ctx{};
    std::vector<pass_step> steps;
    Receiver receiver;

    auto start() noexcept -> void
    {
      Receiver rcvr = std::move(receiver);
      auto const token = ex::get_stop_token(ex::get_env(rcvr));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(rcvr));
          return;
        }
      }

      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        detail::submit_scope scope = detail::submit_scope::open(*ctx);
        for (pass_step const &step : steps) { step.record(*ctx, scope.cmd, scope.cleanup); }
        scope.end_recording();

        VkFence fence{ VK_NULL_HANDLE };
        VkSemaphore done{ VK_NULL_HANDLE };
        VKEXEC_TRY
        {
          done = ctx->submit_async(scope.cmd, &fence);// NOLINT(misc-misplaced-const)
          ctx->enqueue_fence_wait(done,
            fence,
            token,
            [scope = std::move(scope), rcvr = std::move(rcvr)](std::exception_ptr wait_error, bool stopped) mutable
              -> void { detail::release_scope_and_complete(scope, std::move(rcvr), std::move(wait_error), stopped); });
          return;
        }
        VKEXEC_CATCH_ALL
        {
          detail::reclaim_submission_sync(ctx->device(), ctx->compute_queue(), done, fence);
          scope.release();
          // cppcheck-suppress rethrowNoCurrentException
          throw;
        }
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) { ex::set_error(std::move(rcvr), error); }
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      self.ctx,
      std::forward_like<decltype(self)>(self.steps),
      std::move(receiver),
    };
  }
};

template<typename Params, typename Fun> struct compute_pass_closure
{
  std::uint32_t shape{};
  Params params{};
  Fun fun{};
};

struct prebuilt_compute_pass_closure
{
  compute_bind bind{};
  std::vector<std::byte> push;
  dispatch groups{};
  indirect_dispatch indirect{};
  bool is_indirect{ false };
};

template<typename Params, typename Fun>
auto compute_pass(std::uint32_t shape, Params params, Fun fun) -> compute_pass_closure<Params, Fun>
{ return compute_pass_closure<Params, Fun>{ shape, std::move(params), std::move(fun) }; }

template<typename Params>
auto compute_pass(compute_bind bind, Params const &params, dispatch groups) -> prebuilt_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  prebuilt_compute_pass_closure closure{};
  closure.bind = bind;
  closure.groups = groups;
  closure.push.resize(sizeof(Params));
  std::memcpy(closure.push.data(), &params, sizeof(Params));
  return closure;
}

template<typename Params>
auto compute_pass(compute_bind bind, Params const &params, indirect_dispatch groups) -> prebuilt_compute_pass_closure
{
  static_assert(std::is_trivially_copyable_v<Params>);
  prebuilt_compute_pass_closure closure{};
  closure.bind = bind;
  closure.indirect = groups;
  closure.is_indirect = true;
  closure.push.resize(sizeof(Params));
  std::memcpy(closure.push.data(), &params, sizeof(Params));
  return closure;
}

inline auto compute_pass(compute_bind bind, dispatch groups) -> prebuilt_compute_pass_closure
{
  prebuilt_compute_pass_closure closure{};
  closure.bind = bind;
  closure.groups = groups;
  return closure;
}

inline auto compute_pass(compute_bind bind, indirect_dispatch groups) -> prebuilt_compute_pass_closure
{
  prebuilt_compute_pass_closure closure{};
  closure.bind = bind;
  closure.indirect = groups;
  closure.is_indirect = true;
  return closure;
}

template<typename Params>
auto compute_pass(pipeline_resources &pipe, VkDescriptorSet set, Params const &params, dispatch groups)
  -> prebuilt_compute_pass_closure
{ return compute_pass(bind_compute(pipe, set), params, groups); }

namespace detail {

  template<typename Params, typename Fun> auto make_traced_step(compute_pass_closure<Params, Fun> closure) -> pass_step
  {
    return pass_step{ .record = [closure = std::move(closure)](
                                  context &record_ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> void {
      edsl::trace_scope const scope;
      edsl::Int const idx = edsl::Int::param_index();
      auto push = edsl::push_constant<Params>::bind();
      closure.fun(idx, push);

      pipeline_resources &pipe = record_ctx.get_or_compile(scope, closure.shape);
      std::vector<edsl::storage_trace> const buffers = scope.buffers();
      auto const local = scope.local_size_x();

      VkDescriptorSet set = bind_or_allocate_set(record_ctx, pipe, buffers, cleanup);
      std::uint32_t const groups = (closure.shape + local - 1U) / local;
      void const *push_ptr = pipe.push_bytes > 0 ? static_cast<void const *>(&closure.params) : nullptr;
      auto const push_bytes = static_cast<std::uint32_t>(pipe.push_bytes > 0 ? sizeof(Params) : 0);
      record_pass(cmd, pipe, set, push_ptr, push_bytes, dispatch{ .x = groups });
    } };
  }

  inline auto make_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step
  {
    return pass_step{ .record = [closure = std::move(closure)](
                                  context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> void {
      void const *push_ptr = closure.push.empty() ? nullptr : static_cast<void const *>(closure.push.data());
      auto const push_bytes = static_cast<std::uint32_t>(closure.push.size());
      if (closure.is_indirect) {
        record_pass(cmd, closure.bind, push_ptr, push_bytes, closure.indirect);
      } else {
        record_pass(cmd, closure.bind, push_ptr, push_bytes, closure.groups);
      }
    } };
  }

  template<typename Tag> auto make_barrier_step(Tag tag) -> pass_step
  {
    return pass_step{ .record = [tag](context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> void {
      tag(cmd);
    } };
  }

  inline auto append_step(pass_graph_sender graph, pass_step step) -> pass_graph_sender
  {
    graph.steps.push_back(std::move(step));
    return graph;
  }

}// namespace detail

template<typename Params, typename Fun>
auto operator|(schedule_sender snd, compute_pass_closure<Params, Fun> closure) -> pass_graph_sender
{ return pass_graph_sender{ .ctx = snd.ctx, .steps = { detail::make_traced_step(std::move(closure)) } }; }

template<typename Params, typename Fun>
auto operator|(pass_graph_sender graph, compute_pass_closure<Params, Fun> closure) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_traced_step(std::move(closure))); }

inline auto operator|(schedule_sender snd, prebuilt_compute_pass_closure closure) -> pass_graph_sender
{ return pass_graph_sender{ .ctx = snd.ctx, .steps = { detail::make_prebuilt_step(std::move(closure)) } }; }

inline auto operator|(pass_graph_sender graph, prebuilt_compute_pass_closure closure) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_prebuilt_step(std::move(closure))); }

inline auto operator|(pass_graph_sender graph, barrier::transfer_to_compute_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

inline auto operator|(pass_graph_sender graph, barrier::compute_to_compute_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

inline auto operator|(pass_graph_sender graph, barrier::compute_to_graphics_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

inline auto operator|(pass_graph_sender graph, barrier::graphics_to_compute_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

inline auto operator|(pass_graph_sender graph, barrier::compute_read_t tag) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_barrier_step(tag)); }

[[nodiscard]] inline auto operator|(pass_graph_sender &&snd, submit_t /*tag*/) -> pass_graph_async_sender
{ return pass_graph_async_sender{ std::move(snd) }; }

[[nodiscard]] inline auto operator|(pass_graph_sender &snd, submit_t /*tag*/) -> pass_graph_async_sender
{ return pass_graph_async_sender{ snd.ctx, snd.steps }; }

}// namespace vkexec

#endif// VKEXEC_PASS_HPP
