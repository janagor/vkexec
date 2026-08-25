#ifndef VKEXEC_PASS_HPP
#define VKEXEC_PASS_HPP

#include <vkexec/barrier.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/push_data.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <span>
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
  if (bind.layout != VK_NULL_HANDLE) { upload_push_constants(cmd, bind.layout, push, push_bytes); }
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
  if (bind.layout != VK_NULL_HANDLE) { upload_push_constants(cmd, bind.layout, push, push_bytes); }
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
}

[[nodiscard]] inline auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.pipeline);
  if (!push.empty()) {
    if (auto const pushed = cmd_push_data(ctx, cmd, push); !pushed) { return pushed; }
  }
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
  return {};
}

[[nodiscard]] inline auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind.pipeline);
  if (!push.empty()) {
    if (auto const pushed = cmd_push_data(ctx, cmd, push); !pushed) { return pushed; }
  }
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
  return {};
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
  std::function<status(context &, VkCommandBuffer, detail::pass_cleanup &)> record;
};

struct pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::vector<pass_step> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver)
  {
    return ex::connect(ex::let_value(detail::enter_submit_scope(self.ctx),
                         [steps = std::forward_like<decltype(self)>(self.steps)](
                           detail::submit_scope &scope) mutable -> decltype(auto) {
                           for (pass_step const &step : steps) {
                             if (auto const recorded = step.record(*scope.ctx, scope.cmd, scope.cleanup); !recorded) {
                               scope.release();
                               return detail::fail_with(recorded.error());
                             }
                           }
                           if (auto const ended = scope.end_recording(); !ended) {
                             scope.release();
                             return detail::fail_with(ended.error());
                           }
                           return detail::submit_and_wait(std::move(scope));
                         }),
      std::move(receiver));
  }
};

struct pass_graph_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::vector<pass_step> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  explicit pass_graph_async_sender(pass_graph_sender graph) : ctx(graph.ctx), steps(std::move(graph.steps)) {}

  pass_graph_async_sender(context *host, std::vector<pass_step> graph_steps) : ctx(host), steps(std::move(graph_steps))
  {}

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver)
  {
    return ex::connect(ex::let_value(detail::enter_submit_scope(self.ctx),
                         [steps = std::forward_like<decltype(self)>(self.steps)](
                           detail::submit_scope &scope) mutable -> decltype(auto) {
                           for (pass_step const &step : steps) {
                             if (auto const recorded = step.record(*scope.ctx, scope.cmd, scope.cleanup); !recorded) {
                               scope.release();
                               return detail::fail_with(recorded.error());
                             }
                           }
                           if (auto const ended = scope.end_recording(); !ended) {
                             scope.release();
                             return detail::fail_with(ended.error());
                           }
                           return detail::submit_fence(std::move(scope));
                         }),
      std::move(receiver));
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
                                  context &record_ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status {
      edsl::trace_scope const scope;
      edsl::Int const idx = edsl::Int::param_index();
      auto push = edsl::push_constant<Params>::bind();
      closure.fun(idx, push);

      auto const pipe = record_ctx.get_or_compile(scope, closure.shape);
      if (!pipe) { return std::unexpected(pipe.error()); }
      std::vector<edsl::storage_trace> const buffers = scope.buffers();
      auto const local = scope.local_size_x();

      auto const set = bind_or_allocate_set(record_ctx, *pipe, buffers, cleanup);
      if (!set) { return std::unexpected(set.error()); }
      std::uint32_t const groups = (closure.shape + local - 1U) / local;
      void const *push_ptr = pipe->get().push_bytes > 0 ? static_cast<void const *>(&closure.params) : nullptr;
      auto const push_bytes = static_cast<std::uint32_t>(pipe->get().push_bytes > 0 ? sizeof(Params) : 0);
      record_pass(cmd, pipe->get(), *set, push_ptr, push_bytes, dispatch{ .x = groups });
      return {};
    } };
  }

  inline auto make_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step
  {
    return pass_step{ .record = [closure = std::move(closure)](
                                  context &record_ctx, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
      std::span<std::byte const> const push_bytes{ closure.push };
      bool const use_push_data = closure.bind.layout == VK_NULL_HANDLE;
      if (use_push_data) {
        if (closure.is_indirect) {
          return record_heap_pass(record_ctx, cmd, closure.bind, push_bytes, closure.indirect);
        }
        return record_heap_pass(record_ctx, cmd, closure.bind, push_bytes, closure.groups);
      }

      void const *push_ptr = closure.push.empty() ? nullptr : static_cast<void const *>(closure.push.data());
      auto const push_size = static_cast<std::uint32_t>(closure.push.size());
      if (closure.is_indirect) {
        record_pass(cmd, closure.bind, push_ptr, push_size, closure.indirect);
      } else {
        record_pass(cmd, closure.bind, push_ptr, push_size, closure.groups);
      }
      return {};
    } };
  }

  template<typename Tag> auto make_barrier_step(Tag tag) -> pass_step
  {
    return pass_step{ .record = [tag](context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
      tag(cmd);
      return {};
    } };
  }

  inline auto append_step(pass_graph_sender graph, pass_step step) -> pass_graph_sender
  {
    graph.steps.push_back(std::move(step));
    return graph;
  }

}// namespace detail

template<class Pred, class Closure> struct pass_adaptor_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_sender::completion_signatures;

  Pred pred;
  Closure closure;

  [[nodiscard]] auto get_env() const noexcept -> decltype(auto) { return ex::get_env(pred); }
};

template<class Pred, class Closure, class Env>
[[nodiscard]] auto
  lower_vkexec_sender(ex::set_value_t /*tag*/, pass_adaptor_sender<Pred, Closure> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  // NOLINTNEXTLINE(misc-const-correctness)
  context *const ctx = sched.get_context();
  return ex::let_value(
    std::move(sndr.pred), [ctx, closure = std::move(sndr.closure)](auto &&...) mutable -> pass_graph_sender {
      if constexpr (std::is_same_v<std::remove_cvref_t<Closure>, prebuilt_compute_pass_closure>) {
        return pass_graph_sender{ .ctx = ctx, .steps = { detail::make_prebuilt_step(std::move(closure)) } };
      } else {
        return pass_graph_sender{ .ctx = ctx, .steps = { detail::make_traced_step(std::move(closure)) } };
      }
    });
}

template<class Pred, class Closure> struct pass_async_adaptor_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_async_sender::completion_signatures;

  Pred pred;
  Closure closure;

  [[nodiscard]] auto get_env() const noexcept -> decltype(auto) { return ex::get_env(pred); }
};

template<class Pred, class Closure, class Env>
[[nodiscard]] auto
  lower_vkexec_sender(ex::set_value_t /*tag*/, pass_async_adaptor_sender<Pred, Closure> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  // NOLINTNEXTLINE(misc-const-correctness)
  context *const ctx = sched.get_context();
  return ex::let_value(
    std::move(sndr.pred), [ctx, closure = std::move(sndr.closure)](auto &&...) mutable -> pass_graph_async_sender {
      pass_graph_sender graph;
      if constexpr (std::is_same_v<std::remove_cvref_t<Closure>, prebuilt_compute_pass_closure>) {
        graph = pass_graph_sender{ .ctx = ctx, .steps = { detail::make_prebuilt_step(std::move(closure)) } };
      } else {
        graph = pass_graph_sender{ .ctx = ctx, .steps = { detail::make_traced_step(std::move(closure)) } };
      }
      return pass_graph_async_sender{ std::move(graph) };
    });
}

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

template<vkexec_predecessor Pred, typename Params, typename Fun>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>
           && !std::same_as<std::remove_cvref_t<Pred>, pass_graph_sender>)
[[nodiscard]] auto operator|(Pred &&pred, compute_pass_closure<Params, Fun> closure)
  -> pass_adaptor_sender<std::remove_cvref_t<Pred>, compute_pass_closure<Params, Fun>>
{
  return pass_adaptor_sender<std::remove_cvref_t<Pred>, compute_pass_closure<Params, Fun>>{
    .pred = std::forward<Pred>(pred),
    .closure = std::move(closure),
  };
}

template<vkexec_predecessor Pred>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>
           && !std::same_as<std::remove_cvref_t<Pred>, pass_graph_sender>)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] auto operator|(Pred &&pred, prebuilt_compute_pass_closure closure)
  -> pass_adaptor_sender<std::remove_cvref_t<Pred>, prebuilt_compute_pass_closure>
{
  return pass_adaptor_sender<std::remove_cvref_t<Pred>, prebuilt_compute_pass_closure>{
    .pred = std::forward<Pred>(pred),
    .closure = std::move(closure),
  };
}

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

template<class Pred, class Closure>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
[[nodiscard]] auto operator|(pass_adaptor_sender<Pred, Closure> &&snd, submit_t /*tag*/)
  -> pass_async_adaptor_sender<Pred, Closure>
{
  return pass_async_adaptor_sender<Pred, Closure>{
    .pred = std::move(snd.pred),
    .closure = std::move(snd.closure),
  };
}

template<class Pred, class Closure>
[[nodiscard]] auto operator|(pass_adaptor_sender<Pred, Closure> &snd, submit_t /*tag*/)
  -> pass_async_adaptor_sender<Pred, Closure>
{
  return pass_async_adaptor_sender<Pred, Closure>{
    .pred = snd.pred,
    .closure = snd.closure,
  };
}

}// namespace vkexec

#endif// VKEXEC_PASS_HPP
