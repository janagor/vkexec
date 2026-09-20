#ifndef VKEXEC_PASS_HPP
#define VKEXEC_PASS_HPP

//! \file
//! Compute pass recording, pass graphs, and stdexec pipe adaptors (`compute_pass`, `| submit`).

#include <vkexec/barrier.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/submit_scope.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

//! Workgroup counts for `vkCmdDispatch` (X/Y/Z).
struct dispatch
{
  std::uint32_t x{ 1 };
  std::uint32_t y{ 1 };
  std::uint32_t z{ 1 };
};

/**
 * Computes workgroup count X covering `work_count` invocations for local size `local_x`.
 *
 * Y and Z remain 1. A zero `local_x` is treated as 1.
 */
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr auto dispatch_groups_for(std::uint32_t work_count, std::uint32_t local_x) noexcept -> dispatch
{
  std::uint32_t const group_size = local_x == 0U ? 1U : local_x;
  return dispatch{ .x = (work_count + group_size - 1U) / group_size };
}
// NOLINTEND(bugprone-easily-swappable-parameters)

//! Arguments for `vkCmdDispatchIndirect`.
struct indirect_dispatch
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{ 0 };
};

//! Pipeline, layout, and optional descriptor set for one compute dispatch.
struct compute_bind
{
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkPipelineLayout layout{ VK_NULL_HANDLE };
  VkDescriptorSet set{ VK_NULL_HANDLE };
};

//! Builds a `compute_bind` from pipeline resources and an optional set.
[[nodiscard]] auto bind_compute(pipeline_resources const &pipe, VkDescriptorSet set = VK_NULL_HANDLE) -> compute_bind;

/**
 * Records bind, optional push constants, and a direct dispatch on `cmd`.
 *
 * @param cmd Command buffer in the recording state.
 * @param bind Pipeline / layout / descriptor set.
 * @param push Push-constant bytes (may be null when `push_bytes` is 0).
 * @param push_bytes Size of the push-constant blob.
 * @param groups Workgroup counts for `vkCmdDispatch`.
 */
auto record_pass(VkCommandBuffer cmd, compute_bind bind, void const *push, std::uint32_t push_bytes, dispatch groups)
  -> void;

/**
 * Records bind, optional push constants, and an indirect dispatch on `cmd`.
 *
 * @param groups Buffer and offset containing `VkDispatchIndirectCommand`.
 */
auto record_pass(VkCommandBuffer cmd,
  compute_bind bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void;

//! Convenience overload that builds a `compute_bind` from `pipe` and `set`.
auto record_pass(VkCommandBuffer cmd,
  pipeline_resources const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void;

/**
 * One recording step in a pass graph (dispatch, barrier, or custom record).
 *
 * `record` may allocate descriptor sets into the provided cleanup object.
 * `after_gpu` runs after the graph's submit completes successfully (host-side
 * work such as staging->CPU readback).
 */
struct pass_step
{
  std::function<status(context &, VkCommandBuffer, detail::pass_cleanup &)> record;
  std::function<void()> after_gpu;
};

namespace detail {

  //! Records all `steps` into an already-open `scope`.
  [[nodiscard]] auto record_pass_steps(submit_scope &scope, std::span<pass_step const> steps) -> status;

  //! Opens a submit scope on `ctx`, records `steps`, and returns the open scope.
  [[nodiscard]] auto open_and_record_pass(context *ctx, std::span<pass_step const> steps) -> result<submit_scope>;

}// namespace detail

/**
 * Sender that records a list of pass steps, submits, and blocks until the GPU finishes.
 *
 * Built by piping `schedule()` into `compute_pass(...)` and optional barriers.
 * Use `| vkexec::submit` to switch to non-blocking completion.
 *
 * ~~~~~~~~~~~{.cpp}
 * auto graph = ex::schedule(ctx->get_scheduler())
 *   | vkexec::compute_pass(*bound.pipe, bound.set, params, 10000)
 *   | vkexec::barrier::compute_to_compute()
 *   | vkexec::compute_pass(*bound.pipe, bound.set, params, 10000);
 * vkexec::sync_wait(std::move(graph));
 * ~~~~~~~~~~~
 *
 * @see pass_graph_async_sender, compute_pass, submit
 */
struct pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::vector<pass_step> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct after_gpu_receiver
  {
    using receiver_concept = ex::receiver_t;

    Receiver rcvr;
    std::vector<std::function<void()>> after_gpu;

    auto set_value() && noexcept -> void
    {
      for (std::function<void()> const &callback : after_gpu) {
        if (callback) { callback(); }
      }
      ex::set_value(std::move(rcvr));
    }

    auto set_error(error err) && noexcept -> void { ex::set_error(std::move(rcvr), std::move(err)); }

    auto set_stopped() && noexcept -> void { ex::set_stopped(std::move(rcvr)); }

    [[nodiscard]] auto get_env() const noexcept -> decltype(ex::get_env(std::declval<Receiver const &>()))
    { return ex::get_env(rcvr); }
  };

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::vector<pass_step> steps;
    Receiver receiver;
    using submit_op_t = decltype(ex::connect(std::declval<detail::submit_and_wait_sender>(),
      std::declval<after_gpu_receiver<Receiver>>()));
    std::optional<submit_op_t> submit_op;

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

      std::vector<std::function<void()>> after;
      after.reserve(steps.size());
      for (pass_step &step : steps) {
        if (step.after_gpu) { after.push_back(std::move(step.after_gpu)); }
      }

      auto prepared = detail::open_and_record_pass(ctx, steps);
      if (!prepared) {
        ex::set_error(std::move(rcvr), std::move(prepared.error()));
        return;
      }

      submit_op.emplace(ex::connect(detail::submit_and_wait(expected_take(prepared)),
        after_gpu_receiver<Receiver>{ .rcvr = std::move(rcvr), .after_gpu = std::move(after) }));
      ex::start(*submit_op);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .steps = steps,
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .steps = std::move(steps),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

/**
 * Like `pass_graph_sender`, but completes asynchronously via the fence agent.
 *
 * Produced by `pass_graph_sender | vkexec::submit`. Does not block `start()`.
 *
 * @see pass_graph_sender, submit_t
 */
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

  template<class Receiver> struct after_gpu_receiver
  {
    using receiver_concept = ex::receiver_t;

    Receiver rcvr;
    std::vector<std::function<void()>> after_gpu;

    auto set_value() && noexcept -> void
    {
      for (std::function<void()> const &callback : after_gpu) {
        if (callback) { callback(); }
      }
      ex::set_value(std::move(rcvr));
    }

    auto set_error(error err) && noexcept -> void { ex::set_error(std::move(rcvr), std::move(err)); }

    auto set_stopped() && noexcept -> void { ex::set_stopped(std::move(rcvr)); }

    [[nodiscard]] auto get_env() const noexcept -> decltype(ex::get_env(std::declval<Receiver const &>()))
    { return ex::get_env(rcvr); }
  };

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::vector<pass_step> steps;
    Receiver receiver;
    using submit_op_t =
      decltype(ex::connect(std::declval<detail::submit_fence_sender>(), std::declval<after_gpu_receiver<Receiver>>()));
    std::optional<submit_op_t> submit_op;

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

      std::vector<std::function<void()>> after;
      after.reserve(steps.size());
      for (pass_step &step : steps) {
        if (step.after_gpu) { after.push_back(std::move(step.after_gpu)); }
      }

      auto prepared = detail::open_and_record_pass(ctx, steps);
      if (!prepared) {
        ex::set_error(std::move(rcvr), std::move(prepared.error()));
        return;
      }

      submit_op.emplace(ex::connect(detail::submit_fence(expected_take(prepared)),
        after_gpu_receiver<Receiver>{ .rcvr = std::move(rcvr), .after_gpu = std::move(after) }));
      ex::start(*submit_op);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .steps = steps,
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .steps = std::move(steps),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

/**
 * Prebuilt compute dispatch: bind info, optional push bytes, and direct or indirect groups.
 *
 * Created by `compute_pass(...)` overloads and piped onto a schedule or pass graph.
 *
 * @see compute_pass
 */
struct prebuilt_compute_pass_closure
{
  compute_bind bind{};
  std::vector<std::byte> push;
  dispatch groups{};
  indirect_dispatch indirect{};
  bool is_indirect{ false };
};

/**
 * Builds a direct-dispatch compute pass with push constants `params`.
 *
 * @param bind Pipeline / layout / set.
 * @param params Trivially copyable push-constant blob.
 * @param groups Workgroup counts.
 */
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

/**
 * Builds an indirect-dispatch compute pass with push constants `params`.
 *
 * @param groups Buffer containing `VkDispatchIndirectCommand` at `offset`.
 */
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

//! Builds a direct-dispatch compute pass without push constants.
auto compute_pass(compute_bind bind, dispatch groups) -> prebuilt_compute_pass_closure;

//! Builds an indirect-dispatch compute pass without push constants.
auto compute_pass(compute_bind bind, indirect_dispatch groups) -> prebuilt_compute_pass_closure;

//! Returns the specialized local workgroup size as a `dispatch`.
[[nodiscard]] inline auto local_size(pipeline_resources const &pipe) noexcept -> dispatch
{ return dispatch{ .x = pipe.local_size.at(0), .y = pipe.local_size.at(1), .z = pipe.local_size.at(2) }; }

//! Returns workgroup counts covering `work_count` invocations along X.
[[nodiscard]] inline auto groups_for(pipeline_resources const &pipe, std::uint32_t work_count) noexcept -> dispatch
{ return dispatch_groups_for(work_count, pipe.local_size.at(0)); }

//! Convenience overload that binds `pipe` with `set` before building the closure.
template<typename Params>
auto compute_pass(pipeline_resources const &pipe, VkDescriptorSet set, Params const &params, dispatch groups)
  -> prebuilt_compute_pass_closure
{ return compute_pass(bind_compute(pipe, set), params, groups); }

//! Builds a direct-dispatch pass with push constants and automatic group counts.
template<typename Params>
auto compute_pass(pipeline_resources const &pipe, VkDescriptorSet set, Params const &params, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe, set, params, groups_for(pipe, work_count)); }

//! Builds a direct-dispatch pass without push constants using automatic group counts.
inline auto compute_pass(pipeline_resources const &pipe, VkDescriptorSet set, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(bind_compute(pipe, set), groups_for(pipe, work_count)); }

namespace detail {

  //! Wraps a prebuilt closure as a single `pass_step`.
  auto make_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step;

  //! Wraps a barrier tag callable as a `pass_step`.
  template<typename Tag> auto make_barrier_step(Tag tag) -> pass_step
  {
    return pass_step{ .record = [tag](context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
                       tag(cmd);
                       return {};
                     },
      .after_gpu = {} };
  }

  //! Appends `step` to `graph` and returns the updated graph sender.
  auto append_step(pass_graph_sender graph, pass_step step) -> pass_graph_sender;

}// namespace detail

/**
 * Lazy adaptor: after `pred` completes, builds a one-step `pass_graph_sender`.
 *
 * Lowered by the vkexec domain via `lower_vkexec_sender`.
 */
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
      return pass_graph_sender{ .ctx = ctx, .steps = { detail::make_prebuilt_step(std::move(closure)) } };
    });
}

/**
 * Like `pass_adaptor_sender`, but lowers to `pass_graph_async_sender` (non-blocking submit).
 *
 * Produced by `pass_adaptor_sender | vkexec::submit`.
 */
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
      pass_graph_sender graph{
        .ctx = ctx,
        .steps = { detail::make_prebuilt_step(std::move(closure)) },
      };
      return pass_graph_async_sender{ std::move(graph) };
    });
}

//! Starts a pass graph from a `schedule()` sender with one compute step.
auto operator|(schedule_sender snd, prebuilt_compute_pass_closure closure) -> pass_graph_sender;

//! Appends a compute step to an existing pass graph.
auto operator|(pass_graph_sender graph, prebuilt_compute_pass_closure closure) -> pass_graph_sender;

/**
 * Wraps a vkexec predecessor in a lazy pass adaptor (domain-lowered later).
 *
 * Used when the left-hand side is not already a `schedule_sender` or `pass_graph_sender`.
 */
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

//! Appends a transfer->compute barrier step to the graph.
auto operator|(pass_graph_sender graph, barrier::transfer_to_compute_t tag) -> pass_graph_sender;

//! Appends a compute->compute barrier step to the graph.
auto operator|(pass_graph_sender graph, barrier::compute_to_compute_t tag) -> pass_graph_sender;

//! Appends a compute->graphics barrier step to the graph.
auto operator|(pass_graph_sender graph, barrier::compute_to_graphics_t tag) -> pass_graph_sender;

//! Appends a graphics->compute barrier step to the graph.
auto operator|(pass_graph_sender graph, barrier::graphics_to_compute_t tag) -> pass_graph_sender;

//! Appends a compute read-after-write barrier step to the graph.
auto operator|(pass_graph_sender graph, barrier::compute_read_t tag) -> pass_graph_sender;

//! Converts a blocking pass graph into an async fence-wait graph.
[[nodiscard]] auto operator|(pass_graph_sender &&snd, submit_t /*tag*/) -> pass_graph_async_sender;

//! Converts a blocking pass graph into an async fence-wait graph (lvalue overload).
[[nodiscard]] auto operator|(pass_graph_sender &snd, submit_t /*tag*/) -> pass_graph_async_sender;

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
