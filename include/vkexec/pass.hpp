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

[[nodiscard]] auto bind_compute(pipeline_resources const &pipe, VkDescriptorSet set = VK_NULL_HANDLE)
  -> compute_bind;

auto
  record_pass(VkCommandBuffer cmd, compute_bind bind, void const *push, std::uint32_t push_bytes, dispatch groups)
    -> void;

auto record_pass(VkCommandBuffer cmd,
  compute_bind bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void;

[[nodiscard]] auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status;

[[nodiscard]] auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status;

auto record_pass(VkCommandBuffer cmd,
  pipeline_resources const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void;

struct pass_step
{
  std::function<status(context &, VkCommandBuffer, detail::pass_cleanup &)> record;
};

namespace detail {

  [[nodiscard]] auto record_pass_steps(submit_scope &scope, std::span<pass_step const> steps) -> status;

  [[nodiscard]] auto open_and_record_pass(context *ctx, std::span<pass_step const> steps) -> result<submit_scope>;

}// namespace detail

struct pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::vector<pass_step> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::vector<pass_step> steps;
    Receiver receiver;
    using submit_op_t = decltype(ex::connect(std::declval<detail::submit_and_wait_sender>(), std::declval<Receiver>()));
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

      auto prepared = detail::open_and_record_pass(ctx, steps);
      if (!prepared) {
        ex::set_error(std::move(rcvr), to_error(prepared.error()));
        return;
      }

      submit_op.emplace(ex::connect(detail::submit_and_wait(detail::leaf_take(prepared)), std::move(rcvr)));
      ex::start(*submit_op);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .steps = std::forward_like<decltype(self)>(self.steps),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
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

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::vector<pass_step> steps;
    Receiver receiver;
    using submit_op_t = decltype(ex::connect(std::declval<detail::submit_fence_sender>(), std::declval<Receiver>()));
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

      auto prepared = detail::open_and_record_pass(ctx, steps);
      if (!prepared) {
        ex::set_error(std::move(rcvr), to_error(prepared.error()));
        return;
      }

      submit_op.emplace(ex::connect(detail::submit_fence(detail::leaf_take(prepared)), std::move(rcvr)));
      ex::start(*submit_op);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .steps = std::forward_like<decltype(self)>(self.steps),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

struct prebuilt_compute_pass_closure
{
  compute_bind bind{};
  std::vector<std::byte> push;
  dispatch groups{};
  indirect_dispatch indirect{};
  bool is_indirect{ false };
};

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

auto compute_pass(compute_bind bind, dispatch groups) -> prebuilt_compute_pass_closure;

auto compute_pass(compute_bind bind, indirect_dispatch groups) -> prebuilt_compute_pass_closure;

template<typename Params>
auto compute_pass(pipeline_resources &pipe, VkDescriptorSet set, Params const &params, dispatch groups)
  -> prebuilt_compute_pass_closure
{ return compute_pass(bind_compute(pipe, set), params, groups); }

namespace detail {

  auto make_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step;

  template<typename Tag> auto make_barrier_step(Tag tag) -> pass_step
  {
    return pass_step{ .record = [tag](context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
      tag(cmd);
      return {};
    } };
  }

  auto append_step(pass_graph_sender graph, pass_step step) -> pass_graph_sender;

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
      return pass_graph_sender{ .ctx = ctx, .steps = { detail::make_prebuilt_step(std::move(closure)) } };
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
      pass_graph_sender graph{
        .ctx = ctx,
        .steps = { detail::make_prebuilt_step(std::move(closure)) },
      };
      return pass_graph_async_sender{ std::move(graph) };
    });
}

auto operator|(schedule_sender snd, prebuilt_compute_pass_closure closure) -> pass_graph_sender;

auto operator|(pass_graph_sender graph, prebuilt_compute_pass_closure closure) -> pass_graph_sender;

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

auto operator|(pass_graph_sender graph, barrier::transfer_to_compute_t tag) -> pass_graph_sender;

auto operator|(pass_graph_sender graph, barrier::compute_to_compute_t tag) -> pass_graph_sender;

auto operator|(pass_graph_sender graph, barrier::compute_to_graphics_t tag) -> pass_graph_sender;

auto operator|(pass_graph_sender graph, barrier::graphics_to_compute_t tag) -> pass_graph_sender;

auto operator|(pass_graph_sender graph, barrier::compute_read_t tag) -> pass_graph_sender;

[[nodiscard]] auto operator|(pass_graph_sender &&snd, submit_t /*tag*/) -> pass_graph_async_sender;

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
