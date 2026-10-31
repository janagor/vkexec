#ifndef VKEXEC_RECORD_INTO_HPP
#define VKEXEC_RECORD_INTO_HPP

#include <vkexec/pass.hpp>
#include <vkexec/recorded_graph.hpp>

namespace vkexec {
struct record_into_options
{
  queue_ref queue;
};

//! Record into an already recording primary command buffer compatible with queue.
//! Caller owns submission, entry/exit synchronization, and completion. Initial
//! image layouts must describe actual entry layouts. No state persists between
//! calls. On recording failure discard/reset cmd before using it again.
template<class Graph>
[[nodiscard]] auto record_into(context &ctx, VkCommandBuffer cmd, Graph &&expression, record_into_options options)
  -> result<recorded_graph>
{
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    auto state = detail::context_access::state(ctx);
    if (!state || ctx.device() == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE || options.queue.queue == VK_NULL_HANDLE
        || options.queue.family == VK_QUEUE_FAMILY_IGNORED) {
      return fail(errc::invalid_argument, "record_into requires a context, command buffer and queue");
    }
    if (!detail::same_queue(options.queue, ctx.compute_queue_ref())
        && !detail::same_queue(options.queue, ctx.graphics_queue_ref())
        && !detail::same_queue(options.queue, ctx.present_queue_ref())) {
      return fail(errc::invalid_argument, "queue does not belong to context");
    }
    auto graph = [&] {
      if constexpr (!detail::is_pass_graph_sender_v<Graph>) {
        auto env = scheduler_env{ .state = state };
        return detail::materialize_pass_chain(
          detail::collect_pass_chain(detail::normalize_vkexec_expression(std::forward<Graph>(expression), env), env));
      } else {
        return std::forward<Graph>(expression);
      }
    }();
    if (graph.state != state) { return fail(errc::invalid_argument, "graph belongs to a different context"); }
    if (graph.presentation) { return fail(errc::unsupported, "external recording does not support presentation"); }
    auto plan = detail::prepare_graph(ctx, graph.steps, graph.step_queues, graph.step_predecessors, options.queue);
    if (!plan) { return fail(plan); }
    return recorded_graph::record(ctx, cmd, options.queue, *plan, std::move(graph.steps));
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    return unexpected(unexpected_exception_error());
  }
#endif
}
}// namespace vkexec
#endif
