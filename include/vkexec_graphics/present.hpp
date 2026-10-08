#ifndef VKEXEC_GRAPHICS_PRESENT_HPP
#define VKEXEC_GRAPHICS_PRESENT_HPP

//! \file
//! Swapchain acquisition and presentation for a pass graph.

#include <vkexec/pass.hpp>
#include <vkexec_graphics/presenter.hpp>

#include <utility>

namespace vkexec {

namespace detail {

  struct present_graph_t
  {
  };

  struct present_graph_data
  {
    owned::presenter *win;
    acquired_frame frame;
    present_options options;
  };

  template<class Graph>
    requires is_pass_graph_sender_v<Graph>
  auto attach_presentation(Graph &graph, present_graph_data const &data) -> void
  {
    graph_presentation boundary{
      .queue = data.win->ctx().graphics_queue_ref(),
      .image = data.frame.image,
      .prepare = [win = data.win, frame = data.frame]() -> result<graph_presentation_sync> {
        auto sync = win->submission_sync(frame);
        if (!sync) { return fail(sync); }
        return graph_presentation_sync{ .image_available_wait = sync->image_available_wait,
          .render_finished_signal = sync->render_finished_signal,
          .fence = sync->fence };
      },
      .present = [win = data.win, frame = data.frame, options = data.options]() -> status {
        auto presented = win->present_submitted(frame, options);
        if (!presented) { return fail(presented); }
        return {};
      },
      .abort = [win = data.win, frame = data.frame](
                 presentation_abort_state state) -> status { return win->abandon_frame(frame, state); },
      .sync = {},
    };
    graph.presentation = std::move(boundary);
  }

  template<class Child, class Env>
  [[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/,
    sender_expr<present_graph_t, present_graph_data, Child> expr,
    Env const &env)
    -> decltype(materialize_pass_chain(
      collect_pass_chain(normalize_vkexec_expression(std::move(expr.child), env), env)))
  {
    auto normalized = normalize_vkexec_expression(std::move(expr.child), env);
    auto graph = materialize_pass_chain(collect_pass_chain(std::move(normalized), env));
    attach_presentation(graph, expr.data);
    return graph;
  }

}// namespace detail

/**
 * Presents an acquired frame after all graph batches have been submitted.
 * The final graph batch must run on the presenter's graphics queue and record
 * `frame.image` only as a color attachment in that final batch. No earlier
 * batch may access that image. The graph waits for image acquisition at the
 * color-attachment stage on that batch, signals presentation when it finishes,
 * and completes after its fence. The final render pass must leave the image
 * in `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
 *
 * Acquire with `win.acquire_frame()` and pipe the resulting graph into
 * `present(win, *frame)`. The presenter must outlive the graph operation.
 */
[[nodiscard]] inline auto present(owned::presenter &win, acquired_frame frame, present_options options = {})
  -> detail::expr_closure<detail::present_graph_t, detail::present_graph_data>
{
  return detail::make_expr_closure(
    detail::present_graph_t{}, detail::present_graph_data{ .win = &win, .frame = frame, .options = options });
}

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_PRESENT_HPP
