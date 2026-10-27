#ifndef VKEXEC_GRAPHICS_DRAW_LAYERS_HPP
#define VKEXEC_GRAPHICS_DRAW_LAYERS_HPP

#include <vkexec_graphics/draw_basic.hpp>

namespace vkexec {

struct draw_layers_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  std::vector<draw_layer> layers;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    std::vector<draw_layer> layers;
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_sync(state, receiver, [this]() -> void {
        if (layers.empty()) {
          stdexec::set_error(
            std::move(receiver), make_error(errc::invalid_argument, "draw_layers requires at least one layer"));
          return;
        }

        auto frame_result = detail::try_begin_frame(*win);
        if (!frame_result) {
          stdexec::set_error(std::move(receiver), std::move(frame_result.error()));
          return;
        }
        if (!frame_result->has_value()) {
          stdexec::set_value(std::move(receiver));
          return;
        }

        frame const &drawn = **frame_result;
        graphics_pipeline_config const &clear_cfg = layers.front().pipeline->config();
        std::array<VkClearValue, k_graphics_clear_count> const clears = make_clear_values(clear_cfg);

        VkRenderPassBeginInfo rp_begin{};
        rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_begin.renderPass = win->render_pass();
        rp_begin.framebuffer = drawn.framebuffer;
        rp_begin.renderArea.offset = { .x = 0, .y = 0 };
        rp_begin.renderArea.extent = drawn.extent;
        rp_begin.clearValueCount = k_graphics_clear_count;
        rp_begin.pClearValues = clears.data();

        vkCmdBeginRenderPass(drawn.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
        for (draw_layer const &layer : layers) {
          layer.pipeline->record_draw(drawn.command_buffer, drawn.extent, layer.vertex_count);
        }
        vkCmdEndRenderPass(drawn.command_buffer);
        if (auto fence_result = detail::try_end_frame(*win, drawn); !fence_result) {
          stdexec::set_error(std::move(receiver), std::move(fence_result.error()));
          return;
        }
        stdexec::set_value(std::move(receiver));
      });
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) const & noexcept(
    std::is_nothrow_copy_constructible_v<std::vector<draw_layer>> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .layers = layers,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<std::vector<draw_layer>> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .layers = std::move(layers),
      .receiver = std::move(receiver),
    };
  }
};

//! Async variant of `draw_layers_sender` (fence-agent completion via `| submit`).
struct draw_layers_async_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  std::vector<draw_layer> layers;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  explicit draw_layers_async_sender(draw_layers_sender snd)
    : state(std::move(snd.state)), win(snd.win), layers(std::move(snd.layers))
  {}

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    std::vector<draw_layer> layers;
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        state,
        win,
        [this](frame &drawn_frame) -> result<VkFence> {
          if (layers.empty()) { return fail(errc::invalid_argument, "draw_layers requires at least one layer"); }
          graphics_pipeline_config const &clear_cfg = layers.front().pipeline->config();
          std::array<VkClearValue, k_graphics_clear_count> const clears = make_clear_values(clear_cfg);

          VkRenderPassBeginInfo rp_begin{};
          rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
          rp_begin.renderPass = win->render_pass();
          rp_begin.framebuffer = drawn_frame.framebuffer;
          rp_begin.renderArea.offset = { .x = 0, .y = 0 };
          rp_begin.renderArea.extent = drawn_frame.extent;
          rp_begin.clearValueCount = k_graphics_clear_count;
          rp_begin.pClearValues = clears.data();

          vkCmdBeginRenderPass(drawn_frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
          for (draw_layer const &layer : layers) {
            layer.pipeline->record_draw(drawn_frame.command_buffer, drawn_frame.extent, layer.vertex_count);
          }
          vkCmdEndRenderPass(drawn_frame.command_buffer);
          return detail::try_end_frame(*win, drawn_frame);
        },
        receiver);
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(
    std::is_nothrow_copy_constructible_v<std::vector<draw_layer>> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .layers = layers,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<std::vector<draw_layer>> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .layers = std::move(layers),
      .receiver = std::move(receiver),
    };
  }
};

inline auto operator|(schedule_sender snd, draw_closure closure) -> draw_sender
{
  return draw_sender{
    .state = std::move(snd.state),
    .win = closure.win,
    .pipeline = closure.pipeline,
    .vertex_count = closure.vertex_count,
  };
}

/**
 * Sender that draws one mesh frame and blocks until present submit completes.
 *
 * @see draw, draw_mesh_async_sender
 */
struct draw_mesh_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  owned::graphics_pipeline *pipeline{ nullptr };
  mesh_draw const *drawn{ nullptr };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{ nullptr };
    owned::graphics_pipeline *pipeline{ nullptr };
    mesh_draw const *drawn{ nullptr };
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_sync(state, receiver, [this]() -> void {
        auto frame_result = detail::try_begin_frame(*win);
        if (!frame_result) {
          stdexec::set_error(std::move(receiver), std::move(frame_result.error()));
          return;
        }
        if (!frame_result->has_value()) {
          stdexec::set_value(std::move(receiver));
          return;
        }

        frame const &drawn_frame = **frame_result;
        pipeline->record_pass(
          drawn_frame.command_buffer, win->render_pass(), drawn_frame.framebuffer, drawn_frame.extent, *drawn);
        if (auto fence_result = detail::try_end_frame(*win, drawn_frame); !fence_result) {
          stdexec::set_error(std::move(receiver), std::move(fence_result.error()));
          return;
        }
        stdexec::set_value(std::move(receiver));
      });
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) const noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .pipeline = pipeline,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }
};

//! Async variant of `draw_mesh_sender` (fence-agent completion via `| submit`).
struct draw_mesh_async_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  owned::graphics_pipeline *pipeline{ nullptr };
  mesh_draw const *drawn{ nullptr };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  explicit draw_mesh_async_sender(draw_mesh_sender snd)
    : state(std::move(snd.state)), win(snd.win), pipeline(snd.pipeline), drawn(snd.drawn)
  {}

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    owned::graphics_pipeline *pipeline{};
    mesh_draw const *drawn{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        state,
        win,
        [this](frame &drawn_frame) -> result<VkFence> {
          pipeline->record_pass(
            drawn_frame.command_buffer, win->render_pass(), drawn_frame.framebuffer, drawn_frame.extent, *drawn);
          return detail::try_end_frame(*win, drawn_frame);
        },
        receiver);
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .pipeline = pipeline,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .pipeline = pipeline,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }
};

inline auto operator|(schedule_sender snd, draw_mesh_closure closure) -> draw_mesh_sender
{
  return draw_mesh_sender{
    .state = std::move(snd.state),
    .win = closure.win,
    .pipeline = closure.pipeline,
    .drawn = closure.drawn,
  };
}

inline auto operator|(schedule_sender snd, draw_layers_closure closure) -> draw_layers_sender
{ return draw_layers_sender{ .state = std::move(snd.state), .win = closure.win, .layers = std::move(closure.layers) }; }

[[nodiscard]] inline auto operator|(draw_sender snd, submit_t /*tag*/) -> draw_async_sender
{ return draw_async_sender{ std::move(snd) }; }

[[nodiscard]] inline auto operator|(draw_mesh_sender snd, submit_t /*tag*/) -> draw_mesh_async_sender
{ return draw_mesh_async_sender{ std::move(snd) }; }

[[nodiscard]] inline auto operator|(draw_layers_sender &&snd, submit_t /*tag*/) -> draw_layers_async_sender
{ return draw_layers_async_sender{ std::move(snd) }; }

[[nodiscard]] inline auto operator|(draw_layers_sender const &snd, submit_t /*tag*/) -> draw_layers_async_sender
{ return draw_layers_async_sender{ draw_layers_sender{ .state = snd.state, .win = snd.win, .layers = snd.layers } }; }

} // namespace vkexec

#endif// VKEXEC_GRAPHICS_DRAW_LAYERS_HPP
