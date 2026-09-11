#ifndef VKEXEC_GRAPHICS_DRAW_HPP
#define VKEXEC_GRAPHICS_DRAW_HPP

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  [[nodiscard]] inline auto try_begin_frame(window &win) -> result<std::optional<frame>> { return win.begin_frame(); }

  [[nodiscard]] inline auto try_end_frame(window &win, frame const &drawn) -> result<VkFence>
  { return win.end_frame(drawn); }

  template<class Receiver> auto complete_draw(Receiver &&receiver, std::optional<error> failure, bool stopped) -> void
  {
    if (failure) {
      ex::set_error(std::forward<Receiver>(receiver), std::move(*failure));
    } else if (stopped) {
      ex::set_stopped(std::forward<Receiver>(receiver));
    } else {
      ex::set_value(std::forward<Receiver>(receiver));
    }
  }

  template<class WindowOp, class Receiver>
  auto start_draw_async(context *ctx, window *win, WindowOp &&record_and_end, Receiver receiver) -> void
  {
    Receiver rcvr = std::move(receiver);
    auto const token = ex::get_stop_token(ex::get_env(rcvr));
    if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
      if (token.stop_requested()) {
        ex::set_stopped(std::move(rcvr));
        return;
      }
    }

    auto frame_result = try_begin_frame(*win);
    if (!frame_result) {
      ex::set_error(std::move(rcvr), std::move(frame_result.error()));
      return;
    }
    if (!frame_result->has_value()) {
      ex::set_value(std::move(rcvr));
      return;
    }

    auto fence_result = std::forward<WindowOp>(record_and_end)(**frame_result);
    if (!fence_result) {
      ex::set_error(std::move(rcvr), std::move(fence_result.error()));
      return;
    }

    (void)ctx->enqueue_borrowed_fence_wait(
      *fence_result, token, [rcvr = std::move(rcvr)](std::optional<error> wait_error, bool stopped) mutable -> void {
        complete_draw(std::move(rcvr), std::move(wait_error), stopped);
      });
  }

}// namespace detail

struct draw_closure
{
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

struct draw_mesh_closure
{
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  mesh const *drawn{ nullptr };
};

struct draw_layer
{
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

struct draw_layers_closure
{
  window *win{ nullptr };
  std::vector<draw_layer> layers;
};

/// Present one frame: acquire → record draw → submit → present (stdexec sender adaptor).
inline auto draw(window &win, graphics_pipeline &pipeline, std::uint32_t vertex_count) -> draw_closure
{ return draw_closure{ .win = &win, .pipeline = &pipeline, .vertex_count = vertex_count }; }

inline auto draw(window &win, graphics_pipeline &pipeline, mesh const &drawn) -> draw_mesh_closure
{ return draw_mesh_closure{ .win = &win, .pipeline = &pipeline, .drawn = &drawn }; }

/// Present one frame using multiple graphics pipelines in a single render pass.
inline auto draw_layers(window &win, std::initializer_list<draw_layer> layers) -> draw_layers_closure
{ return draw_layers_closure{ .win = &win, .layers = std::vector<draw_layer>(layers) }; }

struct draw_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    window *win;
    graphics_pipeline *pipeline;
    std::uint32_t vertex_count;
    Receiver receiver;

    auto start() noexcept -> void
    {
      auto frame_result = detail::try_begin_frame(*win);
      if (!frame_result) {
        ex::set_error(std::move(receiver), std::move(frame_result.error()));
        return;
      }
      if (!frame_result->has_value()) {
        ex::set_value(std::move(receiver));
        return;
      }

      frame const &drawn = **frame_result;
      if (auto draw_status =
            pipeline->draw(drawn.command_buffer, win->render_pass(), drawn.framebuffer, drawn.extent, vertex_count);
        !draw_status) {
        ex::set_error(std::move(receiver), std::move(draw_status.error()));
        return;
      }
      if (auto fence_result = detail::try_end_frame(*win, drawn); !fence_result) {
        ex::set_error(std::move(receiver), std::move(fence_result.error()));
        return;
      }
      ex::set_value(std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  explicit draw_async_sender(draw_sender snd)
    : ctx(snd.ctx), win(snd.win), pipeline(snd.pipeline), vertex_count(snd.vertex_count)
  {}

  template<class Receiver> struct op_state
  {
    context *ctx{};
    window *win{};
    graphics_pipeline *pipeline{};
    std::uint32_t vertex_count{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        ctx,
        win,
        [this](frame &drawn) -> result<VkFence> {
          if (auto draw_status =
                pipeline->draw(drawn.command_buffer, win->render_pass(), drawn.framebuffer, drawn.extent, vertex_count);
            !draw_status) {
            return fail(draw_status);
          }
          return detail::try_end_frame(*win, drawn);
        },
        std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_layers_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  std::vector<draw_layer> layers;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    window *win{};
    std::vector<draw_layer> layers;
    Receiver receiver;

    auto start() noexcept -> void
    {
      if (layers.empty()) {
        ex::set_error(
          std::move(receiver), make_error(errc::invalid_argument, "draw_layers requires at least one layer"));
        return;
      }

      auto frame_result = detail::try_begin_frame(*win);
      if (!frame_result) {
        ex::set_error(std::move(receiver), std::move(frame_result.error()));
        return;
      }
      if (!frame_result->has_value()) {
        ex::set_value(std::move(receiver));
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
      if (VkResult const end_result = vkEndCommandBuffer(drawn.command_buffer); end_result != VK_SUCCESS) {
        ex::set_error(std::move(receiver), make_vk_error(end_result, "vkEndCommandBuffer failed"));
        return;
      }
      if (auto fence_result = detail::try_end_frame(*win, drawn); !fence_result) {
        ex::set_error(std::move(receiver), std::move(fence_result.error()));
        return;
      }
      ex::set_value(std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  { return op_state<Receiver>{ .win = win, .layers = layers, .receiver = std::move(receiver) }; }
};

struct draw_layers_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  std::vector<draw_layer> layers;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  explicit draw_layers_async_sender(draw_layers_sender snd) : ctx(snd.ctx), win(snd.win), layers(std::move(snd.layers))
  {}

  template<class Receiver> struct op_state
  {
    context *ctx{};
    window *win{};
    std::vector<draw_layer> layers;
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        ctx,
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
          if (VkResult const end_result = vkEndCommandBuffer(drawn_frame.command_buffer); end_result != VK_SUCCESS) {
            return fail(end_result, "vkEndCommandBuffer failed");
          }
          return detail::try_end_frame(*win, drawn_frame);
        },
        std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .win = win,
      .layers = layers,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .win = win,
      .layers = std::move(layers),
      .receiver = std::move(receiver),
    };
  }
};

inline auto operator|(schedule_sender snd, draw_closure closure) -> draw_sender
{
  return draw_sender{
    .ctx = snd.ctx,
    .win = closure.win,
    .pipeline = closure.pipeline,
    .vertex_count = closure.vertex_count,
  };
}

struct draw_mesh_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  mesh const *drawn{ nullptr };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    window *win;
    graphics_pipeline *pipeline;
    mesh const *drawn;
    Receiver receiver;

    auto start() noexcept -> void
    {
      auto frame_result = detail::try_begin_frame(*win);
      if (!frame_result) {
        ex::set_error(std::move(receiver), std::move(frame_result.error()));
        return;
      }
      if (!frame_result->has_value()) {
        ex::set_value(std::move(receiver));
        return;
      }

      frame const &drawn_frame = **frame_result;
      if (auto draw_status = pipeline->draw(
            drawn_frame.command_buffer, win->render_pass(), drawn_frame.framebuffer, drawn_frame.extent, *drawn);
        !draw_status) {
        ex::set_error(std::move(receiver), std::move(draw_status.error()));
        return;
      }
      if (auto fence_result = detail::try_end_frame(*win, drawn_frame); !fence_result) {
        ex::set_error(std::move(receiver), std::move(fence_result.error()));
        return;
      }
      ex::set_value(std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .win = win,
      .pipeline = pipeline,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_mesh_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  mesh const *drawn{ nullptr };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  explicit draw_mesh_async_sender(draw_mesh_sender snd)
    : ctx(snd.ctx), win(snd.win), pipeline(snd.pipeline), drawn(snd.drawn)
  {}

  template<class Receiver> struct op_state
  {
    context *ctx{};
    window *win{};
    graphics_pipeline *pipeline{};
    mesh const *drawn{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        ctx,
        win,
        [this](frame &drawn_frame) -> result<VkFence> {
          if (auto draw_status = pipeline->draw(
                drawn_frame.command_buffer, win->render_pass(), drawn_frame.framebuffer, drawn_frame.extent, *drawn);
            !draw_status) {
            return fail(draw_status);
          }
          return detail::try_end_frame(*win, drawn_frame);
        },
        std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .win = win,
      .pipeline = pipeline,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
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
    .ctx = snd.ctx,
    .win = closure.win,
    .pipeline = closure.pipeline,
    .drawn = closure.drawn,
  };
}

inline auto operator|(schedule_sender snd, draw_layers_closure closure) -> draw_layers_sender
{ return draw_layers_sender{ .ctx = snd.ctx, .win = closure.win, .layers = std::move(closure.layers) }; }

[[nodiscard]] inline auto operator|(draw_sender snd, submit_t /*tag*/) -> draw_async_sender
{ return draw_async_sender{ snd }; }

[[nodiscard]] inline auto operator|(draw_mesh_sender snd, submit_t /*tag*/) -> draw_mesh_async_sender
{ return draw_mesh_async_sender{ snd }; }

[[nodiscard]] inline auto operator|(draw_layers_sender &&snd, submit_t /*tag*/) -> draw_layers_async_sender
{ return draw_layers_async_sender{ std::move(snd) }; }

[[nodiscard]] inline auto operator|(draw_layers_sender const &snd, submit_t /*tag*/) -> draw_layers_async_sender
{ return draw_layers_async_sender{ draw_layers_sender{ .ctx = snd.ctx, .win = snd.win, .layers = snd.layers } }; }

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_DRAW_HPP
