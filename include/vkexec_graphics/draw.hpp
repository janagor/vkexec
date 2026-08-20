#ifndef VKEXEC_GRAPHICS_DRAW_HPP
#define VKEXEC_GRAPHICS_DRAW_HPP

#include <vkexec/detail/config.hpp>
#include <vkexec/detail/submit_scope.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  template<class Receiver>
  auto complete_draw(Receiver &&receiver, std::exception_ptr error, bool stopped) -> void
  { complete_after_reclaim(std::forward<Receiver>(receiver), std::move(error), stopped); }

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

    std::exception_ptr error;
    VKEXEC_TRY
    {
      // cppcheck-suppress throwInNoexceptFunction
      if (auto frame = win->begin_frame()) {
        // NOLINTNEXTLINE(misc-misplaced-const)
        VkFence fence = std::forward<WindowOp>(record_and_end)(*frame);
        ctx->enqueue_borrowed_fence_wait(fence,
          token,
          [rcvr = std::move(rcvr)](std::exception_ptr wait_error, bool stopped) mutable -> void {
            complete_draw(std::move(rcvr), std::move(wait_error), stopped);
          });
        return;
      }
    }
    VKEXEC_CATCH_ALL { error = std::current_exception(); }
    if (error) { ex::set_error(std::move(rcvr), error); }
    else {
      ex::set_value(std::move(rcvr));
    }
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
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

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
      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        if (auto frame = win->begin_frame()) {
          pipeline->draw(frame->command_buffer, win->render_pass(), frame->framebuffer, frame->extent, vertex_count);
          (void)win->end_frame(*frame);
        }
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
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
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

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
      detail::start_draw_async(ctx,
        win,
        [this](frame &drawn) -> VkFence {
          pipeline->draw(drawn.command_buffer, win->render_pass(), drawn.framebuffer, drawn.extent, vertex_count);
          return win->end_frame(drawn);
        },
        std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .win = self.win,
      .pipeline = self.pipeline,
      .vertex_count = self.vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_layers_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

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
      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        if (layers.empty()) { VKEXEC_THROW(std::invalid_argument("draw_layers requires at least one layer")); }
        if (auto frame = win->begin_frame()) {
          graphics_pipeline_config const &clear_cfg = layers.front().pipeline->config();
          std::array<VkClearValue, k_graphics_clear_count> const clears = make_clear_values(clear_cfg);

          VkRenderPassBeginInfo rp_begin{};
          rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
          rp_begin.renderPass = win->render_pass();
          rp_begin.framebuffer = frame->framebuffer;
          rp_begin.renderArea.offset = { .x = 0, .y = 0 };
          rp_begin.renderArea.extent = frame->extent;
          rp_begin.clearValueCount = k_graphics_clear_count;
          rp_begin.pClearValues = clears.data();

          vkCmdBeginRenderPass(frame->command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
          for (draw_layer const &layer : layers) {
            layer.pipeline->record_draw(frame->command_buffer, frame->extent, layer.vertex_count);
          }
          vkCmdEndRenderPass(frame->command_buffer);
          if (vkEndCommandBuffer(frame->command_buffer) != VK_SUCCESS) {
            VKEXEC_THROW(std::runtime_error("vkEndCommandBuffer failed"));
          }
          (void)win->end_frame(*frame);
        }
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  { return op_state<Receiver>{ .win = win, .layers = layers, .receiver = std::move(receiver) }; }
};

struct draw_layers_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  window *win{ nullptr };
  std::vector<draw_layer> layers;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  explicit draw_layers_async_sender(draw_layers_sender snd)
    : ctx(snd.ctx), win(snd.win), layers(std::move(snd.layers))
  {}

  template<class Receiver> struct op_state
  {
    context *ctx{};
    window *win{};
    std::vector<draw_layer> layers;
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(ctx,
        win,
        [this](frame &drawn_frame) -> VkFence {
          if (layers.empty()) { VKEXEC_THROW(std::invalid_argument("draw_layers requires at least one layer")); }
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
          if (vkEndCommandBuffer(drawn_frame.command_buffer) != VK_SUCCESS) {
            VKEXEC_THROW(std::runtime_error("vkEndCommandBuffer failed"));
          }
          return win->end_frame(drawn_frame);
        },
        std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .win = self.win,
      .layers = std::forward_like<decltype(self)>(self.layers),
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
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

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
      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        if (auto frame = win->begin_frame()) {
          pipeline->draw(frame->command_buffer, win->render_pass(), frame->framebuffer, frame->extent, *drawn);
          (void)win->end_frame(*frame);
        }
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
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
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

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
      detail::start_draw_async(ctx,
        win,
        [this](frame &drawn_frame) -> VkFence {
          pipeline->draw(
            drawn_frame.command_buffer, win->render_pass(), drawn_frame.framebuffer, drawn_frame.extent, *drawn);
          return win->end_frame(drawn_frame);
        },
        std::move(receiver));
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .win = self.win,
      .pipeline = self.pipeline,
      .drawn = self.drawn,
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
