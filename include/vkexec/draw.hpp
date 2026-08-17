#ifndef VKEXEC_DRAW_HPP
#define VKEXEC_DRAW_HPP


#include <vkexec/graphics.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

struct draw_closure {
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

struct draw_layer {
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

struct draw_layers_closure {
  window *win{ nullptr };
  std::vector<draw_layer> layers;
};

/// Present one frame: acquire → record draw → submit → present (stdexec sender adaptor).
inline auto draw(window &win, graphics_pipeline &pipeline, std::uint32_t vertex_count) -> draw_closure
{
  return draw_closure{ .win = &win, .pipeline = &pipeline, .vertex_count = vertex_count };
}

/// Present one frame using multiple graphics pipelines in a single render pass.
inline auto draw_layers(window &win, std::initializer_list<draw_layer> layers) -> draw_layers_closure
{
  return draw_layers_closure{ .win = &win, .layers = std::vector<draw_layer>(layers) };
}

struct draw_sender {
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };

  template<class Receiver>
  struct op_state {
    window *win;
    graphics_pipeline *pipeline;
    std::uint32_t vertex_count;
    Receiver receiver;

    auto start() noexcept -> void
    {
      std::exception_ptr error;
      try {
        if (auto frame = win->begin_frame()) {
          pipeline->draw(frame->command_buffer, win->render_pass(), frame->framebuffer, frame->extent, vertex_count);
          win->end_frame(*frame);
        }
      } catch (...) {
        error = std::current_exception();
      }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_layers_sender {
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

  window *win{ nullptr };
  std::vector<draw_layer> layers;

  template<class Receiver>
  struct op_state {
    window *win{};
    std::vector<draw_layer> layers;
    Receiver receiver;

    auto start() noexcept -> void
    {
      std::exception_ptr error;
      try {
        if (layers.empty()) { throw std::invalid_argument("draw_layers requires at least one layer"); }
        if (auto frame = win->begin_frame()) {
          graphics_pipeline_config const &clear_cfg = layers.front().pipeline->config();
          VkClearValue clear{};
          clear.color = { { clear_cfg.clear_r, clear_cfg.clear_g, clear_cfg.clear_b, clear_cfg.clear_a } };

          VkRenderPassBeginInfo rp_begin{};
          rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
          rp_begin.renderPass = win->render_pass();
          rp_begin.framebuffer = frame->framebuffer;
          rp_begin.renderArea.offset = { .x = 0, .y = 0 };
          rp_begin.renderArea.extent = frame->extent;
          rp_begin.clearValueCount = 1;
          rp_begin.pClearValues = &clear;

          vkCmdBeginRenderPass(frame->command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
          for (draw_layer const &layer : layers) {
            layer.pipeline->record_draw(frame->command_buffer, frame->extent, layer.vertex_count);
          }
          vkCmdEndRenderPass(frame->command_buffer);
          if (vkEndCommandBuffer(frame->command_buffer) != VK_SUCCESS) {
            throw std::runtime_error("vkEndCommandBuffer failed");
          }
          win->end_frame(*frame);
        }
      } catch (...) {
        error = std::current_exception();
      }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  {
    return op_state<Receiver>{ .win = win, .layers = layers, .receiver = std::move(receiver) };
  }
};

inline auto operator|(schedule_sender /*snd*/, draw_closure closure) -> draw_sender
{
  return draw_sender{ .win = closure.win, .pipeline = closure.pipeline, .vertex_count = closure.vertex_count };
}

inline auto operator|(schedule_sender /*snd*/, draw_layers_closure closure) -> draw_layers_sender
{
  return draw_layers_sender{ .win = closure.win, .layers = std::move(closure.layers) };
}

} // namespace vkexec

#endif  // VKEXEC_DRAW_HPP
