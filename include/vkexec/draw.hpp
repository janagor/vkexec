#ifndef VKEXEC_DRAW_HPP
#define VKEXEC_DRAW_HPP


#include <vkexec/graphics.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <exception>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

struct draw_closure {
  window *win{ nullptr };
  graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

/// Present one frame: acquire → record draw → submit → present (stdexec sender adaptor).
inline auto draw(window &win, graphics_pipeline &pipeline, std::uint32_t vertex_count) -> draw_closure
{
  return draw_closure{ .win = &win, .pipeline = &pipeline, .vertex_count = vertex_count };
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
      try {
        if (auto frame = win->begin_frame()) {
          pipeline->draw(frame->command_buffer, win->render_pass(), frame->framebuffer, frame->extent, vertex_count);
          win->end_frame(*frame);
        }
        ex::set_value(std::move(receiver));
      } catch (...) {
        ex::set_error(std::move(receiver), std::current_exception());
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

inline auto operator|(schedule_sender /*snd*/, draw_closure closure) -> draw_sender
{
  return draw_sender{ .win = closure.win, .pipeline = closure.pipeline, .vertex_count = closure.vertex_count };
}

} // namespace vkexec

#endif  // VKEXEC_DRAW_HPP
