#pragma once

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
inline draw_closure draw(window &win, graphics_pipeline &pipeline, std::uint32_t vertex_count)
{
  return draw_closure{ &win, &pipeline, vertex_count };
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

    void start() noexcept
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
  auto connect(Receiver receiver) const
  {
    return op_state<Receiver>{ win, pipeline, vertex_count, std::move(receiver) };
  }
};

inline draw_sender operator|(schedule_sender /*snd*/, draw_closure cl)
{
  return draw_sender{ cl.win, cl.pipeline, cl.vertex_count };
}

} // namespace vkexec
