#include "sync_wait_helpers.hpp"

#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include "glfw_presenter.hpp"

#include <stdexec/execution.hpp>

#include <cstdint>
#include <format>
#include <iostream>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_triangle_vertices = 3;
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
static auto run() -> int
{
  auto win = vkexec::examples::glfw_presenter::create(
    { .width = k_window_width, .height = k_window_height, .title = "vkexec triangle", .validation_layers = true });

  auto pipeline = vkexec::examples::sync_wait_value(vkexec::graphics_pipeline::create(
    win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));

  std::cout << std::format("vkexec triangle (stdexec frame pipeline) - close the window to exit\n");
  while (!win.should_close()) {
    win.poll_events();
    vkexec::examples::sync_wait_graph(
      ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win.target(), pipeline, k_triangle_vertices));
  }
  win.wait_idle();
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
