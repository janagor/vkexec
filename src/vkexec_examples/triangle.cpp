#include <vkexec/error.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <print>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_triangle_vertices = 3;
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    auto win = vkexec::value_or_throw(vkexec::window::create({ .width = k_window_width,
      .height = k_window_height,
      .title = "vkexec triangle",
      .validation_layers = true }));

    auto pipeline = vkexec::value_or_throw(vkexec::graphics_pipeline::create(
      win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));

    std::println("vkexec triangle (stdexec frame pipeline) - close the window to exit");
    while (!win.should_close()) {
      win.poll_events();
      (void)vkexec::sync_wait(
        ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices));
    }
    win.wait_idle();
    return 0;
  } catch (vkexec::error const &error) {
    std::println(stderr, "vkexec triangle example failed: {}", error.message());
    return 1;
  }
}
