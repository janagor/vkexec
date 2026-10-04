#include "common/shader_loader.hpp"
#include "glfw_presenter.hpp"
#include "sync_wait_helpers.hpp"

#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_triangle_vertices = 3;

// NOLINTNEXTLINE(bugprone-exception-escape)
auto run() -> int
{
  auto win = vkexec::examples::glfw_presenter::create({
    .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec Vulkan Samples: Hello Triangle",
    .validation_layers = true,
  });

  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  auto const vertex_shader = vkexec::examples::load_spirv(shader_dir / "triangle.vert.spv");
  auto const fragment_shader = vkexec::examples::load_spirv(shader_dir / "triangle.frag.spv");
  auto pipeline = vkexec::examples::sync_wait_value(
    vkexec::factory::make_graphics_pipeline(win.ctx(), win.render_pass(), vertex_shader, fragment_shader));

  std::cout << "vkexec Hello Triangle - close the window to exit\n";
  while (!win.should_close()) {
    win.poll_events();
    vkexec::examples::sync_wait_graph(
      ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win.target(), pipeline, k_triangle_vertices));
  }
  win.wait_idle();
  return 0;
}
}// namespace

auto main() -> int { return vkexec::examples::run_example(run); }
