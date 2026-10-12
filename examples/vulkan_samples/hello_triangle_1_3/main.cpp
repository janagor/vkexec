#include "common/shader_loader.hpp"
#include "common/smoke_frames.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
#include "sync_wait_helpers.hpp"

#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_triangle_vertices = 3;

// NOLINTNEXTLINE(bugprone-exception-escape)
auto run(int argc, char const *const *argv) -> int
{
  auto frames = vkexec::examples::smoke_frames::parse(argc, argv);
  vkexec::examples::smoke_frames::reject_other_arguments(argc, argv);
  auto win = vkexec::examples::glfw_presenter::create({
    .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec Vulkan Samples: Hello Triangle",
    .validation_layers = true,
    .requirements = vkexec::examples::vulkan_sample_requirements(),
  });

  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  auto const vertex_shader = vkexec::examples::load_spirv(shader_dir / "triangle.vert.spv");
  auto const fragment_shader = vkexec::examples::load_spirv(shader_dir / "triangle.frag.spv");
  auto pipeline = vkexec::examples::sync_wait_value(
    vkexec::factory::make_graphics_pipeline(win.ctx(), win.render_pass(), vertex_shader, fragment_shader));

  std::cout << "vkexec Hello Triangle - close the window to exit\n";
  while (!win.should_close() && frames.should_continue()) {
    win.poll_events();
    vkexec::examples::sync_wait_graph(
      stdexec::schedule(win.ctx().get_scheduler()) | vkexec::draw(win.target(), pipeline, k_triangle_vertices));
    frames.schedule();
  }
  win.wait_idle();
  frames.mark_all_drained();
  frames.verify_complete();
  return 0;
}
}// namespace

auto main(int argc, char const *const *argv) -> int
{
  return vkexec::examples::run_example([argc, argv]() -> int { return run(argc, argv); });
}
