#include "sync_wait_helpers.hpp"

#include "glfw_presenter.hpp"
#include <vkexec/result.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

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
  auto win = vkexec::examples::glfw_presenter::create({ .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec graphics execution",
    .validation_layers = true });

  auto resources_result = vkexec::create(win.ctx(),
    win.render_pass(),
    vkexec::graphics_pipeline_config{},
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag);
  if (!resources_result) { vkexec::examples::abort_with_error(resources_result.error()); }
  auto resources = vkexec::expected_take(resources_result);

  std::cout << std::format("vkexec graphics execution (borrowed pipeline handles) - close the window to exit\n");
  while (!win.should_close()) {
    win.poll_events();
    vkexec::examples::sync_wait_graph(ex::schedule(win.ctx().get_scheduler())
                                      | vkexec::draw(win.target(), resources, VK_NULL_HANDLE, k_triangle_vertices));
  }

  win.wait_idle();
  vkexec::destroy(win.ctx(), resources);
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
