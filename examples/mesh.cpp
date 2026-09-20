#include "load_gltf_mesh.hpp"
#include "sync_wait_helpers.hpp"

#include "glfw_presenter.hpp"
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/mesh.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <format>
#include <iostream>
#include <string_view>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;

#ifndef VKEXEC_MESH_GLTF_PATH
#error "VKEXEC_MESH_GLTF_PATH must be defined by the mesh example target"
#endif

constexpr char const *k_gltf_path = VKEXEC_MESH_GLTF_PATH;

constexpr std::string_view k_mesh_vert = R"(
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;
void main() {
  gl_Position = vec4(inPosition, 1.0);
  vColor = inColor;
}
)";

constexpr std::string_view k_mesh_frag = R"(
#version 450
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(vColor, 1.0);
}
)";
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
static auto run() -> int
{
  auto win = vkexec::examples::glfw_presenter::create(
    { .width = k_window_width, .height = k_window_height, .title = "vkexec mesh", .validation_layers = true });
  auto mesh_data = vkexec::examples::sync_wait_value(vkexec::examples::load_gltf_mesh(k_gltf_path));
  auto drawn =
    vkexec::examples::sync_wait_value(vkexec::mesh::create(win.ctx(), mesh_data.vertices, mesh_data.indices));

  vkexec::graphics_pipeline_config const cfg{
    .depth_test = true,
    .use_mesh_vertices = true,
  };
  auto pipeline = vkexec::examples::sync_wait_value(
    vkexec::graphics_pipeline::create(win.ctx(), win.render_pass(), cfg, k_mesh_vert, k_mesh_frag));

  std::cout << std::format("vkexec indexed mesh (gltf: {}) - close the window to exit\n", k_gltf_path);
  while (!win.should_close()) {
    win.poll_events();
    vkexec::examples::sync_wait_graph(
      ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win.target(), pipeline, drawn));
  }
  win.wait_idle();
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
