#include "load_gltf_mesh.hpp"

#include <vkexec_edsl/types.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <memory>
#include <print>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;

#ifndef VKEXEC_MESH_GLTF_PATH
#error "VKEXEC_MESH_GLTF_PATH must be defined by the mesh example target"
#endif

constexpr char const *k_gltf_path = VKEXEC_MESH_GLTF_PATH;
}// namespace

auto main() -> int
{
  auto win_result = vkexec::window::create({ .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec mesh",
    .validation_layers = true });
  if (!win_result) {
    std::println(stderr, "vkexec mesh example failed: {}", win_result.error().message());
    return 1;
  }
  auto win = std::move(*win_result);

  auto mesh_data = vkexec::examples::load_gltf_mesh(k_gltf_path);
  if (!mesh_data) {
    std::println(stderr, "vkexec mesh example failed: {}", mesh_data.error().message());
    return 1;
  }
  auto drawn_result = vkexec::mesh::create(win.ctx(), mesh_data->vertices, mesh_data->indices);
  if (!drawn_result) {
    std::println(stderr, "vkexec mesh example failed: {}", drawn_result.error().message());
    return 1;
  }
  auto &drawn = *drawn_result;

  vkexec::graphics_pipeline_config const cfg{
    .depth_test = true,
    .use_mesh_vertices = true,
  };
  auto pipeline_result = vkexec::graphics_pipeline::create(
    win.ctx(),
    win.render_pass(),
    cfg,
    [](edsl::Int vertex_id, edsl::VertexWriter out) -> void {
      (void)vertex_id;
      out.position(edsl::vec4(edsl::VertexIn::position(), 1.0));
      out.color(edsl::VertexIn::color());
    },
    [](edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void {
      out.color(edsl::vec4(fragment_in.color(), 1.0));
    });
  if (!pipeline_result) {
    std::println(stderr, "vkexec mesh example failed: {}", pipeline_result.error().message());
    return 1;
  }
  auto &pipeline = *pipeline_result;

  std::println("vkexec indexed mesh (gltf: {}) - close the window to exit", k_gltf_path);
  while (!win.should_close()) {
    win.poll_events();
    (void)vkexec::sync_wait(ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, drawn));
  }
  win.wait_idle();
  return 0;
}
