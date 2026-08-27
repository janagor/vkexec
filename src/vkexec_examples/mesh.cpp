#include "load_gltf_mesh.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/leaf/result.hpp>
#include <vkexec/error.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
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
  return vkexec::leaf::try_handle_all(
    []() -> vkexec::leaf::result<int> {
      VKEXEC_LEAF_AUTO(win,
        vkexec::window::create(
          { .width = k_window_width, .height = k_window_height, .title = "vkexec mesh", .validation_layers = true }));
      VKEXEC_LEAF_AUTO(mesh_data, vkexec::examples::load_gltf_mesh(k_gltf_path));
      VKEXEC_LEAF_AUTO(drawn, vkexec::mesh::create(win.ctx(), mesh_data.vertices, mesh_data.indices));

      vkexec::graphics_pipeline_config const cfg{
        .depth_test = true,
        .use_mesh_vertices = true,
      };
      VKEXEC_LEAF_AUTO(pipeline,
        vkexec::graphics_pipeline::create(
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
          }));

      std::println("vkexec indexed mesh (gltf: {}) - close the window to exit", k_gltf_path);
      while (!win.should_close()) {
        win.poll_events();
        (void)vkexec::sync_wait(ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, drawn));
      }
      win.wait_idle();
      return 0;
    },
    [](vkexec::error const &error) -> int {
      std::println(stderr, "vkexec mesh example failed: {}", error.message());
      return 1;
    },
    [] -> int {
      std::println(stderr, "vkexec mesh example failed");
      return 1;
    });
}
