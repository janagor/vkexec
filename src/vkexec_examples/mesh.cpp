#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <print>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr float k_near_depth = 0.3F;
constexpr float k_far_depth = 0.7F;
constexpr std::size_t k_quad_vertex_count = 8;
constexpr std::size_t k_quad_index_count = 12;
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::window win({ .width = k_window_width, .height = k_window_height, .title = "vkexec mesh" });

    std::array<vkexec::mesh_vertex, k_quad_vertex_count> const vertices{{
      { .position = { -0.6F, -0.4F, k_near_depth }, .color = { 0.9F, 0.25F, 0.2F } },
      { .position = { 0.2F, -0.4F, k_near_depth }, .color = { 0.9F, 0.25F, 0.2F } },
      { .position = { 0.2F, 0.4F, k_near_depth }, .color = { 0.9F, 0.25F, 0.2F } },
      { .position = { -0.6F, 0.4F, k_near_depth }, .color = { 0.9F, 0.25F, 0.2F } },
      { .position = { -0.2F, -0.4F, k_far_depth }, .color = { 0.2F, 0.35F, 0.9F } },
      { .position = { 0.6F, -0.4F, k_far_depth }, .color = { 0.2F, 0.35F, 0.9F } },
      { .position = { 0.6F, 0.4F, k_far_depth }, .color = { 0.2F, 0.35F, 0.9F } },
      { .position = { -0.2F, 0.4F, k_far_depth }, .color = { 0.2F, 0.35F, 0.9F } },
    }};
    std::array<std::uint32_t, k_quad_index_count> const indices{ 0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7 };
    vkexec::mesh const drawn(win.ctx(), vertices, indices);

    vkexec::graphics_pipeline_config const cfg{
      .depth_test = true,
      .use_mesh_vertices = true,
    };
    vkexec::graphics_pipeline pipeline(
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

    std::println("vkexec indexed mesh (overlapping quads, depth test) - close the window to exit");
    while (!win.should_close()) {
      win.poll_events();
      (void)ex::sync_wait(ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, drawn));
    }
    win.wait_idle();
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec mesh example failed: {}", ex.what());
    return 1;
  }
}
