#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <exception>
#include <print>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_triangle_vertices = 3;
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::window win({ .width = k_window_width, .height = k_window_height, .title = "vkexec triangle" });

    // Vertex + fragment shaders traced from C++ (AST -> GLSL -> SPIR-V).
    vkexec::graphics_pipeline pipeline(
      win.ctx(),
      win.render_pass(),
      [](edsl::Int vertex_id, edsl::VertexWriter out) -> void {
        edsl::Float2 const pos = edsl::select(vertex_id == edsl::Int::constant(0),
          edsl::vec2(0.0, -0.5),
          edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec2(0.5, 0.5), edsl::vec2(-0.5, 0.5)));
        edsl::Float3 const col = edsl::select(vertex_id == edsl::Int::constant(0),
          edsl::vec3(1.0, 0.2, 0.2),
          edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec3(0.2, 1.0, 0.2), edsl::vec3(0.2, 0.4, 1.0)));
        out.position(pos);
        out.color(col);
      },
      [](edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void {
        out.color(edsl::vec4(fragment_in.color(), 1.0));
      });

    std::println("vkexec traced triangle (stdexec frame pipeline) - close the window to exit");
    while (!win.should_close()) {
      win.poll_events();
      // Same shape as compute: schedule | algorithm | sync_wait
      (void)ex::sync_wait(ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices));
    }
    win.wait_idle();
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec triangle example failed: {}", ex.what());
    return 1;
  }
}
