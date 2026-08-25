#include <vkexec_edsl/types.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
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
constexpr std::uint32_t k_fullscreen_vertices = 3;
constexpr std::uint32_t k_foreground_vertices = 3;
constexpr float k_foreground_alpha = 0.85F;
constexpr float k_clear_r = 0.05F;
constexpr float k_clear_g = 0.05F;
constexpr float k_clear_b = 0.08F;

auto fullscreen_vertex(edsl::Int vertex_id, edsl::VertexWriter out) -> void
{
  edsl::Float2 const pos = edsl::select(vertex_id == edsl::Int::constant(0),
    edsl::vec2(-1.0, -1.0),
    edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec2(3.0, -1.0), edsl::vec2(-1.0, 3.0)));
  edsl::Float3 const corner_color = edsl::select(vertex_id == edsl::Int::constant(0),
    edsl::vec3(0.08, 0.10, 0.35),
    edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec3(0.45, 0.12, 0.55), edsl::vec3(0.10, 0.40, 0.50)));
  out.position(pos);
  out.color(corner_color);
}

auto gradient_fragment(edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void
{ out.color(edsl::vec4(fragment_in.color(), edsl::Float::constant(1.0))); }

auto foreground_vertex(edsl::Int vertex_id, edsl::VertexWriter out) -> void
{
  edsl::Float2 const pos = edsl::select(vertex_id == edsl::Int::constant(0),
    edsl::vec2(0.0, -0.35),
    edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec2(0.35, 0.35), edsl::vec2(-0.35, 0.35)));
  edsl::Float3 const col = edsl::select(vertex_id == edsl::Int::constant(0),
    edsl::vec3(1.0, 0.95, 0.2),
    edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec3(1.0, 0.35, 0.55), edsl::vec3(0.35, 0.85, 1.0)));
  out.position(pos);
  out.color(col);
}

auto tinted_fragment(edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void
{ out.color(edsl::vec4(fragment_in.color(), edsl::Float::constant(static_cast<double>(k_foreground_alpha)))); }
}// namespace

auto main() -> int
{
  auto win_result = vkexec::window::create({ .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec layers",
    .validation_layers = true });
  if (!win_result) {
    std::println(stderr, "vkexec layers example failed: {}", win_result.error().message());
    return 1;
  }
  auto win = std::move(*win_result);

  vkexec::graphics_pipeline_config background_cfg{};
  background_cfg.clear_r = k_clear_r;
  background_cfg.clear_g = k_clear_g;
  background_cfg.clear_b = k_clear_b;

  auto background_result =
    vkexec::graphics_pipeline::create(win.ctx(), win.render_pass(), background_cfg, fullscreen_vertex, gradient_fragment);
  if (!background_result) {
    std::println(stderr, "vkexec layers example failed: {}", background_result.error().message());
    return 1;
  }
  auto &background = *background_result;

  vkexec::graphics_pipeline_config foreground_cfg{};
  foreground_cfg.alpha_blend = true;

  auto foreground_result =
    vkexec::graphics_pipeline::create(win.ctx(), win.render_pass(), foreground_cfg, foreground_vertex, tinted_fragment);
  if (!foreground_result) {
    std::println(stderr, "vkexec layers example failed: {}", foreground_result.error().message());
    return 1;
  }
  auto &foreground = *foreground_result;

  std::println("vkexec layers (two graphics pipelines, four traced shaders) - close the window to exit");

  while (!win.should_close()) {
    win.poll_events();
    (void)vkexec::sync_wait(ex::schedule(win.ctx().get_scheduler())
                            | vkexec::draw_layers(win,
                              {
                                { .pipeline = &background, .vertex_count = k_fullscreen_vertices },
                                { .pipeline = &foreground, .vertex_count = k_foreground_vertices },
                              }));
  }

  win.wait_idle();
  return 0;
}
