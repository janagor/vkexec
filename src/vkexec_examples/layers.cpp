#include <vkexec/error.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <print>
#include <string_view>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_fullscreen_vertices = 3;
constexpr std::uint32_t k_foreground_vertices = 3;
constexpr float k_clear_r = 0.05F;
constexpr float k_clear_g = 0.05F;
constexpr float k_clear_b = 0.08F;

constexpr std::string_view k_background_vert = R"(
#version 450
layout(location = 0) out vec3 vColor;
void main() {
  vec2 pos[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  vec3 col[3] = vec3[](vec3(0.08, 0.10, 0.35), vec3(0.45, 0.12, 0.55), vec3(0.10, 0.40, 0.50));
  gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
  vColor = col[gl_VertexIndex];
}
)";

constexpr std::string_view k_gradient_frag = R"(
#version 450
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(vColor, 1.0);
}
)";

constexpr std::string_view k_foreground_vert = R"(
#version 450
layout(location = 0) out vec3 vColor;
void main() {
  vec2 pos[3] = vec2[](vec2(0.0, -0.35), vec2(0.35, 0.35), vec2(-0.35, 0.35));
  vec3 col[3] = vec3[](vec3(1.0, 0.95, 0.2), vec3(1.0, 0.35, 0.55), vec3(0.35, 0.85, 1.0));
  gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
  vColor = col[gl_VertexIndex];
}
)";

constexpr std::string_view k_tinted_frag = R"(
#version 450
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(vColor, 0.85);
}
)";
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    auto win = vkexec::value_or_throw(vkexec::window::create(
      { .width = k_window_width, .height = k_window_height, .title = "vkexec layers", .validation_layers = true }));

    vkexec::graphics_pipeline_config background_cfg{};
    background_cfg.clear_r = k_clear_r;
    background_cfg.clear_g = k_clear_g;
    background_cfg.clear_b = k_clear_b;

    auto background = vkexec::value_or_throw(vkexec::graphics_pipeline::create(
      win.ctx(), win.render_pass(), background_cfg, k_background_vert, k_gradient_frag));

    vkexec::graphics_pipeline_config foreground_cfg{};
    foreground_cfg.alpha_blend = true;

    auto foreground = vkexec::value_or_throw(vkexec::graphics_pipeline::create(
      win.ctx(), win.render_pass(), foreground_cfg, k_foreground_vert, k_tinted_frag));

    std::println("vkexec layers (two graphics pipelines) - close the window to exit");

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
  } catch (vkexec::error const &error) {
    std::println(stderr, "vkexec layers example failed: {}", error.message());
    return 1;
  }
}
