#include "sync_wait_helpers.hpp"
#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>

#include "glfw_presenter.hpp"
#include <stdexec/execution.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <random>
#include <string_view>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_particle_count = 8192;
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr float k_aspect = static_cast<float>(k_window_height) / static_cast<float>(k_window_width);
constexpr float k_spawn_radius = 0.25F;
constexpr float k_two_pi = 6.28318530718F;
constexpr float k_epsilon = 1.0E-6F;
constexpr float k_initial_speed = 0.25F;
constexpr float k_max_delta_seconds = 0.05F;
constexpr float k_clear_r = 0.02F;
constexpr float k_clear_g = 0.02F;
constexpr float k_clear_b = 0.05F;
constexpr unsigned k_rng_seed = 42;
constexpr std::uint32_t k_local_size_x = 64;

constexpr std::string_view k_particle_update_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer PosX { float data[]; } pos_x;
layout(set = 0, binding = 1) buffer PosY { float data[]; } pos_y;
layout(set = 0, binding = 2) buffer VelX { float data[]; } vel_x;
layout(set = 0, binding = 3) buffer VelY { float data[]; } vel_y;
layout(push_constant) uniform Push {
  float delta_time;
} pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  float px = pos_x.data[i];
  float py = pos_y.data[i];
  float vx = vel_x.data[i];
  float vy = vel_y.data[i];
  px = px + vx * pc.delta_time;
  py = py + vy * pc.delta_time;
  if (px <= -1.0 || px >= 1.0) vx = -vx;
  if (py <= -1.0 || py >= 1.0) vy = -vy;
  pos_x.data[i] = px;
  pos_y.data[i] = py;
  vel_x.data[i] = vx;
  vel_y.data[i] = vy;
}
)";

constexpr std::string_view k_particle_vert = R"(
#version 450
layout(set = 0, binding = 0) readonly buffer PosX { float data[]; } pos_x;
layout(set = 0, binding = 1) readonly buffer PosY { float data[]; } pos_y;
layout(set = 0, binding = 2) readonly buffer ColR { float data[]; } col_r;
layout(set = 0, binding = 3) readonly buffer ColG { float data[]; } col_g;
layout(set = 0, binding = 4) readonly buffer ColB { float data[]; } col_b;
layout(set = 0, binding = 5) readonly buffer ColA { float data[]; } col_a;
layout(location = 0) out vec4 vColor;
void main() {
  uint i = gl_VertexIndex;
  gl_Position = vec4(pos_x.data[i], pos_y.data[i], 0.0, 1.0);
  gl_PointSize = 3.0;
  vColor = vec4(col_r.data[i], col_g.data[i], col_b.data[i], col_a.data[i]);
}
)";

constexpr std::string_view k_particle_frag = R"(
#version 450
layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vColor;
}
)";

struct particle_params
{
  float delta_time;
};
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
static auto run() -> int
{
  auto win = vkexec::examples::glfw_presenter::create(
    { .width = k_window_width, .height = k_window_height, .title = "vkexec particles", .validation_layers = true });
  auto &ctx = win.ctx();

  auto pos_x = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto pos_y = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto vel_x = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto vel_y = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto col_r = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto col_g = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto col_b = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));
  auto col_a = vkexec::examples::sync_wait_value(vkexec::factory::buffer<float>(ctx, k_particle_count));

  {
    // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
    std::mt19937 rng{ k_rng_seed };
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    float *const pos_x_host = pos_x.data();
    float *const pos_y_host = pos_y.data();
    float *const vel_x_host = vel_x.data();
    float *const vel_y_host = vel_y.data();
    float *const col_r_host = col_r.data();
    float *const col_g_host = col_g.data();
    float *const col_b_host = col_b.data();
    float *const col_a_host = col_a.data();
    for (std::uint32_t index = 0; index < k_particle_count; ++index) {
      float const radius = k_spawn_radius * std::sqrt(dist(rng));
      float const theta = dist(rng) * k_two_pi;
      float const spawn_x = radius * std::cos(theta) * k_aspect;
      float const spawn_y = radius * std::sin(theta);
      float const length = std::sqrt((spawn_x * spawn_x) + (spawn_y * spawn_y)) + k_epsilon;
      pos_x_host[index] = spawn_x;
      pos_y_host[index] = spawn_y;
      vel_x_host[index] = (spawn_x / length) * k_initial_speed;
      vel_y_host[index] = (spawn_y / length) * k_initial_speed;
      col_r_host[index] = dist(rng);
      col_g_host[index] = dist(rng);
      col_b_host[index] = dist(rng);
      col_a_host[index] = 1.0F;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  using enum vkexec::buffer_access;
  auto compute_pipe = vkexec::examples::sync_wait_value(vkexec::factory::compute_pipeline(ctx,
    k_particle_update_glsl,
    vkexec::layout_desc{
      .binding_kinds = {},
      .binding_slots = {},
      .bindings = { readwrite, readwrite, readwrite, readwrite },
      .push_constant_size = sizeof(particle_params),
      .specialization = {},
      .local_size = { k_local_size_x, 1, 1 },
    },
    "particle_update.comp"));

  std::array<vkexec::storage_binding, 4> const compute_buffers{
    vkexec::storage_binding{ .buffer = pos_x.vk_buffer(), .byte_size = pos_x.size() * sizeof(float), .binding = 0 },
    vkexec::storage_binding{ .buffer = pos_y.vk_buffer(), .byte_size = pos_y.size() * sizeof(float), .binding = 1 },
    vkexec::storage_binding{ .buffer = vel_x.vk_buffer(), .byte_size = vel_x.size() * sizeof(float), .binding = 2 },
    vkexec::storage_binding{ .buffer = vel_y.vk_buffer(), .byte_size = vel_y.size() * sizeof(float), .binding = 3 },
  };
  auto compute_bound = vkexec::examples::sync_wait_value(vkexec::bind_storage_sender(compute_pipe, compute_buffers));

  vkexec::graphics_pipeline_config graphics_cfg{};
  graphics_cfg.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  graphics_cfg.alpha_blend = true;
  graphics_cfg.clear_r = k_clear_r;
  graphics_cfg.clear_g = k_clear_g;
  graphics_cfg.clear_b = k_clear_b;

  std::array<vkexec::storage_binding, 6> const draw_buffers{
    vkexec::storage_binding{ .buffer = pos_x.vk_buffer(), .byte_size = pos_x.size() * sizeof(float), .binding = 0 },
    vkexec::storage_binding{ .buffer = pos_y.vk_buffer(), .byte_size = pos_y.size() * sizeof(float), .binding = 1 },
    vkexec::storage_binding{ .buffer = col_r.vk_buffer(), .byte_size = col_r.size() * sizeof(float), .binding = 2 },
    vkexec::storage_binding{ .buffer = col_g.vk_buffer(), .byte_size = col_g.size() * sizeof(float), .binding = 3 },
    vkexec::storage_binding{ .buffer = col_b.vk_buffer(), .byte_size = col_b.size() * sizeof(float), .binding = 4 },
    vkexec::storage_binding{ .buffer = col_a.vk_buffer(), .byte_size = col_a.size() * sizeof(float), .binding = 5 },
  };

  auto gfx = vkexec::examples::sync_wait_value(vkexec::factory::graphics_pipeline(
    ctx, win.render_pass(), graphics_cfg, k_particle_vert, k_particle_frag, draw_buffers));

  auto last = std::chrono::steady_clock::now();
  std::cout << std::format("vkexec particles (compute update + point sprites) - close the window to exit\n");

  while (!win.should_close()) {
    win.poll_events();

    auto const now = std::chrono::steady_clock::now();
    float delta_time = std::chrono::duration<float>(now - last).count();
    last = now;
    delta_time = std::min(delta_time, k_max_delta_seconds);
    particle_params const params{ delta_time };

    vkexec::examples::sync_wait_graph(
      ex::schedule(ctx.get_scheduler())
      | vkexec::compute_pass(*compute_bound.pipe, compute_bound.set, params, k_particle_count));

    vkexec::examples::sync_wait_graph(
      ex::schedule(ctx.get_scheduler()) | vkexec::draw(win.target(), gfx, k_particle_count));
  }

  win.wait_idle();
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
