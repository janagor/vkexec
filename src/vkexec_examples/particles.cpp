#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec_edsl/control.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/window.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <print>
#include <random>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::uint32_t k_particle_count = 8192;
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr float k_aspect = static_cast<float>(k_window_height) / static_cast<float>(k_window_width);
constexpr float k_spawn_radius = 0.25F;
constexpr float k_two_pi = 6.28318530718F;
constexpr float k_epsilon = 1.0E-6F;
constexpr float k_initial_speed = 0.25F;
constexpr double k_point_size = 3.0;
constexpr float k_max_delta_seconds = 0.05F;
constexpr float k_clear_r = 0.02F;
constexpr float k_clear_g = 0.02F;
constexpr float k_clear_b = 0.05F;
constexpr unsigned k_rng_seed = 42;
}// namespace

struct particle_params
{
  float delta_time;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(particle_params, (), (delta_time))

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::window win({ .width = k_window_width,
      .height = k_window_height,
      .title = "vkexec particles",
      .validation_layers = true });
    auto &ctx = win.ctx();

    // SoA particle buffers (host-mapped SSBOs shared by compute + vertex stages).
    vkexec::buffer<float> pos_x(ctx, k_particle_count);
    vkexec::buffer<float> pos_y(ctx, k_particle_count);
    vkexec::buffer<float> vel_x(ctx, k_particle_count);
    vkexec::buffer<float> vel_y(ctx, k_particle_count);
    vkexec::buffer<float> col_r(ctx, k_particle_count);
    vkexec::buffer<float> col_g(ctx, k_particle_count);
    vkexec::buffer<float> col_b(ctx, k_particle_count);
    vkexec::buffer<float> col_a(ctx, k_particle_count);

    {
      // Deterministic demo seed (not cryptographic).
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

    // Graphics: point list reading particle SSBOs (tutorial draw path via storage buffers).
    vkexec::graphics_pipeline_config graphics_cfg{};
    graphics_cfg.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    graphics_cfg.alpha_blend = true;
    graphics_cfg.clear_r = k_clear_r;
    graphics_cfg.clear_g = k_clear_g;
    graphics_cfg.clear_b = k_clear_b;

    vkexec::graphics_pipeline gfx(
      ctx,
      win.render_pass(),
      graphics_cfg,
      [&](edsl::Int vertex_id, edsl::VertexWriter out) -> void {
        out.position(edsl::vec2(pos_x[vertex_id], pos_y[vertex_id]));
        out.point_size(edsl::Float::constant(k_point_size));
        out.color(edsl::vec4(col_r[vertex_id], col_g[vertex_id], col_b[vertex_id], col_a[vertex_id]));
      },
      [](edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void { out.color(fragment_in.color4()); });

    auto last = std::chrono::steady_clock::now();
    std::println("vkexec particles (compute update + point sprites) - close the window to exit");

    while (!win.should_close()) {
      win.poll_events();

      auto const now = std::chrono::steady_clock::now();
      float delta_time = std::chrono::duration<float>(now - last).count();
      last = now;
      // Match tutorial feel; clamp spikes when the window stalls.
      delta_time = std::min(delta_time, k_max_delta_seconds);
      particle_params const params{ delta_time };

      // GPU particle update (in-place SSBO), synchronized before draw.
      (void)ex::sync_wait(
        ex::schedule(ctx.get_scheduler())
        | vkexec::bulk(
          k_particle_count, params, [&](edsl::Int const index, edsl::push_constant<particle_params> push) -> void {
            edsl::Float position_x = pos_x[index];
            edsl::Float position_y = pos_y[index];
            edsl::Float velocity_x = vel_x[index];
            edsl::Float velocity_y = vel_y[index];

            position_x = position_x + (velocity_x * push.get<&particle_params::delta_time>());
            position_y = position_y + (velocity_y * push.get<&particle_params::delta_time>());

            edsl::if_then((position_x <= edsl::Float::constant(-1.0)) || (position_x >= edsl::Float::constant(1.0)),
              [&]() -> void { velocity_x = edsl::Float::constant(0.0) - velocity_x; });
            edsl::if_then((position_y <= edsl::Float::constant(-1.0)) || (position_y >= edsl::Float::constant(1.0)),
              [&]() -> void { velocity_y = edsl::Float::constant(0.0) - velocity_y; });

            pos_x[index] = position_x;
            pos_y[index] = position_y;
            vel_x[index] = velocity_x;
            vel_y[index] = velocity_y;
          }));

      (void)ex::sync_wait(ex::schedule(ctx.get_scheduler()) | vkexec::draw(win, gfx, k_particle_count));
    }

    win.wait_idle();
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec particles example failed: {}", ex.what());
    return 1;
  }
}
