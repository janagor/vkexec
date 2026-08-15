#include <vkexec/vkexec.hpp>

#include <stdexec/execution.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace ex = stdexec;

struct ParticleParams {
  float dt;
};

VLK_PUSH_CONSTANT(ParticleParams, float, dt);

int main()
{
  try {
    constexpr std::uint32_t N = 8192;
    constexpr float k_aspect = 600.0f / 800.0f; // height / width like the tutorial init

    vkexec::window win({ .width = 800, .height = 600, .title = "vkexec particles" });
    auto &ctx = win.ctx();

    // SoA particle buffers (host-mapped SSBOs shared by compute + vertex stages).
    vkexec::buffer<float> pos_x(ctx, N);
    vkexec::buffer<float> pos_y(ctx, N);
    vkexec::buffer<float> vel_x(ctx, N);
    vkexec::buffer<float> vel_y(ctx, N);
    vkexec::buffer<float> col_r(ctx, N);
    vkexec::buffer<float> col_g(ctx, N);
    vkexec::buffer<float> col_b(ctx, N);
    vkexec::buffer<float> col_a(ctx, N);

    {
      std::mt19937 rng{ 42 };
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (std::uint32_t i = 0; i < N; ++i) {
        const float r = 0.25f * std::sqrt(dist(rng));
        const float theta = dist(rng) * 6.28318530718f;
        const float x = r * std::cos(theta) * k_aspect;
        const float y = r * std::sin(theta);
        const float len = std::sqrt(x * x + y * y) + 1e-6f;
        pos_x.data()[i] = x;
        pos_y.data()[i] = y;
        vel_x.data()[i] = (x / len) * 0.25f;
        vel_y.data()[i] = (y / len) * 0.25f;
        col_r.data()[i] = dist(rng);
        col_g.data()[i] = dist(rng);
        col_b.data()[i] = dist(rng);
        col_a.data()[i] = 1.0f;
      }
    }

    // Graphics: point list reading particle SSBOs (tutorial draw path via storage buffers).
    vkexec::graphics_pipeline_config gcfg{};
    gcfg.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    gcfg.alpha_blend = true;
    gcfg.clear_r = 0.02f;
    gcfg.clear_g = 0.02f;
    gcfg.clear_b = 0.05f;

    vkexec::graphics_pipeline gfx(ctx,
      win.render_pass(),
      gcfg,
      [&](vlk::Int vid, vlk::VertexWriter out) {
        out.position(vlk::vec2(pos_x[vid], pos_y[vid]));
        out.point_size(vlk::Float::constant(3.0));
        out.color(vlk::vec4(col_r[vid], col_g[vid], col_b[vid], col_a[vid]));
      },
      [](vlk::FragmentReader in, vlk::FragmentWriter out) { out.color(in.color4()); });

    auto last = std::chrono::steady_clock::now();
    std::printf("vkexec particles (compute update + point sprites) — close the window to exit\n");

    while (!win.should_close()) {
      win.poll_events();

      const auto now = std::chrono::steady_clock::now();
      float dt = std::chrono::duration<float>(now - last).count();
      last = now;
      // Match tutorial feel; clamp spikes when the window stalls.
      if (dt > 0.05f) { dt = 0.05f; }
      ParticleParams params{ dt };

      // GPU particle update (in-place SSBO), synchronized before draw.
      ex::sync_wait(ex::schedule(ctx.get_scheduler())
                    | vkexec::bulk(N, params, [&](vlk::Int idx, vlk::PushConstant<ParticleParams> pc) {
                        vlk::Float px = pos_x[idx];
                        vlk::Float py = pos_y[idx];
                        vlk::Float vx = vel_x[idx];
                        vlk::Float vy = vel_y[idx];

                        px = px + (vx * pc.dt);
                        py = py + (vy * pc.dt);

                        vlk::if_then((px <= vlk::Float::constant(-1.0)) || (px >= vlk::Float::constant(1.0)),
                          [&] { vx = vlk::Float::constant(0.0) - vx; });
                        vlk::if_then((py <= vlk::Float::constant(-1.0)) || (py >= vlk::Float::constant(1.0)),
                          [&] { vy = vlk::Float::constant(0.0) - vy; });

                        pos_x[idx] = px;
                        pos_y[idx] = py;
                        vel_x[idx] = vx;
                        vel_y[idx] = vy;
                      }));

      ex::sync_wait(ex::schedule(ctx.get_scheduler()) | vkexec::draw(win, gfx, N));
    }

    win.wait_idle();
    return 0;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "vkexec particles example failed: %s\n", ex.what());
    return 1;
  }
}
