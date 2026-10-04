#include "common/shader_loader.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
#include "sync_wait_helpers.hpp"

#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>

namespace ex = stdexec;

namespace {
constexpr std::uint32_t k_particle_count = 1024;
constexpr std::uint32_t k_local_size_x = 64;
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr float k_two_pi = 6.28318530718F;
constexpr float k_max_delta_seconds = 0.033F;
constexpr float k_clear_red = 0.01F;
constexpr float k_clear_green = 0.01F;
constexpr float k_clear_blue = 0.03F;
constexpr unsigned k_rng_seed = 42;

struct integrate_params
{
  float delta_time;
};

// NOLINTNEXTLINE(bugprone-exception-escape)
auto run() -> int
{
  auto win = vkexec::examples::glfw_presenter::create({
    .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec Vulkan Samples: Compute N-Body",
    .validation_layers = true,
    .requirements = vkexec::examples::vulkan_sample_requirements(),
  });
  auto &ctx = win.ctx();
  auto allocator = vkexec::examples::make_vma_allocator(ctx);
  auto pos_x = vkexec::examples::sync_wait_value(vkexec::factory::make_buffer(allocator, k_particle_count, 0.0F));
  auto pos_y = vkexec::examples::sync_wait_value(vkexec::factory::make_buffer(allocator, k_particle_count, 0.0F));
  auto vel_x = vkexec::examples::sync_wait_value(vkexec::factory::make_buffer(allocator, k_particle_count, 0.0F));
  auto vel_y = vkexec::examples::sync_wait_value(vkexec::factory::make_buffer(allocator, k_particle_count, 0.0F));
  auto acc_x = vkexec::examples::sync_wait_value(vkexec::factory::make_buffer(allocator, k_particle_count, 0.0F));
  auto acc_y = vkexec::examples::sync_wait_value(vkexec::factory::make_buffer(allocator, k_particle_count, 0.0F));

  // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
  std::mt19937 rng{ k_rng_seed };
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::uint32_t index = 0; index < k_particle_count; ++index) {
    float const radius = 0.12F + (0.65F * std::sqrt(uniform(rng)));
    float const angle = k_two_pi * uniform(rng);
    float const position_x = radius * std::cos(angle);
    float const position_y = radius * std::sin(angle);
    float const speed = 0.035F / std::sqrt(radius);
    pos_x.data()[index] = position_x;
    pos_y.data()[index] = position_y;
    vel_x.data()[index] = -position_y * speed / radius;
    vel_y.data()[index] = position_x * speed / radius;
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  using enum vkexec::buffer_access;
  auto calculate = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(ctx,
    vkexec::examples::load_spirv(shader_dir / "particle_calculate.comp.spv"),
    vkexec::layout_desc{
      .binding_kinds = {},
      .binding_slots = {},
      .bindings = { readonly, readonly, writeonly, writeonly },
      .push_constant_size = 0,
      .specialization = {},
      .local_size = { k_local_size_x, 1, 1 },
    }));
  auto integrate = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(ctx,
    vkexec::examples::load_spirv(shader_dir / "particle_integrate.comp.spv"),
    vkexec::layout_desc{
      .binding_kinds = {},
      .binding_slots = {},
      .bindings = { readwrite, readwrite, readwrite, readwrite, readonly, readonly },
      .push_constant_size = sizeof(integrate_params),
      .specialization = {},
      .local_size = { k_local_size_x, 1, 1 },
    }));

  auto const bind = [](auto const &buffer, std::uint32_t slot) -> vkexec::storage_binding {
    return { .buffer = buffer.vk_buffer(), .byte_size = buffer.size() * sizeof(float), .binding = slot };
  };
  std::array<vkexec::storage_binding, 4> const calculate_buffers{
    bind(pos_x, 0), bind(pos_y, 1), bind(acc_x, 2), bind(acc_y, 3),
  };
  std::array<vkexec::storage_binding, 6> const integrate_buffers{
    bind(pos_x, 0), bind(pos_y, 1), bind(vel_x, 2), bind(vel_y, 3), bind(acc_x, 4), bind(acc_y, 5),
  };
  std::array<vkexec::storage_binding, 2> const draw_buffers{ bind(pos_x, 0), bind(pos_y, 1) };
  auto calculate_bound = vkexec::examples::sync_wait_value(vkexec::bind_storage_sender(calculate, calculate_buffers));
  auto integrate_bound = vkexec::examples::sync_wait_value(vkexec::bind_storage_sender(integrate, integrate_buffers));

  vkexec::graphics_pipeline_config graphics_config{};
  graphics_config.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  graphics_config.clear_r = k_clear_red;
  graphics_config.clear_g = k_clear_green;
  graphics_config.clear_b = k_clear_blue;
  auto graphics = vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(ctx,
    win.render_pass(),
    graphics_config,
    vkexec::examples::load_spirv(shader_dir / "particle.vert.spv"),
    vkexec::examples::load_spirv(shader_dir / "particle.frag.spv"),
    draw_buffers));

  auto last = std::chrono::steady_clock::now();
  std::cout << "vkexec Compute N-Body - close the window to exit\n";
  while (!win.should_close()) {
    win.poll_events();
    auto const now = std::chrono::steady_clock::now();
    float const delta_time = std::min(std::chrono::duration<float>(now - last).count(), k_max_delta_seconds);
    last = now;

    vkexec::examples::sync_wait_graph(
      ex::schedule(ctx.get_scheduler())
      | vkexec::compute_pass(*calculate_bound.pipe, calculate_bound.set, k_particle_count)
      | vkexec::barrier::compute_to_compute()
      | vkexec::compute_pass(*integrate_bound.pipe, integrate_bound.set, integrate_params{ delta_time }, k_particle_count));
    vkexec::examples::sync_wait_graph(
      ex::schedule(ctx.get_scheduler()) | vkexec::draw(win.target(), graphics, k_particle_count));
    win.wait_idle();
  }
  win.wait_idle();
  return 0;
}
}// namespace

auto main() -> int { return vkexec::examples::run_example(run); }
