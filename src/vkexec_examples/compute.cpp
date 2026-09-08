#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/sync_wait.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <print>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {
constexpr std::size_t k_element_count = 10000;
constexpr float k_initial_velocity = 1.5F;
constexpr float k_timestep = 0.016F;
constexpr float k_damping = 0.99F;
constexpr float k_epsilon = 1.0E-4F;
constexpr std::uint32_t k_local_size_x = 64;

constexpr std::string_view k_sim_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Positions { float data[]; } positions;
layout(set = 0, binding = 1) buffer Velocities { float data[]; } velocities;
layout(push_constant) uniform Push {
  float dt;
  float damping;
} pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  float velocity = velocities.data[i];
  float position = positions.data[i];
  velocity = velocity * pc.damping;
  position = position + velocity * pc.dt;
  positions.data[i] = position;
  velocities.data[i] = velocity;
}
)";

struct sim_params
{
  float dt;
  float damping;
};
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    auto ctx = vkexec::value_or_throw(vkexec::context::create({ .validation_layers = true }));
    auto positions = vkexec::value_or_throw(vkexec::buffer<float>::create_sync(*ctx, k_element_count, 0.0F));
    auto velocities =
      vkexec::value_or_throw(vkexec::buffer<float>::create_sync(*ctx, k_element_count, k_initial_velocity));

    using enum vkexec::buffer_access;
    auto pipe = vkexec::value_or_throw(vkexec::compute_pipeline::create(*ctx,
      k_sim_glsl,
      vkexec::layout_desc{
        .bindings = { readwrite, readwrite },
        .push_constant_size = sizeof(sim_params),
        .specialization = {},
        .local_size = { k_local_size_x, 1, 1 },
      },
      "sim.comp"));

    auto *set = vkexec::value_or_throw(pipe.allocate_set());
    std::array<vkexec::storage_binding, 2> const buffers{
      vkexec::storage_binding{
        .buffer = positions.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(positions.size() * sizeof(float)) },
      vkexec::storage_binding{
        .buffer = velocities.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(velocities.size() * sizeof(float)) },
    };
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    if (auto updated = pipe.update_set(set, buffers); !updated) { throw std::move(updated.error()); }

    sim_params const params{ .dt = k_timestep, .damping = k_damping };

    auto waited = vkexec::sync_wait(ex::schedule(ctx->get_scheduler())
                                    | vkexec::compute_pass(pipe, set, params, static_cast<std::uint32_t>(k_element_count)));
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    if (!waited.has_value()) { throw vkexec::make_error(vkexec::errc::cancelled, "pipeline was stopped"); }

    float const expected_v = k_initial_velocity * k_damping;
    float const expected_p = expected_v * k_timestep;
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t const index : { std::size_t{ 0 }, k_element_count / 2, k_element_count - 1 }) {
      if (std::fabs(velocities.data()[index] - expected_v) > k_epsilon
          || std::fabs(positions.data()[index] - expected_p) > k_epsilon) {
        std::println(stderr,
          "mismatch at {}: p={} v={} (expected p={} v={})",
          index,
          positions.data()[index],
          velocities.data()[index],
          expected_p,
          expected_v);
        // NOLINTNEXTLINE(hicpp-exception-baseclass)
        throw vkexec::make_error(vkexec::errc::unsupported, "sim result mismatch");
      }
    }

    std::println("vkexec sim ok: p[0]={} v[0]={}", positions.data()[0], velocities.data()[0]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (vkexec::error const &error) {
    std::println(stderr, "vkexec example failed: {}", error.message());
    return 1;
  }
}
