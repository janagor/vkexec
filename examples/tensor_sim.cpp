#include "sync_wait_helpers.hpp"

#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/tensor.hpp>
#include <vkexec/tensor_sync.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/buffer_device_address.hpp>
#include <vkexec_features/feature.hpp>

#include <stdexec/execution.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
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
static auto run() -> int
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::buffer_device_address>(requirements);
  auto ctx = vkexec::examples::sync_wait_value(
    vkexec::factory::context({ .validation_layers = true, .requirements = std::move(requirements) }));
  auto positions = vkexec::examples::sync_wait_value(vkexec::factory::tensor<float>(*ctx, k_element_count, 0.0F));
  auto velocities =
    vkexec::examples::sync_wait_value(vkexec::factory::tensor<float>(*ctx, k_element_count, k_initial_velocity));

  using enum vkexec::buffer_access;
  auto resources_result = vkexec::create_compute_resources(*ctx,
    k_sim_glsl,
    vkexec::layout_desc{
      .binding_kinds = {},
      .binding_slots = {},
      .bindings = { readwrite, readwrite },
      .push_constant_size = sizeof(sim_params),
      .specialization = {},
      .local_size = { k_local_size_x, 1, 1 },
    },
    "tensor_sim.comp");
  if (!resources_result) { vkexec::examples::abort_with_error(resources_result.error()); }
  auto resources = vkexec::expected_take(resources_result);

  std::array const bindings{
    positions.storage_binding(0),
    velocities.storage_binding(1),
  };
  auto bound_result = vkexec::bind_storage(*ctx, resources, bindings);
  if (!bound_result) { vkexec::examples::abort_with_error(bound_result.error()); }
  auto bound = vkexec::expected_take(bound_result);

  sim_params const params{ .dt = k_timestep, .damping = k_damping };

  // Upload -> dispatch -> download on the existing pass graph.
  vkexec::examples::sync_wait_graph(
    ex::schedule(ctx->get_scheduler()) | vkexec::sync_to_device(positions) | vkexec::sync_to_device(velocities)
    | vkexec::compute_pass(resources, bound.set, params, static_cast<std::uint32_t>(k_element_count))
    | vkexec::sync_to_host(positions) | vkexec::sync_to_host(velocities));

  float const expected_v = k_initial_velocity * k_damping;
  float const expected_p = expected_v * k_timestep;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t const index : { std::size_t{ 0 }, k_element_count / 2, k_element_count - 1 }) {
    if (std::fabs(velocities.data()[index] - expected_v) > k_epsilon
        || std::fabs(positions.data()[index] - expected_p) > k_epsilon) {
      std::cerr << std::format("mismatch at {}: p={} v={} (expected p={} v={})\n",
        index,
        positions.data()[index],
        velocities.data()[index],
        expected_p,
        expected_v);
      vkexec::examples::fail_check("tensor sim result mismatch");
    }
  }

  std::cout << std::format("vkexec tensor_sim ok: p[0]={} v[0]={}\n", positions.data()[0], velocities.data()[0]);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  vkexec::free_compute_set(*ctx, resources, bound.set);
  vkexec::destroy_compute_resources(*ctx, resources);
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
