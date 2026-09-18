#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::uint32_t k_local_size = 64;

constexpr std::string_view k_sim_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Positions { float data[]; } positions;
layout(set = 0, binding = 1) buffer Velocities { float data[]; } velocities;
layout(push_constant) uniform Push { float dt; float damping; } pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  float v = velocities.data[i] * pc.damping;
  float p = positions.data[i] + v * pc.dt;
  velocities.data[i] = v;
  positions.data[i] = p;
}
)";

constexpr std::string_view k_add_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Values { float data[]; } values;
layout(push_constant) uniform Push { float value; } pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  values.data[i] = values.data[i] + pc.value;
}
)";

constexpr std::string_view k_scale_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Values { float data[]; } values;
layout(push_constant) uniform Push { float value; } pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  values.data[i] = values.data[i] * pc.value;
}
)";

struct sim_params
{
  float dt;
  float damping;
};

struct pass_params
{
  float value;
};

auto make_one_buffer_layout() -> vkexec::layout_desc
{
  return vkexec::layout_desc{
    .bindings = { vkexec::buffer_access::readwrite },
    .push_constant_size = sizeof(pass_params),
    .specialization = {},
    .local_size = { k_local_size, 1, 1 },
  };
}

auto make_sim_layout() -> vkexec::layout_desc
{
  return vkexec::layout_desc{
    .bindings = { vkexec::buffer_access::readwrite, vkexec::buffer_access::readwrite },
    .push_constant_size = sizeof(sim_params),
    .specialization = {},
    .local_size = { k_local_size, 1, 1 },
  };
}

}// namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("headless compute pipeline updates buffers", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 128;
  constexpr float k_initial_velocity = 1.5F;
  constexpr float k_timestep = 0.016F;
  constexpr float k_damping = 0.99F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx = vkexec::test::require_context();
  auto positions = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, 0.0F));
  auto velocities = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, k_initial_velocity));
  auto pipe =
    vkexec::test::sync_wait_value(vkexec::compute_pipeline::create(*ctx, k_sim_glsl, make_sim_layout(), "sim.comp"));

  auto set_result = pipe.allocate_set();
  REQUIRE(set_result.has_value());
  auto *set = vkexec::expected_take(set_result);
  std::array const bindings{
    vkexec::storage_binding{ .buffer = positions.vk_buffer(), .byte_size = k_count * sizeof(float), .binding = 0 },
    vkexec::storage_binding{ .buffer = velocities.vk_buffer(), .byte_size = k_count * sizeof(float), .binding = 1 },
  };
  REQUIRE(pipe.update_set(set, bindings));

  sim_params const params{ .dt = k_timestep, .damping = k_damping };
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | vkexec::compute_pass(pipe, set, params, static_cast<std::uint32_t>(k_count)));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  float const expected_v = k_initial_velocity * k_damping;
  float const expected_p = expected_v * k_timestep;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(std::fabs(positions.data()[0] - expected_p) <= k_epsilon);
  REQUIRE(std::fabs(velocities.data()[0] - expected_v) <= k_epsilon);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("classic compute borrowable path without owning pipeline", "[vkexec][gpu][execution]")
{
  constexpr std::size_t k_count = 128;
  constexpr float k_initial_velocity = 1.5F;
  constexpr float k_timestep = 0.016F;
  constexpr float k_damping = 0.99F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx = vkexec::test::require_context();
  auto positions = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, 0.0F));
  auto velocities = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, k_initial_velocity));

  auto resources_result = vkexec::create_compute_resources(*ctx, k_sim_glsl, make_sim_layout(), "sim_execution.comp");
  REQUIRE(resources_result.has_value());
  auto resources = vkexec::expected_take(resources_result);

  std::array const bindings{
    vkexec::storage_binding{ .buffer = positions.vk_buffer(), .byte_size = k_count * sizeof(float), .binding = 0 },
    vkexec::storage_binding{ .buffer = velocities.vk_buffer(), .byte_size = k_count * sizeof(float), .binding = 1 },
  };
  auto bound_result = vkexec::bind_storage(*ctx, resources, bindings);
  REQUIRE(bound_result.has_value());
  auto bound = vkexec::expected_take(bound_result);
  REQUIRE(bound.pipe == &resources);
  REQUIRE(bound.set != VK_NULL_HANDLE);
  REQUIRE(bindings.at(0).binding == 0);
  REQUIRE(bindings.at(1).binding == 1);

  sim_params const params{ .dt = k_timestep, .damping = k_damping };
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler())
    | vkexec::compute_pass(resources, bound.set, params, static_cast<std::uint32_t>(k_count)));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  float const expected_v = k_initial_velocity * k_damping;
  float const expected_p = expected_v * k_timestep;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(std::fabs(positions.data()[0] - expected_p) <= k_epsilon);
  REQUIRE(std::fabs(velocities.data()[0] - expected_v) <= k_epsilon);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  vkexec::free_compute_set(*ctx, resources, bound.set);
  vkexec::destroy_compute_resources(*ctx, resources);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("chained compute passes reuse descriptor sets safely", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 64;
  constexpr float k_initial = 1.0F;
  constexpr float k_add = 3.0F;
  constexpr float k_scale = 2.0F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, k_initial));
  auto add_pipe = vkexec::test::sync_wait_value(
    vkexec::compute_pipeline::create(*ctx, k_add_glsl, make_one_buffer_layout(), "add.comp"));
  auto scale_pipe = vkexec::test::sync_wait_value(
    vkexec::compute_pipeline::create(*ctx, k_scale_glsl, make_one_buffer_layout(), "scale.comp"));

  auto set_result = add_pipe.allocate_set();
  REQUIRE(set_result.has_value());
  auto *set = vkexec::expected_take(set_result);
  vkexec::storage_binding const binding{
    .buffer = values.vk_buffer(),
    .byte_size = k_count * sizeof(float),
    .binding = 0,
  };
  REQUIRE(add_pipe.update_set(set, std::span{ &binding, 1 }));
  REQUIRE(scale_pipe.update_set(set, std::span{ &binding, 1 }));

  auto graph =
    ex::schedule(ctx->get_scheduler())
    | vkexec::compute_pass(add_pipe, set, pass_params{ .value = k_add }, static_cast<std::uint32_t>(k_count))
    | vkexec::barrier::compute_to_compute()
    | vkexec::compute_pass(scale_pipe, set, pass_params{ .value = k_scale }, static_cast<std::uint32_t>(k_count));
  auto waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  float const expected = (k_initial + k_add) * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(values.data()[index] - expected) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("chained compute_pass graph completes asynchronously", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 64;
  constexpr float k_initial = 1.0F;
  constexpr float k_add = 3.0F;
  constexpr float k_scale = 2.0F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, k_initial));
  auto add_pipe = vkexec::test::sync_wait_value(
    vkexec::compute_pipeline::create(*ctx, k_add_glsl, make_one_buffer_layout(), "add.comp"));
  auto scale_pipe = vkexec::test::sync_wait_value(
    vkexec::compute_pipeline::create(*ctx, k_scale_glsl, make_one_buffer_layout(), "scale.comp"));

  auto set_result = add_pipe.allocate_set();
  REQUIRE(set_result.has_value());
  auto *set = vkexec::expected_take(set_result);
  vkexec::storage_binding const binding{
    .buffer = values.vk_buffer(),
    .byte_size = k_count * sizeof(float),
    .binding = 0,
  };
  REQUIRE(add_pipe.update_set(set, std::span{ &binding, 1 }));
  REQUIRE(scale_pipe.update_set(set, std::span{ &binding, 1 }));

  auto graph =
    ex::schedule(ctx->get_scheduler())
    | vkexec::compute_pass(add_pipe, set, pass_params{ .value = k_add }, static_cast<std::uint32_t>(k_count))
    | vkexec::barrier::compute_to_compute()
    | vkexec::compute_pass(scale_pipe, set, pass_params{ .value = k_scale }, static_cast<std::uint32_t>(k_count))
    | vkexec::submit;
  auto waited = vkexec::test::sync_wait_sender(graph);
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  float const expected = (k_initial + k_add) * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(values.data()[index] - expected) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("pass graph submit completes with set_stopped when stop is already requested", "[vkexec][pass]")
{
  // Null scheduler: exercises stop handling without allocating a Vulkan context.
  vkexec::scheduler const sched{ nullptr };
  ex::inplace_stop_source source;
  source.request_stop();

  auto sender =
    ex::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }) | vkexec::submit;

  auto const waited =
    vkexec::test::sync_wait_sender(ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() }));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}
