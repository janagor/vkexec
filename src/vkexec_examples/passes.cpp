#include "sync_wait_helpers.hpp"
#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <utility>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string_view>

namespace ex = stdexec;

namespace {
constexpr std::size_t k_element_count = 1024;
constexpr float k_initial = 1.0F;
constexpr float k_add = 3.0F;
constexpr float k_scale = 2.0F;
constexpr float k_epsilon = 1.0E-4F;
constexpr std::uint32_t k_local_size_x = 64;

constexpr std::string_view k_pass_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Values { float data[]; } values;
layout(push_constant) uniform Push {
  float value;
  uint op;
} pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  if (pc.op == 0u) {
    values.data[i] = values.data[i] + pc.value;
  } else {
    values.data[i] = values.data[i] * pc.value;
  }
}
)";

struct pass_params
{
  float value;
  std::uint32_t op;
};

constexpr std::uint32_t k_op_add = 0;
constexpr std::uint32_t k_op_mul = 1;
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
static auto run() -> int
{
  auto ctx = vkexec::examples::sync_wait_value(vkexec::context::create({ .validation_layers = true }));
  auto values = vkexec::examples::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_element_count, k_initial));

  using enum vkexec::buffer_access;
  auto pipe = vkexec::examples::sync_wait_value(vkexec::compute_pipeline::create(*ctx,
    k_pass_glsl,
    vkexec::layout_desc{
      .bindings = { readwrite },
      .push_constant_size = sizeof(pass_params),
      .specialization = {},
      .local_size = { k_local_size_x, 1, 1 },
    },
    "pass.comp"));

  std::array<vkexec::storage_binding, 1> const buffers{ vkexec::storage_binding{
    .buffer = values.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(values.size() * sizeof(float)) } };
  auto bound = vkexec::examples::sync_wait_value(vkexec::bind_storage_sender(pipe, buffers));

  auto graph = ex::schedule(ctx->get_scheduler())
               | vkexec::compute_pass(*bound.pipe,
                 bound.set,
                 pass_params{ .value = k_add, .op = k_op_add },
                 static_cast<std::uint32_t>(k_element_count))
               | vkexec::barrier::compute_to_compute()
               | vkexec::compute_pass(*bound.pipe,
                 bound.set,
                 pass_params{ .value = k_scale, .op = k_op_mul },
                 static_cast<std::uint32_t>(k_element_count));
  vkexec::examples::sync_wait_graph(std::move(graph));

  float const expected = (k_initial + k_add) * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_element_count; ++index) {
    if (std::fabs(values.data()[index] - expected) > k_epsilon) {
      std::cerr << std::format("pass mismatch at {}: got {} expected {}\n", index, values.data()[index], expected);
      vkexec::examples::fail_check("pass result mismatch");
    }
  }
  std::cout << std::format("vkexec passes ok: N={} result={}\n", k_element_count, values.data()[0]);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
