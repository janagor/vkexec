#include <vkexec/barrier.hpp>
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
auto main() -> int
{
  try {
    auto ctx = vkexec::value_or_throw(vkexec::context::create({ .validation_layers = true }));
    auto values = vkexec::value_or_throw(vkexec::buffer<float>::create_sync(*ctx, k_element_count, k_initial));

    using enum vkexec::buffer_access;
    auto pipe = vkexec::value_or_throw(vkexec::compute_pipeline::create(*ctx,
      k_pass_glsl,
      vkexec::layout_desc{
        .bindings = { readwrite },
        .push_constant_size = sizeof(pass_params),
        .specialization = {},
        .local_size = { k_local_size_x, 1, 1 },
      },
      "pass.comp"));

    auto *set = vkexec::value_or_throw(pipe.allocate_set());
    std::array<vkexec::storage_binding, 1> const buffers{ vkexec::storage_binding{
      .buffer = values.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(values.size() * sizeof(float)) } };
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    if (auto updated = pipe.update_set(set, buffers); !updated) { throw std::move(updated.error()); }

    auto graph = ex::schedule(ctx->get_scheduler())
                 | vkexec::compute_pass(pipe,
                   set,
                   pass_params{ .value = k_add, .op = k_op_add },
                   static_cast<std::uint32_t>(k_element_count))
                 | vkexec::barrier::compute_to_compute()
                 | vkexec::compute_pass(pipe,
                   set,
                   pass_params{ .value = k_scale, .op = k_op_mul },
                   static_cast<std::uint32_t>(k_element_count));
    auto waited = vkexec::sync_wait(std::move(graph));
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    if (!waited.has_value()) { throw vkexec::make_error(vkexec::errc::cancelled, "pipeline was stopped"); }

    float const expected = (k_initial + k_add) * k_scale;
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      if (std::fabs(values.data()[index] - expected) > k_epsilon) {
        std::println(stderr, "pass mismatch at {}: got {} expected {}", index, values.data()[index], expected);
        // NOLINTNEXTLINE(hicpp-exception-baseclass)
        throw vkexec::make_error(vkexec::errc::unsupported, "pass result mismatch");
      }
    }
    std::println("vkexec passes ok: N={} result={}", k_element_count, values.data()[0]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (vkexec::error const &error) {
    std::println(stderr, "vkexec passes example failed: {}", error.message());
    return 1;
  }
}
