#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline_cache.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <print>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {
constexpr std::size_t k_element_count = 1024;
constexpr float k_initial = 3.0F;
constexpr float k_scale = 2.0F;
constexpr float k_epsilon = 1.0E-4F;
constexpr std::uint32_t k_local_size_x = 64;

constexpr std::string_view k_scale_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(constant_id = 0) const uint N = 1;
layout(set = 0, binding = 0) readonly buffer In { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer Out { float data[]; } dst;
layout(push_constant) uniform Push { float scale; } pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= N) return;
  dst.data[i] = src.data[i] * pc.scale;
}
)";
}// namespace

struct scale_params
{
  float scale;
};

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::context ctx;
    vkexec::buffer<float> const input(ctx, k_element_count, k_initial);
    vkexec::buffer<float> output(ctx, k_element_count, 0.0F);

    using enum vkexec::buffer_access;
    auto pipe = vkexec::compute_pipeline::from_glsl(ctx,
      k_scale_glsl,
      vkexec::layout_desc{
        .bindings = { readonly, writeonly },
        .push_constant_size = sizeof(scale_params),
        .specialization = { static_cast<std::uint32_t>(k_element_count) },
        .local_size = { k_local_size_x, 1, 1 },
      },
      "scale.comp");

    VkDescriptorSet set = pipe.allocate_set();
    std::array<vkexec::storage_binding, 2> const buffers{
      vkexec::storage_binding{ .buffer = input.vk_buffer(),
        .byte_size = static_cast<VkDeviceSize>(input.size() * sizeof(float)) },
      vkexec::storage_binding{ .buffer = output.vk_buffer(),
        .byte_size = static_cast<VkDeviceSize>(output.size() * sizeof(float)) },
    };
    pipe.update_set(set, buffers);

    auto graph = ex::schedule(ctx.get_scheduler())
                 | vkexec::compute_pass(pipe, set, scale_params{ .scale = k_scale },
                   static_cast<std::uint32_t>(k_element_count));
    ex::sync_wait(std::move(graph));

    float const expected = k_initial * k_scale;
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      if (std::fabs(output.data()[index] - expected) > k_epsilon) {
        std::println(stderr, "spirv pass mismatch at {}: got {} expected {}", index, output.data()[index], expected);
        return 1;
      }
    }
    std::println("vkexec spirv pass ok: N={} result={}", k_element_count, output.data()[0]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec spirv example failed: {}", ex.what());
    return 1;
  }
}
