#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include "sync_wait_helpers.hpp"

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
constexpr std::size_t k_mat4_floats = 16;
constexpr std::size_t k_mat4_bytes = k_mat4_floats * sizeof(float);
constexpr std::size_t k_addr_offset = k_mat4_bytes;
constexpr std::size_t k_count_offset = k_addr_offset + sizeof(std::uint64_t);
constexpr float k_initial = 3.0F;
constexpr float k_scale = 2.0F;
constexpr float k_epsilon = 1.0E-4F;
constexpr std::uint32_t k_local_size_x = 64;
constexpr std::uint64_t k_placeholder_bda = 0x1111222233334444ULL;

constexpr std::string_view k_scale_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer In { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer Out { float data[]; } dst;
layout(push_constant) uniform Push {
  mat4 xform;
  uvec2 addr;
  uint count;
} pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.count) return;
  dst.data[i] = src.data[i] * pc.xform[0][0];
}
)";

/// Host POD matching the GLSL push block
struct host_push
{
  std::array<float, k_mat4_floats> xform{};
  std::uint64_t addr{};
  std::uint32_t count{};
};
static_assert(offsetof(host_push, addr) == k_addr_offset);
static_assert(offsetof(host_push, count) == k_count_offset);
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
static auto run() -> int
{
  auto ctx = vkexec::examples::sync_wait_value(vkexec::context::create({ .validation_layers = true }));
    auto input = vkexec::examples::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_element_count, k_initial));
    auto output = vkexec::examples::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_element_count, 0.0F));

    using enum vkexec::buffer_access;
    auto pipe = vkexec::examples::sync_wait_value(vkexec::compute_pipeline::create(*ctx,
      k_scale_glsl,
      vkexec::layout_desc{
        .bindings = { readonly, writeonly },
        .push_constant_size = sizeof(host_push),
        .specialization = {},
        .local_size = { k_local_size_x, 1, 1 },
      },
      "scale.comp"));

    std::array<vkexec::storage_binding, 2> const buffers{
      vkexec::storage_binding{
        .buffer = input.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(input.size() * sizeof(float)) },
      vkexec::storage_binding{
        .buffer = output.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(output.size() * sizeof(float)) },
    };
    auto bound = vkexec::examples::sync_wait_value(vkexec::bind_storage_sender(pipe, buffers));

    host_push push{};
    push.xform.at(0) = k_scale;
    push.addr = k_placeholder_bda;
    push.count = static_cast<std::uint32_t>(k_element_count);

    auto graph = ex::schedule(ctx->get_scheduler())
                 | vkexec::compute_pass(bound.pipe, bound.set, push, static_cast<std::uint32_t>(k_element_count));
  vkexec::examples::sync_wait_graph(std::move(graph));

  float const expected = k_initial * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_element_count; ++index) {
    if (std::fabs(output.data()[index] - expected) > k_epsilon) {
      std::cerr << std::format("spirv pass mismatch at {}: got {} expected {}\n", index, output.data()[index], expected);
      vkexec::examples::fail_check("spirv pass result mismatch");
    }
  }
  std::cout << std::format("vkexec spirv pass ok: N={} result={}\n", k_element_count, output.data()[0]);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
