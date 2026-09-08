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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <print>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

namespace ex = stdexec;

namespace {
constexpr std::size_t k_element_count = 256;
constexpr unsigned k_rng_seed = 42;
constexpr float k_value_max = 1000.0F;
constexpr float k_epsilon = 1.0E-4F;
constexpr std::uint32_t k_local_size_x = 64;

constexpr std::string_view k_sort_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Data { float data[]; } data_buf;
layout(push_constant) uniform Push {
  int offset;
  int n;
} pc;
void main() {
  uint idx = gl_GlobalInvocationID.x;
  int left = int(2 * idx) + pc.offset;
  int right = left + 1;
  if (right < pc.n) {
    float left_value = data_buf.data[left];
    float right_value = data_buf.data[right];
    if (left_value > right_value) {
      data_buf.data[left] = right_value;
      data_buf.data[right] = left_value;
    }
  }
}
)";

struct sort_params
{
  int offset;
  int n;
};
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    auto ctx = vkexec::value_or_throw(vkexec::context::create({ .validation_layers = true }));
    auto data = vkexec::value_or_throw(vkexec::buffer<float>::create_sync(*ctx, k_element_count, 0.0F));

    // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
    std::mt19937 rng{ k_rng_seed };
    std::uniform_real_distribution<float> dist(0.0F, k_value_max);
    std::vector<float> expected(k_element_count);
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      float const value = dist(rng);
      data.data()[index] = value;
      expected.at(index) = value;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::ranges::sort(expected);

    using enum vkexec::buffer_access;
    auto pipe = vkexec::value_or_throw(vkexec::compute_pipeline::create(*ctx,
      k_sort_glsl,
      vkexec::layout_desc{
        .bindings = { readwrite },
        .push_constant_size = sizeof(sort_params),
        .specialization = {},
        .local_size = { k_local_size_x, 1, 1 },
      },
      "sort.comp"));

    auto *set = vkexec::value_or_throw(pipe.allocate_set());
    std::array<vkexec::storage_binding, 1> const buffers{ vkexec::storage_binding{
      .buffer = data.vk_buffer(), .byte_size = static_cast<VkDeviceSize>(data.size() * sizeof(float)) } };
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    if (auto updated = pipe.update_set(set, buffers); !updated) { throw std::move(updated.error()); }

    auto make_phase = [&](std::size_t phase) -> auto {
      sort_params const params{ .offset = static_cast<int>(phase % 2), .n = static_cast<int>(k_element_count) };
      return vkexec::compute_pass(pipe,
        set,
        params,
        static_cast<std::uint32_t>(k_element_count / 2));
    };

    auto graph = ex::schedule(ctx->get_scheduler()) | make_phase(0);
    for (std::size_t phase = 1; phase < k_element_count; ++phase) {
      graph = std::move(graph) | vkexec::barrier::compute_to_compute() | make_phase(phase);
    }
    auto waited = vkexec::sync_wait(std::move(graph));
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    if (!waited.has_value()) { throw vkexec::make_error(vkexec::errc::cancelled, "pipeline was stopped"); }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      if (std::fabs(data.data()[index] - expected.at(index)) > k_epsilon) {
        std::println(stderr, "sort mismatch at {}: got {} expected {}", index, data.data()[index], expected.at(index));
        // NOLINTNEXTLINE(hicpp-exception-baseclass)
        throw vkexec::make_error(vkexec::errc::unsupported, "sort result mismatch");
      }
    }

    std::println("vkexec sort ok: N={} first={} mid={} last={}",
      k_element_count,
      data.data()[0],
      data.data()[k_element_count / 2],
      data.data()[k_element_count - 1]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (vkexec::error const &error) {
    std::println(stderr, "vkexec sort example failed: {}", error.message());
    return 1;
  }
}
