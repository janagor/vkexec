#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/result.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/extension.hpp>

#include "../../sync_wait_helpers.hpp"

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string_view>

namespace ex = stdexec;

namespace {

constexpr std::size_t k_heap_slots = 1;
constexpr VkDeviceSize k_storage_bytes = 256;
constexpr std::uint32_t k_work_count = 64;

constexpr std::string_view k_heap_glsl = R"(#version 460
layout(local_size_x = 64) in;
void main() {}
)";

struct heap_push
{
  std::uint32_t count;
};

auto make_requirements() -> vkexec::vulkan_requirements
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.optional_device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_12).enable_extension_feature_if_present(features_heap);
  return requirements;
}

auto run() -> int
{
  auto ctx = vkexec::examples::sync_wait_value(
    vkexec::context::create({ .requirements = make_requirements() }));

  if (!vkexec::ext::available<vkexec::ext::descriptor_heap>(*ctx)) {
    std::cout << std::format("descriptor_heap: skipped (extension PFNs unavailable)\n");
    return 0;
  }

  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  if (!layout_result) {
    vkexec::examples::abort_with_error(layout_result.error());
  }
  auto const &layout = vkexec::expected_get(layout_result);

  auto storage = vkexec::examples::sync_wait_value(vkexec::gpu_buffer::create(*ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto heap = vkexec::examples::sync_wait_value(vkexec::gpu_buffer::create(
    *ctx, vkexec::descriptor_heap_byte_size(layout, k_heap_slots), vkexec::gpu_buffer_memory::descriptor_heap));

  auto const storage_addr = storage.device_address();
  if (!storage_addr) { vkexec::examples::fail_check("storage buffer device address unavailable"); }
  auto mapped = heap.mapped();
  if (auto const written = vkexec::write_storage_buffer_descriptor(
        *ctx, *storage_addr, storage.size(), mapped.subspan(0, layout.buffer_descriptor_size));
    !written) {
    vkexec::examples::abort_with_error(written.error());
  }

  auto pipe = vkexec::examples::sync_wait_value(vkexec::compute_pipeline::create(*ctx,
    k_heap_glsl,
    vkexec::layout_desc{
      .bindings = {},
      .push_constant_size = 0,
      .specialization = {},
      .local_size = vkexec::k_default_local_size,
      .descriptor_heap = true,
    },
    "descriptor_heap.comp"));

  heap_push const params{ .count = k_work_count };
  auto outcome = vkexec::try_sync_wait(
    ex::schedule(ctx->get_scheduler()) | vkexec::compute_heap_pass(pipe, params, k_work_count));
  if (outcome.failed() || outcome.stopped || !outcome.values.has_value()) {
    vkexec::examples::fail_check("bindless compute dispatch failed");
  }

  std::cout << std::format("descriptor_heap: bindless compute ok\n");
  return 0;
}

}// namespace

auto main() -> int { return vkexec::examples::run_example(run); }
