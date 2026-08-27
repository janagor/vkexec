#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/descriptor_heap.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t k_slot_count = 2;
constexpr VkDeviceSize k_storage_bytes = 256;

auto skip_if_unavailable(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("descriptor heap layout query and buffer descriptor write", "[vkexec][descriptor_heap][gpu]")
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
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_12).require_extension_feature(features_heap);

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_unavailable(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto const layout_result = vkexec::query_descriptor_heap_layout(ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = *layout_result;
  REQUIRE(layout.descriptor_stride > 0);
  REQUIRE(layout.buffer_descriptor_size > 0);

  auto const heap_bytes = vkexec::descriptor_heap_byte_size(layout, k_slot_count);
  REQUIRE(heap_bytes >= static_cast<VkDeviceSize>(layout.descriptor_stride * k_slot_count));

  auto storage_result = vkexec::gpu_buffer::create(ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    });
  REQUIRE(storage_result.has_value());
  auto heap_result = vkexec::gpu_buffer::create(ctx, heap_bytes, vkexec::gpu_buffer_memory::descriptor_heap);
  REQUIRE(heap_result.has_value());

  std::vector<std::byte> slot(layout.buffer_descriptor_size);
  auto const storage_addr = storage_result->device_address();
  REQUIRE(storage_addr.has_value());
  REQUIRE(vkexec::write_storage_buffer_descriptor(ctx, *storage_addr, storage_result->size(), slot));
  auto mapped = heap_result->mapped();
  REQUIRE(mapped.size() >= slot.size());
  std::ranges::copy(slot, mapped.begin());

  auto cmd_result = ctx.allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  VkCommandBuffer cmd = *cmd_result;
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  auto const reserved_offset = static_cast<VkDeviceSize>(layout.descriptor_stride * k_slot_count);
  auto const aligned_offset =
    (layout.resource_heap_alignment == 0)
      ? reserved_offset
      : ((reserved_offset + layout.resource_heap_alignment - 1) / layout.resource_heap_alignment)
          * layout.resource_heap_alignment;
  auto const heap_addr = heap_result->device_address();
  REQUIRE(heap_addr.has_value());
  REQUIRE(vkexec::cmd_bind_resource_heap(
    ctx, cmd, *heap_addr, heap_result->size(), aligned_offset, layout.min_resource_heap_reserved_range));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx.free_command_buffer(cmd);
}
