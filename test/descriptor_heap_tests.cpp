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
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t k_slot_count = 2;
constexpr VkDeviceSize k_storage_bytes = 256;

auto skip_if_unavailable(std::exception const &error) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + error.what()); }

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

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  auto const layout = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout.descriptor_stride > 0);
  REQUIRE(layout.buffer_descriptor_size > 0);

  auto const heap_bytes = vkexec::descriptor_heap_byte_size(layout, k_slot_count);
  REQUIRE(heap_bytes >= static_cast<VkDeviceSize>(layout.descriptor_stride * k_slot_count));

  auto storage = vkexec::gpu_buffer::create(*ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    });
  auto heap = vkexec::gpu_buffer::create(*ctx, heap_bytes, vkexec::gpu_buffer_memory::descriptor_heap);

  std::vector<std::byte> slot(layout.buffer_descriptor_size);
  vkexec::write_storage_buffer_descriptor(*ctx, storage.device_address(), storage.size(), slot);
  auto mapped = heap.mapped();
  REQUIRE(mapped.size() >= slot.size());
  std::ranges::copy(slot, mapped.begin());

  VkCommandBuffer cmd = ctx->allocate_command_buffer();
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
  vkexec::cmd_bind_resource_heap(
    *ctx, cmd, heap.device_address(), heap.size(), aligned_offset, layout.min_resource_heap_reserved_range);

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx->free_command_buffer(cmd);
}
