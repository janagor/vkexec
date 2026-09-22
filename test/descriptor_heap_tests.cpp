#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/image.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/buffer.hpp>
#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t k_slot_count = 4;
constexpr VkDeviceSize k_storage_bytes = 256;
constexpr std::uint32_t k_image_extent = 64;

}// namespace
// NOLINTBEGIN(readability-function-cognitive-complexity)
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

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));

  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = vkexec::expected_get(layout_result);
  REQUIRE(layout.descriptor_stride > 0);
  REQUIRE(layout.buffer_descriptor_size > 0);
  REQUIRE(layout.sampler_descriptor_size > 0);

  auto const heap_bytes = vkexec::descriptor_heap_byte_size(layout, k_slot_count);
  REQUIRE(heap_bytes >= layout.descriptor_stride * k_slot_count);
  auto const sampler_bytes = vkexec::sampler_heap_byte_size(layout, k_slot_count);
  REQUIRE(sampler_bytes >= layout.sampler_descriptor_size * k_slot_count);

  auto storage = vkexec::test::sync_wait_value(vkexec::factory::make_gpu_buffer(*ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto heap = vkexec::test::sync_wait_value(vkexec::factory::make_descriptor_heap_buffer(*ctx, heap_bytes));
  std::vector<std::byte> slot(layout.buffer_descriptor_size);
  auto const storage_addr = storage.device_address();
  REQUIRE(storage_addr.has_value());
  REQUIRE(vkexec::write_storage_buffer_descriptor(*ctx, *storage_addr, storage.size(), slot));
  auto mapped = heap.mapped();
  REQUIRE(mapped.size() >= slot.size());
  std::ranges::copy(slot, mapped.begin());

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  auto const reserved_offset = layout.descriptor_stride * k_slot_count;
  auto const aligned_offset =
    (layout.resource_heap_alignment == 0)
      ? reserved_offset
      : ((reserved_offset + layout.resource_heap_alignment - 1) / layout.resource_heap_alignment)
          * layout.resource_heap_alignment;
  auto const heap_addr = heap.device_address();
  REQUIRE(heap_addr.has_value());
  REQUIRE(vkexec::cmd_bind_resource_heap(
    *ctx, cmd, *heap_addr, heap.size(), aligned_offset, layout.min_resource_heap_reserved_range));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx->free_command_buffer(cmd);
}
// NOLINTEND(readability-function-cognitive-complexity)

TEST_CASE("write_storage_image_descriptor fills a heap slot", "[vkexec][descriptor_heap][gpu]")
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

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));

  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = vkexec::expected_get(layout_result);
  REQUIRE(layout.image_descriptor_size > 0);

  auto img = vkexec::test::sync_wait_value(vkexec::factory::make_image(*ctx,
    vkexec::image_create_info{
      .width = k_image_extent,
      .height = k_image_extent,
      .usage = vkexec::image_usage::color_storage,
    }));

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = img.handle();
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = img.format();
  view_info.subresourceRange = {
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
  };

  std::vector<std::byte> slot(layout.image_descriptor_size);
  REQUIRE(vkexec::write_storage_image_descriptor(*ctx, view_info, VK_IMAGE_LAYOUT_GENERAL, slot));
}

TEST_CASE("write_sampled_image_descriptor fills a heap slot", "[vkexec][descriptor_heap][gpu]")
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

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));

  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = vkexec::expected_get(layout_result);
  REQUIRE(layout.image_descriptor_size > 0);

  auto img = vkexec::test::sync_wait_value(vkexec::factory::make_image(*ctx,
    vkexec::image_create_info{
      .width = k_image_extent,
      .height = k_image_extent,
      .usage = vkexec::image_usage::color_storage,
    }));

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = img.handle();
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = img.format();
  view_info.subresourceRange = {
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
  };

  std::vector<std::byte> slot(layout.image_descriptor_size);
  REQUIRE(vkexec::write_sampled_image_descriptor(*ctx, view_info, VK_IMAGE_LAYOUT_GENERAL, slot));
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST_CASE("write_sampler_descriptor and cmd_bind_sampler_heap", "[vkexec][descriptor_heap][gpu]")
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

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));

  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = vkexec::expected_get(layout_result);
  REQUIRE(layout.sampler_descriptor_size > 0);

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

  std::vector<std::byte> slot(layout.sampler_descriptor_size);
  REQUIRE(vkexec::write_sampler_descriptor(*ctx, sampler_info, slot));

  auto const heap_bytes = vkexec::sampler_heap_byte_size(layout, k_slot_count);
  auto heap = vkexec::test::sync_wait_value(vkexec::factory::make_descriptor_heap_buffer(*ctx, heap_bytes));
  auto mapped = heap.mapped();
  REQUIRE(mapped.size() >= slot.size());
  std::ranges::copy(slot, mapped.begin());

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  auto const reserved_offset = layout.sampler_descriptor_size * k_slot_count;
  auto const aligned_offset =
    (layout.sampler_heap_alignment == 0)
      ? reserved_offset
      : ((reserved_offset + layout.sampler_heap_alignment - 1) / layout.sampler_heap_alignment)
          * layout.sampler_heap_alignment;
  auto const heap_addr = heap.device_address();
  REQUIRE(heap_addr.has_value());
  REQUIRE(vkexec::cmd_bind_sampler_heap(
    *ctx, cmd, *heap_addr, heap.size(), aligned_offset, layout.min_sampler_heap_reserved_range));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx->free_command_buffer(cmd);
}
// NOLINTEND(readability-function-cognitive-complexity)
