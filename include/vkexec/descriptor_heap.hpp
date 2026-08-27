#ifndef VKEXEC_DESCRIPTOR_HEAP_HPP
#define VKEXEC_DESCRIPTOR_HEAP_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

/// Queried `VK_EXT_descriptor_heap` size/alignment properties for mixed buffer/image heaps.
struct descriptor_heap_layout
{
  std::size_t buffer_descriptor_size{ 0 };
  std::size_t image_descriptor_size{ 0 };
  std::size_t descriptor_stride{ 0 };
  VkDeviceSize resource_heap_alignment{ 0 };
  VkDeviceSize min_resource_heap_reserved_range{ 0 };
};

[[nodiscard]] auto query_descriptor_heap_layout(context const &ctx) -> result<descriptor_heap_layout>;

/// Bytes needed for `descriptor_count` slots plus the implementation reserved range.
[[nodiscard]] auto descriptor_heap_byte_size(descriptor_heap_layout const &layout, std::size_t descriptor_count)
  -> VkDeviceSize;

/// Host write of a storage-buffer descriptor into a slot-sized destination span.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto write_storage_buffer_descriptor(context const &ctx,
  VkDeviceAddress buffer_address,
  VkDeviceSize buffer_size,
  std::span<std::byte> destination) -> status;
// NOLINTEND(bugprone-easily-swappable-parameters)

/// Bind a resource descriptor heap buffer for subsequent bindless dispatches/draws.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto cmd_bind_resource_heap(context const &ctx,
  VkCommandBuffer cmd,
  VkDeviceAddress heap_address,
  VkDeviceSize heap_size,
  VkDeviceSize reserved_range_offset,
  VkDeviceSize reserved_range_size) -> status;
// NOLINTEND(bugprone-easily-swappable-parameters)

}// namespace vkexec

#endif// VKEXEC_DESCRIPTOR_HEAP_HPP
