#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_DESCRIPTOR_HEAP_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_DESCRIPTOR_HEAP_HPP

//! \file
//! Layout query and host/GPU helpers for `VK_EXT_descriptor_heap`.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

/**
 * Queried `VK_EXT_descriptor_heap` size/alignment properties for mixed buffer/image heaps.
 *
 * @see query_descriptor_heap_layout, descriptor_heap_byte_size
 */
struct descriptor_heap_layout
{
  std::size_t buffer_descriptor_size{ 0 };
  std::size_t image_descriptor_size{ 0 };
  std::size_t descriptor_stride{ 0 };
  VkDeviceSize resource_heap_alignment{ 0 };
  VkDeviceSize min_resource_heap_reserved_range{ 0 };
};

//! Queries descriptor-heap layout properties from `ctx`'s physical device.
[[nodiscard]] auto query_descriptor_heap_layout(context const &ctx) -> result<descriptor_heap_layout>;

/**
 * Returns bytes needed for `descriptor_count` slots plus the implementation reserved range.
 *
 * @param layout Layout from `query_descriptor_heap_layout`.
 * @param descriptor_count Number of descriptor slots to allocate.
 */
[[nodiscard]] auto descriptor_heap_byte_size(descriptor_heap_layout const &layout, std::size_t descriptor_count)
  -> VkDeviceSize;

/**
 * Host-writes a storage-buffer descriptor into a slot-sized destination span.
 *
 * @param ctx Context used to resolve extension procs.
 * @param buffer_address Device address of the storage buffer.
 * @param buffer_size Size of the buffer range described by the descriptor.
 * @param destination Host-mapped slot bytes (must be large enough for one buffer descriptor).
 */
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto write_storage_buffer_descriptor(context const &ctx,
  VkDeviceAddress buffer_address,
  VkDeviceSize buffer_size,
  std::span<std::byte> destination) -> status;
// NOLINTEND(bugprone-easily-swappable-parameters)

/**
 * Host-writes a storage-image descriptor into a slot-sized destination span.
 *
 * @param view_info Image view create info describing the storage image.
 * @param layout Image layout expected when the descriptor is used.
 * @param destination Host-mapped slot bytes (must be large enough for one image descriptor).
 */
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto write_storage_image_descriptor(context const &ctx,
  VkImageViewCreateInfo const &view_info,
  VkImageLayout layout,
  std::span<std::byte> destination) -> status;
// NOLINTEND(bugprone-easily-swappable-parameters)

/**
 * Binds a resource descriptor heap buffer for subsequent bindless dispatches/draws.
 *
 * @param cmd Command buffer in the recording state.
 * @param heap_address Device address of the heap buffer.
 * @param heap_size Size of the heap buffer in bytes.
 * @param reserved_range_offset Offset of the implementation reserved range.
 * @param reserved_range_size Size of the implementation reserved range.
 */
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto cmd_bind_resource_heap(context const &ctx,
  VkCommandBuffer cmd,
  VkDeviceAddress heap_address,
  VkDeviceSize heap_size,
  VkDeviceSize reserved_range_offset,
  VkDeviceSize reserved_range_size) -> status;
// NOLINTEND(bugprone-easily-swappable-parameters)

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_DESCRIPTOR_HEAP_HPP
