#include <vkexec/descriptor_heap.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>

namespace vkexec {
namespace {

  [[nodiscard]] auto align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept -> VkDeviceSize
  {
    if (alignment == 0) { return value; }
    return (value + alignment - 1) / alignment * alignment;
  }

}// namespace

auto query_descriptor_heap_layout(context const &ctx) -> result<descriptor_heap_layout>
{
  if (ctx.physical_device() == VK_NULL_HANDLE) {
    return std::unexpected(make_error(errc::invalid_argument, "query_descriptor_heap_layout requires a physical device"));
  }

  VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props{};
  heap_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;

  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &heap_props;
  vkGetPhysicalDeviceProperties2(ctx.physical_device(), &props2);

  descriptor_heap_layout layout{};
  layout.buffer_descriptor_size = static_cast<std::size_t>(heap_props.bufferDescriptorSize);
  layout.image_descriptor_size = static_cast<std::size_t>(heap_props.imageDescriptorSize);
  layout.descriptor_stride =
    static_cast<std::size_t>(std::max(heap_props.bufferDescriptorSize, heap_props.imageDescriptorSize));
  layout.resource_heap_alignment = heap_props.resourceHeapAlignment;
  layout.min_resource_heap_reserved_range = heap_props.minResourceHeapReservedRange;

  if (layout.descriptor_stride == 0) {
    return std::unexpected(
      make_error(errc::unsupported, "descriptor heap properties reported a zero descriptor stride"));
  }
  auto const descriptor_alignment =
    std::max(heap_props.bufferDescriptorAlignment, heap_props.imageDescriptorAlignment);
  if (descriptor_alignment != 0 && layout.descriptor_stride % descriptor_alignment != 0) {
    return std::unexpected(
      make_error(errc::unsupported, "descriptor heap stride is not aligned for buffer/image descriptors"));
  }
  return layout;
}

auto descriptor_heap_byte_size(descriptor_heap_layout const &layout, std::size_t descriptor_count) -> VkDeviceSize
{
  auto const descriptor_region =
    static_cast<VkDeviceSize>(layout.descriptor_stride) * static_cast<VkDeviceSize>(descriptor_count);
  auto const reserved_offset = align_up(descriptor_region, layout.resource_heap_alignment);
  return reserved_offset + layout.min_resource_heap_reserved_range;
}

auto write_storage_buffer_descriptor(context const &ctx,
  VkDeviceAddress buffer_address,
  VkDeviceSize buffer_size,
  std::span<std::byte> destination) -> status
{
  if (ctx.procs().write_resource_descriptors == nullptr) {
    return make_error(errc::unsupported, "vkWriteResourceDescriptorsEXT is unavailable");
  }
  if (destination.empty()) {
    return make_error(errc::invalid_argument, "write_storage_buffer_descriptor destination is empty");
  }

  VkDeviceAddressRangeEXT address_range{};
  address_range.address = buffer_address;
  address_range.size = buffer_size;

  VkResourceDescriptorInfoEXT resource_info{};
  resource_info.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
  resource_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  resource_info.data.pAddressRange = &address_range;

  VkHostAddressRangeEXT host_range{};
  host_range.address = destination.data();
  host_range.size = destination.size();

  VkResult const write_result =
    ctx.procs().write_resource_descriptors(ctx.device(), 1, &resource_info, &host_range);
  if (write_result != VK_SUCCESS) {
    return make_vk_error(write_result, "vkWriteResourceDescriptorsEXT failed");
  }
  return {};
}

auto cmd_bind_resource_heap(context const &ctx,
  VkCommandBuffer cmd,
  VkDeviceAddress heap_address,
  VkDeviceSize heap_size,
  VkDeviceSize reserved_range_offset,
  VkDeviceSize reserved_range_size) -> status
{
  if (ctx.procs().cmd_bind_resource_heap == nullptr) {
    return make_error(errc::unsupported, "vkCmdBindResourceHeapEXT is unavailable");
  }

  VkBindHeapInfoEXT bind_info{};
  bind_info.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
  bind_info.heapRange.address = heap_address;
  bind_info.heapRange.size = heap_size;
  bind_info.reservedRangeOffset = reserved_range_offset;
  bind_info.reservedRangeSize = reserved_range_size;
  ctx.procs().cmd_bind_resource_heap(cmd, &bind_info);
  return {};
}

}// namespace vkexec
