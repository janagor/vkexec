#ifndef VKEXEC_TEST_RECORDING_BUFFER_HPP
#define VKEXEC_TEST_RECORDING_BUFFER_HPP
#include <memory>
#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

namespace vkexec::test {
//! Small native fixture shared by recording tests and benchmarks.
struct recording_buffer
{
  VkDevice device{};
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  static constexpr VkDeviceSize byte_size = 64;
  ~recording_buffer()
  {
    if (buffer) { vkDestroyBuffer(device, buffer, nullptr); }
    if (memory) { vkFreeMemory(device, memory, nullptr); }
  }
  [[nodiscard]] static auto make(context &ctx, VkMemoryPropertyFlags required_memory = 0)
    -> result<std::unique_ptr<recording_buffer>>
  {
    auto resource = std::make_unique<recording_buffer>();
    resource->device = ctx.device();
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = byte_size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device(), &info, nullptr, &resource->buffer) != VK_SUCCESS) {
      return fail(errc::vulkan, "recording fixture buffer allocation failed");
    }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.device(), resource->buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &properties);
    for (std::uint32_t index{}; index < properties.memoryTypeCount; ++index) {
      if ((requirements.memoryTypeBits & (1U << index)) == 0
          || (properties.memoryTypes[index].propertyFlags & required_memory) != required_memory) {
        continue;
      }
      VkMemoryAllocateInfo allocation{};
      allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = index;
      if (vkAllocateMemory(ctx.device(), &allocation, nullptr, &resource->memory) != VK_SUCCESS) {
        return fail(errc::vulkan, "recording fixture memory allocation failed");
      }
      if (vkBindBufferMemory(ctx.device(), resource->buffer, resource->memory, 0) != VK_SUCCESS) {
        return fail(errc::vulkan, "recording fixture buffer binding failed");
      }
      return resource;
    }
    return fail(errc::unsupported, "no buffer memory type");
  }
};
}// namespace vkexec::test
#endif
