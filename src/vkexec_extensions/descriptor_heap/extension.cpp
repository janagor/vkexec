#include <vkexec_extensions/descriptor_heap/extension.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::ext {

auto extension_traits<descriptor_heap>::available(context const &ctx) -> bool
{
  return ctx.procs().write_resource_descriptors != nullptr && ctx.procs().cmd_push_data != nullptr;
}

auto extension_traits<descriptor_heap>::configure(vulkan_requirements &req) -> void
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  req.optional_device_extensions.push_back(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
  req.require_extension_feature(features_12).enable_extension_feature_if_present(features_heap);
}

}// namespace vkexec::ext
