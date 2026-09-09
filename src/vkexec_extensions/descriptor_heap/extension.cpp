#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/descriptor_heap/procs.hpp>

#include <vkexec_features/buffer_device_address.hpp>
#include <vkexec_features/feature.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::ext {

auto extension_traits<descriptor_heap>::available(context const &ctx) -> bool
{ return descriptor_heap_available(descriptor_heap_procs_for(ctx)); }

auto extension_traits<descriptor_heap>::configure(vulkan_requirements &req) -> void
{
  feat::configure<feat::buffer_device_address>(req);

  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  req.optional_device_extensions.push_back(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
  req.enable_extension_feature_if_present(features_heap);
}

}// namespace vkexec::ext
