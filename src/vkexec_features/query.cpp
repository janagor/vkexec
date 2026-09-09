#include <vkexec_features/common.hpp>
#include <vkexec_features/query.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::feat::detail {

auto physical_device_timeline_semaphore(VkPhysicalDevice physical_device, std::uint32_t api_version) -> bool
{
  if (physical_device == VK_NULL_HANDLE) { return false; }

  if (api_at_least(api_version, 1, 2)) {
    VkPhysicalDeviceVulkan12Features features_12{};
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features_12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features_12.timelineSemaphore == VK_TRUE;
  }

  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features_khr;
  vkGetPhysicalDeviceFeatures2(physical_device, &features2);
  return features_khr.timelineSemaphore == VK_TRUE;
}

auto physical_device_buffer_device_address(VkPhysicalDevice physical_device, std::uint32_t api_version) -> bool
{
  if (physical_device == VK_NULL_HANDLE) { return false; }

  if (api_at_least(api_version, 1, 2)) {
    VkPhysicalDeviceVulkan12Features features_12{};
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features_12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features_12.bufferDeviceAddress == VK_TRUE;
  }

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features_khr;
  vkGetPhysicalDeviceFeatures2(physical_device, &features2);
  return features_khr.bufferDeviceAddress == VK_TRUE;
}

auto physical_device_dynamic_rendering(VkPhysicalDevice physical_device, std::uint32_t api_version) -> bool
{
  if (physical_device == VK_NULL_HANDLE) { return false; }

  if (api_at_least(api_version, 1, 3)) {
    VkPhysicalDeviceVulkan13Features features_13{};
    features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features_13;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features_13.dynamicRendering == VK_TRUE;
  }

  VkPhysicalDeviceDynamicRenderingFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features_khr;
  vkGetPhysicalDeviceFeatures2(physical_device, &features2);
  return features_khr.dynamicRendering == VK_TRUE;
}

}// namespace vkexec::feat::detail
