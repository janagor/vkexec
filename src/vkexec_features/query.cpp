#include <vkexec_features/common.hpp>
#include <vkexec_features/query.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

namespace vkexec::feat::detail {

auto physical_device_timeline_semaphore(VkPhysicalDevice physical_device, std::uint32_t api_version) -> bool
{
  if (physical_device == VK_NULL_HANDLE) { return false; }

  // Query the matching feature struct for the API the context negotiated.
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

auto physical_device_synchronization2(VkInstance instance, VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool
{
  if (physical_device == VK_NULL_HANDLE || instance == VK_NULL_HANDLE) { return false; }
  std::uint32_t count = 0;
  if (api_version < VK_API_VERSION_1_3) {
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr) != VK_SUCCESS) { return false; }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data()) != VK_SUCCESS) {
      return false;
    }
    if (!std::ranges::any_of(extensions, [](VkExtensionProperties const &extension) -> bool {
          return std::strcmp(std::data(extension.extensionName), VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0;
        })) {
      return false;
    }
  }

  VkPhysicalDeviceSynchronization2Features features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features;
  if (api_version >= VK_API_VERSION_1_1) {
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
  } else {
    // A Vulkan 1.0 instance needs VK_KHR_get_physical_device_properties2 enabled.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
    if (get_features2 == nullptr) { return false; }
    get_features2(physical_device, &features2);
  }
  return features.synchronization2 == VK_TRUE;
}

}// namespace vkexec::feat::detail
