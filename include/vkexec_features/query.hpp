#ifndef VKEXEC_FEATURES_QUERY_HPP
#define VKEXEC_FEATURES_QUERY_HPP

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec::feat::detail {

[[nodiscard]] auto physical_device_timeline_semaphore(VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool;

[[nodiscard]] auto physical_device_buffer_device_address(VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool;

[[nodiscard]] auto physical_device_dynamic_rendering(VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool;

}// namespace vkexec::feat::detail

#endif// VKEXEC_FEATURES_QUERY_HPP
