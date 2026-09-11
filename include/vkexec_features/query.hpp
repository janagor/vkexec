#ifndef VKEXEC_FEATURES_QUERY_HPP
#define VKEXEC_FEATURES_QUERY_HPP

//! \file
//! Physical-device queries used by feature `available()` implementations.

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec::feat::detail {

//! Returns whether timeline semaphores are supported for `api_version` on `physical_device`.
[[nodiscard]] auto physical_device_timeline_semaphore(VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool;

//! Returns whether buffer device address is supported for `api_version` on `physical_device`.
[[nodiscard]] auto physical_device_buffer_device_address(VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool;

//! Returns whether dynamic rendering is supported for `api_version` on `physical_device`.
[[nodiscard]] auto physical_device_dynamic_rendering(VkPhysicalDevice physical_device, std::uint32_t api_version)
  -> bool;

}// namespace vkexec::feat::detail

#endif// VKEXEC_FEATURES_QUERY_HPP
