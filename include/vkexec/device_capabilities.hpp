#ifndef VKEXEC_DEVICE_CAPABILITIES_HPP
#define VKEXEC_DEVICE_CAPABILITIES_HPP

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec {

//! Features enabled and usable on this logical device.
struct device_capabilities
{
  std::uint32_t api_version{ VK_API_VERSION_1_0 };
  bool synchronization2{ false };
  bool timeline_semaphore{ false };
  bool buffer_device_address{ false };
  bool dynamic_rendering{ false };
};

}// namespace vkexec

#endif// VKEXEC_DEVICE_CAPABILITIES_HPP
