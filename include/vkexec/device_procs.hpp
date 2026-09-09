#ifndef VKEXEC_DEVICE_PROCS_HPP
#define VKEXEC_DEVICE_PROCS_HPP

#include <vulkan/vulkan.h>

namespace vkexec {

/// Core device entry points loaded via `vkGetDeviceProcAddr`. Extension PFNs load in their modules.
struct device_procs
{
  PFN_vkGetBufferDeviceAddress get_buffer_device_address{};
};

}// namespace vkexec

#endif// VKEXEC_DEVICE_PROCS_HPP
