#ifndef VKEXEC_DEVICE_PROCS_HPP
#define VKEXEC_DEVICE_PROCS_HPP

#include <vulkan/vulkan.h>

namespace vkexec {

/**
 * Core device entry points loaded via `vkGetDeviceProcAddr`.
 *
 * Extension PFNs are loaded in their own modules. Null members mean the
 * entry point was not resolved (feature/extension unavailable).
 *
 * @see context::procs, resolve_device_proc
 */
struct device_procs
{
  PFN_vkGetBufferDeviceAddress get_buffer_device_address{};
};

}// namespace vkexec

#endif// VKEXEC_DEVICE_PROCS_HPP
