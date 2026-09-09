#ifndef VKEXEC_DEVICE_PROC_HPP
#define VKEXEC_DEVICE_PROC_HPP

#include <vulkan/vulkan.h>

namespace vkexec {

/// Load a single device entry point via `vkGetDeviceProcAddr`. Returns null when unavailable.
template<typename Fn>
[[nodiscard]] auto resolve_device_proc(VkDevice device, char const *name) noexcept -> Fn
{
  if (device == VK_NULL_HANDLE || name == nullptr) { return Fn{}; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<Fn>(vkGetDeviceProcAddr(device, name));
}

}// namespace vkexec

#endif// VKEXEC_DEVICE_PROC_HPP
