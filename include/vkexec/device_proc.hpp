#ifndef VKEXEC_DEVICE_PROC_HPP
#define VKEXEC_DEVICE_PROC_HPP

#include <vulkan/vulkan.h>

namespace vkexec {

/**
 * Loads a single device entry point via `vkGetDeviceProcAddr`.
 *
 * @param device Logical device (null returns a null function pointer).
 * @param name Entry point name (null returns a null function pointer).
 * @return Function pointer of type `Fn`, or null when unavailable.
 */
template<typename Fn> [[nodiscard]] auto resolve_device_proc(VkDevice device, char const *name) noexcept -> Fn
{
  if (device == VK_NULL_HANDLE || name == nullptr) { return Fn{}; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<Fn>(vkGetDeviceProcAddr(device, name));
}

}// namespace vkexec

#endif// VKEXEC_DEVICE_PROC_HPP
