#ifndef VKEXEC_DETAIL_RETIREMENT_WAIT_HPP
#define VKEXEC_DETAIL_RETIREMENT_WAIT_HPP

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <exception>

namespace vkexec::detail {

// Cleanup cannot release Vulkan loans until a wait succeeds or the device is lost.
// Repeated ordinary failures leave retirement unknown and terminate the process.
// The attempt count bounds retries, not elapsed time: each Vulkan wait can block.
inline constexpr std::uint32_t k_retirement_wait_attempts = 3;

[[nodiscard]] inline auto wait_for_retirement(VkDevice device,
  std::uint32_t count,
  VkFence const *fences,
  PFN_vkWaitForFences wait = vkWaitForFences) noexcept -> VkResult
{
  VkResult first_failure = VK_SUCCESS;
  for (std::uint32_t attempt = 0; attempt < k_retirement_wait_attempts; ++attempt) {
    VkResult const result = wait(device, count, fences, VK_TRUE, UINT64_MAX);
    if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST) {
      return first_failure == VK_SUCCESS ? result : first_failure;
    }
    if (first_failure == VK_SUCCESS) { first_failure = result; }
  }
  std::terminate();
}

[[nodiscard]] inline auto wait_for_queue_retirement(VkQueue queue, PFN_vkQueueWaitIdle wait = vkQueueWaitIdle) noexcept
  -> VkResult
{
  VkResult first_failure = VK_SUCCESS;
  for (std::uint32_t attempt = 0; attempt < k_retirement_wait_attempts; ++attempt) {
    VkResult const result = wait(queue);
    if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST) {
      return first_failure == VK_SUCCESS ? result : first_failure;
    }
    if (first_failure == VK_SUCCESS) { first_failure = result; }
  }
  std::terminate();
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RETIREMENT_WAIT_HPP
