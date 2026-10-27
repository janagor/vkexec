#ifndef VKEXEC_PROFILING_DETAIL_FENCE_STATUS_HPP
#define VKEXEC_PROFILING_DETAIL_FENCE_STATUS_HPP

#include <vulkan/vulkan.h>

namespace vkexec::profiling::detail {

using fence_status_fn = VkResult (*)(VkDevice, VkFence);

[[nodiscard]] auto fence_status(VkDevice device, VkFence fence) -> VkResult;
// Source-private test hook. Overrides only the calling thread.
[[nodiscard]] auto set_fence_status_override(fence_status_fn replacement) noexcept -> fence_status_fn;

}// namespace vkexec::profiling::detail

#endif// VKEXEC_PROFILING_DETAIL_FENCE_STATUS_HPP
