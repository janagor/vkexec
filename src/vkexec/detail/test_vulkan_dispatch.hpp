#ifndef VKEXEC_TEST_VULKAN_DISPATCH_HPP
#define VKEXEC_TEST_VULKAN_DISPATCH_HPP

#include <vkexec/config.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>

namespace vkexec::detail {

#ifdef VKEXEC_TEST_VULKAN_DISPATCH
struct test_vulkan_dispatch
{
  VkDevice device{ VK_NULL_HANDLE };
  std::array<VkQueue, 3> queues{};
  PFN_vkQueueSubmit queue_submit{ vkQueueSubmit };
  PFN_vkQueueSubmit2 queue_submit2{ nullptr };
  PFN_vkGetFenceStatus get_fence_status{ vkGetFenceStatus };
  PFN_vkWaitForFences wait_for_fences{ vkWaitForFences };
  PFN_vkQueueWaitIdle queue_wait_idle{ vkQueueWaitIdle };
#if VKEXEC_HAS_EXCEPTIONS
  bool (*fail_enqueue)() = nullptr;
#endif
};

// Only one dispatch installation may be active per process. Tests using this
// seam must not overlap. Each caller retains a snapshot until its Vulkan call
// returns, even when a test uninstalls the active configuration.
auto set_test_vulkan_dispatch(test_vulkan_dispatch const *dispatch) -> void;
[[nodiscard]] auto current_test_vulkan_dispatch() noexcept -> std::shared_ptr<test_vulkan_dispatch const>;
[[nodiscard]] inline auto targets_queue(test_vulkan_dispatch const &dispatch, VkQueue queue) noexcept -> bool
{
  for (VkQueue const candidate : dispatch.queues) {
    if (candidate != VK_NULL_HANDLE && candidate == queue) { return true; }
  }
  return false;
}
#endif

[[nodiscard]] inline auto
  dispatch_queue_submit(VkQueue queue, std::uint32_t count, VkSubmitInfo const *submits, VkFence fence) -> VkResult
{
#ifdef VKEXEC_TEST_VULKAN_DISPATCH
  if (auto dispatch = current_test_vulkan_dispatch(); dispatch && targets_queue(*dispatch, queue)) {
    return dispatch->queue_submit(queue, count, submits, fence);
  }
#endif
  return vkQueueSubmit(queue, count, submits, fence);
}

[[nodiscard]] inline auto dispatch_queue_submit2(PFN_vkQueueSubmit2 submit2,
  VkQueue queue,
  std::uint32_t count,
  VkSubmitInfo2 const *submits,
  VkFence fence) -> VkResult
{
#ifdef VKEXEC_TEST_VULKAN_DISPATCH
  if (auto dispatch = current_test_vulkan_dispatch();
    dispatch && targets_queue(*dispatch, queue) && dispatch->queue_submit2) {
    return dispatch->queue_submit2(queue, count, submits, fence);
  }
#endif
  return submit2(queue, count, submits, fence);
}

[[nodiscard]] inline auto dispatch_get_fence_status(VkDevice device, VkFence fence) -> VkResult
{
#ifdef VKEXEC_TEST_VULKAN_DISPATCH
  if (auto dispatch = current_test_vulkan_dispatch(); dispatch && dispatch->device == device) {
    return dispatch->get_fence_status(device, fence);
  }
#endif
  return vkGetFenceStatus(device, fence);
}

[[nodiscard]] inline auto dispatch_wait_for_fences(VkDevice device,
  std::uint32_t count,
  VkFence const *fences,
  VkBool32 wait_all,
  std::uint64_t timeout) -> VkResult
{
#ifdef VKEXEC_TEST_VULKAN_DISPATCH
  if (auto dispatch = current_test_vulkan_dispatch(); dispatch && dispatch->device == device) {
    return dispatch->wait_for_fences(device, count, fences, wait_all, timeout);
  }
#endif
  return vkWaitForFences(device, count, fences, wait_all, timeout);
}

[[nodiscard]] inline auto dispatch_queue_wait_idle(VkQueue queue) -> VkResult
{
#ifdef VKEXEC_TEST_VULKAN_DISPATCH
  if (auto dispatch = current_test_vulkan_dispatch(); dispatch && targets_queue(*dispatch, queue)) {
    return dispatch->queue_wait_idle(queue);
  }
#endif
  return vkQueueWaitIdle(queue);
}

}// namespace vkexec::detail

#endif// VKEXEC_TEST_VULKAN_DISPATCH_HPP
