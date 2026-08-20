#ifndef VKEXEC_DETAIL_FENCE_WAIT_HPP
#define VKEXEC_DETAIL_FENCE_WAIT_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/config.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  inline auto destroy_submission_sync(context const &ctx, VkSemaphore semaphore, VkFence fence) noexcept -> void
  {
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(ctx.device(), semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(ctx.device(), fence, nullptr); }
  }

  inline auto wait_fence_or_queue(context const &ctx, VkFence fence) -> void
  {
    if (fence != VK_NULL_HANDLE) {
      if (vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        VKEXEC_THROW(std::runtime_error("vkWaitForFences failed"));
      }
      return;
    }
    if (vkQueueWaitIdle(ctx.compute_queue()) != VK_SUCCESS) {
      VKEXEC_THROW(std::runtime_error("vkQueueWaitIdle failed"));
    }
  }

  inline auto wait_and_release_submission(context const &ctx, VkSemaphore semaphore, VkFence fence) -> void
  {
    VKEXEC_TRY { wait_fence_or_queue(ctx, fence); }
    VKEXEC_CATCH_ALL
    {
      destroy_submission_sync(ctx, semaphore, fence);
      // cppcheck-suppress rethrowNoCurrentException
      throw;
    }
    destroy_submission_sync(ctx, semaphore, fence);
  }

  template<class StopToken>
  auto poll_for_stop_or_fence(context const &ctx, VkFence fence, StopToken token) -> bool
  {
    constexpr std::uint64_t k_poll_timeout_ns = 1'000'000ULL;// 1 ms
    bool stop_seen = token.stop_requested();
    if (fence == VK_NULL_HANDLE) {
      wait_fence_or_queue(ctx, fence);
      return stop_seen || token.stop_requested();
    }
    while (true) {
      stop_seen = stop_seen || token.stop_requested();
      VkResult const result = vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, k_poll_timeout_ns);
      if (result == VK_SUCCESS) { return stop_seen || token.stop_requested(); }
      if (result != VK_TIMEOUT) { VKEXEC_THROW(std::runtime_error("vkWaitForFences failed")); }
    }
  }

  /// Wait for GPU completion, polling so a stop request can be observed.
  /// Always waits for the GPU before destroying sync objects / returning.
  /// Returns true when the caller should complete with `set_stopped`.
  template<class StopToken>
  auto wait_submission_with_stop(context const &ctx, VkSemaphore semaphore, VkFence fence, StopToken token) -> bool
  {
    if constexpr (ex::unstoppable_token<StopToken>) {
      wait_and_release_submission(ctx, semaphore, fence);
      return false;
    } else {
      bool stop_seen = false;
      VKEXEC_TRY { stop_seen = poll_for_stop_or_fence(ctx, fence, token); }
      VKEXEC_CATCH_ALL
      {
        destroy_submission_sync(ctx, semaphore, fence);
        // cppcheck-suppress rethrowNoCurrentException
        throw;
      }
      destroy_submission_sync(ctx, semaphore, fence);
      return stop_seen;
    }
  }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_DETAIL_FENCE_WAIT_HPP
