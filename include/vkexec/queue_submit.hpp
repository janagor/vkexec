#ifndef VKEXEC_QUEUE_SUBMIT_HPP
#define VKEXEC_QUEUE_SUBMIT_HPP

//! \file
//! Mixed binary / timeline queue submit description for `context::submit`.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace vkexec {

/**
 * Wait or signal entry for `queue_submit`.
 *
 * `value` is used when any entry in the submit needs a timeline semaphore;
 * binary semaphores ignore `value`.
 */
struct semaphore_submit
{
  VkSemaphore semaphore{ VK_NULL_HANDLE };
  std::uint64_t value{ 0 };
  VkPipelineStageFlags stage{ VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
};

/**
 * Description of a `vkQueueSubmit` with optional binary and timeline sync.
 *
 * When `queue` is null, `context::submit` uses the context compute queue.
 *
 * @see context::submit, semaphore_submit
 */
struct queue_submit
{
  std::span<VkCommandBuffer const> command_buffers;
  // NOLINTNEXTLINE(readability-redundant-member-init) -- keep for designated-init call sites
  std::span<semaphore_submit const> waits{};
  // NOLINTNEXTLINE(readability-redundant-member-init) -- keep for designated-init call sites
  std::span<semaphore_submit const> signals{};
  VkFence fence{ VK_NULL_HANDLE };
  //! Defaults to the context compute queue when null.
  VkQueue queue{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_QUEUE_SUBMIT_HPP
