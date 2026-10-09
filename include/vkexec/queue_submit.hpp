#ifndef VKEXEC_QUEUE_SUBMIT_HPP
#define VKEXEC_QUEUE_SUBMIT_HPP

//! \file
//! Synchronization2 queue submit description for `context::submit`.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace vkexec {

//! A queue and the family that owns command buffers submitted to it.
struct queue_ref
{
  VkQueue queue{ VK_NULL_HANDLE };
  std::uint32_t family{ VK_QUEUE_FAMILY_IGNORED };
};

/**
 * Wait or signal entry for `queue_submit`.
 *
 * Timeline semaphores use `value`; binary semaphores ignore it.
 */
struct semaphore_submit
{
  VkSemaphore semaphore{ VK_NULL_HANDLE };
  std::uint64_t value{ 0 };
  VkPipelineStageFlags2 stage{ VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
};

/**
 * Description of one Vulkan queue submission with optional binary and
 * timeline synchronization.
 *
 * Stage masks use synchronization2 semantics. On devices without
 * synchronization2, vkexec lowers waits that are representable by legacy
 * queue submission. NONE or HOST wait stages, stage masks with no legacy
 * representation, and non-ALL_COMMANDS signal stages fail with
 * `errc::unsupported`.
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
