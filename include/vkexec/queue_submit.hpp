#ifndef VKEXEC_QUEUE_SUBMIT_HPP
#define VKEXEC_QUEUE_SUBMIT_HPP

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace vkexec {

/// Wait or signal entry for `queue_submit`. `value` is used when any entry needs a timeline.
struct semaphore_submit
{
  VkSemaphore semaphore{ VK_NULL_HANDLE };
  std::uint64_t value{ 0 };
  VkPipelineStageFlags stage{ VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
};

/// Mixed binary / timeline queue submit description.
struct queue_submit
{
  std::span<VkCommandBuffer const> command_buffers;
  std::span<semaphore_submit const> waits{};
  std::span<semaphore_submit const> signals{};
  VkFence fence{ VK_NULL_HANDLE };
  /// Defaults to the context compute queue when null.
  VkQueue queue{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_QUEUE_SUBMIT_HPP
