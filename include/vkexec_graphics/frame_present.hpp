#ifndef VKEXEC_GRAPHICS_FRAME_PRESENT_HPP
#define VKEXEC_GRAPHICS_FRAME_PRESENT_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/frame_ring.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

struct acquired_present_frame
{
  std::size_t slot{};
  std::uint32_t image_index{};
  std::uint64_t timeline_value{};
  frame_ring_submit_sync submit_sync{};
};

enum class present_acquire_status : std::uint8_t { ready, needs_recreate };

struct present_acquire_result
{
  present_acquire_status status{ present_acquire_status::needs_recreate };
  acquired_present_frame frame{};
};

/// Wait for slot reuse, acquire a swapchain image, and build submit wait/signal lists.
[[nodiscard]] auto acquire_present_frame(frame_ring &ring,
  swapchain &chain,
  std::size_t slot,
  VkPipelineStageFlags acquire_wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
  -> detail::result<present_acquire_result>;

/// Submit recorded work with frame-ring sync, mark timeline completion, and present.
/// Returns `false` when the swapchain must be recreated; `true` on success.
[[nodiscard]] auto submit_and_present(context &ctx,
  frame_ring &ring,
  swapchain &chain,
  acquired_present_frame const &frame,
  std::span<VkCommandBuffer const> command_buffers) -> detail::result<bool>;

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_FRAME_PRESENT_HPP
