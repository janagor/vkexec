#ifndef VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_FRAME_PRESENT_HPP
#define VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_FRAME_PRESENT_HPP

//! \file
//! Acquire / submit / present helpers using `frame_ring` + `swapchain`.

#include <vkexec/context.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec_extensions/timeline_semaphore/frame_ring.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

/**
 * Frame identity and submit sync produced by a successful acquire.
 *
 * @see acquire_present_frame, submit_and_present
 */
struct acquired_present_frame
{
  std::size_t slot{};
  std::uint32_t image_index{};
  std::uint64_t timeline_value{};
  frame_ring_submit_sync submit_sync{};
};

//! Outcome of acquire: ready to record, or swapchain needs recreate.
enum class present_acquire_status : std::uint8_t { ready, needs_recreate };

/**
 * Result of `acquire_present_frame`.
 *
 * When `status == ready`, `frame` holds slot/image/timeline/sync for submit.
 */
struct present_acquire_result
{
  present_acquire_status status{ present_acquire_status::needs_recreate };
  acquired_present_frame frame{};
};

/**
 * Waits for slot reuse, acquires a swapchain image, and builds submit wait/signal lists.
 *
 * @param ring Frame ring owning acquire/present/timeline sync.
 * @param chain Swapchain to acquire from.
 * @param slot CPU frame slot index.
 * @param acquire_wait_stage Pipeline stage for the acquire wait.
 * @return `ready` with frame data, `needs_recreate`, or an error.
 */
[[nodiscard]] auto acquire_present_frame(frame_ring &ring,
  swapchain &chain,
  std::size_t slot,
  VkPipelineStageFlags acquire_wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
  -> result<present_acquire_result>;

/**
 * Submits recorded work with frame-ring sync, marks timeline completion, and presents.
 *
 * @param ctx Context used for queue submit.
 * @param ring Frame ring to mark submitted.
 * @param chain Swapchain to present.
 * @param frame Frame from a successful `acquire_present_frame`.
 * @param command_buffers Recorded command buffers to submit.
 * @return `false` when the swapchain must be recreated; `true` on success.
 */
[[nodiscard]] auto submit_and_present(context &ctx,
  frame_ring &ring,
  swapchain &chain,
  acquired_present_frame const &frame,
  std::span<VkCommandBuffer const> command_buffers) -> result<bool>;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_FRAME_PRESENT_HPP
