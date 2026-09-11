#include <vkexec_graphics/frame_present.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec_extensions/timeline_semaphore/frame_ring.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <span>

namespace vkexec {

auto acquire_present_frame(frame_ring &ring,
  swapchain &chain,
  std::size_t slot,
  VkPipelineStageFlags acquire_wait_stage) -> result<present_acquire_result>
{
  VKEXEC_TRY(ring.wait_slot(slot));

  auto acquire_sem = ring.acquire_semaphore(slot);
  if (!acquire_sem) { return fail(acquire_sem); }

  auto image_index = chain.acquire_next_image(*acquire_sem);
  if (!image_index) { return fail(image_index); }
  if (!image_index->has_value()) { return present_acquire_result{ .status = present_acquire_status::needs_recreate }; }

  auto const timeline_value = ring.allocate_signal_value();
  auto submit_sync = ring.make_submit_sync(slot, **image_index, timeline_value, acquire_wait_stage);
  if (!submit_sync) { return fail(submit_sync); }

  return present_acquire_result{
    .status = present_acquire_status::ready,
    .frame =
      acquired_present_frame{
        .slot = slot,
        .image_index = **image_index,
        .timeline_value = timeline_value,
        .submit_sync = *submit_sync,
      },
  };
}

auto submit_and_present(context &ctx,
  frame_ring &ring,
  swapchain &chain,
  acquired_present_frame const &frame,
  std::span<VkCommandBuffer const> command_buffers) -> result<bool>
{
  VkQueue submit_queue = ctx.graphics_queue() != VK_NULL_HANDLE ? ctx.graphics_queue() : ctx.present_queue();
  if (submit_queue == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "submit_and_present requires a graphics or present queue");
  }

  VKEXEC_TRY(ctx.submit(queue_submit{
    .command_buffers = command_buffers,
    .waits = frame.submit_sync.waits,
    .signals = frame.submit_sync.signals,
    .queue = submit_queue,
  }));
  VKEXEC_TRY(ring.mark_submitted(frame.slot, frame.image_index, frame.timeline_value));

  auto const finished = ring.render_finished_semaphore(frame.image_index);
  if (!finished) { return fail(finished); }
  std::array<VkSemaphore, 1> const present_waits{ *finished };
  return chain.present(frame.image_index, present_waits);
}

}// namespace vkexec
