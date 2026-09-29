#include <vkexec/detail/submission.hpp>

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

namespace vkexec::detail {

auto legacy_wait_stage(VkPipelineStageFlags2 stage) -> result<VkPipelineStageFlags>
{
  if (stage == VK_PIPELINE_STAGE_2_NONE) {
    return fail(errc::unsupported, "NONE wait stage cannot be represented by legacy submission");
  }
  if ((stage & VK_PIPELINE_STAGE_2_HOST_BIT) != 0) {
    return fail(errc::unsupported, "HOST wait stage cannot be represented by legacy submission");
  }
  return legacy_stage_mask(stage);
}

auto legacy_signal_stage(VkPipelineStageFlags2 stage) -> status
{
  if (stage != VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) {
    return fail(errc::unsupported, "signal stage cannot be represented by legacy submission");
  }
  return {};
}

auto lower_legacy_submit(queue_submit const &info, bool legacy_timeline_submit_info_available)
  -> result<legacy_submission>
{
  legacy_submission lowered{};
  lowered.legacy_timeline_submit_info_available = legacy_timeline_submit_info_available;
  lowered.wait_semaphores.reserve(info.waits.size());
  lowered.wait_stages.reserve(info.waits.size());
  lowered.wait_values.reserve(info.waits.size());
  lowered.signal_semaphores.reserve(info.signals.size());
  lowered.signal_values.reserve(info.signals.size());
  for (semaphore_submit const &entry : info.waits) {
    VKEXEC_TRY_ASSIGN(stage, legacy_wait_stage(entry.stage));
    lowered.wait_semaphores.push_back(entry.semaphore);
    lowered.wait_stages.push_back(stage);
    lowered.wait_values.push_back(entry.value);
  }
  for (semaphore_submit const &entry : info.signals) {
    VKEXEC_TRY(legacy_signal_stage(entry.stage));
    lowered.signal_semaphores.push_back(entry.semaphore);
    lowered.signal_values.push_back(entry.value);
  }
  return lowered;
}

auto legacy_submission::make_submit_info(queue_submit const &info) const noexcept -> legacy_submission_view
{ return legacy_submission_view{ *this, info }; }

legacy_submission_view::legacy_submission_view(legacy_submission const &lowered, queue_submit const &info) noexcept
  : timeline{ .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreValueCount = static_cast<std::uint32_t>(lowered.wait_values.size()),
      .pWaitSemaphoreValues = lowered.wait_values.empty() ? nullptr : lowered.wait_values.data(),
      .signalSemaphoreValueCount = static_cast<std::uint32_t>(lowered.signal_values.size()),
      .pSignalSemaphoreValues = lowered.signal_values.empty() ? nullptr : lowered.signal_values.data() },
    submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = lowered.legacy_timeline_submit_info_available ? &timeline : nullptr,
      .waitSemaphoreCount = static_cast<std::uint32_t>(lowered.wait_semaphores.size()),
      .pWaitSemaphores = lowered.wait_semaphores.empty() ? nullptr : lowered.wait_semaphores.data(),
      .pWaitDstStageMask = lowered.wait_stages.empty() ? nullptr : lowered.wait_stages.data(),
      .commandBufferCount = static_cast<std::uint32_t>(info.command_buffers.size()),
      .pCommandBuffers = info.command_buffers.data(),
      .signalSemaphoreCount = static_cast<std::uint32_t>(lowered.signal_semaphores.size()),
      .pSignalSemaphores = lowered.signal_semaphores.empty() ? nullptr : lowered.signal_semaphores.data() }
{}

auto lower_synchronization2_submit(queue_submit const &info) -> synchronization2_submission
{
  synchronization2_submission lowered{};
  lowered.commands.reserve(info.command_buffers.size());
  lowered.waits.reserve(info.waits.size());
  lowered.signals.reserve(info.signals.size());
  for (VkCommandBuffer cmd : info.command_buffers) {
    lowered.commands.push_back(VkCommandBufferSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .pNext = nullptr, .commandBuffer = cmd, .deviceMask = 1 });
  }
  for (semaphore_submit const &entry : info.waits) {
    lowered.waits.push_back(VkSemaphoreSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .semaphore = entry.semaphore,
      .value = entry.value,
      .stageMask = entry.stage,
      .deviceIndex = 0 });
  }
  for (semaphore_submit const &entry : info.signals) {
    lowered.signals.push_back(VkSemaphoreSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .semaphore = entry.semaphore,
      .value = entry.value,
      .stageMask = entry.stage,
      .deviceIndex = 0 });
  }
  return lowered;
}

auto synchronization2_submission::submit_info() const noexcept -> VkSubmitInfo2
{
  return VkSubmitInfo2{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
    .pNext = nullptr,
    .flags = 0,
    .waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size()),
    .pWaitSemaphoreInfos = waits.empty() ? nullptr : waits.data(),
    .commandBufferInfoCount = static_cast<std::uint32_t>(commands.size()),
    .pCommandBufferInfos = commands.empty() ? nullptr : commands.data(),
    .signalSemaphoreInfoCount = static_cast<std::uint32_t>(signals.size()),
    .pSignalSemaphoreInfos = signals.empty() ? nullptr : signals.data() };
}

auto submit(queue_submit const &info,
  VkQueue queue,
  synchronization_backend backend,
  bool legacy_timeline_submit_info_available,
  device_procs const &procs,
  std::mutex &host_mutex) -> status
{
  if (info.command_buffers.empty()) {
    return fail(errc::invalid_argument, "queue_submit requires at least one command buffer");
  }
  if (queue == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "queue_submit requires a VkQueue"); }
  if (std::ranges::any_of(info.waits, [](semaphore_submit const &wait) { return wait.semaphore == VK_NULL_HANDLE; })) {
    return fail(errc::invalid_argument, "queue_submit wait semaphore is null");
  }
  if (std::ranges::any_of(
        info.signals, [](semaphore_submit const &signal) { return signal.semaphore == VK_NULL_HANDLE; })) {
    return fail(errc::invalid_argument, "queue_submit signal semaphore is null");
  }

  if (backend != synchronization_backend::legacy) {
    synchronization2_submission const lowered = lower_synchronization2_submit(info);
    VkSubmitInfo2 const submission = lowered.submit_info();
    std::scoped_lock const lock(host_mutex);
    if (VkResult const result = procs.queue_submit2(queue, 1, &submission, info.fence); result != VK_SUCCESS) {
      return fail(result, "vkQueueSubmit2 failed");
    }
    return {};
  }

  VKEXEC_TRY_ASSIGN(lowered, lower_legacy_submit(info, legacy_timeline_submit_info_available));
  legacy_submission_view const view = lowered.make_submit_info(info);
  std::scoped_lock const lock(host_mutex);
  if (VkResult const result = vkQueueSubmit(queue, 1, &view.submit, info.fence); result != VK_SUCCESS) {
    return fail(result, "vkQueueSubmit failed");
  }
  return {};
}

}// namespace vkexec::detail
