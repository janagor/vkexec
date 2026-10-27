#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/presentation_abort_state.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/swapchain.hpp>
#include <vulkan/vulkan_core.h>

namespace vkexec {

namespace {
  constexpr std::uint32_t k_framebuffer_attachment_count = 2;
}// namespace

auto owned::presenter::resize(std::uint32_t width, std::uint32_t height) -> status
{
  if (frame_open_) { return fail(errc::invalid_argument, "resize called while a frame is open"); }
  if (width == 0 || height == 0) {
    suspended_ = true;
    resize_required_ = true;
    return {};
  }
  if (!resize_required_ && !suspended_ && width == cfg_.width && height == cfg_.height) { return {}; }
  return recreate_swapchain(width, height);
}

auto owned::presenter::recreate_swapchain(std::uint32_t width, std::uint32_t height) -> status
{
  cfg_.width = width;
  cfg_.height = height;

  // Idle before destroying framebuffers/views that may still be referenced by in-flight frames.
  if (VkResult const idle = vkDeviceWaitIdle(ctx_->device()); idle != VK_SUCCESS) {
    return fail(idle, "vkDeviceWaitIdle failed before swapchain recreation");
  }

  for (VkFramebuffer framebuffer : framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), framebuffer, nullptr); }
  }
  framebuffers_.clear();
  destroy_depth_resources();

  if (auto swapchain = create_swapchain(); !swapchain) { return fail(swapchain); }
  if (auto depth = create_depth_resources(); !depth) { return fail(depth); }
  if (auto framebuffers = create_framebuffers(); !framebuffers) { return fail(framebuffers); }
  suspended_ = false;
  resize_required_ = false;
  return {};
}

auto owned::presenter::acquire_frame() -> result<std::optional<acquired_frame>>
{
  if (frame_open_) { return fail(errc::invalid_argument, "acquire_frame called while a frame is already open"); }
  if (suspended_ || resize_required_) { return std::optional<acquired_frame>{}; }
  if (!swapchain_) { return fail(errc::invalid_argument, "acquire_frame requires a swapchain"); }
  swapchain &active_swapchain = *swapchain_;

  auto &sync = frames_.at(frame_index_);
  if (VkResult const wait_result = vkWaitForFences(ctx_->device(), 1, &sync.in_flight, VK_TRUE, UINT64_MAX);
    wait_result != VK_SUCCESS) {
    return fail(wait_result, "vkWaitForFences failed");
  }

  auto acquired = active_swapchain.acquire_next_image(sync.image_available);
  if (!acquired) { return fail(acquired); }
  if (!acquired->has_value()) {
    resize_required_ = true;
    return std::optional<acquired_frame>{};
  }
  std::uint32_t const image_index = **acquired;

  // Wait if a previous frame is still using this swapchain image.
  if (images_in_flight_.at(image_index) != VK_NULL_HANDLE) {
    if (VkResult const wait_result =
          vkWaitForFences(ctx_->device(), 1, &images_in_flight_.at(image_index), VK_TRUE, UINT64_MAX);
      wait_result != VK_SUCCESS) {
      return fail(wait_result, "vkWaitForFences failed (swapchain image in flight)");
    }
  }
  images_in_flight_.at(image_index) = sync.in_flight;

  current_image_index_ = image_index;
  frame_open_ = true;
  recording_open_ = false;
  acquired_wait_consumed_ = false;
  return acquired_frame{ .image = active_swapchain.image_at(image_index),
    .image_view = active_swapchain.image_view_at(image_index),
    .framebuffer = framebuffers_.at(image_index),
    .extent = extent(),
    .image_index = image_index };
}

auto owned::presenter::begin_frame() -> result<std::optional<frame>>
{
  VKEXEC_TRY_ASSIGN(acquired, acquire_frame());
  if (!acquired) { return std::optional<frame>{}; }
  VkFence fence = frames_.at(frame_index_).in_flight;
  if (VkResult const reset_result = vkResetFences(ctx_->device(), 1, &fence); reset_result != VK_SUCCESS) {
    if (auto abandoned = abandon_frame(*acquired); !abandoned) { return fail(abandoned); }
    return fail(reset_result, "vkResetFences failed");
  }
  VkCommandBuffer cmd = command_buffers_.at(frame_index_);
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (VkResult const begin_result = vkBeginCommandBuffer(cmd, &begin_info); begin_result != VK_SUCCESS) {
    if (auto abandoned = abandon_frame(*acquired); !abandoned) { return fail(abandoned); }
    return fail(begin_result, "vkBeginCommandBuffer failed");
  }
  recording_open_ = true;
  return frame{ .command_buffer = cmd,
    .framebuffer = acquired->framebuffer,
    .extent = acquired->extent,
    .image_index = acquired->image_index };
}

auto owned::presenter::end_frame(frame const &drawn, present_options options) -> result<VkFence>
{ return end_frame(drawn, frame_submit_options{}, options); }

auto owned::presenter::end_frame(frame const &drawn, frame_submit_options submit_options, present_options options)
  -> result<VkFence>
{
  if (recording_open_) { VKEXEC_TRY(finish_frame_recording(drawn)); }
  VKEXEC_TRY_ASSIGN(sync, submission_sync(drawn));
  std::array<VkCommandBuffer, 1> const commands{ drawn.command_buffer };
  std::vector<semaphore_submit> waits;
  waits.reserve(1 + submit_options.waits.size());
  waits.push_back(sync.image_available_wait);
  waits.insert(waits.end(), submit_options.waits.begin(), submit_options.waits.end());
  std::array<semaphore_submit, 1> const signals{ sync.render_finished_signal };
  VKEXEC_TRY(ctx_->submit(queue_submit{ .command_buffers = commands,
    .waits = waits,
    .signals = signals,
    .fence = sync.fence,
    .queue = ctx_->graphics_queue() }));

  return present_submitted(drawn, options);
}

auto owned::presenter::finish_frame_recording(frame const &drawn) -> status
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (!recording_open_) { return fail(errc::invalid_argument, "frame command buffer is not recording"); }
  if (VkResult const result = vkEndCommandBuffer(drawn.command_buffer); result != VK_SUCCESS) {
    return fail(result, "vkEndCommandBuffer failed");
  }
  recording_open_ = false;
  return {};
}

auto owned::presenter::validate_current_frame(frame const &drawn) const -> status
{
  if (drawn.command_buffer != command_buffers_.at(frame_index_)) {
    return fail(errc::invalid_argument, "operation requires the current open frame");
  }
  return validate_current_frame(
    acquired_frame{ .framebuffer = drawn.framebuffer, .extent = drawn.extent, .image_index = drawn.image_index });
}

auto owned::presenter::validate_current_frame(acquired_frame const &drawn) const -> status
{
  if (!frame_open_ || !swapchain_ || drawn.image_index != current_image_index_
      || drawn.framebuffer != framebuffers_.at(current_image_index_) || drawn.extent.width != extent().width
      || drawn.extent.height != extent().height
      || (drawn.image != VK_NULL_HANDLE && drawn.image != swapchain_->image_at(current_image_index_))
      || (drawn.image_view != VK_NULL_HANDLE && drawn.image_view != swapchain_->image_view_at(current_image_index_))) {
    return fail(errc::invalid_argument, "operation requires the current open frame");
  }
  return {};
}

auto owned::presenter::submission_sync(frame const &drawn) const -> result<frame_submit_sync>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  return submission_sync(
    acquired_frame{ .framebuffer = drawn.framebuffer, .extent = drawn.extent, .image_index = drawn.image_index });
}

auto owned::presenter::submission_sync(acquired_frame const &drawn) const -> result<frame_submit_sync>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  frame_sync const &sync = frames_.at(frame_index_);
  return frame_submit_sync{
    .image_available_wait =
      semaphore_submit{ .semaphore = sync.image_available, .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT },
    .render_finished_signal = semaphore_submit{ .semaphore = render_finished_.at(current_image_index_) },
    .fence = sync.in_flight,
  };
}

auto owned::presenter::present_submitted(frame const &drawn, present_options options) -> result<VkFence>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (recording_open_) { return fail(errc::invalid_argument, "frame command buffer is still recording"); }
  return present_submitted(
    acquired_frame{ .framebuffer = drawn.framebuffer, .extent = drawn.extent, .image_index = drawn.image_index },
    options);
}

auto owned::presenter::present_submitted(acquired_frame const &drawn, present_options options) -> result<VkFence>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (recording_open_) { return fail(errc::invalid_argument, "frame command buffer is still recording"); }
  VKEXEC_TRY_ASSIGN(sync, submission_sync(drawn));
  if (!swapchain_) { return fail(errc::invalid_argument, "present_submitted requires a swapchain"); }
  swapchain &active_swapchain = *swapchain_;

  std::array<VkSemaphore, 1> const wait_semaphores{ sync.render_finished_signal.semaphore };
  auto present_result = active_swapchain.present(current_image_index_, wait_semaphores, options);
  if (!present_result) { return fail(present_result); }
  if (!*present_result) { resize_required_ = true; }

  // Caller may enqueue_borrowed_fence_wait on this fence; presenter retains ownership.
  VkFence submitted = sync.fence;
  frame_index_ = (frame_index_ + 1) % static_cast<std::uint32_t>(k_frames);
  frame_open_ = false;
  recording_open_ = false;
  acquired_wait_consumed_ = false;
  return submitted;
}

auto owned::presenter::abandon_frame(acquired_frame const &drawn, presentation_abort_state state) -> status
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (recording_open_) { return fail(errc::invalid_argument, "cannot abandon a recording frame"); }
  if (VkResult const idle = vkDeviceWaitIdle(ctx_->device()); idle != VK_SUCCESS) {
    return fail(idle, "vkDeviceWaitIdle failed while abandoning frame");
  }
  if (state == presentation_abort_state::acquired_only && !acquired_wait_consumed_) {
    VKEXEC_TRY(consume_acquired_image_wait());
    acquired_wait_consumed_ = true;
  }
  if (state == presentation_abort_state::final_submit_completed) {
    VKEXEC_TRY(replace_abandoned_frame_sync());
    frame_open_ = false;
    acquired_wait_consumed_ = false;
    cleanup_swapchain();
    resize_required_ = true;
    return {};
  }
  if (!swapchain_.has_value()) { return fail(errc::invalid_argument, "abandon_frame requires a swapchain"); }
  auto presented = swapchain_.value().present(current_image_index_, {});
  if (VkResult const idle = vkDeviceWaitIdle(ctx_->device()); idle != VK_SUCCESS) {
    return fail(idle, "vkDeviceWaitIdle failed after abandoned frame presentation");
  }
  VKEXEC_TRY(replace_abandoned_frame_sync());
  frame_open_ = false;
  acquired_wait_consumed_ = false;
  if (!presented || !*presented) {
    cleanup_swapchain();
    resize_required_ = true;
    if (!presented) { return fail(presented); }
  }
  return {};
}

auto owned::presenter::consume_acquired_image_wait() -> status
{
  VkCommandBuffer cmd = command_buffers_.at(frame_index_);
  if (VkResult const reset = vkResetCommandBuffer(cmd, 0); reset != VK_SUCCESS) {
    return fail(reset, "vkResetCommandBuffer failed while abandoning frame");
  }
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (VkResult const begun = vkBeginCommandBuffer(cmd, &begin_info); begun != VK_SUCCESS) {
    return fail(begun, "vkBeginCommandBuffer failed while abandoning frame");
  }
  std::array<VkClearValue, k_framebuffer_attachment_count> const clears{
    VkClearValue{ .color = VkClearColorValue{ .float32 = { 0.0F, 0.0F, 0.0F, 1.0F } } },
    VkClearValue{ .depthStencil = VkClearDepthStencilValue{ .depth = 1.0F, .stencil = 0 } },
  };
  VkRenderPassBeginInfo render_begin{};
  render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_begin.renderPass = render_pass_;
  render_begin.framebuffer = framebuffers_.at(current_image_index_);
  render_begin.renderArea.extent = extent();
  render_begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
  render_begin.pClearValues = clears.data();
  vkCmdBeginRenderPass(cmd, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdEndRenderPass(cmd);
  if (VkResult const ended = vkEndCommandBuffer(cmd); ended != VK_SUCCESS) {
    return fail(ended, "vkEndCommandBuffer failed while abandoning frame");
  }
  VkFenceCreateInfo consume_info{};
  consume_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence consumed{ VK_NULL_HANDLE };
  if (VkResult const created = vkCreateFence(ctx_->device(), &consume_info, nullptr, &consumed);
    created != VK_SUCCESS) {
    return fail(created, "vkCreateFence failed while abandoning frame");
  }
  std::array<VkCommandBuffer, 1> const commands{ cmd };
  std::array<semaphore_submit, 1> const waits{ semaphore_submit{
    .semaphore = frames_.at(frame_index_).image_available } };
  auto submitted = ctx_->submit(
    queue_submit{ .command_buffers = commands, .waits = waits, .fence = consumed, .queue = ctx_->graphics_queue() });
  if (!submitted) {
    vkDestroyFence(ctx_->device(), consumed, nullptr);
    return fail(submitted);
  }
  VkResult const waited = vkWaitForFences(ctx_->device(), 1, &consumed, VK_TRUE, UINT64_MAX);
  vkDestroyFence(ctx_->device(), consumed, nullptr);
  if (waited != VK_SUCCESS) { return fail(waited, "vkWaitForFences failed while abandoning frame"); }
  return {};
}

auto owned::presenter::replace_abandoned_frame_sync() -> status
{
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkSemaphore image_available{ VK_NULL_HANDLE };
  VkSemaphore render_finished{ VK_NULL_HANDLE };
  VkFence in_flight{ VK_NULL_HANDLE };
  auto cleanup = [&]() -> void {
    if (image_available != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), image_available, nullptr); }
    if (render_finished != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), render_finished, nullptr); }
    if (in_flight != VK_NULL_HANDLE) { vkDestroyFence(ctx_->device(), in_flight, nullptr); }
  };
  if (VkResult const created = vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &image_available);
    created != VK_SUCCESS) {
    return fail(created, "vkCreateSemaphore failed while abandoning frame");
  }
  if (VkResult const created = vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &render_finished);
    created != VK_SUCCESS) {
    cleanup();
    return fail(created, "vkCreateSemaphore failed while abandoning frame");
  }
  if (VkResult const created = vkCreateFence(ctx_->device(), &fence_info, nullptr, &in_flight); created != VK_SUCCESS) {
    cleanup();
    return fail(created, "vkCreateFence failed while abandoning frame");
  }

  auto &sync = frames_.at(frame_index_);
  std::ranges::replace(images_in_flight_, sync.in_flight, VkFence{ VK_NULL_HANDLE });
  vkDestroySemaphore(ctx_->device(), sync.image_available, nullptr);
  vkDestroySemaphore(ctx_->device(), render_finished_.at(current_image_index_), nullptr);
  vkDestroyFence(ctx_->device(), sync.in_flight, nullptr);
  sync.image_available = image_available;
  sync.in_flight = in_flight;
  render_finished_.at(current_image_index_) = render_finished;
  return {};
}

}// namespace vkexec
