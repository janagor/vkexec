#include <vkexec/frame_ring.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  auto create_binary_semaphore(VkDevice device) -> result<VkSemaphore>
  {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore semaphore{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreateSemaphore(device, &info, nullptr, &semaphore);
    if (create_result != VK_SUCCESS) {
      return make_vk_error(create_result, "vkCreateSemaphore (binary) failed");
    }
    return semaphore;
  }

}// namespace

auto frame_ring::create(context &ctx, create_info info) -> result<frame_ring>
{
  if (ctx.device() == VK_NULL_HANDLE) {
    return make_error(errc::invalid_argument, "frame_ring requires a VkDevice");
  }
  if (info.slot_count == 0) {
    return make_error(errc::invalid_argument, "frame_ring requires slot_count > 0");
  }

  return timeline_semaphore::create(ctx, 0).and_then([&](timeline_semaphore timeline) -> result<frame_ring> {
    frame_ring ring{ &ctx, std::move(timeline) };
    ring.acquire_.resize(info.slot_count, VK_NULL_HANDLE);
    ring.slot_timeline_value_.assign(info.slot_count, 0);

    for (std::size_t slot = 0; slot < info.slot_count; ++slot) {
      auto semaphore_result = create_binary_semaphore(ctx.device());
      if (!semaphore_result) {
        for (std::size_t index = 0; index < slot; ++index) {
          if (ring.acquire_.at(index) != VK_NULL_HANDLE) {
            vkDestroySemaphore(ctx.device(), ring.acquire_.at(index), nullptr);
            ring.acquire_.at(index) = VK_NULL_HANDLE;
          }
        }
        return std::unexpected(std::move(semaphore_result).error());
      }
      ring.acquire_.at(slot) = *semaphore_result;
    }

    return ring.create_image_semaphores(info.image_count).transform([&] { return std::move(ring); });
  });
}

frame_ring::frame_ring(context *ctx, timeline_semaphore timeline) noexcept
  : ctx_(ctx), timeline_(std::move(timeline))
{}

frame_ring::~frame_ring() { destroy(); }

frame_ring::frame_ring(frame_ring &&other) noexcept
  : ctx_(other.ctx_), timeline_(std::move(other.timeline_)), next_timeline_value_(other.next_timeline_value_),
    acquire_(std::move(other.acquire_)), render_finished_(std::move(other.render_finished_)),
    slot_timeline_value_(std::move(other.slot_timeline_value_)),
    image_timeline_value_(std::move(other.image_timeline_value_))
{
  other.ctx_ = nullptr;
  other.next_timeline_value_ = 0;
}

auto frame_ring::operator=(frame_ring &&other) noexcept -> frame_ring &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  timeline_ = std::move(other.timeline_);
  next_timeline_value_ = other.next_timeline_value_;
  acquire_ = std::move(other.acquire_);
  render_finished_ = std::move(other.render_finished_);
  slot_timeline_value_ = std::move(other.slot_timeline_value_);
  image_timeline_value_ = std::move(other.image_timeline_value_);
  other.ctx_ = nullptr;
  other.next_timeline_value_ = 0;
  return *this;
}

auto frame_ring::resize_images(std::size_t image_count) -> status
{
  destroy_image_semaphores();
  return create_image_semaphores(image_count);
}

auto frame_ring::reset_completion_tracking() -> void
{
  std::fill(slot_timeline_value_.begin(), slot_timeline_value_.end(), 0);
  std::fill(image_timeline_value_.begin(), image_timeline_value_.end(), 0);
  next_timeline_value_ = 0;
}

auto frame_ring::acquire_semaphore(std::size_t slot) const -> result<VkSemaphore>
{
  return check_slot(slot).transform([&] { return acquire_.at(slot); });
}

auto frame_ring::render_finished_semaphore(std::size_t image_index) const -> result<VkSemaphore>
{
  return check_image(image_index).transform([&] { return render_finished_.at(image_index); });
}

auto frame_ring::wait_slot(std::size_t slot) const -> status
{
  return check_slot(slot).and_then([&] { return timeline_.wait(slot_timeline_value_.at(slot)); });
}

auto frame_ring::wait_image(std::size_t image_index) const -> status
{
  return check_image(image_index).and_then([&] { return timeline_.wait(image_timeline_value_.at(image_index)); });
}

auto frame_ring::allocate_signal_value() -> std::uint64_t
{
  ++next_timeline_value_;
  return next_timeline_value_;
}

auto frame_ring::mark_submitted(std::size_t slot, std::size_t image_index, std::uint64_t signal_value) -> status
{
  return check_slot(slot)
    .and_then([&] { return check_image(image_index); })
    .and_then([&]() -> status {
      slot_timeline_value_.at(slot) = signal_value;
      image_timeline_value_.at(image_index) = signal_value;
      return {};
    });
}

auto frame_ring::make_submit_sync(std::size_t slot,
  std::size_t image_index,
  std::uint64_t signal_value,
  VkPipelineStageFlags acquire_wait_stage) const -> result<frame_ring_submit_sync>
{
  return check_slot(slot)
    .and_then([&] { return check_image(image_index); })
    .and_then([&]() -> result<frame_ring_submit_sync> {
      if (signal_value == 0) {
        return make_error(errc::invalid_argument, "frame_ring::make_submit_sync requires signal_value > 0");
      }

      frame_ring_submit_sync sync{};
      sync.waits[0] = semaphore_submit{
        .semaphore = acquire_.at(slot),
        .value = 0,
        .stage = acquire_wait_stage,
      };
      sync.signals[0] = semaphore_submit{
        .semaphore = render_finished_.at(image_index),
        .value = 0,
        .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      };
      sync.signals[1] = semaphore_submit{
        .semaphore = timeline_.handle(),
        .value = signal_value,
        .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      };
      return sync;
    });
}

auto frame_ring::destroy() noexcept -> void
{
  destroy_image_semaphores();
  if (ctx_ != nullptr && ctx_->device() != VK_NULL_HANDLE) {
    for (VkSemaphore semaphore : acquire_) {
      if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), semaphore, nullptr); }
    }
  }
  acquire_.clear();
  slot_timeline_value_.clear();
  next_timeline_value_ = 0;
  ctx_ = nullptr;
}

auto frame_ring::destroy_image_semaphores() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->device() != VK_NULL_HANDLE) {
    for (VkSemaphore semaphore : render_finished_) {
      if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), semaphore, nullptr); }
    }
  }
  render_finished_.clear();
  image_timeline_value_.clear();
}

auto frame_ring::create_image_semaphores(std::size_t image_count) -> status
{
  render_finished_.assign(image_count, VK_NULL_HANDLE);
  image_timeline_value_.assign(image_count, 0);
  for (std::size_t index = 0; index < image_count; ++index) {
    auto semaphore_result = create_binary_semaphore(ctx_->device());
    if (!semaphore_result) {
      destroy_image_semaphores();
      return std::unexpected(semaphore_result.error());
    }
    render_finished_.at(index) = *semaphore_result;
  }
  return {};
}

auto frame_ring::check_slot(std::size_t slot) const -> status
{
  if (slot >= acquire_.size()) { return make_error(errc::out_of_range, "frame_ring slot index out of range"); }
  return {};
}

auto frame_ring::check_image(std::size_t image_index) const -> status
{
  if (image_index >= render_finished_.size()) {
    return make_error(errc::out_of_range, "frame_ring image index out of range");
  }
  return {};
}

}// namespace vkexec
