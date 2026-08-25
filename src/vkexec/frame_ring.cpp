#include <vkexec/frame_ring.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  [[noreturn]] auto fail(char const *what) -> void { VKEXEC_THROW(std::runtime_error(what)); }

  auto create_binary_semaphore(VkDevice device) -> VkSemaphore
  {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore semaphore{ VK_NULL_HANDLE };
    if (vkCreateSemaphore(device, &info, nullptr, &semaphore) != VK_SUCCESS) {
      fail("vkCreateSemaphore (binary) failed");
    }
    return semaphore;
  }

}// namespace

auto frame_ring::create(context &ctx, create_info info) -> frame_ring { return frame_ring{ &ctx, info }; }

frame_ring::frame_ring(context *ctx, create_info info)
  : ctx_(ctx), timeline_(timeline_semaphore::create(*ctx, 0))
{
  if (ctx_ == nullptr || ctx_->device() == VK_NULL_HANDLE) { fail("frame_ring requires a VkDevice"); }
  if (info.slot_count == 0) { fail("frame_ring requires slot_count > 0"); }

  acquire_.resize(info.slot_count, VK_NULL_HANDLE);
  slot_timeline_value_.assign(info.slot_count, 0);
  for (VkSemaphore &semaphore : acquire_) { semaphore = create_binary_semaphore(ctx_->device()); }

  create_image_semaphores(info.image_count);
}

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

auto frame_ring::resize_images(std::size_t image_count) -> void
{
  destroy_image_semaphores();
  create_image_semaphores(image_count);
}

auto frame_ring::reset_completion_tracking() -> void
{
  std::fill(slot_timeline_value_.begin(), slot_timeline_value_.end(), 0);
  std::fill(image_timeline_value_.begin(), image_timeline_value_.end(), 0);
  next_timeline_value_ = 0;
}

auto frame_ring::acquire_semaphore(std::size_t slot) const -> VkSemaphore
{
  check_slot(slot);
  return acquire_.at(slot);
}

auto frame_ring::render_finished_semaphore(std::size_t image_index) const -> VkSemaphore
{
  check_image(image_index);
  return render_finished_.at(image_index);
}

auto frame_ring::wait_slot(std::size_t slot) const -> void
{
  check_slot(slot);
  timeline_.wait(slot_timeline_value_.at(slot));
}

auto frame_ring::wait_image(std::size_t image_index) const -> void
{
  check_image(image_index);
  timeline_.wait(image_timeline_value_.at(image_index));
}

auto frame_ring::allocate_signal_value() -> std::uint64_t
{
  ++next_timeline_value_;
  return next_timeline_value_;
}

auto frame_ring::mark_submitted(std::size_t slot, std::size_t image_index, std::uint64_t signal_value) -> void
{
  check_slot(slot);
  check_image(image_index);
  slot_timeline_value_.at(slot) = signal_value;
  image_timeline_value_.at(image_index) = signal_value;
}

auto frame_ring::make_submit_sync(std::size_t slot,
  std::size_t image_index,
  std::uint64_t signal_value,
  VkPipelineStageFlags acquire_wait_stage) const -> frame_ring_submit_sync
{
  check_slot(slot);
  check_image(image_index);
  if (signal_value == 0) { fail("frame_ring::make_submit_sync requires signal_value > 0"); }

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

auto frame_ring::create_image_semaphores(std::size_t image_count) -> void
{
  render_finished_.assign(image_count, VK_NULL_HANDLE);
  image_timeline_value_.assign(image_count, 0);
  for (VkSemaphore &semaphore : render_finished_) { semaphore = create_binary_semaphore(ctx_->device()); }
}

auto frame_ring::check_slot(std::size_t slot) const -> void
{
  if (slot >= acquire_.size()) { fail("frame_ring slot index out of range"); }
}

auto frame_ring::check_image(std::size_t image_index) const -> void
{
  if (image_index >= render_finished_.size()) { fail("frame_ring image index out of range"); }
}

}// namespace vkexec
