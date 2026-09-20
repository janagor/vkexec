#include <vkexec_extensions/timeline_semaphore/frame_ring.hpp>

#include <vkexec_features/feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/timeline_semaphore/timeline_semaphore.hpp>

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
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateSemaphore (binary) failed"); }
    return semaphore;
  }

}// namespace

auto frame_ring::create(context &ctx, create_info info) -> sender<frame_ring>
{
  return make_sender<frame_ring>([&ctx, info]() -> result<frame_ring> {
    if (ctx.device() == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "frame_ring requires a VkDevice"); }
    if (info.slot_count == 0) { return fail(errc::invalid_argument, "frame_ring requires slot_count > 0"); }
    if (!feat::available<feat::timeline_semaphore>(ctx)) {
      return fail(errc::unsupported, "frame_ring requires feat::timeline_semaphore");
    }

    VKEXEC_TRY_ASSIGN(timeline_sem, detail::make_timeline_semaphore(ctx, 0));
    frame_ring ring{ &ctx, std::move(timeline_sem) };
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
        return fail(semaphore_result);
      }
      ring.acquire_.at(slot) = expected_take(semaphore_result);
    }

    VKEXEC_TRY(ring.create_image_semaphores(info.image_count));
    return ring;
  });
}

frame_ring::frame_ring(context *ctx, timeline_semaphore timeline_sem) noexcept
  : ctx_(ctx), timeline_(std::move(timeline_sem))
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
  // Slot acquire semaphores and the timeline counter stay; only per-image present sync is rebuilt.
  destroy_image_semaphores();
  return create_image_semaphores(image_count);
}

auto frame_ring::reset_completion_tracking() -> void
{
  // Device idle makes the old gates obsolete, but the live Vulkan timeline and
  // its monotonically increasing signal sequence cannot be reset.
  std::ranges::fill(slot_timeline_value_, 0);
  std::ranges::fill(image_timeline_value_, 0);
}

auto frame_ring::acquire_semaphore(std::size_t slot) const -> result<VkSemaphore>
{
  VKEXEC_TRY(check_slot(slot));
  return acquire_.at(slot);
}

auto frame_ring::render_finished_semaphore(std::size_t image_index) const -> result<VkSemaphore>
{
  VKEXEC_TRY(check_image(image_index));
  return render_finished_.at(image_index);
}

auto frame_ring::wait_slot(std::size_t slot) const -> status
{
  VKEXEC_TRY(check_slot(slot));
  return timeline_.wait(slot_timeline_value_.at(slot));
}

auto frame_ring::wait_image(std::size_t image_index) const -> status
{
  VKEXEC_TRY(check_image(image_index));
  return timeline_.wait(image_timeline_value_.at(image_index));
}

auto frame_ring::allocate_signal_value() -> std::uint64_t
{
  // Monotonic counter; never reuse a value while GPU work may still be in flight.
  ++next_timeline_value_;
  return next_timeline_value_;
}

auto frame_ring::mark_submitted(std::size_t slot, std::size_t image_index, std::uint64_t signal_value) -> status
{
  // Subsequent wait_slot / wait_image block until this timeline value is reached.
  VKEXEC_TRY(check_slot(slot));
  VKEXEC_TRY(check_image(image_index));
  slot_timeline_value_.at(slot) = signal_value;
  image_timeline_value_.at(image_index) = signal_value;
  return {};
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto frame_ring::make_submit_sync(std::size_t slot,
  std::size_t image_index,
  std::uint64_t signal_value,
  VkPipelineStageFlags acquire_wait_stage) const -> result<frame_ring_submit_sync>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  VKEXEC_TRY(check_slot(slot));
  VKEXEC_TRY(check_image(image_index));
  if (signal_value == 0) {
    return fail(errc::invalid_argument, "frame_ring::make_submit_sync requires signal_value > 0");
  }

  frame_ring_submit_sync sync{};
  // Binary acquire wait + binary present signal + timeline completion signal.
  sync.waits = { semaphore_submit{
    .semaphore = acquire_.at(slot),
    .value = 0,
    .stage = acquire_wait_stage,
  } };
  sync.signals = {
    semaphore_submit{
      .semaphore = render_finished_.at(image_index),
      .value = 0,
      .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    },
    semaphore_submit{
      .semaphore = timeline_.handle(),
      .value = signal_value,
      .stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    },
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

auto frame_ring::create_image_semaphores(std::size_t image_count) -> status
{
  render_finished_.assign(image_count, VK_NULL_HANDLE);
  image_timeline_value_.assign(image_count, 0);
  for (std::size_t index = 0; index < image_count; ++index) {
    auto semaphore_result = create_binary_semaphore(ctx_->device());
    if (!semaphore_result) {
      destroy_image_semaphores();
      return fail(semaphore_result);
    }
    render_finished_.at(index) = expected_take(semaphore_result);
  }
  return {};
}

auto frame_ring::check_slot(std::size_t slot) const -> status
{
  if (slot >= acquire_.size()) { return fail(errc::out_of_range, "frame_ring slot index out of range"); }
  return {};
}

auto frame_ring::check_image(std::size_t image_index) const -> status
{
  if (image_index >= render_finished_.size()) {
    return fail(errc::out_of_range, "frame_ring image index out of range");
  }
  return {};
}

}// namespace vkexec
