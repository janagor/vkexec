#include <vkexec_profiling/manual_capture.hpp>

#include "detail/profiler_state.hpp"
#include "detail/fence_status.hpp"

#include <vkexec/config.hpp>
#include <vkexec/detail/submission_id.hpp>
#include <vkexec/error_helpers.hpp>

#include <utility>

namespace vkexec::profiling::detail {

namespace {
  thread_local fence_status_fn fence_status_override = nullptr;
}

auto fence_status(VkDevice device, VkFence fence) -> VkResult
{
  if (fence_status_override) { return fence_status_override(device, fence); }
  return vkGetFenceStatus(device, fence);
}

auto set_fence_status_override(fence_status_fn replacement) noexcept -> fence_status_fn
{ return std::exchange(fence_status_override, replacement); }

}// namespace vkexec::profiling::detail

namespace vkexec::profiling {

struct manual_capture::sample
{
  detail::query_allocation allocation;
  std::string name;
};

manual_capture::manual_capture(std::shared_ptr<detail::profiler_state> state, VkCommandBuffer command_buffer,
                               queue_ref queue)
  : state_(std::move(state)), command_buffer_(command_buffer), queue_(queue)
{}

auto manual_capture::create(profiler const &profiler, context &ctx, VkCommandBuffer command_buffer, queue_ref queue)
  -> result<manual_capture>
{
  if (command_buffer == VK_NULL_HANDLE || queue.queue == VK_NULL_HANDLE || !profiler.state_ ||
      profiler.state_->device() != ctx.device()) {
    return fail(errc::invalid_argument, "manual capture requires a command buffer and matching context");
  }
  if (!profiler.supported(queue)) { return fail(errc::unsupported, "queue family cannot record timestamp queries"); }
  return manual_capture{ profiler.state_, command_buffer, queue };
}

manual_capture::manual_capture(manual_capture &&other) noexcept
  : state_(std::move(other.state_)), command_buffer_(std::exchange(other.command_buffer_, VK_NULL_HANDLE)),
    queue_(other.queue_), fence_(std::exchange(other.fence_, VK_NULL_HANDLE)), samples_(std::move(other.samples_)),
    active_regions_(std::exchange(other.active_regions_, 0)), submitted_(std::exchange(other.submitted_, false)),
    resolved_(std::exchange(other.resolved_, false)), submission_id_(std::exchange(other.submission_id_, 0))
{
  if (active_regions_ != 0) { vkexec::detail::contract_violation("cannot move a capture with active regions"); }
}

auto manual_capture::operator=(manual_capture &&other) noexcept -> manual_capture &
{
  if (this == &other) { return *this; }
  release();
  if (other.active_regions_ != 0) { vkexec::detail::contract_violation("cannot move a capture with active regions"); }
  state_ = std::move(other.state_);
  command_buffer_ = std::exchange(other.command_buffer_, VK_NULL_HANDLE);
  queue_ = other.queue_;
  fence_ = std::exchange(other.fence_, VK_NULL_HANDLE);
  samples_ = std::move(other.samples_);
  submitted_ = std::exchange(other.submitted_, false);
  resolved_ = std::exchange(other.resolved_, false);
  submission_id_ = std::exchange(other.submission_id_, 0);
  return *this;
}

manual_capture::~manual_capture() { release(); }

auto manual_capture::scoped_region(std::string name) -> result<region>
{
  constexpr std::uint32_t k_queries_per_region = 2;
  if (!state_ || submitted_) { return fail(errc::invalid_argument, "capture is not recording"); }
  auto reserved = state_->reserve(queue_);
  if (!reserved) { return fail(reserved); }
  auto const index = static_cast<std::uint32_t>(samples_.size());
  auto entry = std::make_unique<sample>();
  entry->allocation = std::move(*reserved);
  entry->name = std::move(name);
  samples_.push_back(std::move(entry));
  auto const &allocation = samples_.back()->allocation;
  vkCmdResetQueryPool(command_buffer_, allocation.page->pool, allocation.first_query, k_queries_per_region);
  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, allocation.page->pool,
                      allocation.first_query);
  ++active_regions_;
  return region{ *this, index };
}

auto manual_capture::end_region(std::uint32_t index) noexcept -> void
{
  constexpr std::uint32_t k_end_offset = 1;
  auto &entry = *samples_[index];
  vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, entry.allocation.page->pool,
                      entry.allocation.first_query + k_end_offset);
  --active_regions_;
}

manual_capture::region::region(region &&other) noexcept
  : capture_(std::exchange(other.capture_, nullptr)), index_(other.index_)
{}

auto manual_capture::region::operator=(region &&other) noexcept -> region &
{
  if (this == &other) { return *this; }
  if (capture_) { capture_->end_region(index_); }
  capture_ = std::exchange(other.capture_, nullptr);
  index_ = other.index_;
  return *this;
}

manual_capture::region::~region()
{
  if (capture_) { capture_->end_region(index_); }
}

auto manual_capture::mark_submitted(VkFence fence) -> status
{
  if (!state_ || submitted_ || active_regions_ != 0 || fence == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "capture requires closed regions and a submission fence");
  }
  submitted_ = true;
  fence_ = fence;
  submission_id_ = vkexec::detail::allocate_submission_id();
  return {};
}

auto manual_capture::resolve_after_gpu() -> status
{
  if (!state_ || !submitted_ || resolved_) { return fail(errc::invalid_argument, "capture has not been submitted"); }
  auto const completion = detail::fence_status(state_->device(), fence_);
  if (completion == VK_NOT_READY) { return fail(errc::out_of_range, "capture submission is still in flight"); }
  if (completion != VK_SUCCESS) {
    if (completion == VK_ERROR_DEVICE_LOST) { quarantine(); }
    return fail(completion, "vkGetFenceStatus");
  }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    for (std::uint32_t index = 0; index < samples_.size(); ++index) {
      auto &entry = *samples_[index];
      state_->publish(state_->resolve(entry.allocation, std::move(entry.name), submission_id_, index));
    }
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    // The fence signaled successfully, so these slots can be recycled.
    resolved_ = true;
    release();
    throw;
  }
#endif
  resolved_ = true;
  release();
  return {};
}

auto manual_capture::discard_after_gpu() -> status
{
  if (!state_ || !submitted_ || resolved_) { return fail(errc::invalid_argument, "capture has not been submitted"); }
  auto const completion = detail::fence_status(state_->device(), fence_);
  if (completion == VK_NOT_READY) { return fail(errc::out_of_range, "capture submission is still in flight"); }
  if (completion != VK_SUCCESS) {
    if (completion == VK_ERROR_DEVICE_LOST) { quarantine(); }
    return fail(completion, "vkGetFenceStatus");
  }
  resolved_ = true;
  release();
  return {};
}

auto manual_capture::quarantine() noexcept -> void
{
  // Device loss is terminal for GPU work. Keep slots out of the free list
  // until the profiler state and device are destroyed.
  for (auto &entry : samples_) { entry->allocation.quarantine(); }
  resolved_ = true;
  release();
}

auto manual_capture::release() noexcept -> void
{
  if (!state_) { return; }
  if (active_regions_ != 0) { vkexec::detail::contract_violation("capture destroyed with active regions"); }
  if (submitted_ && !resolved_) {
    vkexec::detail::contract_violation("submitted capture must be resolved after GPU completion");
  }
  samples_.clear();
  state_.reset();
}

}// namespace vkexec::profiling
