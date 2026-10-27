#ifndef VKEXEC_PROFILING_DETAIL_PROFILED_PASS_STEP_HPP
#define VKEXEC_PROFILING_DETAIL_PROFILED_PASS_STEP_HPP

#include <vkexec/pass.hpp>
#include <vkexec_profiling/detail/profiler_state.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace vkexec::profiling::detail {

// A region is part of its original graph step: no scheduling or resource edge is added.
template<class Step> class profiled_pass_step
{
public:
  profiled_pass_step(Step step, std::string name, profiler profiler)
    : inner_(std::move(step)), name_(std::move(name)), state_(std::move(profiler.state_))
  {}

  profiled_pass_step(profiled_pass_step const &other)
    requires std::copy_constructible<Step>
    : inner_(other.inner_), name_(other.name_), state_(other.state_)
  {
    if (other.allocation_ || other.completed_) {
      vkexec::detail::contract_violation("cannot copy an executing profiled pass");
    }
  }

  auto operator=(profiled_pass_step const &other) -> profiled_pass_step &
    requires std::copyable<Step>
  {
    if (this == &other) { return *this; }
    if (allocation_ || completed_ || other.allocation_ || other.completed_) {
      vkexec::detail::contract_violation("cannot assign an executing profiled pass");
    }
    inner_ = other.inner_;
    name_ = other.name_;
    state_ = other.state_;
    return *this;
  }

  profiled_pass_step(profiled_pass_step &&) noexcept(std::is_nothrow_move_constructible_v<Step>) = default;
  auto operator=(profiled_pass_step &&other) noexcept(std::is_nothrow_move_assignable_v<Step>) -> profiled_pass_step &
    requires std::is_move_assignable_v<Step>
  {
    if (this == &other) { return *this; }
    if (allocation_ || completed_) { vkexec::detail::contract_violation("cannot replace an executing profiled pass"); }
    inner_ = std::move(other.inner_);
    name_ = std::move(other.name_);
    state_ = std::move(other.state_);
    allocation_ = std::move(other.allocation_);
    completed_ = std::move(other.completed_);
    submission_id_ = other.submission_id_;
    region_index_ = other.region_index_;
    return *this;
  }

  ~profiled_pass_step() = default;

  auto record(context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup, pass_recording_info const &info)
    -> status
  {
    constexpr std::uint32_t k_queries_per_region = 2;
    constexpr std::uint32_t k_end_offset = 1;
    if (ctx.device() != state_->device()) {
      return fail(errc::invalid_argument, "profiler belongs to a different Vulkan device");
    }
    auto reserved = state_->reserve(info.queue);
    if (!reserved) { return fail(reserved); }
    allocation_.emplace(std::move(*reserved));
    submission_id_ = info.ensure_submission_id();
    region_index_ = info.allocate_region_index();
    auto *const pool = allocation_->page->pool;
    auto const first = allocation_->first_query;
    vkCmdResetQueryPool(cmd, pool, first, k_queries_per_region);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, first);
    auto recorded = vkexec::detail::record_step(inner_, ctx, cmd, cleanup, info);
    if (!recorded) {
      allocation_.reset();
      return recorded;
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, first + k_end_offset);
    return {};
  }

  auto after_gpu() -> void
  {
    vkexec::detail::run_after_gpu(inner_);
    if (allocation_) {
      completed_ = state_->resolve(*allocation_, name_, submission_id_, region_index_);
      allocation_.reset();
    }
  }

  auto commit_after_gpu() -> void
  {
    vkexec::detail::commit_after_gpu(inner_);
    if (completed_) {
      state_->publish(std::move(*completed_));
      completed_.reset();
    }
  }

  [[nodiscard]] auto has_resource_declarations() const noexcept -> bool
  { return vkexec::detail::step_has_resource_declarations(inner_); }

  [[nodiscard]] auto requires_complete_metadata() const noexcept -> bool
  { return vkexec::detail::step_requires_complete_metadata(inner_); }

  auto declare_resources(vkexec::detail::resource_state_tracker &tracker, std::size_t index, queue_ref queue) -> status
  { return vkexec::detail::declare_step_resources(inner_, tracker, index, queue, false); }

  auto declare_resources(vkexec::detail::resource_state_tracker &tracker, std::size_t index, queue_ref queue) const
    -> status
  { return vkexec::detail::declare_step_resources(inner_, tracker, index, queue, false); }

  auto declare_resources(vkexec::detail::resource_state_tracker &tracker,
    std::size_t index,
    queue_ref queue,
    bool crosses_families) -> status
  { return vkexec::detail::declare_step_resources(inner_, tracker, index, queue, crosses_families); }

  auto declare_resources(vkexec::detail::resource_state_tracker &tracker,
    std::size_t index,
    queue_ref queue,
    bool crosses_families) const -> status
  { return vkexec::detail::declare_step_resources(inner_, tracker, index, queue, crosses_families); }

private:
  Step inner_;
  std::string name_;
  std::shared_ptr<profiler_state> state_;
  std::optional<query_allocation> allocation_;
  std::optional<profile_event> completed_;
  std::uint64_t submission_id_{};
  std::uint32_t region_index_{};
};

}// namespace vkexec::profiling::detail

#endif// VKEXEC_PROFILING_DETAIL_PROFILED_PASS_STEP_HPP
