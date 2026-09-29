#ifndef VKEXEC_DETAIL_SUBMISSION_HPP
#define VKEXEC_DETAIL_SUBMISSION_HPP

#include <vkexec/detail/synchronization.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>

#include <cstdint>
#include <mutex>
#include <vector>

namespace vkexec::detail {

struct legacy_submission_view;

struct legacy_submission
{
  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  std::vector<std::uint64_t> wait_values;
  std::vector<VkSemaphore> signal_semaphores;
  std::vector<std::uint64_t> signal_values;
  bool legacy_timeline_submit_info_available{ false };

  [[nodiscard]] auto make_submit_info(queue_submit const &info) const noexcept -> legacy_submission_view;
};

// Owns the Vulkan submit structs, but borrows arrays from legacy_submission
// and queue_submit; both must outlive this view.
struct legacy_submission_view
{
  VkTimelineSemaphoreSubmitInfo timeline{};
  VkSubmitInfo submit{};

  legacy_submission_view(legacy_submission const &lowered, queue_submit const &info) noexcept;
  legacy_submission_view(legacy_submission_view const &) = delete;
  auto operator=(legacy_submission_view const &) -> legacy_submission_view & = delete;
  legacy_submission_view(legacy_submission_view &&) = delete;
  auto operator=(legacy_submission_view &&) -> legacy_submission_view & = delete;
};

struct synchronization2_submission
{
  std::vector<VkCommandBufferSubmitInfo> commands;
  std::vector<VkSemaphoreSubmitInfo> waits;
  std::vector<VkSemaphoreSubmitInfo> signals;

  [[nodiscard]] auto submit_info() const noexcept -> VkSubmitInfo2;
};

[[nodiscard]] auto lower_synchronization2_submit(queue_submit const &info) -> synchronization2_submission;
[[nodiscard]] auto lower_legacy_submit(queue_submit const &info, bool legacy_timeline_submit_info_available)
  -> result<legacy_submission>;
[[nodiscard]] auto legacy_wait_stage(VkPipelineStageFlags2 stage) -> result<VkPipelineStageFlags>;
[[nodiscard]] auto legacy_signal_stage(VkPipelineStageFlags2 stage) -> status;
[[nodiscard]] auto submit(queue_submit const &info,
  VkQueue queue,
  synchronization_backend backend,
  bool legacy_timeline_submit_info_available,
  device_procs const &procs,
  std::mutex &host_mutex) -> status;

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SUBMISSION_HPP
