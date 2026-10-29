#ifndef VKEXEC_DETAIL_GRAPH_SUBMISSION_SYNC_HPP
#define VKEXEC_DETAIL_GRAPH_SUBMISSION_SYNC_HPP

#include <vkexec/detail/pass_execution.hpp>

#include <algorithm>
#include <span>
#include <vector>

namespace vkexec::detail {

inline constexpr std::uint64_t k_graph_timeline_signal_value = 1;

struct prepared_batch_sync
{
  std::vector<semaphore_submit> waits;
  std::vector<semaphore_submit> signals;
  std::vector<std::size_t> wait_offsets;
  std::vector<std::size_t> signal_offsets;

  [[nodiscard]] auto waits_for(std::size_t batch) const -> std::span<semaphore_submit const>
  {
    return std::span<semaphore_submit const>{ waits }.subspan(
      wait_offsets.at(batch), wait_offsets.at(batch + std::size_t{ 1 }) - wait_offsets.at(batch));
  }

  [[nodiscard]] auto signals_for(std::size_t batch) const -> std::span<semaphore_submit const>
  {
    return std::span<semaphore_submit const>{ signals }.subspan(
      signal_offsets.at(batch), signal_offsets.at(batch + std::size_t{ 1 }) - signal_offsets.at(batch));
  }
};

[[nodiscard]] inline auto graph_dependency_count(execution_plan const &plan) -> std::size_t
{
  std::size_t count{};
  for (auto const &batch : plan.batches) { count += batch.predecessors.size(); }
  return count;
}

inline auto accumulate_batch_sync_offsets(std::vector<std::size_t> &offsets) -> void
{
  std::size_t total{};
  std::ranges::for_each(offsets, [&total](std::size_t &offset) -> void {
    total += offset;
    offset = total;
  });
}

[[nodiscard]] inline auto prepare_batch_sync_storage(execution_plan const &plan,
  bool timeline,
  bool presentation_wait,
  bool presentation_signal) -> prepared_batch_sync
{
  prepared_batch_sync prepared;
  auto const count = plan.batches.size();
  prepared.wait_offsets.resize(count + std::size_t{ 1 });
  prepared.signal_offsets.resize(count + std::size_t{ 1 });
  for (std::size_t destination{}; destination < count; ++destination) {
    prepared.wait_offsets.at(destination + std::size_t{ 1 }) = plan.batches.at(destination).predecessors.size();
    for (auto const source : plan.batches.at(destination).predecessors) {
      auto &signals = prepared.signal_offsets.at(source + std::size_t{ 1 });
      if (timeline) {
        signals = std::size_t{ 1 };
      } else {
        ++signals;
      }
    }
  }
  if (count != 0) {
    if (presentation_wait) { ++prepared.wait_offsets.back(); }
    if (presentation_signal) { ++prepared.signal_offsets.back(); }
  }
  accumulate_batch_sync_offsets(prepared.wait_offsets);
  accumulate_batch_sync_offsets(prepared.signal_offsets);
  prepared.waits.resize(prepared.wait_offsets.back());
  prepared.signals.resize(prepared.signal_offsets.back());
  return prepared;
}

// Binary semaphore order matches destination order, then predecessor order.
// Timeline sources signal once, regardless of their number of outgoing edges.
[[nodiscard]] inline auto prepare_batch_sync(execution_plan const &plan,
  std::span<VkSemaphore const> semaphores,
  bool timeline,
  semaphore_submit const *presentation_wait = nullptr,
  semaphore_submit const *presentation_signal = nullptr) -> prepared_batch_sync
{
  auto prepared =
    prepare_batch_sync_storage(plan, timeline, presentation_wait != nullptr, presentation_signal != nullptr);
  auto const count = plan.batches.size();
  auto signal_cursor = prepared.signal_offsets;
  std::size_t edge{};
  for (std::size_t destination{}; destination < count; ++destination) {
    auto wait_cursor = prepared.wait_offsets.at(destination);
    for (auto const source : plan.batches.at(destination).predecessors) {
      semaphore_submit const entry{ .semaphore = semaphores.subspan(timeline ? source : edge).front(),
        .value = timeline ? k_graph_timeline_signal_value : std::uint64_t{} };
      prepared.waits.at(wait_cursor++) = entry;
      if (!timeline) { prepared.signals.at(signal_cursor.at(source)++) = entry; }
      ++edge;
    }
  }
  if (timeline) {
    for (std::size_t source{}; source < count; ++source) {
      // The final batch may have a presentation signal without outgoing edges.
      if (prepared.signal_offsets.at(source + std::size_t{ 1 }) - prepared.signal_offsets.at(source)
          > static_cast<std::size_t>(presentation_signal != nullptr && source + std::size_t{ 1 } == count)) {
        prepared.signals.at(signal_cursor.at(source)++) =
          semaphore_submit{ .semaphore = semaphores.subspan(source).front(), .value = k_graph_timeline_signal_value };
      }
    }
  }
  if (count != 0) {
    if (presentation_wait != nullptr) { prepared.waits.back() = *presentation_wait; }
    if (presentation_signal != nullptr) { prepared.signals.back() = *presentation_signal; }
  }
  return prepared;
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_GRAPH_SUBMISSION_SYNC_HPP
