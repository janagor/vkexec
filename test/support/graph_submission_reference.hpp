#ifndef VKEXEC_TEST_GRAPH_SUBMISSION_REFERENCE_HPP
#define VKEXEC_TEST_GRAPH_SUBMISSION_REFERENCE_HPP

#include <vkexec/detail/graph_submission_sync.hpp>

#include <algorithm>
#include <cstdint>

namespace vkexec::test {

struct reference_submission_edge
{
  std::size_t source{};
  std::size_t destination{};
  VkSemaphore semaphore{ VK_NULL_HANDLE };
};

[[nodiscard]] inline auto reference_submission_edges(detail::execution_plan const &plan,
  std::span<VkSemaphore const> semaphores,
  bool timeline) -> std::vector<reference_submission_edge>
{
  std::vector<reference_submission_edge> edges;
  for (std::size_t destination{}; destination < plan.batches.size(); ++destination) {
    for (auto const source : plan.batches.at(destination).predecessors) {
      edges.push_back(reference_submission_edge{ .source = source,
        .destination = destination,
        .semaphore = semaphores.subspan(timeline ? source : edges.size()).front() });
    }
  }
  return edges;
}

// Edge-scanning algorithm, retained only as a benchmark/test oracle.
[[nodiscard]] inline auto reference_batch_sync(detail::execution_plan const &plan,
  std::span<VkSemaphore const> semaphores,
  bool timeline,
  semaphore_submit const *presentation_wait = nullptr,
  semaphore_submit const *presentation_signal = nullptr) -> detail::prepared_batch_sync
{
  auto const edges = reference_submission_edges(plan, semaphores, timeline);
  detail::prepared_batch_sync result;
  for (std::size_t batch{}; batch < plan.batches.size(); ++batch) {
    result.wait_offsets.push_back(result.waits.size());
    result.signal_offsets.push_back(result.signals.size());
    std::vector<semaphore_submit> waits;
    std::vector<semaphore_submit> signals;
    for (auto const &dependency : edges) {
      if (dependency.destination == batch) {
        waits.push_back(semaphore_submit{ .semaphore = dependency.semaphore,
          .value = timeline ? detail::k_graph_timeline_signal_value : std::uint64_t{} });
      }
      if (!timeline && dependency.source == batch) {
        signals.push_back(semaphore_submit{ .semaphore = dependency.semaphore });
      }
    }
    if (timeline && std::ranges::any_of(edges, [batch](reference_submission_edge const &dependency) -> bool {
          return dependency.source == batch;
        })) {
      signals.push_back(semaphore_submit{
        .semaphore = semaphores.subspan(batch).front(), .value = detail::k_graph_timeline_signal_value });
    }
    if (batch + std::size_t{ 1 } == plan.batches.size()) {
      if (presentation_wait != nullptr) { waits.push_back(*presentation_wait); }
      if (presentation_signal != nullptr) { signals.push_back(*presentation_signal); }
    }
    result.waits.insert(result.waits.end(), waits.begin(), waits.end());
    result.signals.insert(result.signals.end(), signals.begin(), signals.end());
  }
  result.wait_offsets.push_back(result.waits.size());
  result.signal_offsets.push_back(result.signals.size());
  return result;
}

enum class submission_shape : std::uint8_t { linear, independent, fork_join, dense, fan_in, mixed_queue };

[[nodiscard]] inline auto submission_plan(std::size_t count, submission_shape shape) -> detail::execution_plan
{
  detail::execution_plan plan;
  constexpr std::size_t k_queue_count = 2;
  for (std::size_t batch{}; batch < count; ++batch) {
    detail::execution_batch next;
    next.first_step = batch;
    next.end_step = batch + std::size_t{ 1 };
    if (shape == submission_shape::mixed_queue) {
      next.queue.family = static_cast<std::uint32_t>(batch % k_queue_count);
    }
    if (batch != 0) {
      if (shape == submission_shape::linear || shape == submission_shape::mixed_queue) {
        next.predecessors.push_back(batch - std::size_t{ 1 });
      } else if (shape == submission_shape::dense
                 || ((shape == submission_shape::fan_in || shape == submission_shape::fork_join)
                     && batch + std::size_t{ 1 } == count)) {
        for (std::size_t source{}; source < batch; ++source) { next.predecessors.push_back(source); }
      } else if (shape == submission_shape::fork_join) {
        next.predecessors.push_back(std::size_t{});
      }
    }
    plan.batches.push_back(std::move(next));
  }
  return plan;
}

}// namespace vkexec::test

#endif// VKEXEC_TEST_GRAPH_SUBMISSION_REFERENCE_HPP
