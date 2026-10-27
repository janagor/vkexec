#ifndef VKEXEC_DETAIL_PASS_EXECUTION_HPP
#define VKEXEC_DETAIL_PASS_EXECUTION_HPP

#include <vkexec/detail/pass_steps.hpp>

namespace vkexec::detail {

struct execution_batch
{
  queue_ref queue{};
  std::size_t first_step{};
  std::size_t end_step{};
  std::vector<std::size_t> predecessors;
};

struct execution_plan
{
  std::vector<execution_batch> batches;
};

[[nodiscard]] inline auto same_queue(queue_ref lhs, queue_ref rhs) noexcept -> bool
{ return lhs.queue == rhs.queue && lhs.family == rhs.family; }

[[nodiscard]] inline auto plan_batches(std::span<queue_affinity const> queues, queue_ref default_queue)
  -> execution_plan
{
  execution_plan plan;
  std::size_t index{};
  for (queue_affinity const &affinity : queues) {
    queue_ref const queue = affinity.value_or(default_queue);
    if (plan.batches.empty() || !same_queue(plan.batches.back().queue, queue)) {
      plan.batches.push_back(execution_batch{
        .queue = queue, .first_step = index, .end_step = index + std::size_t{ 1 }, .predecessors = {} });
      if (plan.batches.size() > std::size_t{ 1 }) {
        plan.batches.back().predecessors.push_back(plan.batches.size() - std::size_t{ 2 });
      }
    } else {
      plan.batches.back().end_step = index + 1;
    }
    ++index;
  }
  return plan;
}

//! DAG nodes are appended in topological order. Each node gets its own batch
//! so independent nodes can run on separate queues without an implicit wait.
[[nodiscard]] inline auto plan_dag(std::span<queue_affinity const> queues,
  std::span<std::vector<std::size_t> const> predecessors,
  queue_ref default_queue) -> result<execution_plan>
{
  if (queues.size() != predecessors.size()) { return fail(errc::invalid_argument, "DAG queue count mismatch"); }
  execution_plan plan;
  plan.batches.reserve(queues.size());
  for (std::size_t index{}; index < queues.size(); ++index) {
    auto const &incoming = predecessors.subspan(index).front();
    for (std::size_t const predecessor : incoming) {
      if (predecessor >= index) { return fail(errc::invalid_argument, "DAG predecessor must precede its node"); }
      if (std::ranges::count(incoming, predecessor) != std::size_t{ 1 }) {
        return fail(errc::invalid_argument, "duplicate DAG predecessor");
      }
    }
    plan.batches.push_back(execution_batch{ .queue = queues.subspan(index).front().value_or(default_queue),
      .first_step = index,
      .end_step = index + std::size_t{ 1 },
      .predecessors = incoming });
  }
  return plan;
}

[[nodiscard]] inline auto plan_execution(std::size_t step_count,
  std::span<queue_affinity const> queues,
  queue_ref default_queue) -> result<execution_plan>
{
  if (!queues.empty() && queues.size() != step_count) {
    return fail(errc::invalid_argument, "pass queue count mismatch");
  }
  std::vector<queue_affinity> default_queues;
  if (queues.empty()) { default_queues.resize(step_count); }
  return plan_batches(queues.empty() ? std::span<queue_affinity const>{ default_queues } : queues, default_queue);
}

template<class StepStorage>
[[nodiscard]] constexpr auto pass_step_count(StepStorage const &steps) noexcept -> std::size_t
{
  if constexpr (requires { steps.size(); }) {
    return steps.size();
  } else {
    return std::tuple_size_v<StepStorage>;
  }
}

template<class StepStorage>
[[nodiscard]] auto
  record_batch_steps(context &facade, submit_scope &scope, execution_batch const &batch, StepStorage &steps) -> status
{
  auto const info = recording_info(scope.queue);
  if constexpr (requires { steps.size(); }) {
    for (std::size_t index = batch.first_step; index < batch.end_step; ++index) {
      if (auto recorded = record_step(steps.at(index), facade, scope.cmd, scope.cleanup, info); !recorded) {
        return fail(recorded);
      }
    }
    return {};
  } else {
    status recorded{};
    std::size_t index{};
    auto record_one = [&](auto &step) -> void {
      if (recorded && index >= batch.first_step && index < batch.end_step) {
        recorded = record_step(step, facade, scope.cmd, scope.cleanup, info);
      }
      ++index;
    };
    std::apply([&](auto &...step) -> void { (record_one(step), ...); }, steps);
    return recorded;
  }
}

[[nodiscard]] inline auto validate_execution_batches(execution_plan const &plan, std::size_t step_count) -> status
{
  if (plan.batches.empty() && step_count != 0) { return fail(errc::invalid_argument, "invalid empty execution plan"); }
  std::size_t validated_step{};
  for (std::size_t batch_index{}; batch_index < plan.batches.size(); ++batch_index) {
    auto const &batch = plan.batches.at(batch_index);
    if (batch.first_step != validated_step || batch.end_step <= batch.first_step || batch.end_step > step_count) {
      return fail(errc::invalid_argument, "invalid execution batch");
    }
    validated_step = batch.end_step;
    if (std::ranges::any_of(
          batch.predecessors, [batch_index](std::size_t predecessor) -> bool { return predecessor >= batch_index; })) {
      return fail(errc::invalid_argument, "invalid execution dependency");
    }
  }
  if (validated_step != step_count) { return fail(errc::invalid_argument, "incomplete execution plan"); }
  return {};
}

[[nodiscard]] inline auto crosses_queue_families(execution_plan const &execution) -> bool
{
  for (std::size_t index = 1; index < execution.batches.size(); ++index) {
    if (execution.batches.at(index - 1).queue.family != execution.batches.at(index).queue.family) { return true; }
  }
  return false;
}

[[nodiscard]] inline auto terminal_batches(execution_plan const &plan) -> std::vector<std::size_t>
{
  std::vector<bool> has_successor(plan.batches.size());
  for (auto const &batch : plan.batches) {
    for (std::size_t const predecessor : batch.predecessors) { has_successor.at(predecessor) = true; }
  }
  std::vector<std::size_t> terminals;
  for (std::size_t index{}; index < has_successor.size(); ++index) {
    if (!has_successor.at(index)) { terminals.push_back(index); }
  }
  return terminals;
}

template<class Step>
[[nodiscard]] auto declare_step_resources(Step &step,
  resource_state_tracker &tracker,
  std::size_t index,
  queue_ref queue,
  bool crosses_families) -> status
{
  if constexpr (requires { step.declare_resources(tracker, index, queue, crosses_families); }) {
    return step.declare_resources(tracker, index, queue, crosses_families);
  } else if constexpr (requires { step.resources; }) {
    return tracker.use(step.resources, index, queue.family, queue.queue);
  } else if constexpr (requires { step.declare_resources(tracker, index, queue); }) {
    if constexpr (requires { step.has_resource_declarations(); }) {
      if (!step.has_resource_declarations()) {
        return fail(errc::invalid_argument, "pass step lacks complete resource declarations");
      }
    } else if constexpr (requires { step.bind.metadata; }) {
      if (step.bind.metadata != resource_metadata_state::complete) {
        return fail(errc::invalid_argument, "pass step lacks complete resource declarations");
      }
    } else if (crosses_families) {
      return fail(errc::unsupported, "cross-family pass steps require resource declarations");
    }
    return step.declare_resources(tracker, index, queue);
  } else {
    if (crosses_families) { return fail(errc::unsupported, "cross-family pass steps require resource declarations"); }
    return {};
  }
}

[[nodiscard]] inline auto declare_step_resources(dynamic_pass_step &step,
  resource_state_tracker &tracker,
  std::size_t index,
  queue_ref queue,
  bool crosses_families) -> status
{
  if (!step.has_resource_declarations() && (crosses_families || step.requires_complete_metadata())) {
    return fail(errc::invalid_argument, "pass step lacks complete resource declarations");
  }
  return step.declare_resources(tracker, index, queue, crosses_families);
}

template<class... Steps>
[[nodiscard]] auto declare_indexed_step(std::tuple<Steps...> &steps,
  resource_state_tracker &tracker,
  std::size_t index,
  queue_ref queue,
  bool crosses_families) -> status
{
  status declared{};
  std::size_t position{};
  std::apply(
    [&](auto &...step) -> void {
      [[maybe_unused]] auto declare_one = [&](auto &current) -> void {
        if (position == index) { declared = declare_step_resources(current, tracker, index, queue, crosses_families); }
        ++position;
      };
      (declare_one(step), ...);
    },
    steps);
  return declared;
}

template<class StepStorage>
[[nodiscard]] auto plan_resource_sync(execution_plan const &execution, StepStorage &steps) -> result<resource_sync_plan>
{
  resource_state_tracker tracker{ pass_step_count(steps) };
  bool const crosses_families = crosses_queue_families(execution);
  for (auto const &batch : execution.batches) {
    for (std::size_t index = batch.first_step; index < batch.end_step; ++index) {
      status declared{};
      if constexpr (requires { steps.size(); }) {
        declared = declare_step_resources(steps.at(index), tracker, index, batch.queue, crosses_families);
      } else {
        declared = declare_indexed_step(steps, tracker, index, batch.queue, crosses_families);
      }
      if (!declared) { return fail(declared); }
    }
  }
  return std::move(tracker).finish();
}

[[nodiscard]] auto record_sync(context &facade,
  VkCommandBuffer cmd,
  std::vector<image_barrier_params> const &images,
  std::vector<buffer_barrier_params> const &buffers) -> status;

template<class StepStorage>
[[nodiscard]] auto record_sync_batch(context &facade,
  submit_scope &scope,
  execution_batch const &batch,
  resource_sync_plan const &sync,
  StepStorage &steps) -> status
{
  auto const info = recording_info(scope.queue);
  for (std::size_t index = batch.first_step; index < batch.end_step; ++index) {
    auto const &point = sync.steps.at(index);
    VKEXEC_TRY(record_sync(facade, scope.cmd, point.images_before, point.buffers_before));
    if constexpr (requires { steps.size(); }) {
      VKEXEC_TRY(record_step(steps.at(index), facade, scope.cmd, scope.cleanup, info));
    } else {
      status recorded{};
      std::size_t position{};
      std::apply(
        [&](auto &...step) -> void {
          [[maybe_unused]] auto record_one = [&](auto &current) -> void {
            if (position == index) { recorded = record_step(current, facade, scope.cmd, scope.cleanup, info); }
            ++position;
          };
          (record_one(step), ...);
        },
        steps);
      if (!recorded) { return unexpected(std::move(recorded.error())); }
    }
    VKEXEC_TRY(record_sync(facade, scope.cmd, point.images_after, point.buffers_after));
  }
  return {};
}

[[nodiscard]] inline auto
  validate_presentation_image(resource_sync_plan const &sync, execution_plan const &plan, VkImage image) -> status
{
  if (image == VK_NULL_HANDLE) { return {}; }
  if (plan.batches.empty()) { return fail(errc::invalid_argument, "presented image requires a final batch"); }
  bool saw_image{};
  for (auto const &use : sync.image_uses) {
    if (use.image != image) { continue; }
    saw_image = true;
    constexpr std::uint32_t k_single_subresource = 1;
    if (use.range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || use.range.baseMipLevel != 0
        || (use.range.levelCount != k_single_subresource && use.range.levelCount != VK_REMAINING_MIP_LEVELS)
        || use.range.baseArrayLayer != 0
        || (use.range.layerCount != k_single_subresource && use.range.layerCount != VK_REMAINING_ARRAY_LAYERS)) {
      return fail(errc::invalid_argument, "presented image use must cover the swapchain color subresource");
    }
    if (use.step < plan.batches.back().first_step || use.usage != image_usage::color_attachment
        || use.final_layout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
      return fail(errc::invalid_argument,
        "presented image must be used as a color attachment only in the final batch and finish in present layout");
    }
  }
  if (!saw_image) { return fail(errc::invalid_argument, "presented image has no declared graph use"); }
  return {};
}

inline auto add_resource_dependencies(execution_plan &plan, resource_sync_plan const &sync) -> void
{
  for (auto const &[source, destination] : sync.dependencies) {
    std::size_t source_batch{};
    std::size_t destination_batch{};
    for (std::size_t index{}; index < plan.batches.size(); ++index) {
      auto const &batch = plan.batches.at(index);
      if (source >= batch.first_step && source < batch.end_step) { source_batch = index; }
      if (destination >= batch.first_step && destination < batch.end_step) { destination_batch = index; }
    }
    if (source_batch == destination_batch) { continue; }
    auto &incoming = plan.batches.at(destination_batch).predecessors;
    if (std::ranges::find(incoming, source_batch) == incoming.end()) { incoming.push_back(source_batch); }
  }
}

template<class StepStorage>
[[nodiscard]] auto record_execution(context_handle const &state,
  execution_plan &plan,
  StepStorage &steps,
  VkImage presentation_image = VK_NULL_HANDLE) -> result<std::vector<submit_scope>>
{
  if (!state) { return fail(errc::invalid_argument, "pass graph requires a context"); }
  auto const step_count = pass_step_count(steps);
  if (auto validated = validate_execution_batches(plan, step_count); !validated) { return fail(validated); }
  auto synchronization = plan_resource_sync(plan, steps);
  if (!synchronization) { return fail(synchronization); }
  add_resource_dependencies(plan, *synchronization);
  VKEXEC_TRY(validate_presentation_image(*synchronization, plan, presentation_image));
  auto facade = context_access::facade(state);
  std::vector<submit_scope> scopes;
  scopes.reserve(plan.batches.empty() ? std::size_t{ 1 } : plan.batches.size());
  for (auto const &batch : plan.batches) {
    auto opened = submit_scope::open(facade, batch.queue);
    if (!opened) { return fail(opened); }
    scopes.push_back(expected_take(opened));
    auto &scope = scopes.back();
    auto recorded = record_sync_batch(facade, scope, batch, *synchronization, steps);
    if (!recorded) { return fail(recorded); }
    if (auto ended = scope.end_recording(); !ended) { return fail(ended); }
  }
  if (scopes.empty()) {
    auto opened = submit_scope::open(facade);
    if (!opened) { return fail(opened); }
    scopes.push_back(expected_take(opened));
    if (auto ended = scopes.back().end_recording(); !ended) { return fail(ended); }
  }
  return scopes;
}

template<class... Steps> auto run_after_gpu(std::tuple<Steps...> &steps) -> void
{
  std::apply([](auto &...step) -> void { (detail::run_after_gpu(step), ...); }, steps);
}

inline auto run_after_gpu(std::vector<dynamic_pass_step> &steps) -> void
{
  for (auto &step : steps) { step.after_gpu(); }
}

template<class... Steps> auto commit_after_gpu(std::tuple<Steps...> &steps) -> void
{
  std::apply([](auto &...step) -> void { (detail::commit_after_gpu(step), ...); }, steps);
}

inline auto commit_after_gpu(std::vector<dynamic_pass_step> &steps) -> void
{
  for (auto &step : steps) { step.commit_after_gpu(); }
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_PASS_EXECUTION_HPP
