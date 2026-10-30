#include <vkexec/detail/pass_execution.hpp>
#include <vkexec/detail/pass_types.hpp>
#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/detail/resource_state_tracker_helpers.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_one_step = 1;
constexpr std::size_t k_two_steps = 2;
constexpr std::size_t k_four_steps = 4;
constexpr std::size_t k_run_length = 8;
constexpr std::size_t k_dense_width = 8;
constexpr std::size_t k_fan_in_width = 4;
constexpr std::size_t k_fan_in_group = k_fan_in_width + k_one_step;
constexpr VkDeviceSize k_buffer_stride = 64;
constexpr std::uint32_t k_first_family = 0;
constexpr std::uint32_t k_second_family = 1;
constexpr std::size_t k_eight_nodes = 8;
constexpr std::size_t k_thirty_two_nodes = 32;
constexpr std::size_t k_one_hundred_twenty_eight_nodes = 128;
constexpr std::size_t k_five_hundred_twelve_nodes = 512;

enum class planning_stage : std::uint8_t {
  dag,
  resource_sync,
  resource_dependencies,
  complete,
};

enum class shape : std::uint8_t {
  linear,
  independent,
  fork_join,
  alternating_queues,
  same_queue_runs,
  resource_hazard_chain,
  dense_dag,
  mixed_queue_hazards,
  resource_fan_in,
};

enum class queue_layout : std::uint8_t {
  same_family,
  different_families,
};

struct resource_step
{
  std::array<vkexec::buffer_use, k_fan_in_width> resources{};
  std::size_t resource_count{};

  [[nodiscard]] auto declare_resources(vkexec::detail::resource_state_tracker &tracker,
    std::size_t index,
    vkexec::queue_ref queue,
    bool /*crosses_families*/) -> vkexec::status
  {
    for (std::size_t resource{}; resource < resource_count; ++resource) {
      VKEXEC_TRY(tracker.use(resources.at(resource), index, queue.family, queue.queue));
    }
    return {};
  }
};

struct graph_input
{
  std::vector<vkexec::queue_affinity> queues;
  std::vector<std::vector<std::size_t>> predecessors;
  std::vector<resource_step> steps;
  vkexec::queue_ref default_queue;
};

[[nodiscard]] auto uses_second_queue(shape topology, std::size_t index) -> bool
{
  if (topology == shape::alternating_queues || topology == shape::mixed_queue_hazards) {
    return index % k_two_steps == k_one_step;
  }
  if (topology == shape::same_queue_runs) { return index / k_run_length % k_two_steps == k_one_step; }
  if (topology == shape::fork_join) { return index % k_four_steps == k_two_steps; }
  return topology == shape::resource_fan_in && index % k_fan_in_group == k_fan_in_width;
}

auto add_predecessors(std::vector<std::size_t> &incoming, shape topology, std::size_t index) -> void
{
  if (index == k_first_step) { return; }
  switch (topology) {
  case shape::independent:
  case shape::resource_hazard_chain:
  case shape::mixed_queue_hazards:
  case shape::resource_fan_in:
    break;
  case shape::fork_join:
    if (index % k_four_steps == k_one_step || index % k_four_steps == k_two_steps) {
      incoming.push_back(index - (index % k_four_steps == k_two_steps ? k_two_steps : k_one_step));
    } else if (index % k_four_steps == k_four_steps - k_one_step) {
      incoming.push_back(index - k_one_step);
      incoming.push_back(index - k_two_steps);
    } else {
      incoming.push_back(index - k_one_step);
    }
    break;
  case shape::dense_dag:
    for (std::size_t distance = k_one_step; distance <= k_dense_width && distance <= index; ++distance) {
      incoming.push_back(index - distance);
    }
    break;
  default:
    incoming.push_back(index - k_one_step);
    break;
  }
}

[[nodiscard]] auto make_resource_step(VkBuffer buffer, shape topology, std::size_t index) -> resource_step
{
  resource_step step{};
  if (topology == shape::resource_fan_in && index % k_fan_in_group == k_fan_in_width) {
    for (std::size_t distance = k_fan_in_width; distance != 0; --distance) {
      step.resources.at(step.resource_count++) = vkexec::read(buffer,
        static_cast<VkDeviceSize>(index - distance) * k_buffer_stride,
        k_buffer_stride,
        vkexec::buffer_usage::uniform_compute);
    }
  } else {
    bool const shared = topology == shape::resource_hazard_chain || topology == shape::mixed_queue_hazards;
    step.resources.front() = vkexec::write(buffer,
      shared ? VkDeviceSize{} : static_cast<VkDeviceSize>(index) * k_buffer_stride,
      k_buffer_stride,
      vkexec::buffer_usage::storage_compute);
    step.resource_count = k_one_step;
  }
  return step;
}

[[nodiscard]] auto make_input(std::size_t count, shape topology, queue_layout layout) -> graph_input
{
  static char first_queue_storage{};
  static char second_queue_storage{};
  static char buffer_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque benchmark-only queue handle
  vkexec::queue_ref const first{ .queue = reinterpret_cast<VkQueue>(&first_queue_storage), .family = k_first_family };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque benchmark-only queue handle
  vkexec::queue_ref const second{ .queue = reinterpret_cast<VkQueue>(&second_queue_storage),
    .family = layout == queue_layout::same_family ? k_first_family : k_second_family };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque benchmark-only buffer handle
  auto *const buffer = reinterpret_cast<VkBuffer>(&buffer_storage);
  graph_input input{ .queues = {}, .predecessors = {}, .steps = {}, .default_queue = first };
  input.queues.reserve(count);
  input.predecessors.resize(count);
  input.steps.reserve(count);
  for (std::size_t index{}; index < count; ++index) {
    input.queues.emplace_back(uses_second_queue(topology, index) ? second : first);
    add_predecessors(input.predecessors.at(index), topology, index);
    input.steps.push_back(make_resource_step(buffer, topology, index));
  }
  return input;
}

[[nodiscard]] auto edge_count(vkexec::detail::execution_plan const &plan) -> std::size_t
{
  std::size_t total{};
  for (auto const &batch : plan.batches) { total += batch.predecessors.size(); }
  return total;
}

auto set_counters(benchmark::State &state,
  graph_input const &input,
  vkexec::detail::execution_plan const &plan,
  vkexec::detail::resource_sync_plan const &sync) -> void
{
  std::size_t explicit_dependencies{};
  std::size_t resource_uses{};
  for (auto const &incoming : input.predecessors) { explicit_dependencies += incoming.size(); }
  for (auto const &step : input.steps) { resource_uses += step.resource_count; }
  state.counters.insert_or_assign("steps", static_cast<double>(input.steps.size()));
  state.counters.insert_or_assign("resource_use_count", static_cast<double>(resource_uses));
  state.counters.insert_or_assign("resource_dependencies", static_cast<double>(sync.dependencies.size()));
  state.counters.insert_or_assign("explicit_dependencies", static_cast<double>(explicit_dependencies));
  state.counters.insert_or_assign("batch_count", static_cast<double>(plan.batches.size()));
  state.counters.insert_or_assign("dependency_edge_count", static_cast<double>(edge_count(plan)));
  std::size_t queue_count{};
  if (!input.queues.empty()) {
    queue_count = k_one_step;
    for (auto const &queue : input.queues) {
      if (!vkexec::detail::same_queue(
            queue.value_or(input.default_queue), input.queues.front().value_or(input.default_queue))) {
        queue_count = k_two_steps;
        break;
      }
    }
  }
  state.counters.insert_or_assign("queue_count", static_cast<double>(queue_count));
}

[[nodiscard]] auto plan_graph(graph_input const &input) -> vkexec::result<vkexec::detail::execution_plan>
{ return vkexec::detail::plan_dag(input.queues, input.predecessors, input.default_queue); }

[[nodiscard]] auto expected_resource_dependencies(std::size_t count, shape topology) -> std::size_t
{
  if (topology == shape::resource_hazard_chain || topology == shape::mixed_queue_hazards) {
    return count == 0 ? 0 : count - k_one_step;
  }
  if (topology == shape::resource_fan_in) { return count / k_fan_in_group * k_fan_in_width; }
  return 0;
}

auto run_benchmark(benchmark::State &state, planning_stage stage) -> void
{
  auto const count = static_cast<std::size_t>(state.range(0));
  auto const topology = static_cast<shape>(state.range(1));
  auto const layout = static_cast<queue_layout>(state.range(2));
  auto input = make_input(count, topology, layout);
  auto initial = plan_graph(input);
  if (!initial) {
    state.SkipWithError(initial.error().message());
    return;
  }
  auto sync = vkexec::detail::plan_resource_sync(*initial, input.steps);
  if (!sync) {
    state.SkipWithError(sync.error().message());
    return;
  }
  if (sync->dependencies.size() != expected_resource_dependencies(count, topology)) {
    state.SkipWithError("unexpected resource dependency count");
    return;
  }
  vkexec::detail::execution_plan last_plan = *initial;

  // NOLINTNEXTLINE(readability-identifier-length) -- Google Benchmark keep-running idiom
  for ([[maybe_unused]] auto _ : state) {
    if (stage == planning_stage::dag) {
      auto planned = plan_graph(input);
      benchmark::DoNotOptimize(planned);
    } else if (stage == planning_stage::resource_sync) {
      auto planned_sync = vkexec::detail::plan_resource_sync(*initial, input.steps);
      benchmark::DoNotOptimize(planned_sync);
    } else if (stage == planning_stage::resource_dependencies) {
      state.PauseTiming();
      last_plan = *initial;
      state.ResumeTiming();
      vkexec::detail::add_resource_dependencies(last_plan, *sync);
      benchmark::DoNotOptimize(last_plan);
    } else {
      auto planned = plan_graph(input);
      if (!planned) {
        state.SkipWithError(planned.error().message());
        return;
      }
      auto planned_sync = vkexec::detail::plan_resource_sync(*planned, input.steps);
      if (!planned_sync) {
        state.SkipWithError(planned_sync.error().message());
        return;
      }
      vkexec::detail::add_resource_dependencies(*planned, *planned_sync);
      benchmark::DoNotOptimize(planned);
      benchmark::DoNotOptimize(planned_sync);
    }
  }
  if (stage == planning_stage::complete) { vkexec::detail::add_resource_dependencies(last_plan, *sync); }
  set_counters(state, input, last_plan, *sync);
}

auto bm_plan_dag(benchmark::State &state) -> void { run_benchmark(state, planning_stage::dag); }
auto bm_resource_sync(benchmark::State &state) -> void { run_benchmark(state, planning_stage::resource_sync); }
auto bm_add_resource_dependencies(benchmark::State &state) -> void
{ run_benchmark(state, planning_stage::resource_dependencies); }
auto bm_complete_planning(benchmark::State &state) -> void { run_benchmark(state, planning_stage::complete); }

auto register_cases(benchmark::Benchmark *entry) -> void
{
  constexpr std::array k_sizes{
    k_eight_nodes, k_thirty_two_nodes, k_one_hundred_twenty_eight_nodes, k_five_hundred_twelve_nodes
  };
  constexpr std::array k_shapes{ shape::linear,
    shape::independent,
    shape::fork_join,
    shape::alternating_queues,
    shape::same_queue_runs,
    shape::resource_hazard_chain,
    shape::dense_dag,
    shape::mixed_queue_hazards,
    shape::resource_fan_in };
  constexpr std::array k_layouts{ queue_layout::same_family, queue_layout::different_families };
  for (auto const size : k_sizes) {
    for (auto const topology : k_shapes) {
      for (auto const layout : k_layouts) {
        bool const uses_second_queue = topology == shape::fork_join || topology == shape::alternating_queues
                                       || topology == shape::mixed_queue_hazards || topology == shape::resource_fan_in
                                       || (topology == shape::same_queue_runs && size > k_run_length);
        if (layout == queue_layout::different_families && !uses_second_queue) { continue; }
        entry->Args(
          { static_cast<std::int64_t>(size), static_cast<std::int64_t>(topology), static_cast<std::int64_t>(layout) });
      }
    }
  }
  entry->Unit(benchmark::kMicrosecond);
}

BENCHMARK(bm_plan_dag)->Apply(register_cases);
BENCHMARK(bm_resource_sync)->Apply(register_cases);
BENCHMARK(bm_add_resource_dependencies)->Apply(register_cases);
BENCHMARK(bm_complete_planning)->Apply(register_cases);

}// namespace
