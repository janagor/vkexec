#include "allocation_counter.hpp"
#include "representation_candidates.hpp"

#include <vkexec/pass.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct noop_step
{
  std::uintptr_t payload{};
  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  {
    benchmark::DoNotOptimize(payload);
    return {};
  }
};

struct representative_step
{
  VkPipeline pipeline{};
  VkPipelineLayout layout{};
  VkDescriptorSet set{};
  VkBuffer buffer{};
  std::uint32_t x{};
  std::uint32_t y{};
  std::uint32_t z{};
  std::uintptr_t payload{};
  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  {
    benchmark::DoNotOptimize(*this);
    return {};
  }
};

struct oversized_step
{
  std::array<std::byte, 256> data{};
  std::uintptr_t payload{};
  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  {
    benchmark::DoNotOptimize(*this);
    return {};
  }
};

template<class Step> [[nodiscard]] auto make_step(std::uintptr_t payload) -> Step
{
  Step step{};
  step.payload = payload;
  return step;
}

template<class Step, std::size_t... I>
[[nodiscard]] auto make_static_steps(std::uintptr_t seed, std::index_sequence<I...>)
{ return std::tuple{ make_step<Step>(seed + I)... }; }

template<class Step>
[[nodiscard]] auto make_dynamic_graph(std::size_t count, bool reserve, std::uintptr_t seed)
  -> vkexec::dynamic_pass_graph_sender
{
  auto graph = vkexec::make_dynamic_pass_graph(vkexec::scheduler{ nullptr }.schedule());
  if (reserve) { graph.reserve(count); }
  for (std::size_t i = 0; i < count; ++i) { graph.append(vkexec::make_pass_adaptor(make_step<Step>(seed + i))); }
  return graph;
}

auto report_allocations(benchmark::State &state, vkexec_benchmark::allocation_counts counts, std::size_t steps) -> void
{
  auto const iterations = static_cast<double>(state.iterations());
  state.counters["allocs/graph"] = static_cast<double>(counts.allocations) / iterations;
  state.counters["bytes/graph"] = static_cast<double>(counts.bytes) / iterations;
  state.counters["allocs/step"] = static_cast<double>(counts.allocations) / (iterations * static_cast<double>(steps));
}

template<class Step, std::size_t N> auto lifecycle_static(benchmark::State &state) -> void
{
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  for (auto _ : state) {
    auto steps = make_static_steps<Step>(seed, std::make_index_sequence<N>{});
    benchmark::DoNotOptimize(steps);
  }
  state.counters["steps"] = static_cast<double>(N);
}

template<class Step, std::size_t N, bool Reserve> auto lifecycle_dynamic(benchmark::State &state) -> void
{
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  vkexec_benchmark::allocation_counts total{};
  for (auto _ : state) {
    vkexec_benchmark::allocation_counter counter{};
    auto graph = make_dynamic_graph<Step>(N, Reserve, seed);
    auto const counts = counter.counts();
    total.allocations += counts.allocations;
    total.bytes += counts.bytes;
    benchmark::DoNotOptimize(graph);
  }
  report_allocations(state, total, N);
  state.counters["steps"] = static_cast<double>(N);
}

template<class Step, std::size_t N> auto record_static(benchmark::State &state) -> void
{
  auto steps = make_static_steps<Step>(reinterpret_cast<std::uintptr_t>(&state), std::make_index_sequence<N>{});
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    auto recorded = vkexec::detail::record_static_steps(ctx, VK_NULL_HANDLE, cleanup, steps);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(N);
}

template<class Step, std::size_t N> auto record_dynamic(benchmark::State &state) -> void
{
  auto graph = make_dynamic_graph<Step>(N, true, reinterpret_cast<std::uintptr_t>(&state));
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    auto recorded = vkexec::detail::record_dynamic_steps(ctx, VK_NULL_HANDLE, cleanup, graph.steps);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(N);
}

template<class Step, std::size_t N> auto build_record_static(benchmark::State &state) -> void
{
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  for (auto _ : state) {
    auto steps = make_static_steps<Step>(seed, std::make_index_sequence<N>{});
    auto recorded = vkexec::detail::record_static_steps(ctx, VK_NULL_HANDLE, cleanup, steps);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(N);
}

template<class Step, std::size_t N> auto build_record_dynamic(benchmark::State &state) -> void
{
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  for (auto _ : state) {
    auto graph = make_dynamic_graph<Step>(N, true, seed);
    auto recorded = vkexec::detail::record_dynamic_steps(ctx, VK_NULL_HANDLE, cleanup, graph.steps);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(N);
}

template<class Step, std::size_t InlineBytes, std::size_t N, bool Reserve>
auto sbo_lifecycle(benchmark::State &state) -> void
{
  using erased_step = vkexec::detail::basic_dynamic_pass_step<InlineBytes>;
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  vkexec_benchmark::allocation_counts total{};
  for (auto _ : state) {
    vkexec_benchmark::allocation_counter counter{};
    std::vector<erased_step> steps;
    if constexpr (Reserve) { steps.reserve(N); }
    for (std::size_t i = 0; i < N; ++i) { steps.emplace_back(make_step<Step>(seed + i)); }
    auto const counts = counter.counts();
    total.allocations += counts.allocations;
    total.bytes += counts.bytes;
    benchmark::DoNotOptimize(steps);
  }
  report_allocations(state, total, N);
  state.counters["step_bytes"] = static_cast<double>(sizeof(erased_step));
  state.counters["model_bytes"] = static_cast<double>(erased_step::template model_size<Step>());
  state.counters["model_align"] = static_cast<double>(erased_step::template model_alignment<Step>());
}

template<class Step, std::size_t InlineBytes, std::size_t N> auto sbo_record(benchmark::State &state) -> void
{
  using erased_step = vkexec::detail::basic_dynamic_pass_step<InlineBytes>;
  std::vector<erased_step> steps;
  steps.reserve(N);
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  for (std::size_t i = 0; i < N; ++i) { steps.emplace_back(make_step<Step>(seed + i)); }
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    for (auto &step : steps) {
      auto recorded = step.record(ctx, VK_NULL_HANDLE, cleanup);
      benchmark::DoNotOptimize(recorded);
    }
  }
  state.counters["steps"] = static_cast<double>(N);
  state.counters["step_bytes"] = static_cast<double>(sizeof(erased_step));
}

template<class Step, bool Reserve> auto compact_lifecycle(benchmark::State &state) -> void
{
  constexpr std::size_t count = 1000;
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  vkexec_benchmark::allocation_counts total{};
  for (auto _ : state) {
    vkexec_benchmark::allocation_counter counter{};
    std::vector<vkexec_benchmark::compact_step> steps;
    if constexpr (Reserve) { steps.reserve(count); }
    for (std::size_t i = 0; i < count; ++i) { steps.emplace_back(make_step<Step>(seed + i)); }
    auto const counts = counter.counts();
    total.allocations += counts.allocations;
    total.bytes += counts.bytes;
    benchmark::DoNotOptimize(steps);
  }
  report_allocations(state, total, count);
  state.counters["step_bytes"] = static_cast<double>(sizeof(vkexec_benchmark::compact_step));
  state.counters["model_bytes"] = static_cast<double>(sizeof(Step));
  state.counters["model_align"] = static_cast<double>(alignof(Step));
}

template<class Step> auto compact_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = 1000;
  std::vector<vkexec_benchmark::compact_step> steps;
  steps.reserve(count);
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  for (std::size_t i = 0; i < count; ++i) { steps.emplace_back(make_step<Step>(seed + i)); }
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    for (auto &step : steps) {
      auto recorded = step.record(ctx, VK_NULL_HANDLE, cleanup);
      benchmark::DoNotOptimize(recorded);
    }
  }
  state.counters["steps"] = static_cast<double>(count);
}

template<class Step> auto compact_build_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = 1000;
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    std::vector<vkexec_benchmark::compact_step> steps;
    steps.reserve(count);
    for (std::size_t i = 0; i < count; ++i) { steps.emplace_back(make_step<Step>(seed + i)); }
    for (auto &step : steps) {
      auto recorded = step.record(ctx, VK_NULL_HANDLE, cleanup);
      benchmark::DoNotOptimize(recorded);
    }
  }
  state.counters["steps"] = static_cast<double>(count);
}

template<class Step, bool Reserve> auto arena_lifecycle(benchmark::State &state) -> void
{
  constexpr std::size_t count = 1000;
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  vkexec_benchmark::allocation_counts total{};
  std::size_t upstream_allocations = 0;
  std::size_t upstream_bytes = 0;
  for (auto _ : state) {
    vkexec_benchmark::allocation_counter counter{};
    vkexec_benchmark::arena_graph graph;
    if constexpr (Reserve) { graph.reserve(count); }
    for (std::size_t i = 0; i < count; ++i) { graph.append(make_step<Step>(seed + i)); }
    auto const counts = counter.counts();
    total.allocations += counts.allocations;
    total.bytes += counts.bytes;
    upstream_allocations += graph.upstream_allocations();
    upstream_bytes += graph.upstream_bytes();
    benchmark::DoNotOptimize(graph.size());
  }
  report_allocations(state, total, count);
  auto const iterations = static_cast<double>(state.iterations());
  state.counters["upstream_allocs/graph"] = static_cast<double>(upstream_allocations) / iterations;
  state.counters["upstream_bytes/graph"] = static_cast<double>(upstream_bytes) / iterations;
  state.counters["step_bytes"] = static_cast<double>(vkexec_benchmark::arena_graph::step_bytes());
}

template<class Step> auto arena_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = 1000;
  vkexec_benchmark::arena_graph graph;
  graph.reserve(count);
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  for (std::size_t i = 0; i < count; ++i) { graph.append(make_step<Step>(seed + i)); }
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    auto recorded = graph.record(ctx, VK_NULL_HANDLE, cleanup);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(count);
}

template<class Step> auto arena_build_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = 1000;
  auto const seed = reinterpret_cast<std::uintptr_t>(&state);
  auto ctx = vkexec::detail::context_access::facade(nullptr);
  vkexec::detail::pass_cleanup cleanup{};
  for (auto _ : state) {
    vkexec_benchmark::arena_graph graph;
    graph.reserve(count);
    for (std::size_t i = 0; i < count; ++i) { graph.append(make_step<Step>(seed + i)); }
    auto recorded = graph.record(ctx, VK_NULL_HANDLE, cleanup);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(count);
}

#define VKEXEC_GRAPH_BENCHMARKS(Step, Label, N)                                                              \
  BENCHMARK_TEMPLATE(lifecycle_static, Step, N)->Name("lifecycle/static/" Label "/" #N);                     \
  BENCHMARK_TEMPLATE(lifecycle_dynamic, Step, N, true)->Name("lifecycle/dynamic/" Label "/" #N);             \
  BENCHMARK_TEMPLATE(lifecycle_dynamic, Step, N, false)->Name("lifecycle/dynamic_no_reserve/" Label "/" #N); \
  BENCHMARK_TEMPLATE(record_static, Step, N)->Name("record/static/" Label "/" #N);                           \
  BENCHMARK_TEMPLATE(record_dynamic, Step, N)->Name("record/dynamic/" Label "/" #N);                         \
  BENCHMARK_TEMPLATE(build_record_static, Step, N)->Name("build_record/static/" Label "/" #N);               \
  BENCHMARK_TEMPLATE(build_record_dynamic, Step, N)->Name("build_record/dynamic/" Label "/" #N)

VKEXEC_GRAPH_BENCHMARKS(noop_step, "noop", 1);
VKEXEC_GRAPH_BENCHMARKS(noop_step, "noop", 10);
VKEXEC_GRAPH_BENCHMARKS(noop_step, "noop", 100);
VKEXEC_GRAPH_BENCHMARKS(noop_step, "noop", 1000);
VKEXEC_GRAPH_BENCHMARKS(representative_step, "representative", 1);
VKEXEC_GRAPH_BENCHMARKS(representative_step, "representative", 10);
VKEXEC_GRAPH_BENCHMARKS(representative_step, "representative", 100);
VKEXEC_GRAPH_BENCHMARKS(representative_step, "representative", 1000);

#define VKEXEC_SBO_CAPACITY(Step, Label, Capacity)                                                                     \
  BENCHMARK_TEMPLATE(sbo_lifecycle, Step, Capacity, 1000, true)->Name("sbo_reserved/" Label "/" #Capacity "/1000");    \
  BENCHMARK_TEMPLATE(sbo_lifecycle, Step, Capacity, 1000, false)->Name("sbo_no_reserve/" Label "/" #Capacity "/1000"); \
  BENCHMARK_TEMPLATE(sbo_record, Step, Capacity, 1000)->Name("sbo_record/" Label "/" #Capacity "/1000")

#define VKEXEC_SBO_BENCHMARKS(Step, Label) \
  VKEXEC_SBO_CAPACITY(Step, Label, 0);     \
  VKEXEC_SBO_CAPACITY(Step, Label, 16);    \
  VKEXEC_SBO_CAPACITY(Step, Label, 32);    \
  VKEXEC_SBO_CAPACITY(Step, Label, 64)

VKEXEC_SBO_BENCHMARKS(noop_step, "noop");
VKEXEC_SBO_BENCHMARKS(representative_step, "representative");
VKEXEC_SBO_BENCHMARKS(oversized_step, "oversized");

#define VKEXEC_CANDIDATE_BENCHMARKS(Step, Label)                                                 \
  BENCHMARK_TEMPLATE(compact_lifecycle, Step, true)->Name("compact_reserved/" Label "/1000");    \
  BENCHMARK_TEMPLATE(compact_lifecycle, Step, false)->Name("compact_no_reserve/" Label "/1000"); \
  BENCHMARK_TEMPLATE(compact_record, Step)->Name("compact_record/" Label "/1000");               \
  BENCHMARK_TEMPLATE(compact_build_record, Step)->Name("compact_build_record/" Label "/1000");   \
  BENCHMARK_TEMPLATE(arena_lifecycle, Step, true)->Name("arena_reserved/" Label "/1000");        \
  BENCHMARK_TEMPLATE(arena_lifecycle, Step, false)->Name("arena_no_reserve/" Label "/1000");     \
  BENCHMARK_TEMPLATE(arena_record, Step)->Name("arena_record/" Label "/1000");                   \
  BENCHMARK_TEMPLATE(arena_build_record, Step)->Name("arena_build_record/" Label "/1000")

VKEXEC_CANDIDATE_BENCHMARKS(noop_step, "noop");
VKEXEC_CANDIDATE_BENCHMARKS(representative_step, "representative");
VKEXEC_CANDIDATE_BENCHMARKS(oversized_step, "oversized");

#undef VKEXEC_GRAPH_BENCHMARKS
#undef VKEXEC_SBO_CAPACITY
#undef VKEXEC_SBO_BENCHMARKS
#undef VKEXEC_CANDIDATE_BENCHMARKS

}// namespace
