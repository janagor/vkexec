#include "pass_graph_common.hpp"

namespace {

using vkexec_benchmark::noop_step;
using vkexec_benchmark::representative_step;
using vkexec_benchmark::make_static_steps;
using vkexec_benchmark::make_dynamic_graph;
using vkexec_benchmark::report_allocations;

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

#define VKEXEC_LIFECYCLE_BENCHMARKS(Step, Label, N)                                              \
  BENCHMARK_TEMPLATE(lifecycle_static, Step, N)->Name("lifecycle/static/" Label "/" #N);         \
  BENCHMARK_TEMPLATE(lifecycle_dynamic, Step, N, true)->Name("lifecycle/dynamic/" Label "/" #N); \
  BENCHMARK_TEMPLATE(lifecycle_dynamic, Step, N, false)->Name("lifecycle/dynamic_no_reserve/" Label "/" #N)

VKEXEC_LIFECYCLE_BENCHMARKS(noop_step, "noop", 5);
VKEXEC_LIFECYCLE_BENCHMARKS(noop_step, "noop", 50);
VKEXEC_LIFECYCLE_BENCHMARKS(noop_step, "noop", 500);
VKEXEC_LIFECYCLE_BENCHMARKS(representative_step, "representative", 5);
VKEXEC_LIFECYCLE_BENCHMARKS(representative_step, "representative", 50);
VKEXEC_LIFECYCLE_BENCHMARKS(representative_step, "representative", 500);

#undef VKEXEC_LIFECYCLE_BENCHMARKS

}// namespace
