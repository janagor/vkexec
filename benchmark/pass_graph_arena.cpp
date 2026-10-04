#include "pass_graph_common.hpp"
#include "representation_candidates.hpp"

#include <vector>

namespace {

using vkexec_benchmark::noop_step;
using vkexec_benchmark::representative_step;
using vkexec_benchmark::oversized_step;
using vkexec_benchmark::make_step;
using vkexec_benchmark::report_allocations;

template<class Step, std::size_t N, bool Reserve> auto arena_lifecycle(benchmark::State &state) -> void
{
  constexpr std::size_t count = N;
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

template<class Step, std::size_t N> auto arena_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = N;
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

template<class Step, std::size_t N> auto arena_build_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = N;
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

#define VKEXEC_ARENA_BENCHMARKS(Step, Label, N)                                                \
  BENCHMARK_TEMPLATE(arena_lifecycle, Step, N, true)->Name("arena_reserved/" Label "/" #N);    \
  BENCHMARK_TEMPLATE(arena_lifecycle, Step, N, false)->Name("arena_no_reserve/" Label "/" #N); \
  BENCHMARK_TEMPLATE(arena_record, Step, N)->Name("arena_record/" Label "/" #N);               \
  BENCHMARK_TEMPLATE(arena_build_record, Step, N)->Name("arena_build_record/" Label "/" #N)

VKEXEC_ARENA_BENCHMARKS(noop_step, "noop", 5);
VKEXEC_ARENA_BENCHMARKS(noop_step, "noop", 50);
VKEXEC_ARENA_BENCHMARKS(noop_step, "noop", 500);
VKEXEC_ARENA_BENCHMARKS(representative_step, "representative", 5);
VKEXEC_ARENA_BENCHMARKS(representative_step, "representative", 50);
VKEXEC_ARENA_BENCHMARKS(representative_step, "representative", 500);
VKEXEC_ARENA_BENCHMARKS(oversized_step, "oversized", 5);
VKEXEC_ARENA_BENCHMARKS(oversized_step, "oversized", 50);
VKEXEC_ARENA_BENCHMARKS(oversized_step, "oversized", 500);

#undef VKEXEC_ARENA_BENCHMARKS

}// namespace
