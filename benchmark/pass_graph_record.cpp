#include "pass_graph_common.hpp"

namespace {

using vkexec_benchmark::noop_step;
using vkexec_benchmark::representative_step;
using vkexec_benchmark::make_static_steps;
using vkexec_benchmark::make_dynamic_graph;

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
    auto recorded = vkexec::detail::record_dynamic_steps(ctx, VK_NULL_HANDLE, cleanup, *graph.steps);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(N);
}

#define VKEXEC_RECORD_BENCHMARKS(Step, Label, N)                                   \
  BENCHMARK_TEMPLATE(record_static, Step, N)->Name("record/static/" Label "/" #N); \
  BENCHMARK_TEMPLATE(record_dynamic, Step, N)->Name("record/dynamic/" Label "/" #N)

VKEXEC_RECORD_BENCHMARKS(noop_step, "noop", 5);
VKEXEC_RECORD_BENCHMARKS(noop_step, "noop", 50);
VKEXEC_RECORD_BENCHMARKS(noop_step, "noop", 500);
VKEXEC_RECORD_BENCHMARKS(representative_step, "representative", 5);
VKEXEC_RECORD_BENCHMARKS(representative_step, "representative", 50);
VKEXEC_RECORD_BENCHMARKS(representative_step, "representative", 500);

#undef VKEXEC_RECORD_BENCHMARKS

}// namespace
