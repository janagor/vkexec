#include "pass_graph_common.hpp"

namespace {

using vkexec_benchmark::noop_step;
using vkexec_benchmark::representative_step;
using vkexec_benchmark::make_static_steps;
using vkexec_benchmark::make_dynamic_graph;

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
    auto recorded = vkexec::detail::record_dynamic_steps(ctx, VK_NULL_HANDLE, cleanup, *graph.steps);
    benchmark::DoNotOptimize(recorded);
  }
  state.counters["steps"] = static_cast<double>(N);
}

#define VKEXEC_BUILD_RECORD_BENCHMARKS(Step, Label, N)                                         \
  BENCHMARK_TEMPLATE(build_record_static, Step, N)->Name("build_record/static/" Label "/" #N); \
  BENCHMARK_TEMPLATE(build_record_dynamic, Step, N)->Name("build_record/dynamic/" Label "/" #N)

VKEXEC_BUILD_RECORD_BENCHMARKS(noop_step, "noop", 5);
VKEXEC_BUILD_RECORD_BENCHMARKS(noop_step, "noop", 50);
VKEXEC_BUILD_RECORD_BENCHMARKS(noop_step, "noop", 500);
VKEXEC_BUILD_RECORD_BENCHMARKS(representative_step, "representative", 5);
VKEXEC_BUILD_RECORD_BENCHMARKS(representative_step, "representative", 50);
VKEXEC_BUILD_RECORD_BENCHMARKS(representative_step, "representative", 500);

#undef VKEXEC_BUILD_RECORD_BENCHMARKS

}// namespace
