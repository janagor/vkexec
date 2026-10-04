#include "pass_graph_common.hpp"
#include "representation_candidates.hpp"

#include <vector>

namespace {

using vkexec_benchmark::noop_step;
using vkexec_benchmark::representative_step;
using vkexec_benchmark::oversized_step;
using vkexec_benchmark::make_step;
using vkexec_benchmark::report_allocations;

template<class Step, std::size_t N, bool Reserve> auto compact_lifecycle(benchmark::State &state) -> void
{
  constexpr std::size_t count = N;
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

template<class Step, std::size_t N> auto compact_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = N;
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

template<class Step, std::size_t N> auto compact_build_record(benchmark::State &state) -> void
{
  constexpr std::size_t count = N;
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

#define VKEXEC_COMPACT_BENCHMARKS(Step, Label, N)                                                  \
  BENCHMARK_TEMPLATE(compact_lifecycle, Step, N, true)->Name("compact_reserved/" Label "/" #N);    \
  BENCHMARK_TEMPLATE(compact_lifecycle, Step, N, false)->Name("compact_no_reserve/" Label "/" #N); \
  BENCHMARK_TEMPLATE(compact_record, Step, N)->Name("compact_record/" Label "/" #N);               \
  BENCHMARK_TEMPLATE(compact_build_record, Step, N)->Name("compact_build_record/" Label "/" #N)

VKEXEC_COMPACT_BENCHMARKS(noop_step, "noop", 5);
VKEXEC_COMPACT_BENCHMARKS(noop_step, "noop", 50);
VKEXEC_COMPACT_BENCHMARKS(noop_step, "noop", 500);
VKEXEC_COMPACT_BENCHMARKS(representative_step, "representative", 5);
VKEXEC_COMPACT_BENCHMARKS(representative_step, "representative", 50);
VKEXEC_COMPACT_BENCHMARKS(representative_step, "representative", 500);
VKEXEC_COMPACT_BENCHMARKS(oversized_step, "oversized", 5);
VKEXEC_COMPACT_BENCHMARKS(oversized_step, "oversized", 50);
VKEXEC_COMPACT_BENCHMARKS(oversized_step, "oversized", 500);

#undef VKEXEC_COMPACT_BENCHMARKS

}// namespace
