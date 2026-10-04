#ifndef VKEXEC_BENCHMARK_PASS_GRAPH_COMMON_HPP
#define VKEXEC_BENCHMARK_PASS_GRAPH_COMMON_HPP

#include "allocation_counter.hpp"

#include <vkexec/pass.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace vkexec_benchmark {

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

inline auto report_allocations(benchmark::State &state, vkexec_benchmark::allocation_counts counts, std::size_t steps)
  -> void
{
  auto const iterations = static_cast<double>(state.iterations());
  state.counters["allocs/graph"] = static_cast<double>(counts.allocations) / iterations;
  state.counters["bytes/graph"] = static_cast<double>(counts.bytes) / iterations;
  state.counters["allocs/step"] = static_cast<double>(counts.allocations) / (iterations * static_cast<double>(steps));
}

}// namespace vkexec_benchmark

#endif// VKEXEC_BENCHMARK_PASS_GRAPH_COMMON_HPP
