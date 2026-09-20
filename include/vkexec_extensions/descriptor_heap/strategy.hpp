#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP

#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_extensions/descriptor_heap/resource_table.hpp>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

struct descriptor_heap_t
{
};
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr descriptor_heap_t descriptor_heap{};

//! Descriptor-heap resource binding step for a pass graph.
struct descriptor_heap_bind_resources_closure
{
  pipeline_resources const *pipe{ nullptr };
  resource_table table;
  heap_table_lower_env env;
  std::vector<std::byte> push;
};

[[nodiscard]] auto bind_resources(descriptor_heap_t /*strategy*/,
  pipeline_resources const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const> push = {}) -> descriptor_heap_bind_resources_closure;

template<class Params>
  requires std::is_trivially_copyable_v<Params>
[[nodiscard]] auto bind_resources(descriptor_heap_t strategy,
  pipeline_resources const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  Params const &params) -> descriptor_heap_bind_resources_closure
{ return bind_resources(strategy, pipe, table, env, std::as_bytes(std::span{ &params, 1 })); }

[[nodiscard]] auto operator|(schedule_sender snd, descriptor_heap_bind_resources_closure closure) -> pass_graph_sender;
[[nodiscard]] auto operator|(pass_graph_sender graph, descriptor_heap_bind_resources_closure closure)
  -> pass_graph_sender;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
