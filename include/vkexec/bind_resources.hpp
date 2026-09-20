#ifndef VKEXEC_BIND_RESOURCES_HPP
#define VKEXEC_BIND_RESOURCES_HPP

//! \file
//! Pipeable resource-table lowering and descriptor binding.

#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/scheduler.hpp>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

//! Descriptor-set resource binding step for a pass graph.
struct bind_resources_closure
{
  pipeline_resources const *pipe{ nullptr };
  resource_table table;
  std::vector<std::byte> push;
};

//! Builds a descriptor-set-backed resource-table graph step.
[[nodiscard]] auto bind_resources(pipeline_resources const &pipe,
  resource_table const &table,
  std::span<std::byte const> push = {}) -> bind_resources_closure;

//! Typed push-data overload for descriptor-set resource binding.
template<class Params>
  requires std::is_trivially_copyable_v<Params>
           && (!std::same_as<std::remove_cvref_t<Params>, std::span<std::byte const>>)
[[nodiscard]] auto bind_resources(pipeline_resources const &pipe, resource_table const &table, Params const &params)
  -> bind_resources_closure
{ return bind_resources(pipe, table, std::as_bytes(std::span{ &params, 1 })); }

[[nodiscard]] auto operator|(schedule_sender snd, bind_resources_closure closure) -> pass_graph_sender;
[[nodiscard]] auto operator|(pass_graph_sender graph, bind_resources_closure closure) -> pass_graph_sender;

}// namespace vkexec

#endif// VKEXEC_BIND_RESOURCES_HPP
