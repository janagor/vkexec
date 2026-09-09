#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP

#include <vkexec/pass.hpp>
#include <vkexec_extensions/descriptor_heap/push_data.hpp>

#include <span>

namespace vkexec {

[[nodiscard]] auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status;

[[nodiscard]] auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status;

/// Embedder alias for bindless `record_heap_pass` (bind + push-data + dispatch).
[[nodiscard]] inline auto record_bindless_compute_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{ return record_heap_pass(ctx, cmd, bind, push, groups); }

[[nodiscard]] inline auto record_bindless_compute_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{ return record_heap_pass(ctx, cmd, bind, push, groups); }

/// Wraps a prebuilt compute pass recorded with push-data instead of push constants.
struct heap_compute_pass_closure
{
  prebuilt_compute_pass_closure inner;
};

namespace detail {

  [[nodiscard]] auto make_heap_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step;

}// namespace detail

[[nodiscard]] auto operator|(schedule_sender snd, heap_compute_pass_closure closure) -> pass_graph_sender;

[[nodiscard]] auto operator|(pass_graph_sender graph, heap_compute_pass_closure closure) -> pass_graph_sender;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP
