#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP

//! \file
//! Descriptor-heap compute/graphics recording and pipeable compute pass closure.

#include <vkexec/pass.hpp>
#include <vkexec_extensions/descriptor_heap/push_data.hpp>

#include <span>

namespace vkexec {

/**
 * Records a bindless compute pass: bind pipeline, `cmd_push_data`, direct dispatch.
 *
 * @param ctx Context used to resolve push-data procs.
 * @param cmd Command buffer in the recording state.
 * @param bind Bindless compute bind (null layout/set).
 * @param push Host push-data bytes.
 * @param groups Workgroup counts.
 */
[[nodiscard]] auto record_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status;

/**
 * Records a bindless compute pass with an indirect dispatch.
 *
 * @param groups Buffer containing `VkDispatchIndirectCommand`.
 */
[[nodiscard]] auto record_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status;

/**
 * Records a bindless graphics draw: bind pipeline, set dynamic viewport/scissor, draw.
 *
 * Does not begin/end rendering and does not bind heaps (`cmd_bind_resource_heap` is
 * the caller's responsibility). Use inside `cmd_begin_rendering`…`cmd_end_rendering`.
 *
 * @param bind Null-layout graphics bind from `bind_compute` on a descriptor graphics bag.
 * @param extent Dynamic viewport and scissor extent.
 * @param vertex_count Vertices for `vkCmdDraw`.
 */
auto record_draw(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  VkExtent2D extent,
  std::uint32_t vertex_count) -> void;

/**
 * Records a bindless graphics draw from a `VkDrawIndirectCommand` buffer.
 *
 * @param buffer Buffer containing `VkDrawIndirectCommand`.
 * @param offset Byte offset into `buffer`.
 */
auto record_draw_indirect(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  VkExtent2D extent,
  VkBuffer buffer,
  VkDeviceSize offset) -> void;

[[deprecated("use record_draw(ctx, ...)")]] inline auto record_heap_draw(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  VkExtent2D extent,
  std::uint32_t vertex_count) -> void
{ record_draw(ctx, cmd, bind, extent, vertex_count); }

[[deprecated("use record_draw_indirect(ctx, ...)")]] inline auto record_heap_draw_indirect(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  VkExtent2D extent,
  VkBuffer buffer,
  VkDeviceSize offset) -> void
{ record_draw_indirect(ctx, cmd, bind, extent, buffer, offset); }

//! Embedder alias for descriptor-heap `record_pass` (bind + push-data + dispatch).
[[nodiscard]] inline auto record_bindless_compute_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{ return record_pass(ctx, cmd, bind, push, groups); }

//! Embedder alias for descriptor-heap indirect `record_pass`.
[[nodiscard]] inline auto record_bindless_compute_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{ return record_pass(ctx, cmd, bind, push, groups); }

/**
 * Wraps a prebuilt compute pass recorded with push-data instead of push constants.
 *
 * Pipe onto `schedule()` or a `pass_graph_sender` like `prebuilt_compute_pass_closure`.
 *
 * @see compute_pass
 */
struct descriptor_compute_pass_closure
{
  prebuilt_compute_pass_closure inner;
};

using heap_compute_pass_closure [[deprecated("use descriptor_compute_pass_closure")]] = descriptor_compute_pass_closure;

namespace detail {

  //! Wraps a prebuilt closure as a heap (push-data) `pass_step`.
  [[nodiscard]] auto make_descriptor_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step;

}// namespace detail

//! Starts a pass graph from `schedule()` with one bindless compute step.
[[nodiscard]] auto operator|(schedule_sender snd, descriptor_compute_pass_closure closure) -> pass_graph_sender;

//! Appends a bindless compute step to an existing pass graph.
[[nodiscard]] auto operator|(pass_graph_sender graph, descriptor_compute_pass_closure closure) -> pass_graph_sender;

[[deprecated("use record_pass(ctx, ...)")]] [[nodiscard]] inline auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{ return record_pass(ctx, cmd, bind, push, groups); }

[[deprecated("use record_pass(ctx, ...)")]] [[nodiscard]] inline auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{ return record_pass(ctx, cmd, bind, push, groups); }

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP
