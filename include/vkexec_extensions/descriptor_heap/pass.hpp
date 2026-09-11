#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP

//! \file
//! Bindless compute pass recording and pipeable `heap_compute_pass_closure`.

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
[[nodiscard]] auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status;

/**
 * Records a bindless compute pass with an indirect dispatch.
 *
 * @param groups Buffer containing `VkDispatchIndirectCommand`.
 */
[[nodiscard]] auto record_heap_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status;

//! Embedder alias for bindless `record_heap_pass` (bind + push-data + dispatch).
[[nodiscard]] inline auto record_bindless_compute_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{ return record_heap_pass(ctx, cmd, bind, push, groups); }

//! Embedder alias for bindless indirect `record_heap_pass`.
[[nodiscard]] inline auto record_bindless_compute_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{ return record_heap_pass(ctx, cmd, bind, push, groups); }

/**
 * Wraps a prebuilt compute pass recorded with push-data instead of push constants.
 *
 * Pipe onto `schedule()` or a `pass_graph_sender` like `prebuilt_compute_pass_closure`.
 *
 * @see compute_heap_pass
 */
struct heap_compute_pass_closure
{
  prebuilt_compute_pass_closure inner;
};

namespace detail {

  //! Wraps a prebuilt closure as a heap (push-data) `pass_step`.
  [[nodiscard]] auto make_heap_prebuilt_step(prebuilt_compute_pass_closure closure) -> pass_step;

}// namespace detail

//! Starts a pass graph from `schedule()` with one bindless compute step.
[[nodiscard]] auto operator|(schedule_sender snd, heap_compute_pass_closure closure) -> pass_graph_sender;

//! Appends a bindless compute step to an existing pass graph.
[[nodiscard]] auto operator|(pass_graph_sender graph, heap_compute_pass_closure closure) -> pass_graph_sender;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PASS_HPP
