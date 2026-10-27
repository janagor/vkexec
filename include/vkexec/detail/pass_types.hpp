#ifndef VKEXEC_DETAIL_PASS_TYPES_HPP
#define VKEXEC_DETAIL_PASS_TYPES_HPP

#include <vkexec/barrier_params.hpp>
#include <vkexec/config.hpp>
#include <vkexec/detail/attributes.hpp>
#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/detail/sender_expr.hpp>
#include <vkexec/detail/submission_id.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/presentation_abort_state.hpp>
#include <vkexec/push.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_scope.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

//! An unset affinity uses the graph's default queue during execution planning.
using queue_affinity = std::optional<queue_ref>;

//! Queue and identity of the execution batch currently recording a pass.
//! `submission_id` identifies a submission attempt, including attempts that
//! fail before queue submission. IDs are process-local and are not persistent.
//! IDs are allocated lazily. A step that needs an identity must call
//! `ensure_submission_id()` before copying this structure for `after_gpu()`;
//! otherwise its copy retains the zero, unassigned ID. The reference passed
//! to `record()` is valid only for that call.
struct pass_recording_info
{
  queue_ref queue{};
  mutable std::uint64_t submission_id{};
  mutable std::uint32_t next_region_index{};

  [[nodiscard]] auto ensure_submission_id() const noexcept -> std::uint64_t
  {
    if (submission_id == 0) { submission_id = detail::allocate_submission_id(); }
    return submission_id;
  }

  [[nodiscard]] auto allocate_region_index() const noexcept -> std::uint32_t { return next_region_index++; }
};

//! Synchronization borrowed from a presenter for one graph execution.
struct graph_presentation_sync
{
  semaphore_submit image_available_wait;
  semaphore_submit render_finished_signal;
  VkFence fence{ VK_NULL_HANDLE };
};

//! External swapchain boundary attached to a complete pass graph.
struct graph_presentation
{
  queue_ref queue;
  VkImage image{ VK_NULL_HANDLE };
  std::function<result<graph_presentation_sync>()> prepare;
  std::function<status()> present;
  std::function<status(presentation_abort_state)> abort;
  graph_presentation_sync sync;
};

//! Workgroup counts for `vkCmdDispatch` (X/Y/Z).
struct dispatch
{
  std::uint32_t x{ 1 };
  std::uint32_t y{ 1 };
  std::uint32_t z{ 1 };
};

/**
 * Computes workgroup count X covering `work_count` invocations for local size `local_x`.
 *
 * Y and Z remain 1.
 *
 * @pre local_x > 0
 */
[[nodiscard]] constexpr auto dispatch_groups_for(std::uint32_t work_count, std::uint32_t local_x) noexcept -> dispatch
{ return dispatch{ .x = (work_count / local_x) + static_cast<std::uint32_t>(work_count % local_x != 0U) }; }

//! Arguments for `vkCmdDispatchIndirect`.
struct indirect_dispatch
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{ 0 };
};

//! Pipeline, layout, and optional descriptor set for one compute dispatch.
struct compute_bind
{
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkPipelineLayout layout{ VK_NULL_HANDLE };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  std::vector<image_use> images;
  std::vector<buffer_use> buffers;
  resource_metadata_state metadata{ resource_metadata_state::none };

  //! The caller must include every resource the shader may access, including every
  //! potentially indexed bindless entry and device-address target. Keep them alive
  //! and do not update their descriptors until the GPU has finished using them.
  [[nodiscard]] auto with_resources(std::vector<image_use> declared_images,
    std::vector<buffer_use> declared_buffers) const -> compute_bind
  {
    auto result = *this;
    result.images = std::move(declared_images);
    result.buffers = std::move(declared_buffers);
    result.metadata = resource_metadata_state::complete;
    return result;
  }
};

//! Builds a `compute_bind` from pipeline resources and an optional set.
//! Raw descriptor sets and descriptor-heap binds have no resource metadata;
//! graph synchronization cannot infer their descriptor accesses.
[[nodiscard]] auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set = VK_NULL_HANDLE)
  -> compute_bind;

//! Builds a binding that retains the descriptor resources for graph planning.
[[nodiscard]] auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set, resource_table const &table)
  -> compute_bind;

//! Returns workgroup counts covering `work_count` invocations along X.
[[nodiscard]] auto groups_for(handles::compute_pipeline const &pipe, std::uint32_t work_count) noexcept -> dispatch;

/**
 * Records bind, optional push constants, and a direct dispatch on `cmd`.
 *
 * @param cmd Command buffer in the recording state.
 * @param bind Pipeline / layout / descriptor set.
 * @param push Push-constant bytes (may be null when `push_bytes` is 0).
 * @param push_bytes Size of the push-constant blob.
 * @param groups Workgroup counts for `vkCmdDispatch`.
 */
auto record_pass(VkCommandBuffer cmd,
  compute_bind const &bind,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void;

/**
 * Records bind, optional push constants, and an indirect dispatch on `cmd`.
 *
 * @param groups Buffer and offset containing `VkDispatchIndirectCommand`.
 */
auto record_pass(VkCommandBuffer cmd,
  compute_bind const &bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void;

//! Convenience overload that builds a `compute_bind` from `pipe` and `set`.
auto record_pass(VkCommandBuffer cmd,
  handles::compute_pipeline const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void;

} // namespace vkexec

#endif// VKEXEC_DETAIL_PASS_TYPES_HPP
