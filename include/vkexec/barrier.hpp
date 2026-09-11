#ifndef VKEXEC_BARRIER_HPP
#define VKEXEC_BARRIER_HPP

//! \file
//! Global memory and image barriers, plus pipeable stage presets for pass graphs.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

//! Parameters for a global `VkMemoryBarrier` recorded with `memory_barrier`.
struct memory_barrier_params
{
  VkPipelineStageFlags src_stage{};
  VkPipelineStageFlags dst_stage{};
  VkAccessFlags src_access{};
  VkAccessFlags dst_access{};
};

/**
 * Records a global memory barrier on `cmd`.
 *
 * @param cmd Command buffer in the recording state.
 * @param params Source/destination stages and access masks.
 */
auto memory_barrier(VkCommandBuffer cmd, memory_barrier_params params) -> void;

//! Parameters for a single-image layout transition recorded with `image_barrier`.
struct image_barrier_params
{
  VkImage image{ VK_NULL_HANDLE };
  VkImageAspectFlags aspect{ VK_IMAGE_ASPECT_COLOR_BIT };
  VkImageLayout old_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  VkImageLayout new_layout{ VK_IMAGE_LAYOUT_GENERAL };
  VkPipelineStageFlags src_stage{ VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };
  VkPipelineStageFlags dst_stage{ VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
  VkAccessFlags src_access{ 0 };
  VkAccessFlags dst_access{ 0 };
};

/**
 * Records an image memory barrier / layout transition on `cmd`.
 *
 * @param cmd Command buffer in the recording state.
 * @param params Image, layouts, and stage/access masks.
 */
auto image_barrier(VkCommandBuffer cmd, image_barrier_params params) -> void;

/**
 * Pipeable barrier presets for chaining between compute/graphics pass adaptors.
 *
 * Tag types are invoked as callables on a command buffer, or used as
 * `| barrier::compute_to_compute()` style steps in a pass graph.
 */
namespace barrier {

  //! Casts bit flags to `VkFlags` for designated-init call sites.
  [[nodiscard]] inline auto flags(std::uint32_t bits) -> VkFlags { return static_cast<VkFlags>(bits); }

  //! Transfer writes → compute shader reads/writes.
  struct transfer_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
  };

  //! Compute → compute (shader write to shader read/write).
  struct compute_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
  };

  //! Compute → graphics (shader write to vertex/fragment read).
  struct compute_to_graphics_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
  };

  //! Graphics → compute.
  struct graphics_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
  };

  //! Compute shader write → compute shader read (read-after-write).
  struct compute_read_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
  };

  [[nodiscard]] inline auto transfer_to_compute() -> transfer_to_compute_t { return {}; }
  [[nodiscard]] inline auto compute_to_compute() -> compute_to_compute_t { return {}; }
  [[nodiscard]] inline auto compute_to_graphics() -> compute_to_graphics_t { return {}; }
  [[nodiscard]] inline auto graphics_to_compute() -> graphics_to_compute_t { return {}; }
  [[nodiscard]] inline auto compute_read() -> compute_read_t { return {}; }

  auto transfer_to_compute(VkCommandBuffer cmd) -> void;
  auto compute_to_compute(VkCommandBuffer cmd) -> void;
  auto compute_to_graphics(VkCommandBuffer cmd) -> void;
  auto graphics_to_compute(VkCommandBuffer cmd) -> void;
  auto compute_read(VkCommandBuffer cmd) -> void;

}// namespace barrier

}// namespace vkexec

#endif// VKEXEC_BARRIER_HPP
