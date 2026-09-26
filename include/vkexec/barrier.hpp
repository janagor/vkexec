#ifndef VKEXEC_BARRIER_HPP
#define VKEXEC_BARRIER_HPP

//! \file
//! Global memory and image barriers, plus pipeable stage presets for pass graphs.

#include <vkexec/pass.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <utility>

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

  //! Transfer writes -> compute shader reads/writes.
  struct transfer_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<transfer_to_compute_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<transfer_to_compute_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Compute -> compute (shader write to shader read/write).
  struct compute_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<compute_to_compute_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<compute_to_compute_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Compute -> graphics (shader write to vertex/fragment read).
  struct compute_to_graphics_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<compute_to_graphics_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<compute_to_graphics_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Graphics -> compute.
  struct graphics_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<graphics_to_compute_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<graphics_to_compute_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Compute shader write -> compute shader read (read-after-write).
  struct compute_read_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<compute_read_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<compute_read_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr transfer_to_compute_t transfer_to_compute{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr compute_to_compute_t compute_to_compute{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr compute_to_graphics_t compute_to_graphics{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr graphics_to_compute_t graphics_to_compute{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr compute_read_t compute_read{};

}// namespace barrier

namespace detail {

  template<class Tag, class Env>
    requires std::same_as<Tag, barrier::transfer_to_compute_t> || std::same_as<Tag, barrier::compute_to_compute_t>
             || std::same_as<Tag, barrier::compute_to_graphics_t> || std::same_as<Tag, barrier::graphics_to_compute_t>
             || std::same_as<Tag, barrier::compute_read_t>
  [[nodiscard]] auto lower_vkexec_pass_step(Tag tag, empty_data /*data*/, Env const & /*env*/) -> barrier_step<Tag>
  { return make_barrier_step(std::move(tag)); }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_BARRIER_HPP
