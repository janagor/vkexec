#ifndef VKEXEC_GRAPHICS_PASS_HPP
#define VKEXEC_GRAPHICS_PASS_HPP

//! \file
//! Graphics draw steps with inferred attachment and mesh resource uses.

#include <vkexec/pass.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>

#include <vulkan/vulkan.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace vkexec {

//! One framebuffer attachment. Access includes loadOp, depth testing, and blending reads.
//! The caller supplies the render pass's finalLayout; graph-entry layout is independent.
struct graphics_attachment
{
  VkImage image{ VK_NULL_HANDLE };
  VkImageSubresourceRange range{};
  //! Layout before the first graph use.
  VkImageLayout initial_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  //! VkAttachmentDescription::initialLayout, before the render pass's implicit transition.
  std::optional<VkImageLayout> render_pass_initial_layout;
  //! VkAttachmentDescription::finalLayout, established by vkCmdEndRenderPass.
  VkImageLayout final_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  //! Includes render-pass load and blending reads when applicable.
  resource_access access{ resource_access::write };
};

//! Render-pass handles and the underlying images touched by one framebuffer.
struct graphics_target
{
  VkRenderPass render_pass{ VK_NULL_HANDLE };
  VkFramebuffer framebuffer{ VK_NULL_HANDLE };
  VkExtent2D extent{};
  std::vector<graphics_attachment> colors;
  std::optional<graphics_attachment> depth;
};

namespace detail {

  [[nodiscard]] inline auto declare_graphics_attachment(resource_state_tracker &tracker,
    graphics_attachment const &attachment,
    image_usage usage,
    std::size_t step,
    queue_ref queue) -> status
  {
    if (!attachment.render_pass_initial_layout || attachment.final_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      return fail(errc::invalid_argument, "graphics attachment requires render-pass initial and final layouts");
    }
    auto const use = image_use{ .image = attachment.image,
      .range = attachment.range,
      .usage = usage,
      .access = attachment.access,
      .initial_layout = attachment.initial_layout,
      .final_layout = attachment.final_layout };
    return tracker.use_attachment(use, *attachment.render_pass_initial_layout, step, queue.family, queue.queue);
  }

  template<class Draw> struct graphics_pass_step
  {
    graphics_target target;
    graphics_bind bind;
    Draw draw;

    [[nodiscard]] auto has_resource_declarations() const noexcept -> bool
    {
      if (!bind.resource_metadata || !bind.complete_resource_metadata
          || target.colors.size() != bind.config.color_attachment_count || (bind.config.depth_test && !target.depth)) {
        return false;
      }
      for (auto const &color : target.colors) {
        if (!color.render_pass_initial_layout || color.final_layout == VK_IMAGE_LAYOUT_UNDEFINED) { return false; }
      }
      return !target.depth
             || (target.depth->render_pass_initial_layout && target.depth->final_layout != VK_IMAGE_LAYOUT_UNDEFINED);
    }

    auto declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) const -> status
    {
      if (!bind.resource_metadata || !bind.complete_resource_metadata) {
        return fail(errc::invalid_argument, "graphics binding lacks resource or layout metadata");
      }
      if (target.colors.size() != bind.config.color_attachment_count) {
        return fail(errc::invalid_argument, "graphics target color images must match the pipeline attachment count");
      }
      if (bind.config.depth_test && !target.depth) {
        return fail(errc::invalid_argument, "depth-tested graphics pass requires depth attachment metadata");
      }
      VKEXEC_TRY(declare_bound_resources(tracker, step, queue));
      VKEXEC_TRY(declare_attachments(tracker, step, queue));
      VKEXEC_TRY(declare_mesh_resources(tracker, step, queue));
      return {};
    }

    auto declare_bound_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) const -> status
    {
      for (auto const &image : bind.images) { VKEXEC_TRY(tracker.use(image, step, queue.family, queue.queue)); }
      for (auto const &buffer : bind.buffers) { VKEXEC_TRY(tracker.use(buffer, step, queue.family, queue.queue)); }
      return {};
    }

    auto declare_attachments(resource_state_tracker &tracker, std::size_t step, queue_ref queue) const -> status
    {
      for (auto const &color : target.colors) {
        VKEXEC_TRY(declare_graphics_attachment(tracker, color, image_usage::color_attachment, step, queue));
      }
      if (target.depth) {
        VKEXEC_TRY(declare_graphics_attachment(tracker, *target.depth, image_usage::depth_attachment, step, queue));
      }
      return {};
    }

    auto declare_mesh_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) const -> status
    {
      if constexpr (std::same_as<Draw, mesh_draw>) {
        VKEXEC_TRY(tracker.use(read(draw.vertex_buffer, buffer_usage::vertex), step, queue.family, queue.queue));
        VKEXEC_TRY(tracker.use(read(draw.index_buffer, buffer_usage::index), step, queue.family, queue.queue));
      }
      return {};
    }

    auto record(context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status
    {
      record_draw_pass(cmd, target.render_pass, target.framebuffer, target.extent, bind.config, bind, draw);
      return {};
    }
  };

  template<class Draw>
  using graphics_pass_closure = expr_closure<raw_pass_step_t, raw_pass_step_data<graphics_pass_step<Draw>>>;

}// namespace detail

//! Adds a graphics draw. Bind descriptor metadata and target attachments are required.
[[nodiscard]] inline auto graphics_pass(graphics_target target, graphics_bind bind, std::uint32_t vertex_count)
  -> detail::graphics_pass_closure<std::uint32_t>
{
  return make_pass_adaptor(detail::graphics_pass_step<std::uint32_t>{
    .target = std::move(target), .bind = std::move(bind), .draw = vertex_count });
}

//! Adds an indexed graphics draw with inferred vertex and index buffer reads.
[[nodiscard]] inline auto graphics_pass(graphics_target target, graphics_bind bind, mesh_draw draw)
  -> detail::graphics_pass_closure<mesh_draw>
{
  return make_pass_adaptor(
    detail::graphics_pass_step<mesh_draw>{ .target = std::move(target), .bind = std::move(bind), .draw = draw });
}

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_PASS_HPP
