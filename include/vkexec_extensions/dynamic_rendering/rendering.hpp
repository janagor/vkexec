#ifndef VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_RENDERING_HPP
#define VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_RENDERING_HPP

//! \file
//! Helpers for `vkCmdBeginRendering` / `vkCmdEndRendering` (dynamic rendering).

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace vkexec {

//! Color attachment description for `rendering_info`.
struct color_attachment
{
  VkImageView view{ VK_NULL_HANDLE };
  VkImageLayout layout{ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
  VkAttachmentLoadOp load_op{ VK_ATTACHMENT_LOAD_OP_CLEAR };
  VkAttachmentStoreOp store_op{ VK_ATTACHMENT_STORE_OP_STORE };
  VkClearValue clear{};
};

//! Optional depth attachment description for `rendering_info`.
struct depth_attachment
{
  VkImageView view{ VK_NULL_HANDLE };
  VkImageLayout layout{ VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL };
  VkAttachmentLoadOp load_op{ VK_ATTACHMENT_LOAD_OP_CLEAR };
  VkAttachmentStoreOp store_op{ VK_ATTACHMENT_STORE_OP_DONT_CARE };
  VkClearValue clear{};
};

/**
 * Parameters for beginning a dynamic rendering pass.
 *
 * `depth` may be null when no depth attachment is used.
 *
 * @see cmd_begin_rendering, cmd_end_rendering
 */
struct rendering_info
{
  VkExtent2D extent{};
  std::span<color_attachment const> color;
  depth_attachment const *depth{ nullptr };
  std::uint32_t layer_count{ 1 };
};

/**
 * Begins dynamic rendering on `cmd` with the given attachments.
 *
 * Requires dynamic rendering support on the device.
 *
 * @param cmd Command buffer in the recording state.
 * @param info Extent and color/depth attachments.
 */
[[nodiscard]] auto cmd_begin_rendering(VkCommandBuffer cmd, rendering_info const &info) -> status;

//! Ends the dynamic rendering pass previously begun on `cmd`.
[[nodiscard]] auto cmd_end_rendering(VkCommandBuffer cmd) -> status;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_RENDERING_HPP
