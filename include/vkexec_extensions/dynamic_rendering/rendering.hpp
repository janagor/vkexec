#ifndef VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_RENDERING_HPP
#define VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_RENDERING_HPP

#include <vkexec/result.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

namespace vkexec {

struct color_attachment
{
  VkImageView view{ VK_NULL_HANDLE };
  VkImageLayout layout{ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
  VkAttachmentLoadOp load_op{ VK_ATTACHMENT_LOAD_OP_CLEAR };
  VkAttachmentStoreOp store_op{ VK_ATTACHMENT_STORE_OP_STORE };
  VkClearValue clear{};
};

struct depth_attachment
{
  VkImageView view{ VK_NULL_HANDLE };
  VkImageLayout layout{ VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL };
  VkAttachmentLoadOp load_op{ VK_ATTACHMENT_LOAD_OP_CLEAR };
  VkAttachmentStoreOp store_op{ VK_ATTACHMENT_STORE_OP_DONT_CARE };
  VkClearValue clear{};
};

struct rendering_info
{
  VkExtent2D extent{};
  std::span<color_attachment const> color;
  depth_attachment const *depth{ nullptr };
  std::uint32_t layer_count{ 1 };
};

[[nodiscard]] auto cmd_begin_rendering(VkCommandBuffer cmd, rendering_info const &info) -> status;
[[nodiscard]] auto cmd_end_rendering(VkCommandBuffer cmd) -> status;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_RENDERING_HPP
