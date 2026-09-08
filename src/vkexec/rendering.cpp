#include <vkexec/rendering.hpp>

#include <vkexec/error.hpp>
#include <vkexec/detail/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <vector>

namespace vkexec {

auto cmd_begin_rendering(VkCommandBuffer cmd, rendering_info const &info) -> detail::status
{
  if (cmd == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "cmd_begin_rendering requires a command buffer");
  }
  if (info.extent.width == 0 || info.extent.height == 0) {
    return fail(errc::invalid_argument, "cmd_begin_rendering requires a non-zero extent");
  }
  if (info.color.empty() && info.depth == nullptr) {
    return fail(errc::invalid_argument, "cmd_begin_rendering requires at least one attachment");
  }

  std::vector<VkRenderingAttachmentInfo> color_infos;
  color_infos.reserve(info.color.size());
  for (color_attachment const &attachment : info.color) {
    if (attachment.view == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "color attachment view is null");
    }
    VkRenderingAttachmentInfo color{};
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = attachment.view;
    color.imageLayout = attachment.layout;
    color.loadOp = attachment.load_op;
    color.storeOp = attachment.store_op;
    color.clearValue = attachment.clear;
    color_infos.push_back(color);
  }

  VkRenderingAttachmentInfo depth_info{};
  depth_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  if (info.depth != nullptr) {
    if (info.depth->view == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "depth attachment view is null");
    }
    depth_info.imageView = info.depth->view;
    depth_info.imageLayout = info.depth->layout;
    depth_info.loadOp = info.depth->load_op;
    depth_info.storeOp = info.depth->store_op;
    depth_info.clearValue = info.depth->clear;
  }

  VkRenderingInfo rendering{};
  rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  rendering.renderArea.offset = { .x = 0, .y = 0 };
  rendering.renderArea.extent = info.extent;
  rendering.layerCount = info.layer_count == 0 ? 1U : info.layer_count;
  rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_infos.size());
  rendering.pColorAttachments = color_infos.empty() ? nullptr : color_infos.data();
  rendering.pDepthAttachment = info.depth != nullptr ? &depth_info : nullptr;

  vkCmdBeginRendering(cmd, &rendering);
  return {};
}

auto cmd_end_rendering(VkCommandBuffer cmd) -> detail::status
{
  if (cmd == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "cmd_end_rendering requires a command buffer");
  }
  vkCmdEndRendering(cmd);
  return {};
}

}// namespace vkexec
