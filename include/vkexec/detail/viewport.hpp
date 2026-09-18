#ifndef VKEXEC_DETAIL_VIEWPORT_HPP
#define VKEXEC_DETAIL_VIEWPORT_HPP

#include <vulkan/vulkan_core.h>

namespace vkexec::detail {

inline auto set_dynamic_viewport_scissor(VkCommandBuffer cmd, VkExtent2D extent) -> void
{
  VkViewport viewport{};
  viewport.width = static_cast<float>(extent.width);
  viewport.height = static_cast<float>(extent.height);
  viewport.maxDepth = 1.0F;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.extent = extent;
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_VIEWPORT_HPP
