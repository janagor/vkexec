#include <vkexec/push.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec {

auto upload_push_constants(VkCommandBuffer cmd, VkPipelineLayout layout, void const *data, std::uint32_t bytes) -> void
{
  if (data == nullptr || bytes == 0U) { return; }
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, data);
}

}// namespace vkexec
