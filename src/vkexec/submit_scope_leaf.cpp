#include <vkexec/error_helpers.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan.h>

namespace vkexec::detail {

// Out-of-line so clang CSA (per-TU) cannot see through LEAF's opaque result<> into
// submit_scope's move assign — a known false positive with Boost.LEAF.
auto submit_scope::open(context &host) -> result<submit_scope>
{
  auto cmd = host.allocate_command_buffer();
  if (!cmd) { return cmd.error(); }

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (VkResult const result = vkBeginCommandBuffer(*cmd, &begin); result != VK_SUCCESS) {
    host.free_command_buffer(*cmd);
    return make_vk_error(result, "vkBeginCommandBuffer failed");
  }

  submit_scope scope;
  scope.ctx = &host;
  scope.cmd = *cmd;
  return scope;
}

void move_from_leaf_submit_scope(submit_scope &dest, leaf::result<submit_scope> &src)
{
  dest = std::move(*src);
}

}// namespace vkexec::detail
