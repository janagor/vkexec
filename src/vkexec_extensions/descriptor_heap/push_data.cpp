#include <vkexec_extensions/descriptor_heap/push_data.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

auto cmd_push_data(context const &ctx, VkCommandBuffer cmd, std::span<std::byte const> bytes, std::uint32_t offset)
  -> status
{
  if (ctx.procs().cmd_push_data == nullptr) { return fail(errc::unsupported, "vkCmdPushDataEXT is unavailable"); }
  if (bytes.empty()) { return fail(errc::invalid_argument, "cmd_push_data requires a non-empty payload"); }

  VkPushDataInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
  info.offset = offset;
  info.data.address = bytes.data();
  info.data.size = bytes.size();
  ctx.procs().cmd_push_data(cmd, &info);
  return {};
}

}// namespace vkexec
