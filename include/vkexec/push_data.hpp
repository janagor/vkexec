#ifndef VKEXEC_PUSH_DATA_HPP
#define VKEXEC_PUSH_DATA_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>
#include <type_traits>

namespace vkexec {

/// Record `vkCmdPushDataEXT` for a host-visible byte range.
[[nodiscard]] inline auto cmd_push_data(context const &ctx,
  VkCommandBuffer cmd,
  std::span<std::byte const> bytes,
  std::uint32_t offset = 0) -> status
{
  if (ctx.procs().cmd_push_data == nullptr) {
    return make_error(errc::unsupported, "vkCmdPushDataEXT is unavailable");
  }
  if (bytes.empty()) { return make_error(errc::invalid_argument, "cmd_push_data requires a non-empty payload"); }

  VkPushDataInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
  info.offset = offset;
  info.data.address = bytes.data();
  info.data.size = bytes.size();
  ctx.procs().cmd_push_data(cmd, &info);
  return {};
}

/// Record `vkCmdPushDataEXT` for a trivially copyable POD.
template<typename T>
[[nodiscard]] auto cmd_push_data(context const &ctx, VkCommandBuffer cmd, T const &value, std::uint32_t offset = 0)
  -> status
{
  static_assert(std::is_trivially_copyable_v<T>);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto const *const bytes = reinterpret_cast<std::byte const *>(&value);
  return cmd_push_data(ctx, cmd, std::span<std::byte const>{ bytes, sizeof(T) }, offset);
}

}// namespace vkexec

#endif// VKEXEC_PUSH_DATA_HPP
