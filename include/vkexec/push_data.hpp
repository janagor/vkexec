#ifndef VKEXEC_PUSH_DATA_HPP
#define VKEXEC_PUSH_DATA_HPP

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>
#include <type_traits>

namespace vkexec {

/// Record `vkCmdPushDataEXT` for a host-visible byte range.
[[nodiscard]] auto
  cmd_push_data(context const &ctx, VkCommandBuffer cmd, std::span<std::byte const> bytes, std::uint32_t offset = 0)
    -> status;

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
