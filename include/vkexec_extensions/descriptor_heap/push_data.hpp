#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PUSH_DATA_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PUSH_DATA_HPP

//! \file
//! `vkCmdPushDataEXT` helpers for bindless compute passes.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>
#include <type_traits>

namespace vkexec {

/**
 * Records `vkCmdPushDataEXT` for a host-visible byte range.
 *
 * @param ctx Context used to resolve extension procs.
 * @param cmd Command buffer in the recording state.
 * @param bytes Host bytes to push.
 * @param offset Destination offset in the push-data range.
 */
[[nodiscard]] auto
  cmd_push_data(context const &ctx, VkCommandBuffer cmd, std::span<std::byte const> bytes, std::uint32_t offset = 0)
    -> status;

/**
 * Records `vkCmdPushDataEXT` for a trivially copyable POD.
 *
 * @param value Host POD copied as bytes.
 * @param offset Destination offset in the push-data range.
 */
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

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PUSH_DATA_HPP
