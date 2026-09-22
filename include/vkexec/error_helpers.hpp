#ifndef VKEXEC_ERROR_HELPERS_HPP
#define VKEXEC_ERROR_HELPERS_HPP

//! \file
//! Helpers that turn Vulkan / VkBootstrap failures into `error` and `unexpected`.

#include <vkexec/config.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <VkBootstrap.h>

#include <vulkan/vulkan.h>

#include <string>
#include <string_view>
#include <utility>

namespace vkexec {

/**
 * Builds an `error` from a failed `VkResult` and a short context string.
 *
 * @param result Vulkan result (typically not `VK_SUCCESS`).
 * @param context Prefix describing the failing call (e.g. `"vkCreateBuffer"`).
 */
[[nodiscard]] auto make_vk_error(VkResult result, std::string_view context) -> error;

//! Returns `unexpected` wrapping `make_vk_error(result, context)`.
[[nodiscard]] inline auto fail(VkResult result, std::string_view context = {}) -> unexpected<error>
{ return fail(make_vk_error(result, context)); }

//! Dereferences a successful `vkb::Result` (caller must check success first).
template<typename T>[[nodiscard VKEXEC_CLANG_SUPPRESS]] auto vkb_take(vkb::Result<T> const &result) -> T
{ return *result; }

/**
 * Converts a failed `vkb::Result` into a vkexec `error`.
 *
 * @param result Failed VkBootstrap result.
 * @param what Short description of the operation that failed.
 */
template<typename T>
[[nodiscard]] inline auto make_error_from_vkb(vkb::Result<T> const &result, char const *what) -> error
{
  std::string detail = what;
  detail += ": ";
  detail += result.error().message();
  if (result.vk_result() != VK_SUCCESS) {
    detail += " (";
    detail += std::to_string(result.vk_result());
    detail += ')';
  }
  return make_error(errc::unsupported, std::move(detail));
}

}// namespace vkexec

#endif// VKEXEC_ERROR_HELPERS_HPP
