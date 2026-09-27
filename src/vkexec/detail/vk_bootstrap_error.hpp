#ifndef VKEXEC_DETAIL_VK_BOOTSTRAP_ERROR_HPP
#define VKEXEC_DETAIL_VK_BOOTSTRAP_ERROR_HPP

#include <VkBootstrap.h>
#include <vkexec/error_helpers.hpp>

#include <string>
#include <utility>

namespace vkexec::detail {

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


}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_VK_BOOTSTRAP_ERROR_HPP
