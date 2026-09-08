#ifndef VKEXEC_ERROR_HELPERS_HPP
#define VKEXEC_ERROR_HELPERS_HPP

#include <vkexec/error.hpp>

#include <VkBootstrap.h>

#include <vulkan/vulkan.h>

#include <string>
#include <string_view>
#include <utility>

namespace vkexec {

[[nodiscard]] auto make_vk_error(VkResult result, std::string_view context) -> error;

[[nodiscard]] inline auto fail(VkResult result, std::string_view context = {}) -> std::unexpected<error>
{ return fail(make_vk_error(result, context)); }

template<typename T>
[[nodiscard, clang::suppress]] auto vkb_take(vkb::Result<T> const &result) -> T
{ return *result; }

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
