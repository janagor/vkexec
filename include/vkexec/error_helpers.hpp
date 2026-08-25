#ifndef VKEXEC_ERROR_HELPERS_HPP
#define VKEXEC_ERROR_HELPERS_HPP

#include <vkexec/error.hpp>

#include <VkBootstrap.h>

#include <vulkan/vulkan.h>

#include <string>
#include <string_view>
#include <utility>

namespace vkexec {

[[nodiscard]] inline auto make_vk_error(VkResult result, std::string_view context) -> error
{
  if (context.empty()) {
    return error{
      .code = MakeVkErrorCode(static_cast<int>(result)),
      .detail = {},
    };
  }
  return error{
    .code = MakeVkErrorCode(static_cast<int>(result)),
    .detail = std::string(context),
  };
}

template<typename T>
[[nodiscard]] inline auto make_error_from_vkb(vkb::Result<T> const &result, char const *what) -> error
{
  std::string detail = what;
  if (!result.vk_result()) {
    detail += ": ";
    detail += result.error().message();
    detail += " (";
    detail += std::to_string(result.vk_result());
    detail += ')';
  } else {
    detail += ": ";
    detail += result.error().message();
  }
  return make_error(errc::unsupported, std::move(detail));
}

[[nodiscard]] inline auto propagate_error(error err) -> std::unexpected<error> { return std::unexpected(std::move(err)); }

template<typename T>
[[nodiscard]] inline auto propagate(result<T> const &value) -> std::unexpected<error>
{
  return std::unexpected(value.error());
}

template<typename T>
[[nodiscard]] inline auto propagate(result<T> &&value) -> std::unexpected<error>
{
  return std::unexpected(std::move(value.error()));
}

}// namespace vkexec

#endif// VKEXEC_ERROR_HELPERS_HPP
