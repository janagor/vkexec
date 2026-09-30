#ifndef VKEXEC_ERROR_HPP
#define VKEXEC_ERROR_HPP

//! \file
//! Error codes, standard categories, and the `error` value used by senders.

#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace vkexec {

/**
 * Portable vkexec error codes (`std::error_code` enum).
 *
 * `errc::vulkan` is a category sentinel; concrete `VkResult` failures use
 * `vulkan_category()` via `make_vk_error_code`.
 */
// NOLINTNEXTLINE(performance-enum-size)
enum class errc {
  invalid_argument = 1,
  io_error,
  parse_error,
  unsupported,
  out_of_range,
  empty_result,
  cancelled,
  vulkan,
  unexpected_exception,
};

//! Returns the singleton `vkexec` error category.
[[nodiscard]] auto category() noexcept -> std::error_category const &;

//! Returns the singleton Vulkan `VkResult` error category.
[[nodiscard]] auto vulkan_category() noexcept -> std::error_category const &;

//! Builds an `error_code` in `category()` from `errc`.
[[nodiscard]] auto make_error_code(errc error) noexcept -> std::error_code;

//! Builds an error code from a failed `VkResult` integer (precondition: negative).
[[nodiscard]] auto make_vk_error_code(int vk_result) noexcept -> std::error_code;

/**
 * Rich error value completed through senders and returned from `result` APIs.
 *
 * Prefer `message()` for display: it returns `detail` when set, otherwise the
 * category message for `code`.
 */
struct error
{
  std::error_code code;
  std::string detail;

  //! Returns `detail` if non-empty; otherwise `code.message()`.
  [[nodiscard]] auto message() const -> std::string
  {
    if (!detail.empty()) { return detail; }
    return code.message();
  }
};

static_assert(std::is_nothrow_move_constructible_v<error>);
static_assert(std::is_nothrow_move_assignable_v<error>);

//! Returns the non-throwing error used when a sender catches an exception.
[[nodiscard]] inline auto unexpected_exception_error() noexcept -> error
{ return error{ .code = make_error_code(errc::unexpected_exception), .detail = {} }; }

/**
 * Constructs an `error` with a vkexec `errc` and optional detail string.
 *
 * @param code Portable error code.
 * @param detail Optional human-readable context (may be empty).
 */
[[nodiscard]] auto make_error(errc code, std::string detail = {}) -> error;

//! Formats `err` for logging (typically `code` plus detail).
[[nodiscard]] auto to_string(error const &err) -> std::string;

}// namespace vkexec

namespace std {

template<> struct is_error_code_enum<vkexec::errc> : std::true_type
{
};

}// namespace std

#endif// VKEXEC_ERROR_HPP
