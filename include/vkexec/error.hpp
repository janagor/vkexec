#ifndef VKEXEC_ERROR_HPP
#define VKEXEC_ERROR_HPP

//! \file
//! Error codes, Boost.System categories, and the `error` value used by senders.

#include <boost/system/error_code.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vkexec {

namespace sys = boost::system;

/**
 * Portable vkexec error codes (Boost.System `error_code` enum).
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
};

//! Stable category id for `vkexec_error_category`.
inline constexpr std::uint64_t k_vkexec_error_category_id = 0x9f3c2a7b1e8d4056ULL;
//! Stable category id for `vulkan_error_category`.
inline constexpr std::uint64_t k_vulkan_error_category_id = 0x4b71e90c6d2a83f5ULL;

/**
 * Boost.System category for `errc` values.
 *
 * @see category, make_error_code
 */
// Boost.System categories are never deleted through a base pointer; the base
// dtor is intentionally non-virtual. Silence GCC -Wnon-virtual-dtor here.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif
class vkexec_error_category final : public sys::error_category
{
public:
  constexpr vkexec_error_category() noexcept : sys::error_category(k_vkexec_error_category_id) {}

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value) const -> std::string override { return message(error_value, nullptr, 0); }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value, char * /*buffer*/, std::size_t /*len*/) const noexcept
    -> char const * override
  {
    switch (static_cast<errc>(error_value)) {
    case errc::invalid_argument:
      return "invalid argument";
    case errc::io_error:
      return "I/O error";
    case errc::parse_error:
      return "parse error";
    case errc::unsupported:
      return "unsupported operation";
    case errc::out_of_range:
      return "out of range";
    case errc::empty_result:
      return "empty result";
    case errc::cancelled:
      return "cancelled";
    case errc::vulkan:
      return "vulkan error";
    default:
      return "unknown vkexec error";
    }
  }
};

/**
 * Boost.System category whose values are raw `VkResult` integers.
 *
 * Negative `VkResult` values are treated as failures via `failed()`.
 *
 * @see vulkan_category, make_vk_error_code
 */
class vulkan_error_category final : public sys::error_category
{
public:
  constexpr vulkan_error_category() noexcept : sys::error_category(k_vulkan_error_category_id) {}

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec.vulkan"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value) const -> std::string override;

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value, char * /*buffer*/, std::size_t /*len*/) const noexcept
    -> char const * override;

  // VkResult: negative values are errors; non-negative includes success and status codes.
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto failed(int error_value) const noexcept -> bool override { return error_value < 0; }
};
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

//! Returns the singleton `vkexec` error category.
[[nodiscard]] auto category() noexcept -> sys::error_category const &;

//! Returns the singleton Vulkan `VkResult` error category.
[[nodiscard]] auto vulkan_category() noexcept -> sys::error_category const &;

//! Builds an `error_code` in `category()` from `errc`.
[[nodiscard]] auto make_error_code(errc error) noexcept -> sys::error_code;

//! Builds an `error_code` in `vulkan_category()` from a `VkResult` integer.
[[nodiscard]] auto make_vk_error_code(int vk_result) noexcept -> sys::error_code;

/**
 * Rich error value completed through senders and returned from `result` APIs.
 *
 * Prefer `message()` for display: it returns `detail` when set, otherwise the
 * category message for `code`.
 */
struct error
{
  sys::error_code code{};
  std::string detail;

  //! Returns `detail` if non-empty; otherwise `code.message()`.
  [[nodiscard]] auto message() const -> std::string
  {
    if (!detail.empty()) { return detail; }
    return code.message();
  }
};

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

namespace boost::system {

template<> struct is_error_code_enum<vkexec::errc> : std::true_type
{
};

}// namespace boost::system

#endif// VKEXEC_ERROR_HPP
