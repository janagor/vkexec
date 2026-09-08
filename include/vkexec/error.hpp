#ifndef VKEXEC_ERROR_HPP
#define VKEXEC_ERROR_HPP

#include <boost/system/error_code.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vkexec {

namespace sys = boost::system;

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

inline constexpr std::uint64_t k_vkexec_error_category_id = 0x9f3c2a7b1e8d4056ULL;
inline constexpr std::uint64_t k_vulkan_error_category_id = 0x4b71e90c6d2a83f5ULL;

class vkexec_error_category final : public sys::error_category
{
public:
  constexpr vkexec_error_category() noexcept : sys::error_category(k_vkexec_error_category_id) {}

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value) const -> std::string override
  {
    return message(error_value, nullptr, 0);
  }

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

class vulkan_error_category final : public sys::error_category
{
public:
  constexpr vulkan_error_category() noexcept : sys::error_category(k_vulkan_error_category_id) {}

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec.vulkan"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value) const -> std::string override;

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value, char *buffer, std::size_t len) const noexcept
    -> char const * override;

  // VkResult: negative values are errors; non-negative includes success and status codes.
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto failed(int error_value) const noexcept -> bool override { return error_value < 0; }
};

[[nodiscard]] auto category() noexcept -> sys::error_category const &;

[[nodiscard]] auto vulkan_category() noexcept -> sys::error_category const &;

[[nodiscard]] auto make_error_code(errc error) noexcept -> sys::error_code;

[[nodiscard]] auto make_vk_error_code(int vk_result) noexcept -> sys::error_code;

struct error
{
  sys::error_code code{};
  std::string detail;

  [[nodiscard]] auto message() const -> std::string
  {
    if (!detail.empty()) { return detail; }
    return code.message();
  }
};

[[nodiscard]] auto make_error(errc code, std::string detail = {}) -> error;

[[nodiscard]] auto to_string(error const &err) -> std::string;

}// namespace vkexec

namespace boost::system {

template<> struct is_error_code_enum<vkexec::errc> : std::true_type
{
};

}// namespace boost::system

#endif// VKEXEC_ERROR_HPP
