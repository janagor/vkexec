#ifndef VKEXEC_ERROR_HPP
#define VKEXEC_ERROR_HPP

#include <cx_system_error/system_error.hpp>

#include <expected>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace vkexec {

// NOLINTNEXTLINE(performance-enum-size)
enum class errc
{
  invalid_argument = 1,
  io_error,
  parse_error,
  unsupported,
  out_of_range,
  empty_result,
};

class vkexec_error_category final : public cx::ErrorCategory
{
public:
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto Name() const noexcept -> const char * override { return "vkexec"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto Message(int error_value) const noexcept -> std::string_view override
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
    default:
      return "unknown vkexec error";
    }
  }
};

[[nodiscard]] inline auto category() noexcept -> cx::ErrorCategory const &
{
  static vkexec_error_category const k_instance{};
  return k_instance;
}

// NOLINTNEXTLINE(readability-identifier-naming)
[[nodiscard]] inline auto MakeErrorCode(errc error) noexcept -> cx::ErrorCode
{
  return cx::ErrorCode{ static_cast<int>(error), category() };
}

struct error
{
  cx::ErrorCode code{};
  std::string detail;

  [[nodiscard]] auto message() const noexcept -> std::string_view
  {
    return detail.empty() ? code.Message() : std::string_view(detail);
  }
};

template<typename T> using result = std::expected<T, error>;
using status = std::expected<void, error>;

[[nodiscard]] inline auto make_error(errc code, std::string const &detail = {}) -> error
{
  return error{
    .code = MakeErrorCode(code),
    .detail = detail,
  };
}

}// namespace vkexec

namespace cx {

// NOLINTNEXTLINE(readability-identifier-naming)
template<> struct IsErrorCodeEnum<vkexec::errc> : std::true_type
{
};

}// namespace cx

#endif// VKEXEC_ERROR_HPP
