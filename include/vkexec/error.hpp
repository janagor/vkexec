#ifndef VKEXEC_ERROR_HPP
#define VKEXEC_ERROR_HPP

#ifndef VKEXEC_ENABLE_EXCEPTIONS
#define VKEXEC_ENABLE_EXCEPTIONS 1
#endif

#include <system_error>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace vkexec {

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

class vkexec_error_category final : public std::error_category
{
public:
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value) const -> std::string override;
};

class vulkan_error_category final : public std::error_category
{
public:
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec.vulkan"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto message(int error_value) const -> std::string override;
};

[[nodiscard]] auto category() noexcept -> std::error_category const &;

[[nodiscard]] auto vulkan_category() noexcept -> std::error_category const &;

[[nodiscard]] auto make_error_code(errc error) noexcept -> std::error_code;

[[nodiscard]] auto make_vk_error_code(int vk_result) noexcept -> std::error_code;

struct error
{
  std::error_code code{};
  std::string detail;

  [[nodiscard]] auto message() const -> std::string
  {
    if (!detail.empty()) { return detail; }
    return code.message();
  }
};

template<typename T> using result = std::expected<T, error>;

using status = std::expected<void, error>;

[[nodiscard]] auto make_error(errc code, std::string detail = {}) -> error;

[[nodiscard]] inline auto fail(error err) -> std::unexpected<error> { return std::unexpected(std::move(err)); }

[[nodiscard]] inline auto fail(errc code, std::string detail = {}) -> std::unexpected<error>
{ return fail(make_error(code, std::move(detail))); }

template<typename T> [[nodiscard]] inline auto fail(result<T> &value) -> std::unexpected<error>
{ return std::unexpected(std::move(value.error())); }

template<typename T> [[nodiscard]] inline auto fail(result<T> const &value) -> std::unexpected<error>
{ return std::unexpected(value.error()); }

namespace detail {

  template<typename T> [[nodiscard, clang::suppress]] auto expected_take(result<T> &value) -> T
  { return std::move(*value); }

  template<typename T> [[nodiscard, clang::suppress]] auto expected_take(result<T> const &value) -> T
  { return *value; }

  template<typename T> [[nodiscard, clang::suppress]] auto expected_get(result<T> &value) -> T & { return *value; }

  template<typename T> [[nodiscard]] auto expected_get(result<T> const &value) -> T const & { return *value; }

}// namespace detail

// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define VKEXEC_TRY_ASSIGN(v, r)                                                \
  auto vkexec_try_tmp_##v = (r);                                              \
  if (!vkexec_try_tmp_##v) { return std::unexpected(std::move(vkexec_try_tmp_##v.error())); } \
  auto &v = ::vkexec::detail::expected_get(vkexec_try_tmp_##v)

#define VKEXEC_TRY(r)                                                          \
  if (auto vkexec_try_chk = (r); !vkexec_try_chk) {                            \
    return std::unexpected(std::move(vkexec_try_chk.error()));                  \
  }
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)

[[nodiscard]] auto to_string(error const &err) -> std::string;

#if VKEXEC_ENABLE_EXCEPTIONS
// stdexec throws custom completion error types; vkexec::error is intentionally not std::exception-derived.
// NOLINTBEGIN(hicpp-exception-baseclass,cert-err09-cpp,cert-err61-cpp,misc-throw-by-value-catch-by-reference,cppcoreguidelines-rvalue-reference-param-not-moved)
template<typename T> [[nodiscard]] auto value_or_throw(result<T> &value) -> T &
{
  if (!value) { throw error{ value.error() }; }
  return *value;
}

template<typename T> [[nodiscard]] auto value_or_throw(result<T> const &value) -> T const &
{
  if (!value) { throw error{ value.error() }; }
  return *value;
}

template<typename T> [[nodiscard]] auto value_or_throw(result<T> &&value) -> T
{
  if (!value) { throw error{ std::move(value.error()) }; }
  return std::move(*value);
}

inline auto value_or_throw(status const &value) -> void
{
  if (!value) { throw error{ value.error() }; }
}
// NOLINTEND(hicpp-exception-baseclass,cert-err09-cpp,cert-err61-cpp,misc-throw-by-value-catch-by-reference,cppcoreguidelines-rvalue-reference-param-not-moved)
#endif

}// namespace vkexec

namespace std {

template<> struct is_error_code_enum<vkexec::errc> : true_type
{
};

}// namespace std

#endif// VKEXEC_ERROR_HPP
