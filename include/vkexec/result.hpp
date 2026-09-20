#ifndef VKEXEC_RESULT_HPP
#define VKEXEC_RESULT_HPP

//! \file
//! Public expected-style `result` / `status` and early-return helpers.

#include <vkexec/error.hpp>

#include <optional>
#include <type_traits>
#include <utility>

namespace vkexec {

using error_type = vkexec::error;

/**
 * Holds an error for constructing a failed `result` (`return fail(...)`).
 *
 * @see fail, result
 */
template<typename E> class unexpected
{
public:
  unexpected() = delete;

  explicit unexpected(E err) : error_(std::move(err)) {}

  [[nodiscard]] auto error() & -> E & { return error_; }
  [[nodiscard]] auto error() const & -> E const & { return error_; }
  [[nodiscard]] auto error() && -> E && { return std::move(error_); }

private:
  E error_;
};

/**
 * Value-or-error result similar to `std::expected<T, error>`.
 *
 * Implicitly constructible from a value or from `unexpected<error_type>`.
 * Accessors require `has_value()` / `operator bool` to be true first.
 */
template<typename T> class result
{
public:
  result() : value_(std::in_place) {}

  result(result const &) = default;
  auto operator=(result const &) -> result & = default;
  result(result &&) = default;
  auto operator=(result &&) -> result & = default;
  ~result() = default;

  // Implicit value/error conversions match prior std::expected call sites (`return value`, `return fail(...)`).
  template<typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, result> && std::constructible_from<T, U>)
  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  result(U &&value) : value_(std::in_place, std::forward<U>(value))
  {}

  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  result(unexpected<error_type> err) : error_(std::in_place, std::move(err.error())) {}

  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] auto has_value() const noexcept -> bool { return value_.has_value(); }

  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  [[nodiscard]] auto error() & -> error_type & { return *error_; }
  [[nodiscard]] auto error() const & -> error_type const & { return *error_; }
  [[nodiscard]] auto error() && -> error_type && { return std::move(*error_); }
  // NOLINTEND(bugprone-unchecked-optional-access)

  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  [[nodiscard]] auto operator*() & -> T & { return *value_; }
  [[nodiscard]] auto operator*() const & -> T const & { return *value_; }
  [[nodiscard]] auto operator*() && -> T && { return std::move(*value_); }

  [[nodiscard]] auto operator->() -> T * { return &(*value_); }
  [[nodiscard]] auto operator->() const -> T const * { return &(*value_); }

  [[nodiscard]] auto value() & -> T & { return *value_; }
  [[nodiscard]] auto value() const & -> T const & { return *value_; }
  [[nodiscard]] auto value() && -> T && { return std::move(*value_); }
  // NOLINTEND(bugprone-unchecked-optional-access)

private:
  std::optional<T> value_{};
  std::optional<error_type> error_;
};

/**
 * Void specialization: success (`has_value()`) or an error.
 *
 * Used as `status` for operations that do not produce a value.
 */
template<> class result<void>
{
public:
  result() = default;

  result(result const &) = default;
  auto operator=(result const &) -> result & = default;
  result(result &&) = default;
  auto operator=(result &&) -> result & = default;
  ~result() = default;

  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  result(unexpected<error_type> err) : error_(std::in_place, std::move(err.error())) {}

  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] auto has_value() const noexcept -> bool { return !error_.has_value(); }

  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  [[nodiscard]] auto error() & -> error_type & { return *error_; }
  [[nodiscard]] auto error() const & -> error_type const & { return *error_; }
  [[nodiscard]] auto error() && -> error_type && { return std::move(*error_); }
  // NOLINTEND(bugprone-unchecked-optional-access)

private:
  std::optional<error_type> error_;
};

//! Alias for `result<void>` (success or error, no value).
using status = result<void>;

//! Wraps `err` as `unexpected` for `return fail(...)`.
[[nodiscard]] inline auto fail(error_type err) -> unexpected<error_type> { return unexpected(std::move(err)); }

//! Builds `unexpected` from an `errc` and optional detail string.
[[nodiscard]] inline auto fail(errc code, std::string detail = {}) -> unexpected<error_type>
{ return fail(make_error(code, std::move(detail))); }

//! Propagates the error from a failed `result`.
template<typename T> [[nodiscard]] inline auto fail(result<T> &value) -> unexpected<error_type>
{ return unexpected(std::move(value.error())); }

//! Propagates the error from a failed `result` (const).
template<typename T> [[nodiscard]] inline auto fail(result<T> const &value) -> unexpected<error_type>
{ return unexpected(value.error()); }

//! Moves the success value out of `value` (caller must check success).
template<typename T> [[nodiscard, clang::suppress]] auto expected_take(result<T> &value) -> T
{ return std::move(*value); }

//! Copies the success value out of `value` (caller must check success).
template<typename T> [[nodiscard, clang::suppress]] auto expected_take(result<T> const &value) -> T { return *value; }

//! Returns a reference to the success value (caller must check success).
template<typename T> [[nodiscard, clang::suppress]] auto expected_get(result<T> &value) -> T & { return *value; }

//! Returns a const reference to the success value (caller must check success).
template<typename T> [[nodiscard]] auto expected_get(result<T> const &value) -> T const & { return *value; }

}// namespace vkexec

// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define VKEXEC_TRY_ASSIGN(v, r)                                                                    \
  auto vkexec_try_tmp_##v = (r);                                                                   \
  if (!vkexec_try_tmp_##v) { return ::vkexec::unexpected(std::move(vkexec_try_tmp_##v.error())); } \
  auto &v = ::vkexec::expected_get(vkexec_try_tmp_##v)

#define VKEXEC_TRY(r) \
  if (auto vkexec_try_chk = (r); !vkexec_try_chk) { return ::vkexec::unexpected(std::move(vkexec_try_chk.error())); }
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)

#endif// VKEXEC_RESULT_HPP
