#ifndef VKEXEC_DETAIL_RESULT_HPP
#define VKEXEC_DETAIL_RESULT_HPP

#include <vkexec/error.hpp>

#include <expected>
#include <utility>

namespace vkexec::detail {

template<typename T> using result = std::expected<T, error>;

using status = std::expected<void, error>;

[[nodiscard]] inline auto fail(error err) -> std::unexpected<error> { return std::unexpected(std::move(err)); }

[[nodiscard]] inline auto fail(errc code, std::string detail = {}) -> std::unexpected<error>
{ return fail(make_error(code, std::move(detail))); }

template<typename T> [[nodiscard]] inline auto fail(result<T> &value) -> std::unexpected<error>
{ return std::unexpected(std::move(value.error())); }

template<typename T> [[nodiscard]] inline auto fail(result<T> const &value) -> std::unexpected<error>
{ return std::unexpected(value.error()); }

template<typename T> [[nodiscard, clang::suppress]] auto expected_take(result<T> &value) -> T
{ return std::move(*value); }

template<typename T> [[nodiscard, clang::suppress]] auto expected_take(result<T> const &value) -> T
{ return *value; }

template<typename T> [[nodiscard, clang::suppress]] auto expected_get(result<T> &value) -> T & { return *value; }

template<typename T> [[nodiscard]] auto expected_get(result<T> const &value) -> T const & { return *value; }

}// namespace vkexec::detail

namespace vkexec {
using detail::fail;
}// namespace vkexec

// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define VKEXEC_TRY_ASSIGN(v, r)                                                                   \
  auto vkexec_try_tmp_##v = (r);                                                                 \
  if (!vkexec_try_tmp_##v) { return std::unexpected(std::move(vkexec_try_tmp_##v.error())); }    \
  auto &v = ::vkexec::detail::expected_get(vkexec_try_tmp_##v)

#define VKEXEC_TRY(r)                                                                             \
  if (auto vkexec_try_chk = (r); !vkexec_try_chk) {                                               \
    return std::unexpected(std::move(vkexec_try_chk.error()));                                    \
  }
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)

#endif// VKEXEC_DETAIL_RESULT_HPP
