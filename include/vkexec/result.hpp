#ifndef VKEXEC_RESULT_HPP
#define VKEXEC_RESULT_HPP

//! \file
//! Public aliases for expected-style `result` / `status` and early-return helpers.

#include <vkexec/detail/result.hpp>

namespace vkexec {

//! Expected-style value-or-error result used throughout the public API.
template<typename T> using result = detail::result<T>;

//! Unexpected wrapper for constructing failed `result` / `status` values.
template<typename E> using unexpected = detail::unexpected<E>;

//! Void specialization used for success-or-error operations without a value.
using status = detail::status;

using detail::expected_get;
using detail::expected_take;
using detail::fail;

}// namespace vkexec

// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
/**
 * Evaluates `r`, returns early on failure, and binds `v` to the success value.
 *
 * Example: `VKEXEC_TRY_ASSIGN(buf, allocate(...));`
 */
#define VKEXEC_TRY_ASSIGN(v, r)                                                                            \
  auto vkexec_try_tmp_##v = (r);                                                                           \
  if (!vkexec_try_tmp_##v) { return ::vkexec::detail::unexpected(std::move(vkexec_try_tmp_##v.error())); } \
  auto &v = ::vkexec::detail::expected_get(vkexec_try_tmp_##v)

//! Evaluates `r` and returns early with its error when the result is failed.
#define VKEXEC_TRY(r)                                                       \
  if (auto vkexec_try_chk = (r); !vkexec_try_chk) {                         \
    return ::vkexec::detail::unexpected(std::move(vkexec_try_chk.error())); \
  }
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)

#endif// VKEXEC_RESULT_HPP
