#ifndef VKEXEC_RESULT_HPP
#define VKEXEC_RESULT_HPP

#include <vkexec/detail/result.hpp>

namespace vkexec {

template<typename T> using result = detail::result<T>;

template<typename E> using unexpected = detail::unexpected<E>;

using status = detail::status;

using detail::expected_get;
using detail::expected_take;
using detail::fail;

}// namespace vkexec

// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define VKEXEC_TRY_ASSIGN(v, r)                                                                            \
  auto vkexec_try_tmp_##v = (r);                                                                           \
  if (!vkexec_try_tmp_##v) { return ::vkexec::detail::unexpected(std::move(vkexec_try_tmp_##v.error())); } \
  auto &v = ::vkexec::detail::expected_get(vkexec_try_tmp_##v)

#define VKEXEC_TRY(r)                                                       \
  if (auto vkexec_try_chk = (r); !vkexec_try_chk) {                         \
    return ::vkexec::detail::unexpected(std::move(vkexec_try_chk.error())); \
  }
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)

#endif// VKEXEC_RESULT_HPP
