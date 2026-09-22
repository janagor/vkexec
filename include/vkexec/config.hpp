#ifndef VKEXEC_CONFIG_HPP
#define VKEXEC_CONFIG_HPP

#include <cassert>
#include <cstdlib>

// Clang-only attribute used to silence false positives in checked-take helpers.
// GCC treats unknown scoped attributes as errors under -Werror=attributes.
#ifdef __clang__
#define VKEXEC_CLANG_SUPPRESS , clang::suppress
#else
#define VKEXEC_CLANG_SUPPRESS
#endif

namespace vkexec::detail {

/**
 * Reports an unrecoverable internal invariant violation.
 *
 * Debug builds assert first; then the process aborts. Prefer returning
 * `result` / `status` failures for recoverable API errors.
 *
 * @param message Human-readable reason (must not be null).
 */
[[noreturn]] inline auto contract_violation(char const *message) noexcept -> void
{
  assert(message != nullptr && false);
  (void)message;
  std::abort();
}

}// namespace vkexec::detail

#endif// VKEXEC_CONFIG_HPP
