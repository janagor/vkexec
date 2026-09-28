#ifndef VKEXEC_CONFIG_HPP
#define VKEXEC_CONFIG_HPP

#include <cassert>
#include <cstdlib>

// Clang-only attribute used to silence false positives in checked-take helpers.
// Feature-test it because older supported Clang versions do not provide it.
#if defined(__clang__) && __has_cpp_attribute(clang::suppress)
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
