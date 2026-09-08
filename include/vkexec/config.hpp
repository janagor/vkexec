#ifndef VKEXEC_CONFIG_HPP
#define VKEXEC_CONFIG_HPP

#include <cassert>
#include <cstdlib>

namespace vkexec::detail {

/// Unrecoverable internal invariant violation (debug builds assert first).
[[noreturn]] inline auto contract_violation(char const *message) noexcept -> void
{
  assert(message != nullptr && false);
  (void)message;
  std::abort();
}

}// namespace vkexec::detail

#endif// VKEXEC_CONFIG_HPP
