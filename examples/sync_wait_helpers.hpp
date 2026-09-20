#ifndef VKEXEC_EXAMPLES_SYNC_WAIT_HELPERS_HPP
#define VKEXEC_EXAMPLES_SYNC_WAIT_HELPERS_HPP

#include <vkexec/detail/sync_wait_outcome.hpp>
#include <vkexec/error.hpp>
#include <vkexec/sync_wait.hpp>

#include <format>
#include <iostream>
#include <utility>

namespace vkexec::examples {

[[noreturn]] inline auto abort_with_error(error const &err) -> void
{
  std::cerr << std::format("{}\n", err.message());
  std::abort();
}

[[noreturn]] inline auto abort_stopped() -> void
{ abort_with_error(make_error(errc::cancelled, "sender completed with set_stopped")); }

template<class Sender> [[nodiscard]] auto sync_wait_value(Sender &&sender)
{
#if VKEXEC_ENABLE_EXCEPTIONS
  return vkexec::sync_wait_value(std::forward<Sender>(sender));
#else
  auto outcome = vkexec::try_sync_wait_value(std::forward<Sender>(sender));
  if (!outcome) { abort_with_error(outcome.error()); }
  return std::move(*outcome);
#endif
}

template<class Sender> auto sync_wait_graph(Sender &&sender) -> void
{
  auto outcome = try_sync_wait(std::forward<Sender>(sender));
  if (outcome.failed()) {
#if VKEXEC_ENABLE_EXCEPTIONS
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    throw outcome.take_error();
#else
    abort_with_error(outcome.take_error());
#endif
  }
  if (outcome.stopped || !outcome.values.has_value()) {
#if VKEXEC_ENABLE_EXCEPTIONS
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    throw make_error(errc::cancelled, "pipeline was stopped");
#else
    abort_stopped();
#endif
  }
}

[[noreturn]] inline auto fail_check(char const *message) -> void
{
#if VKEXEC_ENABLE_EXCEPTIONS
  // NOLINTNEXTLINE(hicpp-exception-baseclass)
  throw make_error(errc::unsupported, message);
#else
  abort_with_error(make_error(errc::unsupported, message));
#endif
}

template<class RunFunc> auto run_example(RunFunc &&run_func) -> int
{
#if VKEXEC_ENABLE_EXCEPTIONS
  try {
    return std::forward<RunFunc>(run_func)();
  } catch (error const &err) {
    std::cerr << std::format("vkexec example failed: {}\n", err.message());
    return 1;
  }
#else
  return std::forward<RunFunc>(run_func)();
#endif
}

}// namespace vkexec::examples

#endif// VKEXEC_EXAMPLES_SYNC_WAIT_HELPERS_HPP
