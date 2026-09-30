#ifndef VKEXEC_EXAMPLES_SYNC_WAIT_HELPERS_HPP
#define VKEXEC_EXAMPLES_SYNC_WAIT_HELPERS_HPP

#include <vkexec/error.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/sync_wait_outcome.hpp>
#include <vkexec_vma/resources.hpp>

#include <iostream>
#include <system_error>
#include <utility>

namespace vkexec::examples {

[[nodiscard]] inline auto make_vma_allocator(context &ctx) -> vma::allocator
{ return sync_wait_value(vma::factory::make_allocator(ctx)); }

[[noreturn]] inline auto abort_with_error(error const &err) -> void
{
  std::cerr << err.message() << '\n';
  std::abort();
}

[[noreturn]] inline auto abort_stopped() -> void
{ abort_with_error(make_error(errc::cancelled, "sender completed with set_stopped")); }

#if VKEXEC_HAS_EXCEPTIONS
[[noreturn]] inline auto throw_example_error(error const &err) -> void
{
  if (err.detail.empty()) { throw std::system_error(err.code); }
  throw std::system_error(err.code, err.detail);
}
#endif

template<class Sender>
[[nodiscard]] auto sync_wait_value(Sender &&sender) -> decltype(vkexec::sync_wait_value(std::forward<Sender>(sender)))
{
#if VKEXEC_HAS_EXCEPTIONS
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
#if VKEXEC_HAS_EXCEPTIONS
    throw_example_error(outcome.take_error());
#else
    abort_with_error(outcome.take_error());
#endif
  }
  if (outcome.stopped || !outcome.values.has_value()) {
#if VKEXEC_HAS_EXCEPTIONS
    throw_example_error(make_error(errc::cancelled, "pipeline was stopped"));
#else
    abort_stopped();
#endif
  }
}

[[noreturn]] inline auto fail_check(char const *message) -> void
{
#if VKEXEC_HAS_EXCEPTIONS
  throw_example_error(make_error(errc::unsupported, message));
#else
  abort_with_error(make_error(errc::unsupported, message));
#endif
}

template<class RunFunc> auto run_example(RunFunc &&run_func) -> int
{
#if VKEXEC_HAS_EXCEPTIONS
  try {
    return std::forward<RunFunc>(run_func)();
  } catch (std::system_error const &err) {
    std::cerr << "vkexec example failed: " << err.what() << '\n';
    return 1;
  }
#else
  return std::forward<RunFunc>(run_func)();
#endif
}

}// namespace vkexec::examples

#endif// VKEXEC_EXAMPLES_SYNC_WAIT_HELPERS_HPP
