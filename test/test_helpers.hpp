#ifndef VKEXEC_TEST_HELPERS_HPP
#define VKEXEC_TEST_HELPERS_HPP

#include <vkexec/context.hpp>
#include <vkexec/sync_wait.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec::test {

[[noreturn]] inline auto skip_if_no_vulkan(error const &err) -> void
{
  SKIP(std::string("Vulkan unavailable: ") + err.message());
}

[[noreturn]] inline auto skip_if_unavailable(error const &err) -> void
{
  SKIP(std::string("Unavailable: ") + err.message());
}

template<class Sender> [[nodiscard]] auto sync_wait_value(Sender &&sender)
{
  auto outcome = vkexec::try_sync_wait_value(std::forward<Sender>(sender));
  if (!outcome) { skip_if_no_vulkan(outcome.error()); }
  return std::move(*outcome);
}

template<class... Values>
[[nodiscard]] inline auto sync_wait_completed(sync_wait_outcome<Values...> const &outcome) -> bool
{ return !outcome.failed() && outcome.values.has_value() && !outcome.stopped; }

template<class... Values>
[[nodiscard]] inline auto sync_wait_stopped(sync_wait_outcome<Values...> const &outcome) -> bool
{ return !outcome.failed() && (outcome.stopped || !outcome.values.has_value()); }

template<class Sender> [[nodiscard]] auto sync_wait_sender(Sender &&sender)
{ return try_sync_wait(std::forward<Sender>(sender)); }

[[nodiscard]] inline auto require_context(scheduler_options const &opts = {}) -> std::unique_ptr<context>
{ return sync_wait_value(factory::make_context(opts)); }

}// namespace vkexec::test

#endif// VKEXEC_TEST_HELPERS_HPP
