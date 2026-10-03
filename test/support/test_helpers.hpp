#ifndef VKEXEC_TEST_HELPERS_HPP
#define VKEXEC_TEST_HELPERS_HPP

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/sync_wait.hpp>

#include <catch2/catch_test_macros.hpp>

#include <exception>
#include <memory>
#if VKEXEC_HAS_EXCEPTIONS
#include <stdexcept>
#endif
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec::test {

[[noreturn]] inline auto skip_if_no_vulkan(error const &err) -> void
{
  SKIP(std::string("Vulkan unavailable: ") + err.message());
#ifndef _MSC_VER
  std::terminate();
#endif
}

[[noreturn]] inline auto skip_if_unavailable(error const &err) -> void
{
  SKIP(std::string("Unavailable: ") + err.message());
#ifndef _MSC_VER
  std::terminate();
#endif
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
{
  auto configured = opts;
  bool const require_validation =
#ifdef VKEXEC_TEST_VALIDATION
    true;
#else
    false;
#endif
  if (require_validation) { configured.validation_layers = true; }
  auto outcome = vkexec::try_sync_wait_value(factory::make_context(configured));
  if (outcome) { return std::move(*outcome); }

  if (require_validation) {
#if VKEXEC_HAS_EXCEPTIONS
    throw std::runtime_error(std::string("Validation context unavailable: ") + outcome.error().message());
#else
    FAIL(std::string("Validation context unavailable: ") + outcome.error().message());
#endif
  }
  skip_if_no_vulkan(outcome.error());
}

}// namespace vkexec::test

#endif// VKEXEC_TEST_HELPERS_HPP
