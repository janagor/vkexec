#ifndef VKEXEC_SYNC_WAIT_HPP
#define VKEXEC_SYNC_WAIT_HPP

//! \file
//! Blocking wait helpers for vkexec senders (`sync_wait`, `try_sync_wait`, …).

#include <vkexec/config.hpp>
#include <vkexec/detail/stdexec_compat.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sync_wait_outcome.hpp>

#include <stdexec/execution.hpp>

#include <cstdlib>
#if VKEXEC_HAS_EXCEPTIONS
#include <exception>
#endif
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

#if VKEXEC_HAS_EXCEPTIONS
  [[noreturn]] inline auto throw_error(error const &err) -> void
  {
    if (err.detail.empty()) { throw std::system_error(err.code); }
    throw std::system_error(err.code, err.detail);
  }
#endif

  struct sync_wait_env
  {
    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr auto query(ex::get_scheduler_t /*query*/) const noexcept -> ex::run_loop::scheduler
    { return loop->get_scheduler(); }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr auto query(ex::get_start_scheduler_t /*query*/) const noexcept -> ex::run_loop::scheduler
    { return loop->get_scheduler(); }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr auto query(ex::get_delegation_scheduler_t /*query*/) const noexcept
      -> ex::run_loop::scheduler
    { return loop->get_scheduler(); }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] static constexpr auto query(stdexec_compat::root_t /*query*/) noexcept -> bool { return true; }

    ex::run_loop *loop{ nullptr };
  };

  struct sync_wait_state
  {
    std::optional<error> wait_error;
    bool stopped{ false };
    ex::run_loop loop;
  };

  template<class... Values> struct sync_wait_receiver
  {
    using receiver_concept = ex::receiver_t;

    sync_wait_state *state{ nullptr };
    std::optional<std::tuple<Values...>> *values{ nullptr };

    template<class... As> auto set_value(As &&...args) noexcept -> void
    {
#if VKEXEC_HAS_EXCEPTIONS
      try {
        values->emplace(std::forward<As>(args)...);
      } catch (...) {
        state->wait_error = unexpected_exception_error();
      }
#else
      values->emplace(std::forward<As>(args)...);
#endif
      state->loop.finish();
    }

    auto set_error(error &&err) noexcept -> void
    {
      state->wait_error = std::move(err);
      state->loop.finish();
    }

    auto set_error(error const &err) noexcept -> void
    {
#if VKEXEC_HAS_EXCEPTIONS
      try {
        state->wait_error.emplace(err);
      } catch (...) {
        state->wait_error = unexpected_exception_error();
      }
#else
      state->wait_error.emplace(err);
#endif
      state->loop.finish();
    }

    // stdexec adaptors (then, etc.) still advertise exception_ptr even under -fno-exceptions.
    auto set_error(std::exception_ptr const & /*exception*/) noexcept -> void
    {
      state->wait_error = unexpected_exception_error();
      state->loop.finish();
    }

    auto set_stopped() noexcept -> void
    {
      state->stopped = true;
      state->loop.finish();
    }

    [[nodiscard]] auto get_env() const noexcept -> sync_wait_env { return sync_wait_env{ &state->loop }; }
  };

  template<class... Ts> using decayed_tuple_t = std::tuple<std::decay_t<Ts>...>;

  template<class... Ts> using decayed_sync_wait_receiver_t = sync_wait_receiver<std::decay_t<Ts>...>;

  template<class CvSender>
  using sync_wait_value_tuple_t = ex::value_types_of_t<CvSender, sync_wait_env, decayed_tuple_t, std::type_identity_t>;

  template<class Tuple> struct sync_wait_outcome_from_tuple;

  template<class... Values> struct sync_wait_outcome_from_tuple<std::tuple<Values...>>
  {
    using type = sync_wait_outcome<Values...>;
  };

  // The dependent ::type names the outcome; without typename this aliases the trait itself.
  // NOLINTNEXTLINE(readability-redundant-typename)
  template<class Tuple> using sync_wait_outcome_from_tuple_t = typename sync_wait_outcome_from_tuple<Tuple>::type;

  template<class CvSender>
  using sync_wait_outcome_t = sync_wait_outcome_from_tuple_t<sync_wait_value_tuple_t<CvSender>>;

  template<class CvSender>
  using sync_wait_receiver_t =
    ex::value_types_of_t<CvSender, sync_wait_env, decayed_sync_wait_receiver_t, std::type_identity_t>;

  template<class CvSender>
  concept sync_waitable_sender =
    ex::sender_in<CvSender, sync_wait_env> && ex::sender_to<CvSender, sync_wait_receiver_t<CvSender>>;

  template<sync_waitable_sender CvSender>
  auto sync_wait_outcome_impl(CvSender &&sender) -> sync_wait_outcome_t<CvSender>
  {
    sync_wait_state state{};
    std::optional<sync_wait_value_tuple_t<CvSender>> values{};

    auto operation = ex::connect(std::forward<CvSender>(sender), sync_wait_receiver_t<CvSender>{ &state, &values });
    ex::start(operation);
#if defined(__GNUC__) && !defined(__clang__)
    // GCC -Wnull-dereference false positive on inlined stdexec::run_loop task dispatch.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
    state.loop.run();
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

    if (state.wait_error) { return { .values = std::nullopt, .error = std::move(state.wait_error), .stopped = false }; }
    if (state.stopped) { return { .values = std::nullopt, .error = std::nullopt, .stopped = true }; }
    return { .values = std::move(values), .error = std::nullopt, .stopped = false };
  }

}// namespace detail

/**
 * Blocks until `sender` completes.
 *
 * Throws `std::system_error` on `set_error` when exceptions are available,
 * otherwise terminates. Returns a disengaged optional on `set_stopped`.
 *
 * @param sender Sender to start and wait for.
 */
template<detail::sync_waitable_sender Sender>
[[nodiscard]] auto sync_wait(Sender &&sender) -> std::optional<detail::sync_wait_value_tuple_t<Sender>>
{
  auto outcome = detail::sync_wait_outcome_impl(std::forward<Sender>(sender));
  if (outcome.failed()) {
#if VKEXEC_HAS_EXCEPTIONS
    detail::throw_error(outcome.take_error());
#else
    std::terminate();
#endif
  }
  return std::move(outcome.values);
}

/**
 * Blocks until `sender` completes and returns sender error completion as an outcome.
 *
 * Prefer this in tests and `-fno-exceptions` call sites that must inspect errors.
 *
 * @param sender Sender to start and wait for.
 */
template<detail::sync_waitable_sender Sender>
[[nodiscard]] auto try_sync_wait(Sender &&sender) -> detail::sync_wait_outcome_t<Sender>
{ return detail::sync_wait_outcome_impl(std::forward<Sender>(sender)); }

/**
 * Blocks until `sender` completes and unwraps its single value completion.
 *
 * @param sender Sender that completes with exactly one value type.
 * @return The unwrapped value, or `result` failure on error/stop.
 */
template<detail::sync_waitable_sender Sender>
[[nodiscard]] auto try_sync_wait_value(Sender &&sender)
  -> result<detail::sync_unwrapped_value_t<detail::sync_wait_value_tuple_t<Sender>>>
{
  auto outcome = try_sync_wait(std::forward<Sender>(sender));
  if (outcome.failed()) { return unexpected(outcome.take_error()); }
  if (outcome.stopped || !outcome.values.has_value()) {
    return fail(errc::cancelled, "sender completed with set_stopped");
  }
  return detail::take_sync_value(std::move(*outcome.values));
}

/**
 * Blocks until `sender` completes and returns its single value.
 *
 * Throws `std::system_error` on failure/stop when exceptions are enabled; otherwise
 * calls `std::terminate`.
 *
 * @param sender Sender that completes with exactly one value type.
 */
template<detail::sync_waitable_sender Sender>
[[nodiscard]] auto sync_wait_value(Sender &&sender)
  -> detail::sync_unwrapped_value_t<detail::sync_wait_value_tuple_t<Sender>>
{
  auto outcome = try_sync_wait_value(std::forward<Sender>(sender));
  if (!outcome) {
#if VKEXEC_HAS_EXCEPTIONS
    detail::throw_error(outcome.error());
#else
    std::terminate();
#endif
  }
  return std::move(*outcome);
}

}// namespace vkexec

#endif// VKEXEC_SYNC_WAIT_HPP
