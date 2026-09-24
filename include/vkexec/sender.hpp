#ifndef VKEXEC_SENDER_HPP
#define VKEXEC_SENDER_HPP

//! \file
//! Synchronous factory senders that run a `result`-returning callable in `start()`.

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <stdexec/execution.hpp>

#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#ifndef VKEXEC_ENABLE_EXCEPTIONS
#define VKEXEC_ENABLE_EXCEPTIONS 1
#endif

namespace vkexec {

namespace ex = stdexec;

/**
 * Sender that invokes `factory` synchronously in `start()` and completes with its `result<Value>`.
 *
 * Honours stop tokens with `set_stopped`. Used by most `factory::make_*` CPOs.
 *
 * @see make_sender
 */
template<class Value> struct sender
{
  using factory_type = std::function<result<std::remove_cvref_t<Value>>()>;
  using sender_concept = ex::sender_t;
  using value_type = std::remove_cvref_t<Value>;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(value_type), ex::set_error_t(error), ex::set_stopped_t()>;

  factory_type factory;

  template<class Receiver> struct op_state
  {
    factory_type factory;
    Receiver receiver;

    auto start() noexcept -> void
    {
      auto const token = ex::get_stop_token(ex::get_env(receiver));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(receiver));
          return;
        }
      }

      std::optional<value_type> value;
      std::optional<error> failure;
#if VKEXEC_ENABLE_EXCEPTIONS
      try {
#endif
        auto produced = factory();
        if (produced) {
          value.emplace(expected_take(produced));
        } else {
          failure.emplace(std::move(produced.error()));
        }
#if VKEXEC_ENABLE_EXCEPTIONS
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
        return;
      }
#endif

      if (value) {
        ex::set_value(std::move(receiver), std::move(*value));
      } else {
        ex::set_error(std::move(receiver), std::move(*failure));
      }
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(
    std::is_nothrow_copy_constructible_v<factory_type> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<factory_type> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = std::move(factory), .receiver = std::move(receiver) }; }
};

//! Builds a `sender` from a factory returning `result<Value>`.
template<class Value>
[[nodiscard]] auto make_sender(std::function<result<std::remove_cvref_t<Value>>()> factory) -> sender<Value>
{ return sender<Value>{ .factory = std::move(factory) }; }

/**
 * Sender that invokes a `status`-returning factory in `start()`.
 *
 * Completes with `set_value()` on success, `set_error` on failure, or `set_stopped`.
 */
struct void_sender
{
  using factory_type = std::function<status()>;
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  factory_type factory;

  template<class Receiver> struct op_state
  {
    factory_type factory;
    Receiver receiver;

    auto start() noexcept -> void
    {
      auto const token = ex::get_stop_token(ex::get_env(receiver));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(receiver));
          return;
        }
      }

      std::optional<error> failure;
#if VKEXEC_ENABLE_EXCEPTIONS
      try {
#endif
        auto done = factory();
        if (!done) { failure.emplace(std::move(done.error())); }
#if VKEXEC_ENABLE_EXCEPTIONS
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
        return;
      }
#endif

      if (failure) {
        ex::set_error(std::move(receiver), std::move(*failure));
      } else {
        ex::set_value(std::move(receiver));
      }
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(
    std::is_nothrow_copy_constructible_v<factory_type> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<factory_type> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = std::move(factory), .receiver = std::move(receiver) }; }
};

//! Builds a `void_sender` from a factory returning `status`.
[[nodiscard]] inline auto make_void_sender(std::function<status()> factory) -> void_sender
{ return void_sender{ .factory = std::move(factory) }; }

}// namespace vkexec

#endif// VKEXEC_SENDER_HPP
