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

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  template<class Receiver> auto set_factory_exception(Receiver &&receiver) noexcept -> void
  {
    // start() is noexcept; map unexpected throws to set_error without allocating.
    ex::set_error(std::forward<Receiver>(receiver), error{ .code = make_error_code(errc::io_error), .detail = {} });
  }

}// namespace detail

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
      Receiver rcvr = std::move(receiver);
      auto const token = ex::get_stop_token(ex::get_env(rcvr));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(rcvr));
          return;
        }
      }

      std::optional<result<value_type>> produced;
      try {
        produced.emplace(factory());
      } catch (...) {
        detail::set_factory_exception(std::move(rcvr));
        return;
      }
      if (*produced) {
        ex::set_value(std::move(rcvr), expected_take(*produced));
      } else {
        ex::set_error(std::move(rcvr), std::move(produced->error()));
      }
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) &
    noexcept(std::is_nothrow_copy_constructible_v<factory_type>
             && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) &&
    noexcept(std::is_nothrow_move_constructible_v<factory_type>
             && std::is_nothrow_move_constructible_v<Receiver>)
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
      Receiver rcvr = std::move(receiver);
      auto const token = ex::get_stop_token(ex::get_env(rcvr));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(rcvr));
          return;
        }
      }

      status done;
      try {
        done = factory();
      } catch (...) {
        detail::set_factory_exception(std::move(rcvr));
        return;
      }
      if (done) {
        ex::set_value(std::move(rcvr));
      } else {
        ex::set_error(std::move(rcvr), std::move(done.error()));
      }
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) &
    noexcept(std::is_nothrow_copy_constructible_v<factory_type>
             && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) &&
    noexcept(std::is_nothrow_move_constructible_v<factory_type>
             && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = std::move(factory), .receiver = std::move(receiver) }; }
};

//! Builds a `void_sender` from a factory returning `status`.
[[nodiscard]] inline auto make_void_sender(std::function<status()> factory) -> void_sender
{ return void_sender{ .factory = std::move(factory) }; }

}// namespace vkexec

#endif// VKEXEC_SENDER_HPP
