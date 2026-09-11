#ifndef VKEXEC_DETAIL_SYNC_SENDER_HPP
#define VKEXEC_DETAIL_SYNC_SENDER_HPP

#include <vkexec/detail/result.hpp>
#include <vkexec/error.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace vkexec::detail {

namespace ex = stdexec;

template<class Value, class Factory>
  requires std::invocable<Factory>
           && std::same_as<detail::result<std::remove_cvref_t<Value>>, std::invoke_result_t<Factory>>
struct sync_sender
{
  using sender_concept = ex::sender_t;
  using value_type = std::remove_cvref_t<Value>;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(value_type), ex::set_error_t(error), ex::set_stopped_t()>;

  Factory factory{};

  template<class Receiver> struct op_state
  {
    Factory factory;
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

      if (result<value_type> produced = factory(); produced) {
        ex::set_value(std::move(rcvr), detail::expected_take(produced));
      } else {
        ex::set_error(std::move(rcvr), std::move(produced.error()));
      }
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = std::move(factory), .receiver = std::move(receiver) }; }
};

template<class Value, class Factory>
[[nodiscard]] auto make_sync_sender(Factory &&factory) -> sync_sender<Value, std::remove_cvref_t<Factory>>
{ return sync_sender<Value, std::remove_cvref_t<Factory>>{ .factory = std::forward<Factory>(factory) }; }

template<class Factory>
  requires std::invocable<Factory> && std::same_as<detail::status, std::invoke_result_t<Factory>>
struct sync_void_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  Factory factory{};

  template<class Receiver> struct op_state
  {
    Factory factory;
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

      if (status done = factory(); done) {
        ex::set_value(std::move(rcvr));
      } else {
        ex::set_error(std::move(rcvr), std::move(done.error()));
      }
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = std::move(factory), .receiver = std::move(receiver) }; }
};

template<class Factory>
[[nodiscard]] auto make_sync_void_sender(Factory &&factory) -> sync_void_sender<std::remove_cvref_t<Factory>>
{ return sync_void_sender<std::remove_cvref_t<Factory>>{ .factory = std::forward<Factory>(factory) }; }

template<class Value>
using sync_sender_fn =
  sync_sender<std::remove_cvref_t<Value>, std::function<detail::result<std::remove_cvref_t<Value>>()>>;

using sync_void_sender_fn = sync_void_sender<std::function<detail::status()>>;

template<class Value>
[[nodiscard]] inline auto make_sync_sender_fn(std::function<detail::result<std::remove_cvref_t<Value>>()> factory)
  -> sync_sender_fn<Value>
{ return sync_sender_fn<Value>{ .factory = std::move(factory) }; }

[[nodiscard]] inline auto make_sync_void_sender_fn(std::function<detail::status()> factory) -> sync_void_sender_fn
{ return sync_void_sender_fn{ .factory = std::move(factory) }; }

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SYNC_SENDER_HPP
