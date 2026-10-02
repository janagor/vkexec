#ifndef VKEXEC_DETAIL_NORMALIZE_ERRORS_HPP
#define VKEXEC_DETAIL_NORMALIZE_ERRORS_HPP

//! \file
//! Maps stdexec composition exceptions back to vkexec's error channel.

#include <vkexec/error.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#if VKEXEC_HAS_EXCEPTIONS
#include <exception>
#endif
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec::detail {

template<class Tuple> struct single_sender_value;

template<class Value> struct single_sender_value<std::tuple<Value>>
{
  using type = Value;
};

template<class Sender>
using single_sender_value_t =
  single_sender_value<stdexec::value_types_of_t<Sender, stdexec::env<>, std::tuple, std::type_identity_t>>::type;

template<class Sender> class normalized_error_sender
{
public:
  using sender_concept = stdexec::sender_t;
  using value_type = single_sender_value_t<Sender>;
  using completion_signatures = stdexec::
    completion_signatures<stdexec::set_value_t(value_type), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  explicit normalized_error_sender(Sender sender) : sender_(std::move(sender)) {}

  [[nodiscard]] auto get_env() const noexcept -> decltype(stdexec::get_env(std::declval<Sender const &>()))
  { return stdexec::get_env(std::as_const(sender_)); }

  template<class Receiver> struct child_receiver
  {
    using receiver_concept = stdexec::receiver_t;
    Receiver receiver;

    template<class Value> auto set_value(Value &&value) && noexcept -> void
    { stdexec::set_value(std::move(receiver), std::forward<Value>(value)); }

    auto set_error(error err) && noexcept -> void { stdexec::set_error(std::move(receiver), std::move(err)); }

    auto set_error(std::exception_ptr const & /*exception*/) && noexcept -> void
    { stdexec::set_error(std::move(receiver), unexpected_exception_error()); }

    auto set_stopped() && noexcept -> void { stdexec::set_stopped(std::move(receiver)); }

    [[nodiscard]] auto get_env() const noexcept -> decltype(stdexec::get_env(std::declval<Receiver const &>()))
    { return stdexec::get_env(std::as_const(receiver)); }
  };

  template<class Receiver, class Source> struct op_state
  {
    using child_receiver_type = child_receiver<Receiver>;
    using child_operation_type =
      decltype(stdexec::connect(std::declval<Source>(), std::declval<child_receiver_type>()));

    child_operation_type child;

    // Forwarding preserves lvalue connections; an unconditional move would break them.
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    explicit op_state(Source &&source, Receiver receiver) noexcept(
      noexcept(stdexec::connect(std::forward<Source>(source), child_receiver_type{ std::move(receiver) })))
      : child(stdexec::connect(std::forward<Source>(source), child_receiver_type{ std::move(receiver) }))
    {}

    auto start() noexcept -> void { stdexec::start(child); }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(
    noexcept(op_state<Receiver, Sender &>{ sender_, std::move(receiver) })) -> op_state<Receiver, Sender &>
  { return op_state<Receiver, Sender &>{ sender_, std::move(receiver) }; }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    noexcept(op_state<Receiver, Sender>{ std::move(sender_), std::move(receiver) })) -> op_state<Receiver, Sender>
  { return op_state<Receiver, Sender>{ std::move(sender_), std::move(receiver) }; }

private:
  Sender sender_;
};

template<class Sender>
[[nodiscard]] auto normalize_errors(Sender &&sender) -> normalized_error_sender<std::decay_t<Sender>>
{ return normalized_error_sender<std::decay_t<Sender>>{ std::forward<Sender>(sender) }; }

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_NORMALIZE_ERRORS_HPP
