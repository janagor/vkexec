#ifndef VKEXEC_SENDER_HPP
#define VKEXEC_SENDER_HPP

//! \file
//! Synchronous factory senders that run a `result`-returning callable in `start()`.

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <exception>
#include <optional>
#include <type_traits>
#include <utility>

#ifndef VKEXEC_ENABLE_EXCEPTIONS
#define VKEXEC_ENABLE_EXCEPTIONS 1
#endif

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  // NOLINTNEXTLINE(readability-identifier-naming)
  template<class T> inline constexpr bool is_factory_result_v = false;

  // NOLINTNEXTLINE(readability-identifier-naming)
  template<class T> inline constexpr bool is_factory_result_v<result<T>> = true;

  template<class T>
  concept factory_result = is_factory_result_v<std::remove_cvref_t<T>>;

  template<class T> struct factory_result_traits;

  template<class T> struct factory_result_traits<result<T>>
  {
    using value_type = T;
    using completion_signatures =
      ex::completion_signatures<ex::set_value_t(T), ex::set_error_t(error), ex::set_stopped_t()>;
  };

  template<> struct factory_result_traits<result<void>>
  {
    using value_type = void;
    using completion_signatures =
      ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;
  };

  template<class F> using factory_result_type = std::remove_cvref_t<decltype(std::declval<F &>()())>;

  template<class F> using factory_value_t = factory_result_traits<factory_result_type<F>>::value_type;

  template<class T> struct factory_completion
  {
    std::optional<T> value;
    std::optional<error> failure;
  };

  template<> struct factory_completion<void>
  {
    std::optional<error> failure;
  };

  template<class F> auto invoke_factory(F &factory) -> factory_completion<factory_value_t<F>>
  {
    using value_type = factory_value_t<F>;

    auto produced = factory();

    if (!produced) {
      if constexpr (std::is_void_v<value_type>) {
        return factory_completion<value_type>{ .failure = std::move(produced.error()) };
      } else {
        return factory_completion<value_type>{ .value = std::nullopt, .failure = std::move(produced.error()) };
      }
    }

    if constexpr (std::is_void_v<value_type>) {
      return factory_completion<value_type>{};
    } else {
      return factory_completion<value_type>{ .value = expected_take(produced), .failure = std::nullopt };
    }
  }

  template<class Receiver, class T>
  // The completion is consumed member-by-member without moving the aggregate itself.
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  auto deliver_factory_completion(Receiver &&receiver, factory_completion<T> &&completion) noexcept -> void
  {
    if (completion.failure) {
      ex::set_error(std::forward<Receiver>(receiver), std::move(*completion.failure));
      return;
    }

    if constexpr (std::is_void_v<T>) {
      ex::set_value(std::forward<Receiver>(receiver));
    } else {
      if (completion.value) {
        ex::set_value(std::forward<Receiver>(receiver), std::move(*completion.value));
        return;
      }
      std::terminate();
    }
  }

}// namespace detail

template<class F>
concept factory_callable = detail::factory_result<decltype(std::declval<F &>()())>;

/**
 * Synchronous sender backed by a concrete result-returning callable.
 *
 * The callable is invoked when the operation is started. Its `result<T>` or
 * `status` return value is mapped to `set_value` / `set_error`, with pre-start
 * cancellation mapped to `set_stopped`.
 *
 * Prefer `make_sender()` for construction. This type is an implementation
 * utility for synchronous factory CPOs rather than a primary user-facing API.
 */
template<factory_callable F> struct factory_sender
{
  using factory_type = F;
  using result_type = detail::factory_result_type<F>;
  using traits = detail::factory_result_traits<result_type>;
  using value_type = traits::value_type;
  using sender_concept = ex::sender_t;
  using completion_signatures = traits::completion_signatures;

  [[no_unique_address]] F factory;

  template<class Receiver> struct op_state
  {
    [[no_unique_address]] F factory;
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

      std::optional<detail::factory_completion<value_type>> completion;
#if VKEXEC_ENABLE_EXCEPTIONS
      try {
#endif
        completion.emplace(detail::invoke_factory(factory));
#if VKEXEC_ENABLE_EXCEPTIONS
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
        return;
      }
#endif

      detail::deliver_factory_completion(std::move(receiver), std::move(*completion));
    }
  };

  template<class Receiver>
    requires std::copy_constructible<F>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(
    std::is_nothrow_copy_constructible_v<F> && std::is_nothrow_move_constructible_v<Receiver>) -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = factory, .receiver = std::move(receiver) }; }

  template<class Receiver>
    requires std::move_constructible<F>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<F> && std::is_nothrow_move_constructible_v<Receiver>) -> op_state<Receiver>
  { return op_state<Receiver>{ .factory = std::move(factory), .receiver = std::move(receiver) }; }
};

//! Builds a `factory_sender` from a callable returning `result<T>` or `status`.
template<class F>
  requires factory_callable<std::decay_t<F>>
[[nodiscard]] auto make_sender(F &&factory) -> factory_sender<std::decay_t<F>>
{ return factory_sender<std::decay_t<F>>{ .factory = std::forward<F>(factory) }; }

}// namespace vkexec

#endif// VKEXEC_SENDER_HPP
