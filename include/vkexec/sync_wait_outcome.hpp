#ifndef VKEXEC_SYNC_WAIT_OUTCOME_HPP
#define VKEXEC_SYNC_WAIT_OUTCOME_HPP

//! \file
//! Non-throwing outcome type and helpers for `try_sync_wait` / `sync_wait_value`.

#include <vkexec/error.hpp>

#include <cstdlib>
#include <optional>
#include <tuple>

namespace vkexec {

/**
 * Outcome of a blocking wait: values, error, or stopped (exactly one path).
 *
 * @see try_sync_wait, sync_wait_outcome
 */
template<class... Values> struct sync_wait_outcome
{
  std::optional<std::tuple<Values...>> values{};
  std::optional<error> error;
  bool stopped{ false };

  //! True when the sender completed with `set_value`.
  [[nodiscard]] auto has_value() const noexcept -> bool { return values.has_value(); }
  //! True when the sender completed with `set_error`.
  [[nodiscard]] auto failed() const noexcept -> bool { return error.has_value(); }

  /**
   * Returns the stored error.
   *
   * Terminates if `failed()` is false (programming error).
   */
  [[nodiscard]] auto take_error() const -> vkexec::error
  {
    if (!error.has_value()) { std::terminate(); }
    return *error;
  }
};

//! Trait: unwraps nested single-element tuples to the underlying value type.
namespace detail {

  template<class Value> struct sync_unwrapped_value
  {
    using type = Value;
  };

  template<class Head, class... Rest> struct sync_unwrapped_value<std::tuple<Head, Rest...>>
  {
    using type = sync_unwrapped_value<Head>::type;
  };

  template<class Value> using sync_unwrapped_value_t = sync_unwrapped_value<Value>::type;

  /**
   * Recursively unwraps a single-element tuple completion to the inner value.
   *
   * Used by `try_sync_wait_value` / `sync_wait_value`.
   */
  template<class Value> [[nodiscard]] inline auto take_sync_value(Value &&value) -> decltype(auto)
  {
    if constexpr (requires { std::get<0>(value); }) {
      return take_sync_value(std::get<0>(std::forward<Value>(value)));
    } else {
      return std::forward<Value>(value);
    }
  }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_SYNC_WAIT_OUTCOME_HPP
