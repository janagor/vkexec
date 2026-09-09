#ifndef VKEXEC_DETAIL_SYNC_WAIT_OUTCOME_HPP
#define VKEXEC_DETAIL_SYNC_WAIT_OUTCOME_HPP

#include <vkexec/error.hpp>

#include <cstdlib>
#include <optional>
#include <tuple>

namespace vkexec::detail {

template<class... Values> struct sync_wait_outcome
{
  std::optional<std::tuple<Values...>> values{};
  std::optional<error> error;
  bool stopped{ false };

  [[nodiscard]] auto has_value() const noexcept -> bool { return values.has_value(); }
  [[nodiscard]] auto failed() const noexcept -> bool { return error.has_value(); }

  [[nodiscard]] auto take_error() const -> vkexec::error
  {
    if (!error.has_value()) { std::terminate(); }
    return *error;
  }
};

template<class Value> struct sync_unwrapped_value
{
  using type = Value;
};

template<class Head, class... Rest> struct sync_unwrapped_value<std::tuple<Head, Rest...>>
{
  using type = sync_unwrapped_value<Head>::type;
};

template<class Value> using sync_unwrapped_value_t = sync_unwrapped_value<Value>::type;

template<class Value> [[nodiscard]] inline auto take_sync_value(Value &&value) -> decltype(auto)
{
  if constexpr (requires { std::get<0>(value); }) {
    return take_sync_value(std::get<0>(std::forward<Value>(value)));
  } else {
    return std::forward<Value>(value);
  }
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SYNC_WAIT_OUTCOME_HPP
