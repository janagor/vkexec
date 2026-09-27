#ifndef VKEXEC_DETAIL_STOP_HPP
#define VKEXEC_DETAIL_STOP_HPP

//! \file
//! Shared receiver stop-token queries for vkexec senders.

#include <stdexec/execution.hpp>

#include <type_traits>

namespace vkexec::detail {

namespace ex = stdexec;

template<class Receiver> [[nodiscard]] auto receiver_stop_token(Receiver &receiver) noexcept
{ return ex::get_stop_token(ex::get_env(receiver)); }

template<class StopToken> [[nodiscard]] constexpr auto stop_requested(StopToken const &token) noexcept -> bool
{
  if constexpr (ex::unstoppable_token<std::remove_cvref_t<StopToken>>) {
    return false;
  } else {
    return token.stop_requested();
  }
}

template<class Receiver> [[nodiscard]] auto receiver_stop_requested(Receiver &receiver) noexcept -> bool
{ return stop_requested(receiver_stop_token(receiver)); }

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_STOP_HPP
