#ifndef VKEXEC_DETAIL_WORKER_CALLBACKS_HPP
#define VKEXEC_DETAIL_WORKER_CALLBACKS_HPP

//! \file
//! Non-throwing callbacks dispatched by the context worker agents.

#include <vkexec/detail/move_only_function.hpp>
#include <vkexec/error.hpp>

#include <optional>
#include <type_traits>

namespace vkexec::detail {

using host_task_fn = move_only_function<void() noexcept>;
using stop_fn = move_only_function<bool() noexcept>;
using done_fn = move_only_function<void(std::optional<error> &&, bool) noexcept>;

static_assert(std::is_nothrow_move_constructible_v<host_task_fn>);
static_assert(std::is_nothrow_move_constructible_v<stop_fn>);
static_assert(std::is_nothrow_move_constructible_v<done_fn>);
static_assert(std::is_nothrow_destructible_v<host_task_fn>);
static_assert(std::is_nothrow_destructible_v<stop_fn>);
static_assert(std::is_nothrow_destructible_v<done_fn>);
static_assert(std::is_nothrow_move_constructible_v<std::optional<error>>);

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_WORKER_CALLBACKS_HPP
