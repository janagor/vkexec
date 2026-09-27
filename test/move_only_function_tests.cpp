#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/move_only_function.hpp>
#include <vkexec/detail/worker_callbacks.hpp>
#include <vkexec/error.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <concepts>
#include <optional>
#include <stop_token>
#include <type_traits>
#include <utility>

namespace {

struct nothrow_task
{
  auto operator()() noexcept -> void {}
};

struct throwing_task
{
  auto operator()() -> void {}
};

struct throwing_destructor_task
{
  throwing_destructor_task() = default;
  throwing_destructor_task(throwing_destructor_task const &) = default;
  auto operator=(throwing_destructor_task const &) -> throwing_destructor_task & = default;
  throwing_destructor_task(throwing_destructor_task &&) = default;
  auto operator=(throwing_destructor_task &&) -> throwing_destructor_task & = default;
  ~throwing_destructor_task() noexcept(false) = default;
  auto operator()() noexcept -> void {}
};

struct throwing_move_task
{
  throwing_move_task() = default;
  throwing_move_task(throwing_move_task const &) = delete;
  auto operator=(throwing_move_task const &) -> throwing_move_task & = delete;
  throwing_move_task(throwing_move_task &&) noexcept(false) = default;
  auto operator=(throwing_move_task &&) -> throwing_move_task & = delete;
  ~throwing_move_task() noexcept = default;
  auto operator()() noexcept -> void {}
};

struct throwing_stop_token
{
  template<class Callback> using callback_type = std::stop_callback<Callback>;

  bool requested{ false };
  bool possible{ true };

  [[nodiscard]] auto stop_requested() const -> bool { return requested; }
  [[nodiscard]] auto stop_possible() const noexcept -> bool { return possible; }

  friend auto operator==(throwing_stop_token const &, throwing_stop_token const &) -> bool = default;
};

struct nothrow_stop
{
  auto operator()() noexcept -> bool { return false; }
};

struct throwing_stop
{
  auto operator()() -> bool { return false; }
};

struct nothrow_done
{
  auto operator()(std::optional<vkexec::error> &&failure, bool /*stopped*/) noexcept -> void
  {
    auto consumed = std::move(failure);
    (void)consumed;
  }
};

struct throwing_done
{
  auto operator()(std::optional<vkexec::error> &&failure, bool /*stopped*/) -> void
  {
    auto consumed = std::move(failure);
    (void)consumed;
  }
};

template<class Task>
concept host_enqueuable = requires(vkexec::context &ctx, Task task) { ctx.enqueue_host(std::move(task)); };

template<class Token>
concept fence_enqueuable = requires(vkexec::context &ctx, Token token, nothrow_done done) {
  ctx.enqueue_fence_wait(VK_NULL_HANDLE, VK_NULL_HANDLE, std::move(token), done);
  ctx.enqueue_borrowed_fence_wait(VK_NULL_HANDLE, std::move(token), done);
};

using task_fn = vkexec::detail::host_task_fn;
using stop_fn = vkexec::detail::stop_fn;
using done_fn = vkexec::detail::done_fn;
using value_fn = vkexec::detail::move_only_function<int()>;

static_assert(std::constructible_from<task_fn, nothrow_task>);
static_assert(!std::constructible_from<task_fn, throwing_task>);
static_assert(!std::constructible_from<task_fn, throwing_destructor_task>);
static_assert(!std::constructible_from<value_fn, nothrow_task>);
static_assert(!std::is_nothrow_move_constructible_v<throwing_move_task>);
static_assert(std::constructible_from<task_fn, throwing_move_task>);
static_assert(noexcept(std::declval<task_fn &>()()));
static_assert(std::constructible_from<stop_fn, nothrow_stop>);
static_assert(!std::constructible_from<stop_fn, throwing_stop>);
static_assert(noexcept(std::declval<stop_fn &>()()));
static_assert(std::constructible_from<done_fn, nothrow_done>);
static_assert(!std::constructible_from<done_fn, throwing_done>);
static_assert(noexcept(std::declval<done_fn &>()(std::declval<std::optional<vkexec::error> &&>(), false)));
static_assert(host_enqueuable<nothrow_task>);
static_assert(!host_enqueuable<throwing_task>);
static_assert(!host_enqueuable<throwing_destructor_task>);
static_assert(!stdexec::stoppable_token<throwing_stop_token>);
static_assert(throwing_stop_token{} == throwing_stop_token{});
static_assert(!fence_enqueuable<throwing_stop_token>);
static_assert(fence_enqueuable<std::stop_token>);

TEST_CASE("noexcept move-only callback moves and invokes", "[vkexec][callback]")
{
  int calls = 0;
  task_fn first{ [&calls]() noexcept -> void { ++calls; } };
  task_fn second{ std::move(first) };
  REQUIRE_FALSE(first);
  second();
  REQUIRE(calls == 1);
}

TEST_CASE("noexcept callback invokes heap-backed callable", "[vkexec][callback]")
{
  std::array<int, 32> values{};
  constexpr int k_expected = 42;
  values.front() = k_expected;
  int result = 0;
  task_fn callback{ [values, &result]() noexcept -> void { result = values.front(); } };
  callback();
  REQUIRE(result == k_expected);
}

TEST_CASE("noexcept callback stores a callable with a throwing move", "[vkexec][callback]")
{
  task_fn callback{ throwing_move_task{} };
  REQUIRE(callback);
  callback();
}

}// namespace
