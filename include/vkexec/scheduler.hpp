#ifndef VKEXEC_SCHEDULER_HPP
#define VKEXEC_SCHEDULER_HPP

//! \file
//! stdexec scheduler that completes on a `context` host agent.

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/stop.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

class scheduler;

namespace detail {
  struct scheduler_access;

  /**
   * Sender environment that advertises the vkexec domain but no completion scheduler.
   *
   * Use for senders that are vkexec-domain but do not guarantee where any completion
   * signal is delivered.
   */
  struct domain_env
  {
    // cppcheck-suppress functionStatic
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr auto query(ex::get_completion_domain_t<ex::set_value_t> /*tag*/) const noexcept -> domain
    { return {}; }

    // cppcheck-suppress functionStatic
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr auto query(ex::get_domain_t /*tag*/) const noexcept -> domain { return {}; }
  };

}// namespace detail

/**
 * Attributes for senders whose set_value completion is guaranteed to occur on
 * the context host scheduler.
 *
 * No scheduling guarantee is made for set_error or set_stopped.
 *
 * Also advertises the vkexec `domain` so algorithms can lower to Vulkan-native
 * senders when overloads exist.
 */
struct scheduler_env
{
  detail::context_handle state;

  [[nodiscard]] auto query(ex::get_completion_scheduler_t<ex::set_value_t> /*tag*/) const noexcept -> scheduler;

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto query(ex::get_completion_domain_t<ex::set_value_t> /*tag*/) const noexcept -> domain
  { return {}; }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto query(ex::get_domain_t /*tag*/) const noexcept -> domain { return {}; }
};

/**
 * Sender produced by `scheduler::schedule()`.
 *
 * A pre-requested stop completes inline with `set_stopped`. Otherwise the
 * queued task checks for stop again before delivering `set_value` on the host
 * agent. An empty state completes inline with `set_error(errc::invalid_argument)`.
 * A pre-requested stop takes precedence. No scheduling guarantee is made for
 * `set_error` or `set_stopped`.
 */
struct schedule_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  detail::context_handle state;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    Receiver receiver;

    auto start() noexcept -> void
    {
      if (detail::receiver_stop_requested(receiver)) {
        ex::set_stopped(std::move(receiver));
        return;
      }

      if (!state) {
        ex::set_error(std::move(receiver), error{ .code = make_error_code(errc::invalid_argument), .detail = {} });
        return;
      }

#if VKEXEC_HAS_EXCEPTIONS
      std::optional<status> enqueued;
      try {
        enqueued.emplace(detail::enqueue_host(state, [this]() noexcept -> void {
          if (detail::receiver_stop_requested(receiver)) {
            ex::set_stopped(std::move(receiver));
          } else {
            ex::set_value(std::move(receiver));
          }
        }));
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
        return;
      }
#else
      auto enqueued = detail::enqueue_host(state, [this]() noexcept -> void {
        if (detail::receiver_stop_requested(receiver)) {
          ex::set_stopped(std::move(receiver));
        } else {
          ex::set_value(std::move(receiver));
        }
      });
#endif

#if VKEXEC_HAS_EXCEPTIONS
      auto &enqueue_status = *enqueued;
#else
      auto &enqueue_status = enqueued;
#endif
      if (!enqueue_status) { ex::set_error(std::move(receiver), std::move(enqueue_status.error())); }
    }
  };

  template<class Receiver>
  // cppcheck-suppress functionStatic
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ state, std::move(receiver) }; }

  template<class Receiver>
  // cppcheck-suppress functionStatic
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  { return op_state<Receiver>{ std::move(state), std::move(receiver) }; }
};

/**
 * stdexec scheduler bound to a shared context runtime.
 *
 * `schedule()` starts work on the context host agent. Equality compares the
 * underlying runtime state.
 *
 * @see context::get_scheduler, schedule_sender, domain
 */
class scheduler
{
public:
  //! A null context is representable for composition; executing it reports invalid_argument.
  explicit scheduler(context const *ctx) noexcept
    : state_(ctx != nullptr ? detail::context_access::state(*ctx) : detail::context_handle{})
  {}

  //! Returns a sender that completes on this scheduler's host agent.
  [[nodiscard]] auto schedule() const noexcept -> schedule_sender { return schedule_sender{ .state = state_ }; }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto query(ex::get_completion_domain_t<ex::set_value_t> /*tag*/) const noexcept -> domain
  { return {}; }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto query(ex::get_domain_t /*tag*/) const noexcept -> domain { return {}; }

  friend auto operator==(scheduler const &lhs, scheduler const &rhs) noexcept -> bool
  { return lhs.state_.get() == rhs.state_.get(); }
  friend struct detail::scheduler_access;

private:
  explicit scheduler(detail::context_handle state) noexcept : state_(std::move(state)) {}
  detail::context_handle state_;
};

namespace detail {
  struct scheduler_access
  {
    [[nodiscard]] static auto state(scheduler const &sched) noexcept -> context_handle { return sched.state_; }
    [[nodiscard]] static auto make(context_handle state) noexcept -> scheduler { return scheduler{ std::move(state) }; }
  };
}// namespace detail

inline auto scheduler_env::query(ex::get_completion_scheduler_t<ex::set_value_t> /*tag*/) const noexcept -> scheduler
{ return detail::scheduler_access::make(state); }

inline auto context::get_scheduler() noexcept -> scheduler { return detail::scheduler_access::make(impl_); }

/**
 * True when `Pred` is a sender whose value completion is guaranteed to happen
 * on a `vkexec::scheduler` (not merely a sender associated with vkexec).
 *
 * Used by pass and graphics adaptors to require a scheduler-completing predecessor.
 */
template<class Pred>
concept vkexec_predecessor = ex::sender<Pred> && requires(Pred const &pred) {
  { ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred)) } -> std::same_as<scheduler>;
};

}// namespace vkexec

#endif// VKEXEC_SCHEDULER_HPP
