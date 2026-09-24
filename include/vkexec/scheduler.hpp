#ifndef VKEXEC_SCHEDULER_HPP
#define VKEXEC_SCHEDULER_HPP

//! \file
//! stdexec scheduler that completes on a `context` host agent.

#include <vkexec/context.hpp>
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

/**
 * Sender environment: value completions finish on this context's scheduler.
 *
 * Also advertises the vkexec `domain` so algorithms can lower to Vulkan-native
 * senders when overloads exist.
 */
struct scheduler_env
{
  context *ctx{ nullptr };

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
 * Completes with `set_value()` on the context host agent, or `set_error` if
 * enqueueing the host task fails. A null `ctx` completes inline with `set_value`.
 */
struct schedule_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error)>;

  context *ctx{ nullptr };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    Receiver receiver;

    auto start() noexcept -> void
    {
      if (ctx == nullptr) {
        ex::set_value(std::move(receiver));
        return;
      }

#if VKEXEC_ENABLE_EXCEPTIONS
      std::optional<status> enqueued;
      try {
        enqueued.emplace(ctx->enqueue_host([this]() noexcept -> void {
          ex::set_value(std::move(receiver));
        }));
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
        return;
      }
#else
      auto enqueued = ctx->enqueue_host([this]() noexcept -> void {
        ex::set_value(std::move(receiver));
      });
#endif

#if VKEXEC_ENABLE_EXCEPTIONS
      auto &enqueue_status = *enqueued;
#else
      auto &enqueue_status = enqueued;
#endif
      if (!enqueue_status) {
        ex::set_error(std::move(receiver), std::move(enqueue_status.error()));
      }
    }
  };

  template<class Receiver>
  // cppcheck-suppress functionStatic
  [[nodiscard]] auto connect(Receiver receiver) & noexcept -> op_state<Receiver>
  { return op_state<Receiver>{ ctx, std::move(receiver) }; }

  template<class Receiver>
  // cppcheck-suppress functionStatic
  [[nodiscard]] auto connect(Receiver receiver) && noexcept -> op_state<Receiver>
  { return op_state<Receiver>{ ctx, std::move(receiver) }; }
};

/**
 * stdexec scheduler bound to a `context`.
 *
 * `schedule()` starts work on the context host agent. Equality compares the
 * underlying `context*` pointers.
 *
 * @see context::get_scheduler, schedule_sender, domain
 */
class scheduler
{
public:
  //! Constructs a scheduler for `ctx` (may be null for a no-op schedule).
  explicit scheduler(context *ctx) noexcept : ctx_(ctx) {}

  //! Returns a sender that completes on this scheduler's host agent.
  [[nodiscard]] auto schedule() const noexcept -> schedule_sender { return schedule_sender{ .ctx = ctx_ }; }
  //! Returns the bound context pointer (may be null).
  [[nodiscard]] auto get_context() const noexcept -> context * { return ctx_; }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto query(ex::get_completion_domain_t<ex::set_value_t> /*tag*/) const noexcept -> domain
  { return {}; }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto query(ex::get_domain_t /*tag*/) const noexcept -> domain { return {}; }

  friend auto operator==(scheduler const &lhs, scheduler const &rhs) noexcept -> bool { return lhs.ctx_ == rhs.ctx_; }

private:
  context *ctx_{ nullptr };
};

inline auto scheduler_env::query(ex::get_completion_scheduler_t<ex::set_value_t> /*tag*/) const noexcept -> scheduler
{ return scheduler{ ctx }; }

inline auto context::get_scheduler() noexcept -> scheduler { return scheduler{ this }; }

/**
 * True when `Pred` is a sender whose value completion scheduler is `vkexec::scheduler`.
 *
 * Used by pass and graphics adaptors to require a vkexec schedule predecessor.
 */
template<class Pred>
concept vkexec_predecessor = ex::sender<Pred> && requires(Pred const &pred) {
  { ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred)) } -> std::same_as<scheduler>;
};

}// namespace vkexec

#endif// VKEXEC_SCHEDULER_HPP
