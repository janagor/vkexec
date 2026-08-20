#ifndef VKEXEC_SCHEDULER_HPP
#define VKEXEC_SCHEDULER_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/domain.hpp>

#include <stdexec/execution.hpp>

#include <exception>
#include <concepts>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

class scheduler;

/// Sender attributes: value completions finish on this context's scheduler.
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

struct schedule_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

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

      Receiver rcvr = std::move(receiver);
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        ctx->enqueue_host([rcvr = std::move(rcvr)]() mutable -> void { ex::set_value(std::move(rcvr)); });
      }
      VKEXEC_CATCH_ALL { ex::set_error(std::move(rcvr), std::current_exception()); }
    }
  };

  // cppcheck-suppress functionStatic
  template<class Receiver> auto connect(this auto &&self, Receiver receiver) noexcept -> op_state<Receiver>
  { return op_state<Receiver>{ self.ctx, std::move(receiver) }; }
};

class scheduler
{
public:
  explicit scheduler(context *ctx) noexcept : ctx_(ctx) {}

  [[nodiscard]] auto schedule() const noexcept -> schedule_sender { return schedule_sender{ .ctx = ctx_ }; }
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

template<class Pred>
concept vkexec_predecessor = ex::sender<Pred> && requires(Pred const &pred) {
  {
    ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred))
  } -> std::same_as<scheduler>;
};

}// namespace vkexec

#endif// VKEXEC_SCHEDULER_HPP
