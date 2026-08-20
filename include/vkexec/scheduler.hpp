#ifndef VKEXEC_SCHEDULER_HPP
#define VKEXEC_SCHEDULER_HPP

#include <vkexec/context.hpp>

#include <stdexec/execution.hpp>

#include <utility>

namespace vkexec {

namespace ex = stdexec;

class scheduler;

/// Sender attributes: value completions finish on this context's scheduler.
struct scheduler_env
{
  context *ctx{ nullptr };

  [[nodiscard]] auto query(ex::get_completion_scheduler_t<ex::set_value_t> /*tag*/) const noexcept -> scheduler;
};

struct schedule_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

  context *ctx{ nullptr };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    Receiver receiver;
    auto start() noexcept -> void { ex::set_value(std::move(receiver)); }
  };

  // cppcheck-suppress functionStatic
  template<class Receiver> auto connect(Receiver receiver) const noexcept
  { return op_state<Receiver>{ std::move(receiver) }; }
};

class scheduler
{
public:
  explicit scheduler(context *ctx) noexcept : ctx_(ctx) {}

  [[nodiscard]] auto schedule() const noexcept -> schedule_sender { return schedule_sender{ .ctx = ctx_ }; }
  [[nodiscard]] auto get_context() const noexcept -> context * { return ctx_; }

  friend auto operator==(scheduler const &lhs, scheduler const &rhs) noexcept -> bool { return lhs.ctx_ == rhs.ctx_; }

private:
  context *ctx_{ nullptr };
};

inline auto scheduler_env::query(ex::get_completion_scheduler_t<ex::set_value_t> /*tag*/) const noexcept -> scheduler
{ return scheduler{ ctx }; }

inline auto context::get_scheduler() noexcept -> scheduler { return scheduler{ this }; }

}// namespace vkexec

#endif// VKEXEC_SCHEDULER_HPP
