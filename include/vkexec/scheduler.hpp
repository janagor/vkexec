#ifndef VKEXEC_SCHEDULER_HPP
#define VKEXEC_SCHEDULER_HPP

#include <vkexec/context.hpp>

#include <stdexec/execution.hpp>

#include <exception>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

struct schedule_sender {
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

  context *ctx{ nullptr };

  template<class Receiver>
  struct op_state {
    Receiver receiver;
    auto start() noexcept -> void
    {
      try {
        ex::set_value(std::move(receiver));
      } catch (...) {
        ex::set_error(std::move(receiver), std::current_exception());
      }
    }
  };

  template<class Receiver>
  auto connect(Receiver receiver) const noexcept
  {
    return op_state<Receiver>{ std::move(receiver) };
  }
};

class scheduler {
public:
  explicit scheduler(context *ctx) noexcept : ctx_(ctx) {}

  [[nodiscard]] auto schedule() const noexcept -> schedule_sender { return schedule_sender{ ctx_ }; }
  [[nodiscard]] auto get_context() const noexcept -> context * { return ctx_; }

  friend auto operator==(scheduler const &lhs, scheduler const &rhs) noexcept -> bool
  {
    return lhs.ctx_ == rhs.ctx_;
  }

private:
  context *ctx_{ nullptr };
};

inline auto context::get_scheduler() noexcept -> scheduler { return scheduler{ this }; }

} // namespace vkexec

#endif // VKEXEC_SCHEDULER_HPP
