#pragma once

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
    void start() noexcept
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

  [[nodiscard]] schedule_sender schedule() const noexcept { return schedule_sender{ ctx_ }; }
  [[nodiscard]] context *get_context() const noexcept { return ctx_; }

  friend bool operator==(scheduler a, scheduler b) noexcept { return a.ctx_ == b.ctx_; }

private:
  context *ctx_{ nullptr };
};

inline scheduler context::get_scheduler() noexcept { return scheduler{ this }; }

} // namespace vkexec
