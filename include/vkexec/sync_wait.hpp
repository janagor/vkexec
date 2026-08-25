#ifndef VKEXEC_SYNC_WAIT_HPP
#define VKEXEC_SYNC_WAIT_HPP

#include <vkexec/error.hpp>

#include <stdexec/execution.hpp>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  struct sync_wait_env
  {
    template<ex::one_of<ex::get_scheduler_t, ex::get_start_scheduler_t, ex::get_delegation_scheduler_t> Query>
    [[nodiscard]] constexpr auto query(Query) const noexcept -> ex::run_loop::scheduler
    { return loop_->get_scheduler(); }

    [[nodiscard]] constexpr auto query(ex::__root_t) const noexcept -> bool { return true; }

    ex::run_loop *loop_{ nullptr };
  };

  struct sync_wait_state
  {
    std::optional<error> error_{};
    bool stopped_{ false };
    ex::run_loop loop_{};
  };

  template<class... Values> struct sync_wait_receiver
  {
    using receiver_concept = ex::receiver_t;

    sync_wait_state *state_{ nullptr };
    std::optional<std::tuple<Values...>> *values_{ nullptr };

    template<class... As> auto set_value(As &&...args) noexcept -> void
    {
      values_->emplace(std::forward<As>(args)...);
      state_->loop_.finish();
    }

    auto set_error(error err) noexcept -> void
    {
      state_->error_ = std::move(err);
      state_->loop_.finish();
    }

    auto set_stopped() noexcept -> void
    {
      state_->stopped_ = true;
      state_->loop_.finish();
    }

    [[nodiscard]] auto get_env() const noexcept -> sync_wait_env { return sync_wait_env{ &state_->loop_ }; }
  };

  template<class CvSender, class Continuation>
  using sync_wait_result_t = ex::__value_types_of_t<CvSender,
    sync_wait_env,
    ex::__mtransform<ex::__q<std::decay_t>, Continuation>,
    ex::__q<ex::__msingle>>;

  template<class CvSender>
  using sync_wait_value_tuple_t = sync_wait_result_t<CvSender, ex::__qq<std::tuple>>;

  template<class CvSender>
  using sync_wait_receiver_t = sync_wait_result_t<CvSender, ex::__q<sync_wait_receiver>>;

  template<class CvSender>
  concept sync_waitable_sender = ex::sender_in<CvSender, sync_wait_env> && requires {
    { ex::__count_of<ex::set_value_t, CvSender, sync_wait_env>::value } -> std::same_as<const int>;
  } && ex::__count_of<ex::set_value_t, CvSender, sync_wait_env>::value == 1 && ex::sender_to<CvSender,
    sync_wait_receiver_t<CvSender>>;

  template<sync_waitable_sender CvSender>
  auto sync_wait_impl(CvSender &&sender) -> result<std::optional<sync_wait_value_tuple_t<CvSender>>>
  {
    sync_wait_state state{};
    std::optional<sync_wait_value_tuple_t<CvSender>> values{};

    auto op = ex::connect(std::forward<CvSender>(sender), sync_wait_receiver_t<CvSender>{ &state, &values });
    ex::start(op);
    state.loop_.run();

    if (state.error_) { return std::unexpected(*std::move(state.error_)); }
    if (state.stopped_) { return std::optional<sync_wait_value_tuple_t<CvSender>>{}; }
    return values;
  }

}// namespace detail

/// Exception-free blocking wait for vkexec senders using `set_error_t(vkexec::error)`.
///
/// - Success: `result` holds `optional` with the value tuple.
/// - Stopped: `result` holds disengaged `optional` (not an error).
/// - Failure: `unexpected(error)`.
template<detail::sync_waitable_sender Sender>
[[nodiscard]] auto sync_wait(Sender &&sender) -> result<std::optional<detail::sync_wait_value_tuple_t<Sender>>>
{
  return detail::sync_wait_impl(std::forward<Sender>(sender));
}

}// namespace vkexec

#endif// VKEXEC_SYNC_WAIT_HPP
