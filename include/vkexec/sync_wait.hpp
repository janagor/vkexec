#ifndef VKEXEC_SYNC_WAIT_HPP
#define VKEXEC_SYNC_WAIT_HPP

#include <vkexec/error.hpp>

#include <stdexec/execution.hpp>

#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  struct sync_wait_env
  {
    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    template<ex::__one_of<ex::get_scheduler_t, ex::get_start_scheduler_t, ex::get_delegation_scheduler_t> Query>
    [[nodiscard]] constexpr auto query(Query /*query*/) const noexcept -> ex::run_loop::scheduler
    { return loop->get_scheduler(); }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] static constexpr auto query(ex::__root_t /*query*/) noexcept -> bool { return true; }

    ex::run_loop *loop{ nullptr };
  };

  struct sync_wait_state
  {
    std::optional<error> wait_error;
    bool stopped{ false };
    ex::run_loop loop;
  };

  template<class... Values> struct sync_wait_receiver
  {
    using receiver_concept = ex::receiver_t;

    sync_wait_state *state{ nullptr };
    std::optional<std::tuple<Values...>> *values{ nullptr };

    template<class... As> auto set_value(As &&...args) noexcept -> void
    {
      values->emplace(std::forward<As>(args)...);
      state->loop.finish();
    }

    auto set_error(error &&err) noexcept -> void
    {
      state->wait_error.emplace(std::move(err));
      state->loop.finish();
    }

    auto set_error(error const &err) noexcept -> void
    {
      state->wait_error.emplace(err);
      state->loop.finish();
    }

    // stdexec adaptors (then, etc.) still advertise exception_ptr even under -fno-exceptions.
    auto set_error(std::exception_ptr const & /*exception*/) noexcept -> void
    {
      state->wait_error.emplace(error{
        .code = make_error_code(errc::unsupported),
        .detail = "sender completed with exception_ptr",
      });
      state->loop.finish();
    }

    auto set_stopped() noexcept -> void
    {
      state->stopped = true;
      state->loop.finish();
    }

    [[nodiscard]] auto get_env() const noexcept -> sync_wait_env { return sync_wait_env{ &state->loop }; }
  };

  template<class CvSender, class Continuation>
  using sync_wait_result_t = ex::__value_types_of_t<CvSender,
    sync_wait_env,
    ex::__mtransform<ex::__q<std::decay_t>, Continuation>,
    ex::__q<ex::__msingle>>;

  template<class CvSender> using sync_wait_value_tuple_t = sync_wait_result_t<CvSender, ex::__qq<std::tuple>>;

  template<class CvSender> using sync_wait_receiver_t = sync_wait_result_t<CvSender, ex::__q<sync_wait_receiver>>;

  template<class CvSender>
  concept sync_waitable_sender =
    ex::sender_in<CvSender, sync_wait_env> && ex::sender_to<CvSender, sync_wait_receiver_t<CvSender>>;

  template<sync_waitable_sender CvSender>
  auto sync_wait_impl(CvSender &&sender) -> result<std::optional<sync_wait_value_tuple_t<CvSender>>>
  {
    sync_wait_state state{};
    std::optional<sync_wait_value_tuple_t<CvSender>> values{};

    auto operation = ex::connect(std::forward<CvSender>(sender), sync_wait_receiver_t<CvSender>{ &state, &values });
    ex::start(operation);
    state.loop.run();

    if (state.wait_error) { return leaf::new_error(std::move(*state.wait_error)); }
    if (state.stopped) { return std::optional<sync_wait_value_tuple_t<CvSender>>{}; }
    return values;
  }

}// namespace detail

/// Exception-free blocking wait for vkexec senders using `set_error_t(vkexec::error)`.
///
/// - Success: `result` holds `optional` with the value tuple.
/// - Stopped: `result` holds disengaged `optional` (not an error).
/// - Failure: failed `leaf::result` carrying `vkexec::error`.
template<detail::sync_waitable_sender Sender>
[[nodiscard]] auto sync_wait(Sender &&sender) -> result<std::optional<detail::sync_wait_value_tuple_t<Sender>>>
{ return detail::sync_wait_impl(std::forward<Sender>(sender)); }

}// namespace vkexec

#endif// VKEXEC_SYNC_WAIT_HPP
