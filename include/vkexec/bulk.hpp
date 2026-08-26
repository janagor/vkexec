#ifndef VKEXEC_BULK_HPP
#define VKEXEC_BULK_HPP

#include <vkexec/error.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/config.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  struct bulk_traced_state
  {
    pipeline_resources *pipe{ nullptr };
    std::vector<edsl::storage_trace> buffers;
    std::uint32_t local_size_x{ k_default_local_size_x };
  };

  template<typename Params, typename Fun>
  auto trace_bulk_kernel(context &ctx, std::uint32_t shape, Fun &fun) -> result<bulk_traced_state>
  {
    edsl::trace_scope const scope;
    edsl::Int const idx = edsl::Int::param_index();
    auto push_proxy = edsl::push_constant<Params>::bind();
    fun(idx, push_proxy);
    auto const pipe = ctx.get_or_compile(scope, shape);
    if (!pipe) { return std::unexpected(pipe.error()); }
    return bulk_traced_state{
      .pipe = &pipe->get(),
      .buffers = scope.buffers(),
      .local_size_x = scope.local_size_x(),
    };
  }

  template<typename Params, typename Fun>
  auto record_bulk_into(submit_scope &scope, std::uint32_t shape, Params const &params, Fun &fun) -> status
  {
    auto const traced = trace_bulk_kernel<Params>(*scope.ctx, shape, fun);
    if (!traced) { return std::unexpected(traced.error()); }
    auto const set = allocate_compute_set(*scope.ctx, *traced->pipe, traced->buffers);
    if (!set) { return std::unexpected(set.error()); }
    scope.track_set(*traced->pipe, *set);

    vkCmdBindPipeline(scope.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced->pipe->pipeline);
    vkCmdBindDescriptorSets(
      scope.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced->pipe->pipeline_layout, 0, 1, &*set, 0, nullptr);
    if (traced->pipe->push_bytes > 0) { upload_push_constants(scope.cmd, *traced->pipe, params); }

    std::uint32_t const groups = (shape + traced->local_size_x - 1U) / traced->local_size_x;
    vkCmdDispatch(scope.cmd, groups, 1, 1);
    return {};
  }

  template<typename Params, typename Fun>
  [[nodiscard]] auto open_and_record_bulk(context *ctx, std::uint32_t shape, Params &params, Fun &fun)
    -> result<submit_scope>
  {
    auto opened = submit_scope::open(*ctx);
    if (!opened) { return std::unexpected(opened.error()); }

    submit_scope scope = std::move(*opened);
    if (auto const recorded = record_bulk_into<Params>(scope, shape, params, fun); !recorded) {
      scope.release();
      return std::unexpected(recorded.error());
    }
    if (auto const ended = scope.end_recording(); !ended) {
      scope.release();
      return std::unexpected(ended.error());
    }
    return scope;
  }

}// namespace detail

template<typename Params, typename Fun> struct bulk_closure
{
  std::uint32_t shape{};
  Params params{};
  Fun fun{};
};

template<typename Params, typename Fun> auto bulk(std::uint32_t shape, Params params, Fun fun)
{ return bulk_closure<Params, Fun>{ shape, std::move(params), std::move(fun) }; }

template<typename Params, typename Fun> struct bulk_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::uint32_t shape{};
  Params params{};
  Fun fun{};

  bulk_sender(context *host, std::uint32_t work_count, Params push, Fun kernel)
    : ctx(host), shape(work_count), params(std::move(push)), fun(std::move(kernel))
  {}

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::uint32_t shape{};
    Params params{};
    Fun fun{};
    Receiver receiver;
    using submit_op_t = decltype(ex::connect(std::declval<detail::submit_and_wait_sender>(), std::declval<Receiver>()));
    std::optional<submit_op_t> submit_op;

    auto start() noexcept -> void
    {
      Receiver rcvr = std::move(receiver);
      auto const token = ex::get_stop_token(ex::get_env(rcvr));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(rcvr));
          return;
        }
      }

      auto prepared = detail::open_and_record_bulk<Params>(ctx, shape, params, fun);
      if (!prepared) {
        ex::set_error(std::move(rcvr), std::move(prepared).error());
        return;
      }

      submit_op.emplace(ex::connect(detail::submit_and_wait(std::move(*prepared)), std::move(rcvr)));
      ex::start(*submit_op);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .shape = self.shape,
      .params = std::forward_like<decltype(self)>(self.params),
      .fun = std::forward_like<decltype(self)>(self.fun),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

template<class Pred, typename Params, typename Fun> struct bulk_adaptor_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = bulk_sender<Params, Fun>::completion_signatures;

  Pred pred;
  bulk_closure<Params, Fun> closure;

  [[nodiscard]] auto get_env() const noexcept -> decltype(auto) { return ex::get_env(pred); }
};

template<class Pred, typename Params, typename Fun, class Env>
[[nodiscard]] auto
  lower_vkexec_sender(ex::set_value_t /*tag*/, bulk_adaptor_sender<Pred, Params, Fun> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  context *const ctx = sched.get_context();
  return ex::let_value(std::move(sndr.pred),
    [ctx, shape = sndr.closure.shape, params = std::move(sndr.closure.params), fun = std::move(sndr.closure.fun)](
      auto &&...) mutable -> bulk_sender<Params, Fun> {
      return bulk_sender<Params, Fun>(ctx, shape, std::move(params), std::move(fun));
    });
}

template<typename Params, typename Fun> struct bulk_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::uint32_t shape{};
  Params params{};
  Fun fun{};

  bulk_async_sender(context *host, std::uint32_t work_count, Params push, Fun kernel)
    : ctx(host), shape(work_count), params(std::move(push)), fun(std::move(kernel))
  {}

  explicit bulk_async_sender(bulk_sender<Params, Fun> snd)
    : ctx(snd.ctx), shape(snd.shape), params(std::move(snd.params)), fun(std::move(snd.fun))
  {}

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::uint32_t shape{};
    Params params{};
    Fun fun{};
    Receiver receiver;
    using submit_op_t = decltype(ex::connect(std::declval<detail::submit_fence_sender>(), std::declval<Receiver>()));
    std::optional<submit_op_t> submit_op;

    auto start() noexcept -> void
    {
      Receiver rcvr = std::move(receiver);
      auto const token = ex::get_stop_token(ex::get_env(rcvr));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(rcvr));
          return;
        }
      }

      auto prepared = detail::open_and_record_bulk<Params>(ctx, shape, params, fun);
      if (!prepared) {
        ex::set_error(std::move(rcvr), std::move(prepared).error());
        return;
      }

      submit_op.emplace(ex::connect(detail::submit_fence(std::move(*prepared)), std::move(rcvr)));
      ex::start(*submit_op);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = self.ctx,
      .shape = self.shape,
      .params = std::forward_like<decltype(self)>(self.params),
      .fun = std::forward_like<decltype(self)>(self.fun),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

template<class Pred, typename Params, typename Fun> struct bulk_async_adaptor_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = bulk_async_sender<Params, Fun>::completion_signatures;

  Pred pred;
  bulk_closure<Params, Fun> closure;

  [[nodiscard]] auto get_env() const noexcept -> decltype(auto) { return ex::get_env(pred); }
};

template<class Pred, typename Params, typename Fun, class Env>
[[nodiscard]] auto
  lower_vkexec_sender(ex::set_value_t /*tag*/, bulk_async_adaptor_sender<Pred, Params, Fun> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  context *const ctx = sched.get_context();
  return ex::let_value(std::move(sndr.pred),
    [ctx, shape = sndr.closure.shape, params = std::move(sndr.closure.params), fun = std::move(sndr.closure.fun)](
      auto &&...) mutable -> bulk_async_sender<Params, Fun> {
      return bulk_async_sender<Params, Fun>(ctx, shape, std::move(params), std::move(fun));
    });
}

/// Pipe after `bulk` to submit without blocking `start()`.
/// The context completion agent delivers the receiver once the GPU fence signals (or with `set_stopped`).
template<typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_sender<Params, Fun> &&snd, submit_t /*tag*/) -> bulk_async_sender<Params, Fun>
{ return bulk_async_sender<Params, Fun>(std::move(snd)); }

template<typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_sender<Params, Fun> &snd, submit_t /*tag*/) -> bulk_async_sender<Params, Fun>
{ return bulk_async_sender<Params, Fun>{ snd.ctx, snd.shape, snd.params, snd.fun }; }

template<class Pred, typename Params, typename Fun>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
[[nodiscard]] auto operator|(bulk_adaptor_sender<Pred, Params, Fun> &&snd, submit_t /*tag*/)
  -> bulk_async_adaptor_sender<Pred, Params, Fun>
{
  return bulk_async_adaptor_sender<Pred, Params, Fun>{
    .pred = std::move(snd.pred),
    .closure = std::move(snd.closure),
  };
}

template<class Pred, typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_adaptor_sender<Pred, Params, Fun> &snd, submit_t /*tag*/)
  -> bulk_async_adaptor_sender<Pred, Params, Fun>
{
  return bulk_async_adaptor_sender<Pred, Params, Fun>{
    .pred = snd.pred,
    .closure = snd.closure,
  };
}

template<typename Params, typename Fun>
[[nodiscard]] auto operator|(schedule_sender snd, bulk_closure<Params, Fun> closure) -> bulk_sender<Params, Fun>
{ return bulk_sender<Params, Fun>(snd.ctx, closure.shape, std::move(closure.params), std::move(closure.fun)); }

template<vkexec_predecessor Pred, typename Params, typename Fun>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>)
[[nodiscard]] auto operator|(Pred &&pred, bulk_closure<Params, Fun> closure)
  -> bulk_adaptor_sender<std::remove_cvref_t<Pred>, Params, Fun>
{
  return bulk_adaptor_sender<std::remove_cvref_t<Pred>, Params, Fun>{
    .pred = std::forward<Pred>(pred),
    .closure = std::move(closure),
  };
}

}// namespace vkexec

#endif// VKEXEC_BULK_HPP
