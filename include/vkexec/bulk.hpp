#ifndef VKEXEC_BULK_HPP
#define VKEXEC_BULK_HPP

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
  auto trace_bulk_kernel(context &ctx, std::uint32_t shape, Fun &fun) -> bulk_traced_state
  {
    edsl::trace_scope const scope;
    edsl::Int const idx = edsl::Int::param_index();
    auto push_proxy = edsl::push_constant<Params>::bind();
    fun(idx, push_proxy);
    return bulk_traced_state{
      .pipe = &ctx.get_or_compile(scope, shape),
      .buffers = scope.buffers(),
      .local_size_x = scope.local_size_x(),
    };
  }

  template<typename Params, typename Fun>
  auto record_bulk_into(submit_scope &scope, std::uint32_t shape, Params const &params, Fun &fun) -> void
  {
    bulk_traced_state const traced = trace_bulk_kernel<Params>(*scope.ctx, shape, fun);
    VkDescriptorSet set = allocate_compute_set(*scope.ctx, *traced.pipe, traced.buffers);
    scope.track_set(*traced.pipe, set);

    vkCmdBindPipeline(scope.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced.pipe->pipeline);
    vkCmdBindDescriptorSets(
      scope.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced.pipe->pipeline_layout, 0, 1, &set, 0, nullptr);
    if (traced.pipe->push_bytes > 0) { upload_push_constants(scope.cmd, *traced.pipe, params); }

    std::uint32_t const groups = (shape + traced.local_size_x - 1U) / traced.local_size_x;
    vkCmdDispatch(scope.cmd, groups, 1, 1);
  }

  template<typename Params, typename Fun, class SubmitFactory>
  [[nodiscard]] auto make_bulk_pipeline(context *ctx, std::uint32_t shape, Params params, Fun fun, SubmitFactory submit)
  {
    return ex::let_value(enter_submit_scope(ctx),
      [shape, params = std::move(params), fun = std::move(fun), submit = std::move(submit)](
        submit_scope &scope) mutable -> decltype(auto) {
        record_bulk_into<Params>(scope, shape, params, fun);
        scope.end_recording();
        return submit(std::move(scope));
      });
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
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::uint32_t shape{};
  Params params{};
  Fun fun{};

  bulk_sender(context *host, std::uint32_t work_count, Params push, Fun kernel)
    : ctx(host), shape(work_count), params(std::move(push)), fun(std::move(kernel))
  {}

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver)
  {
    return ex::connect(detail::make_bulk_pipeline<Params>(self.ctx,
                         self.shape,
                         std::forward_like<decltype(self)>(self.params),
                         std::forward_like<decltype(self)>(self.fun),
                         [](detail::submit_scope scope) -> detail::submit_and_wait_sender {
                           return detail::submit_and_wait(std::move(scope));
                         }),
      std::move(receiver));
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
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>;

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

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver)
  {
    return ex::connect(detail::make_bulk_pipeline<Params>(self.ctx,
                         self.shape,
                         std::forward_like<decltype(self)>(self.params),
                         std::forward_like<decltype(self)>(self.fun),
                         [](detail::submit_scope scope) -> detail::submit_fence_sender {
                           return detail::submit_fence(std::move(scope));
                         }),
      std::move(receiver));
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
