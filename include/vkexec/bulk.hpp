#ifndef VKEXEC_BULK_HPP
#define VKEXEC_BULK_HPP

#include <vkexec/buffer.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/detail/submit_scope.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <exception>
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
  auto record_bulk_dispatch(context &ctx, std::uint32_t shape, Params const &params, Fun &fun) -> submit_scope
  {
    bulk_traced_state const traced = trace_bulk_kernel<Params>(ctx, shape, fun);
    submit_scope scope = submit_scope::open(ctx);
    VkDescriptorSet set = allocate_compute_set(ctx, *traced.pipe, traced.buffers);
    scope.track_set(*traced.pipe, set);

    vkCmdBindPipeline(scope.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced.pipe->pipeline);
    vkCmdBindDescriptorSets(
      scope.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced.pipe->pipeline_layout, 0, 1, &set, 0, nullptr);
    if (traced.pipe->push_bytes > 0) { upload_push_constants(scope.cmd, *traced.pipe, params); }

    std::uint32_t const groups = (shape + traced.local_size_x - 1U) / traced.local_size_x;
    vkCmdDispatch(scope.cmd, groups, 1, 1);
    scope.end_recording();
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
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

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

    void start() noexcept
    {
      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        run();
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
    }

    void run()
    {
      detail::submit_scope scope = detail::record_bulk_dispatch<Params>(*ctx, shape, params, fun);
      ctx->submit_and_wait(scope.cmd);
      scope.release();
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      self.ctx,
      self.shape,
      std::forward_like<decltype(self)>(self.params),
      std::forward_like<decltype(self)>(self.fun),
      std::move(receiver),
    };
  }
};

template<typename Params, typename Fun> struct bulk_async_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(),
    ex::set_error_t(std::exception_ptr),
    ex::set_stopped_t()>;

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

    void start() noexcept
    {
      Receiver rcvr = std::move(receiver);
      auto const token = ex::get_stop_token(ex::get_env(rcvr));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(rcvr));
          return;
        }
      }

      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        detail::submit_scope scope = detail::record_bulk_dispatch<Params>(*ctx, shape, params, fun);
        VkFence fence{ VK_NULL_HANDLE };
        VkSemaphore done = ctx->submit_async(scope.cmd, &fence);// NOLINT(misc-misplaced-const)
        ctx->enqueue_fence_wait(done,
          fence,
          token,
          [scope = std::move(scope), rcvr = std::move(rcvr)](std::exception_ptr wait_error, bool stopped) mutable -> void {
            scope.release();
            if (wait_error) {
              ex::set_error(std::move(rcvr), wait_error);
            } else if (stopped) {
              ex::set_stopped(std::move(rcvr));
            } else {
              ex::set_value(std::move(rcvr));
            }
          });
        return;
      }
      VKEXEC_CATCH_ALL { error = std::current_exception(); }
      if (error) { ex::set_error(std::move(rcvr), error); }
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
  {
    return op_state<Receiver>{
      self.ctx,
      self.shape,
      std::forward_like<decltype(self)>(self.params),
      std::forward_like<decltype(self)>(self.fun),
      std::move(receiver),
    };
  }
};

/// Pipe after `bulk` to submit without blocking `start()`.
/// The context completion agent delivers the receiver once the GPU fence signals (or with `set_stopped`).
template<typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_sender<Params, Fun> &&snd, submit_t /*tag*/) -> bulk_async_sender<Params, Fun>
{ return bulk_async_sender<Params, Fun>(std::move(snd)); }

template<typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_sender<Params, Fun> &snd, submit_t /*tag*/) -> bulk_async_sender<Params, Fun>
{ return bulk_async_sender<Params, Fun>{ snd.ctx, snd.shape, snd.params, snd.fun }; }

template<typename Params, typename Fun> auto operator|(schedule_sender snd, bulk_closure<Params, Fun> closure)
{ return bulk_sender<Params, Fun>(snd.ctx, closure.shape, std::move(closure.params), std::move(closure.fun)); }

}// namespace vkexec

#endif// VKEXEC_BULK_HPP
