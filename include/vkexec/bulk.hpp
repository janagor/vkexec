#ifndef VKEXEC_BULK_HPP
#define VKEXEC_BULK_HPP


#include <vkexec/buffer.hpp>
#include <vkexec/detail/fence_wait.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_async.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  inline auto
    write_traced_descriptors(VkDevice device, VkDescriptorSet set, std::span<edsl::storage_trace const> buffers) -> void
  {
    if (buffers.empty()) { return; }
    std::vector<VkDescriptorBufferInfo> buf_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    std::size_t index = 0;
    for (edsl::storage_trace const &buffer : buffers) {
      buf_infos.at(index).buffer = static_cast<VkBuffer>(buffer.vk_buffer);
      buf_infos.at(index).offset = 0;
      buf_infos.at(index).range = buffer.byte_size;
      writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes.at(index).dstSet = set;
      writes.at(index).dstBinding = static_cast<std::uint32_t>(buffer.binding);
      writes.at(index).descriptorCount = 1;
      writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes.at(index).pBufferInfo = &buf_infos.at(index);
      ++index;
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  inline auto allocate_traced_set(context const &ctx,
    pipeline_resources &pipe,
    std::span<edsl::storage_trace const> buffers) -> VkDescriptorSet
  {
    std::unique_lock const lock = ctx.lock_host();
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pipe.descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pipe.set_layout;
    VkDescriptorSet set{ VK_NULL_HANDLE };
    if (vkAllocateDescriptorSets(ctx.device(), &dsai, &set) != VK_SUCCESS) {
      VKEXEC_THROW(std::runtime_error("vkAllocateDescriptorSets failed"));
    }
    write_traced_descriptors(ctx.device(), set, buffers);
    return set;
  }

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

  struct bulk_gpu_work
  {
    context *ctx{ nullptr };
    pipeline_resources *pipe{ nullptr };
    VkCommandBuffer cmd{ VK_NULL_HANDLE };
    VkDescriptorSet set{ VK_NULL_HANDLE };
  };

  inline auto release_bulk_gpu_work(bulk_gpu_work const &work) noexcept -> void
  {
    if (work.ctx == nullptr) { return; }
    work.ctx->free_command_buffer(work.cmd);
    if (work.set != VK_NULL_HANDLE && work.pipe != nullptr) {
      std::unique_lock const lock = work.ctx->lock_host();
      vkFreeDescriptorSets(work.ctx->device(), work.pipe->descriptor_pool, 1, &work.set);
    }
  }

  template<typename Params, typename Fun>
  auto record_bulk_dispatch(context &ctx, std::uint32_t shape, Params const &params, Fun &fun) -> bulk_gpu_work
  {
    bulk_traced_state const traced = trace_bulk_kernel<Params>(ctx, shape, fun);
    VkDescriptorSet set = allocate_traced_set(ctx, *traced.pipe, traced.buffers);

    VkCommandBuffer cmd = ctx.allocate_command_buffer();
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
      {
        std::unique_lock const lock = ctx.lock_host();
        vkFreeDescriptorSets(ctx.device(), traced.pipe->descriptor_pool, 1, &set);
      }
      ctx.free_command_buffer(cmd);
      VKEXEC_THROW(std::runtime_error("vkBeginCommandBuffer failed"));
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced.pipe->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, traced.pipe->pipeline_layout, 0, 1, &set, 0, nullptr);
    if (traced.pipe->push_bytes > 0) { upload_push_constants(cmd, *traced.pipe, params); }

    std::uint32_t const groups = (shape + traced.local_size_x - 1U) / traced.local_size_x;
    vkCmdDispatch(cmd, groups, 1, 1);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
      {
        std::unique_lock const lock = ctx.lock_host();
        vkFreeDescriptorSets(ctx.device(), traced.pipe->descriptor_pool, 1, &set);
      }
      ctx.free_command_buffer(cmd);
      VKEXEC_THROW(std::runtime_error("vkEndCommandBuffer failed"));
    }

    return bulk_gpu_work{ .ctx = &ctx, .pipe = traced.pipe, .cmd = cmd, .set = set };
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
      detail::bulk_gpu_work const work = detail::record_bulk_dispatch<Params>(*ctx, shape, params, fun);
      VKEXEC_TRY { ctx->submit_and_wait(work.cmd); }
      VKEXEC_CATCH_ALL
      {
        detail::release_bulk_gpu_work(work);
        // cppcheck-suppress rethrowNoCurrentException
        throw;
      }
      detail::release_bulk_gpu_work(work);
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
    std::optional<std::jthread> waiter;

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
        detail::bulk_gpu_work const work = detail::record_bulk_dispatch<Params>(*ctx, shape, params, fun);
        VkFence fence{ VK_NULL_HANDLE };
        VkSemaphore done = ctx->submit_async(work.cmd, &fence);// NOLINT(misc-misplaced-const)
        waiter.emplace([work, done, fence, token, rcvr = std::move(rcvr)]() mutable -> void {
          std::exception_ptr wait_error;
          bool stopped = false;
          VKEXEC_TRY
          {
            stopped = detail::wait_submission_with_stop(*work.ctx, done, fence, token);
          }
          VKEXEC_CATCH_ALL { wait_error = std::current_exception(); }
          detail::release_bulk_gpu_work(work);
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
      std::nullopt,
    };
  }
};

/// Pipe after `bulk` to submit without blocking `start()`.
/// A waiter thread completes the receiver once the GPU fence signals (or with `set_stopped`).
template<typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_sender<Params, Fun> &&snd, submit_async_t /*tag*/) -> bulk_async_sender<Params, Fun>
{ return bulk_async_sender<Params, Fun>(std::move(snd)); }

template<typename Params, typename Fun>
[[nodiscard]] auto operator|(bulk_sender<Params, Fun> &snd, submit_async_t /*tag*/) -> bulk_async_sender<Params, Fun>
{ return bulk_async_sender<Params, Fun>{ snd.ctx, snd.shape, snd.params, snd.fun }; }

template<typename Params, typename Fun> auto operator|(schedule_sender snd, bulk_closure<Params, Fun> closure)
{ return bulk_sender<Params, Fun>(snd.ctx, closure.shape, std::move(closure.params), std::move(closure.fun)); }

}// namespace vkexec

#endif// VKEXEC_BULK_HPP
