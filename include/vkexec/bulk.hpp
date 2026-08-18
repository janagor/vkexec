#ifndef VKEXEC_BULK_HPP
#define VKEXEC_BULK_HPP


#include <vkexec/detail/config.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
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

  inline auto allocate_traced_set(context &ctx, pipeline_resources &pipe, std::span<edsl::storage_trace const> buffers)
    -> VkDescriptorSet
  {
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
  pipeline_resources *pipe{ nullptr };
  std::vector<edsl::storage_trace> buffers;
  std::uint32_t local_size_x{ k_default_local_size_x };

  bulk_sender(context *host, std::uint32_t work_count, Params push, Fun fun)
    : ctx(host), shape(work_count), params(std::move(push))
  {
    edsl::trace_scope const scope;
    edsl::Int const idx = edsl::Int::param_index();
    auto push_proxy = edsl::push_constant<Params>::bind();
    fun(idx, push_proxy);
    pipe = &ctx->get_or_compile(scope, shape);
    buffers = scope.buffers();
    local_size_x = scope.local_size_x();
  }

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::uint32_t shape{};
    Params params{};
    pipeline_resources *pipe{ nullptr };
    std::vector<edsl::storage_trace> buffers;
    std::uint32_t local_size_x{ k_default_local_size_x };
    Receiver receiver;

    void start() noexcept
    {
      std::exception_ptr error;
      VKEXEC_TRY
      {
        // cppcheck-suppress throwInNoexceptFunction
        run();
      }
      VKEXEC_CATCH_ALL
      {
        error = std::current_exception();
      }
      if (error) {
        ex::set_error(std::move(receiver), error);
      } else {
        ex::set_value(std::move(receiver));
      }
    }

    void run()
    {
      VkDescriptorSet set = detail::allocate_traced_set(*ctx, *pipe, buffers);

      VkCommandBuffer cmd = ctx->allocate_command_buffer();
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cmd, &begin);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline_layout, 0, 1, &set, 0, nullptr);
      if (pipe->push_bytes > 0) { upload_push_constants(cmd, *pipe, params); }

      std::uint32_t const groups = (shape + local_size_x - 1U) / local_size_x;
      vkCmdDispatch(cmd, groups, 1, 1);
      vkEndCommandBuffer(cmd);

      ctx->submit_and_wait(cmd);
      ctx->free_command_buffer(cmd);
      vkFreeDescriptorSets(ctx->device(), pipe->descriptor_pool, 1, &set);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  { return op_state<Receiver>{ ctx, shape, params, pipe, buffers, local_size_x, std::move(receiver) }; }
};

template<typename Params, typename Fun> auto operator|(schedule_sender snd, bulk_closure<Params, Fun> closure)
{ return bulk_sender<Params, Fun>(snd.ctx, closure.shape, std::move(closure.params), std::move(closure.fun)); }

/// Submit without blocking; returns a binary semaphore signaled on compute completion.
template<typename Params, typename Fun> auto submit_async(bulk_sender<Params, Fun> sender) -> VkSemaphore
{
  VkDescriptorSet set = detail::allocate_traced_set(*sender.ctx, *sender.pipe, sender.buffers);

  VkCommandBuffer cmd = sender.ctx->allocate_command_buffer();
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sender.pipe->pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sender.pipe->pipeline_layout, 0, 1, &set, 0, nullptr);
  if (sender.pipe->push_bytes > 0) { upload_push_constants(cmd, *sender.pipe, sender.params); }
  std::uint32_t const groups = (sender.shape + sender.local_size_x - 1U) / sender.local_size_x;
  vkCmdDispatch(cmd, groups, 1, 1);
  vkEndCommandBuffer(cmd);

  return sender.ctx->submit_async(cmd);
}

}// namespace vkexec

#endif// VKEXEC_BULK_HPP
