#ifndef VKEXEC_BULK_HPP
#define VKEXEC_BULK_HPP


#include <vkexec/buffer.hpp>
#include <vkexec/pipeline_cache.hpp>
#include <vkexec/push.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_edsl/push_constant.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

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

  template<class Receiver> struct op_state
  {
    context *ctx;
    std::uint32_t shape;
    Params params;
    Fun fun;
    Receiver receiver;

    void start() noexcept
    {
      std::exception_ptr error;
      try {
        run();
      } catch (...) {
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
      edsl::ASTContext ast_ctx;
      {
        edsl::ASTScope const scope(ast_ctx);
        edsl::Int const idx = edsl::Int::param_index();
        auto push = edsl::push_constant<Params>::bind();
        fun(idx, push);
      }

      auto &pipe = ctx->get_pipeline_cache().get_or_compile(ast_ctx, shape);

      VkDescriptorSetAllocateInfo dsai{};
      dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      dsai.descriptorPool = pipe.descriptor_pool;
      dsai.descriptorSetCount = 1;
      dsai.pSetLayouts = &pipe.set_layout;
      VkDescriptorSet set{ VK_NULL_HANDLE };
      if (vkAllocateDescriptorSets(ctx->device(), &dsai, &set) != VK_SUCCESS) {
        throw std::runtime_error("vkAllocateDescriptorSets failed");
      }

      std::vector<VkDescriptorBufferInfo> buf_infos(ast_ctx.buffers.size());
      std::vector<VkWriteDescriptorSet> writes(ast_ctx.buffers.size());
      for (std::size_t index = 0; index < ast_ctx.buffers.size(); ++index) {
        auto *vkbuf = static_cast<VkBuffer>(ast_ctx.buffers.at(index).vk_buffer);
        buf_infos.at(index).buffer = vkbuf;
        buf_infos.at(index).offset = 0;
        buf_infos.at(index).range = ast_ctx.buffers.at(index).byte_size;
        writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes.at(index).dstSet = set;
        writes.at(index).dstBinding = static_cast<std::uint32_t>(ast_ctx.buffers.at(index).binding);
        writes.at(index).descriptorCount = 1;
        writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes.at(index).pBufferInfo = &buf_infos.at(index);
      }
      if (!writes.empty()) {
        vkUpdateDescriptorSets(ctx->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
      }

      VkCommandBuffer cmd = ctx->allocate_command_buffer();
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cmd, &begin);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout, 0, 1, &set, 0, nullptr);
      if (pipe.push_bytes > 0) { upload_push_constants(cmd, pipe, params); }

      auto local = static_cast<std::uint32_t>(ast_ctx.local_size_x);
      std::uint32_t const groups = (shape + local - 1U) / local;
      vkCmdDispatch(cmd, groups, 1, 1);
      vkEndCommandBuffer(cmd);

      ctx->submit_and_wait(cmd);
      ctx->free_command_buffer(cmd);
      vkFreeDescriptorSets(ctx->device(), pipe.descriptor_pool, 1, &set);
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) const -> op_state<Receiver>
  { return op_state<Receiver>{ ctx, shape, params, fun, std::move(receiver) }; }
};

template<typename Params, typename Fun> auto operator|(schedule_sender snd, bulk_closure<Params, Fun> closure)
{ return bulk_sender<Params, Fun>{ snd.ctx, closure.shape, std::move(closure.params), std::move(closure.fun) }; }

/// Submit without blocking; returns a binary semaphore signaled on compute completion.
template<typename Params, typename Fun> auto submit_async(bulk_sender<Params, Fun> sender) -> VkSemaphore
{
  edsl::ASTContext ast_ctx;
  {
    edsl::ASTScope const scope(ast_ctx);
    edsl::Int const idx = edsl::Int::param_index();
    auto push = edsl::push_constant<Params>::bind();
    sender.fun(idx, push);
  }

  auto &pipe = sender.ctx->get_pipeline_cache().get_or_compile(ast_ctx, sender.shape);

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pipe.descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.set_layout;
  VkDescriptorSet set{ VK_NULL_HANDLE };
  if (vkAllocateDescriptorSets(sender.ctx->device(), &dsai, &set) != VK_SUCCESS) {
    throw std::runtime_error("vkAllocateDescriptorSets failed");
  }

  std::vector<VkDescriptorBufferInfo> buf_infos(ast_ctx.buffers.size());
  std::vector<VkWriteDescriptorSet> writes(ast_ctx.buffers.size());
  for (std::size_t index = 0; index < ast_ctx.buffers.size(); ++index) {
    buf_infos.at(index).buffer = static_cast<VkBuffer>(ast_ctx.buffers.at(index).vk_buffer);
    buf_infos.at(index).offset = 0;
    buf_infos.at(index).range = ast_ctx.buffers.at(index).byte_size;
    writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.at(index).dstSet = set;
    writes.at(index).dstBinding = static_cast<std::uint32_t>(ast_ctx.buffers.at(index).binding);
    writes.at(index).descriptorCount = 1;
    writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.at(index).pBufferInfo = &buf_infos.at(index);
  }
  if (!writes.empty()) {
    vkUpdateDescriptorSets(sender.ctx->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  VkCommandBuffer cmd = sender.ctx->allocate_command_buffer();
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout, 0, 1, &set, 0, nullptr);
  if (pipe.push_bytes > 0) { upload_push_constants(cmd, pipe, sender.params); }
  auto local = static_cast<std::uint32_t>(ast_ctx.local_size_x);
  std::uint32_t const groups = (sender.shape + local - 1U) / local;
  vkCmdDispatch(cmd, groups, 1, 1);
  vkEndCommandBuffer(cmd);

  return sender.ctx->submit_async(cmd);
}

}// namespace vkexec

#endif// VKEXEC_BULK_HPP
