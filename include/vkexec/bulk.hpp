#pragma once

#include <vkexec/buffer.hpp>
#include <vkexec/detail/glsl_emit.hpp>
#include <vkexec/detail/pipeline_cache.hpp>
#include <vkexec/detail/push_constant.hpp>
#include <vkexec/scheduler.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

template<typename Params, typename Fun>
struct bulk_closure {
  std::uint32_t shape{};
  Params params{};
  Fun fun{};
};

template<typename Params, typename Fun>
auto bulk(std::uint32_t shape, Params params, Fun fun)
{
  return bulk_closure<Params, Fun>{ shape, std::move(params), std::move(fun) };
}

template<typename Params, typename Fun>
struct bulk_sender {
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

  context *ctx{ nullptr };
  std::uint32_t shape{};
  Params params{};
  Fun fun{};

  template<class Receiver>
  struct op_state {
    context *ctx;
    std::uint32_t shape;
    Params params;
    Fun fun;
    Receiver receiver;

    void start() noexcept
    {
      try {
        run();
        ex::set_value(std::move(receiver));
      } catch (...) {
        ex::set_error(std::move(receiver), std::current_exception());
      }
    }

    void run()
    {
      vlk::ASTContext ast_ctx;
      {
        vlk::ASTScope scope(ast_ctx);
        vlk::Int idx = vlk::Int::param_index();
        auto pc = vlk::PushConstant<Params>::bind();
        fun(idx, pc);
      }

      auto &pipe = ctx->pipeline_cache().get_or_compile(ast_ctx, shape);

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
      for (std::size_t i = 0; i < ast_ctx.buffers.size(); ++i) {
        auto *vkbuf = static_cast<VkBuffer>(ast_ctx.buffers[i].vk_buffer);
        buf_infos[i].buffer = vkbuf;
        buf_infos[i].offset = 0;
        buf_infos[i].range = ast_ctx.buffers[i].byte_size;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<std::uint32_t>(ast_ctx.buffers[i].binding);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
      }
      if (!writes.empty()) { vkUpdateDescriptorSets(ctx->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr); }

      VkCommandBuffer cmd = ctx->allocate_command_buffer();
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cmd, &begin);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout, 0, 1, &set, 0, nullptr);
      if (pipe.push_bytes > 0) {
        vkCmdPushConstants(cmd, pipe.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
          static_cast<std::uint32_t>(sizeof(Params)), &params);
      }

      const std::uint32_t local = static_cast<std::uint32_t>(ast_ctx.local_size_x);
      const std::uint32_t groups = (shape + local - 1u) / local;
      vkCmdDispatch(cmd, groups, 1, 1);
      vkEndCommandBuffer(cmd);

      ctx->submit_and_wait(cmd);
      ctx->free_command_buffer(cmd);
      vkFreeDescriptorSets(ctx->device(), pipe.descriptor_pool, 1, &set);
    }
  };

  template<class Receiver>
  auto connect(Receiver receiver) const
  {
    return op_state<Receiver>{ ctx, shape, params, fun, std::move(receiver) };
  }
};

template<typename Params, typename Fun>
auto operator|(schedule_sender snd, bulk_closure<Params, Fun> cl)
{
  return bulk_sender<Params, Fun>{ snd.ctx, cl.shape, std::move(cl.params), std::move(cl.fun) };
}

/// Submit without blocking; returns a binary semaphore signaled on compute completion.
template<typename Params, typename Fun>
VkSemaphore submit_async(bulk_sender<Params, Fun> sender)
{
  vlk::ASTContext ast_ctx;
  {
    vlk::ASTScope scope(ast_ctx);
    vlk::Int idx = vlk::Int::param_index();
    auto pc = vlk::PushConstant<Params>::bind();
    sender.fun(idx, pc);
  }

  auto &pipe = sender.ctx->pipeline_cache().get_or_compile(ast_ctx, sender.shape);

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
  for (std::size_t i = 0; i < ast_ctx.buffers.size(); ++i) {
    buf_infos[i].buffer = static_cast<VkBuffer>(ast_ctx.buffers[i].vk_buffer);
    buf_infos[i].offset = 0;
    buf_infos[i].range = ast_ctx.buffers[i].byte_size;
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = static_cast<std::uint32_t>(ast_ctx.buffers[i].binding);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buf_infos[i];
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
  if (pipe.push_bytes > 0) {
    vkCmdPushConstants(cmd, pipe.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
      static_cast<std::uint32_t>(sizeof(Params)), &sender.params);
  }
  const std::uint32_t local = static_cast<std::uint32_t>(ast_ctx.local_size_x);
  const std::uint32_t groups = (sender.shape + local - 1u) / local;
  vkCmdDispatch(cmd, groups, 1, 1);
  vkEndCommandBuffer(cmd);

  return sender.ctx->submit_async(cmd);
}

} // namespace vkexec
