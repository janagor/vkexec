#ifndef VKEXEC_GRAPHICS_GRAPHICS_HPP
#define VKEXEC_GRAPHICS_GRAPHICS_HPP


#include <vkexec/context.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexec {

constexpr float k_default_clear_r = 0.08F;
constexpr float k_default_clear_g = 0.09F;
constexpr float k_default_clear_b = 0.12F;
constexpr float k_default_clear_a = 1.0F;

struct graphics_pipeline_config
{
  VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
  bool alpha_blend{ false };
  float clear_r{ k_default_clear_r };
  float clear_g{ k_default_clear_g };
  float clear_b{ k_default_clear_b };
  float clear_a{ k_default_clear_a };
};

/// Graphics pipeline built by tracing vertex/fragment eDSL lambdas to GLSL/SPIR-V.
class graphics_pipeline
{
public:
  template<typename VertexFn, typename FragmentFn>
  graphics_pipeline(context &ctx,
    VkRenderPass render_pass,
    graphics_pipeline_config cfg,
    VertexFn &&vertex_fn,
    FragmentFn &&fragment_fn)
    : device_(ctx.device()), cfg_(cfg)
  { build(ctx, render_pass, std::forward<VertexFn>(vertex_fn), std::forward<FragmentFn>(fragment_fn)); }

  template<typename VertexFn, typename FragmentFn>
  graphics_pipeline(context &ctx, VkRenderPass render_pass, VertexFn &&vertex_fn, FragmentFn &&fragment_fn)
    : graphics_pipeline(ctx,
        render_pass,
        graphics_pipeline_config{},
        std::forward<VertexFn>(vertex_fn),
        std::forward<FragmentFn>(fragment_fn))
  {}

  ~graphics_pipeline() { destroy(); }

  graphics_pipeline(graphics_pipeline const &) = delete;
  auto operator=(graphics_pipeline const &) -> graphics_pipeline & = delete;

  graphics_pipeline(graphics_pipeline &&other) noexcept
    : device_(other.device_), cfg_(other.cfg_), layout_(other.layout_), pipeline_(other.pipeline_),
      set_layout_(other.set_layout_), descriptor_pool_(other.descriptor_pool_), descriptor_set_(other.descriptor_set_),
      buffers_(std::move(other.buffers_))
  { other.release(); }

  auto operator=(graphics_pipeline &&other) noexcept -> graphics_pipeline &
  {
    if (this == &other) { return *this; }
    destroy();
    device_ = other.device_;
    cfg_ = other.cfg_;
    layout_ = other.layout_;
    pipeline_ = other.pipeline_;
    set_layout_ = other.set_layout_;
    descriptor_pool_ = other.descriptor_pool_;
    descriptor_set_ = other.descriptor_set_;
    buffers_ = std::move(other.buffers_);
    other.release();
    return *this;
  }

  [[nodiscard]] auto pipeline() const noexcept -> VkPipeline { return pipeline_; }
  [[nodiscard]] auto config() const noexcept -> graphics_pipeline_config const & { return cfg_; }

  /// Bind pipeline, descriptors, viewport/scissor, and issue `vkCmdDraw` (no render-pass management).
  auto record_draw(VkCommandBuffer cmd, VkExtent2D extent, std::uint32_t vertex_count) const -> void
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    if (descriptor_set_ != VK_NULL_HANDLE) {
      std::vector<VkDescriptorBufferInfo> infos(buffers_.size());
      std::vector<VkWriteDescriptorSet> writes(buffers_.size());
      for (std::size_t index = 0; index < buffers_.size(); ++index) {
        infos.at(index).buffer = buffers_.at(index).buffer;
        infos.at(index).offset = 0;
        infos.at(index).range = buffers_.at(index).byte_size;
        writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes.at(index).dstSet = descriptor_set_;
        writes.at(index).dstBinding = buffers_.at(index).binding;
        writes.at(index).descriptorCount = 1;
        writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes.at(index).pBufferInfo = &infos.at(index);
      }
      vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &descriptor_set_, 0, nullptr);
    }

    VkViewport viewport{};
    viewport.x = 0.0F;
    viewport.y = 0.0F;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { .x = 0, .y = 0 };
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, vertex_count, 1, 0, 0);
  }

  /// Begin the render pass, bind this pipeline, draw `vertex_count` verts, end the pass, and end the cmd buffer.
  auto draw(VkCommandBuffer cmd,
    VkRenderPass render_pass,
    VkFramebuffer framebuffer,
    VkExtent2D extent,
    std::uint32_t vertex_count) const -> void
  {
    VkClearValue clear{};
    clear.color = { { cfg_.clear_r, cfg_.clear_g, cfg_.clear_b, cfg_.clear_a } };

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = { .x = 0, .y = 0 };
    rp_begin.renderArea.extent = extent;
    rp_begin.clearValueCount = 1;
    rp_begin.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    record_draw(cmd, extent, vertex_count);
    vkCmdEndRenderPass(cmd);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) { throw std::runtime_error("vkEndCommandBuffer failed"); }
  }

private:
  struct bound_buffer
  {
    std::uint32_t binding{ 0 };
    VkBuffer buffer{ VK_NULL_HANDLE };
    VkDeviceSize byte_size{ 0 };
  };

  auto destroy() noexcept -> void
  {
    if (device_ == VK_NULL_HANDLE) { return; }
    if (pipeline_ != VK_NULL_HANDLE) { vkDestroyPipeline(device_, pipeline_, nullptr); }
    if (layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, layout_, nullptr); }
    if (set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr); }
    if (descriptor_pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr); }
  }

  auto release() noexcept -> void
  {
    device_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    descriptor_pool_ = VK_NULL_HANDLE;
    descriptor_set_ = VK_NULL_HANDLE;
  }

  template<typename VertexFn, typename FragmentFn>
  auto build(context &ctx, VkRenderPass render_pass, VertexFn &&vertex_fn, FragmentFn &&fragment_fn) -> void
  {
    std::vector<std::uint32_t> vs_spv;
    std::vector<edsl::storage_trace> vs_buffers;
    {
      edsl::trace_scope const vertex_trace;
      edsl::Int const vertex_id = edsl::Int::vertex_index();
      edsl::VertexWriter const vertex_out;
      std::forward<VertexFn>(vertex_fn)(vertex_id, vertex_out);
      vs_spv = edsl::compile_vertex_spirv(vertex_trace);
      vs_buffers = vertex_trace.buffers();
    }
    std::vector<std::uint32_t> fs_spv;
    {
      edsl::trace_scope const fragment_trace;
      edsl::FragmentReader const fragment_in;
      edsl::FragmentWriter const fragment_out;
      std::forward<FragmentFn>(fragment_fn)(fragment_in, fragment_out);
      fs_spv = edsl::compile_fragment_spirv(fragment_trace);
    }
    complete(ctx, render_pass, vs_spv, fs_spv, vs_buffers);
  }

  auto complete(context &ctx,
    VkRenderPass render_pass,
    std::vector<std::uint32_t> const &vs_spv,
    std::vector<std::uint32_t> const &fs_spv,
    std::span<edsl::storage_trace const> buffers) -> void;

  [[nodiscard]] auto create_module(std::vector<std::uint32_t> const &spirv) const -> VkShaderModule;

  VkDevice device_{ VK_NULL_HANDLE };
  graphics_pipeline_config cfg_{};
  VkPipelineLayout layout_{ VK_NULL_HANDLE };
  VkPipeline pipeline_{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout_{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool_{ VK_NULL_HANDLE };
  VkDescriptorSet descriptor_set_{ VK_NULL_HANDLE };
  std::vector<bound_buffer> buffers_;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_GRAPHICS_HPP
